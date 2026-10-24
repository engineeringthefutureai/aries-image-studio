// tonic::Status is the error type of every gRPC call; boxing it buys nothing here.
#![allow(clippy::result_large_err)]

use crate::config::bearer_matches;
use crate::db::{
    now_s, AgentRecord, AttemptFailure, ExecutorRecord, ExecutorState, LimitDecl, ModelDecl,
    ProvenanceRecord, RequirementDecl, StageCapDecl,
};
use crate::proto;
use crate::proto::agent_gateway_server::AgentGateway;
use crate::scheduler::{proto_to_stage, run_scheduler_tick, tick, AppState};
use crate::scraper::scrape_single_agent;
use crate::storage::{ascii_json, inject_png_text_chunk, validate_enhanced_json};
use anyhow::Result;
use axum::body::Bytes;
use serde_json::json;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};
use tracing::{info, warn};

pub struct AgentGatewayService {
    pub state: Arc<AppState>,
}

impl AgentGatewayService {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    fn check_auth(&self, req: &Request<Streaming<proto::AgentMessage>>) -> Result<(), Status> {
        let Some(val) = req.metadata().get("authorization") else {
            return Err(Status::unauthenticated("missing authorization header"));
        };
        let Ok(val_str) = val.to_str() else {
            return Err(Status::unauthenticated("invalid authorization header"));
        };
        if !bearer_matches(val_str, &self.state.config.agent_token) {
            return Err(Status::unauthenticated("invalid bearer token"));
        }
        Ok(())
    }
}

#[tonic::async_trait]
impl AgentGateway for AgentGatewayService {
    type ConnectStream =
        Pin<Box<dyn Stream<Item = Result<proto::BackendMessage, Status>> + Send + 'static>>;

    async fn connect(
        &self,
        request: Request<Streaming<proto::AgentMessage>>,
    ) -> Result<Response<Self::ConnectStream>, Status> {
        self.check_auth(&request)?;
        let mut inbound = request.into_inner();
        let state = self.state.clone();

        // Read first message synchronously so we can reject invalid Register with a gRPC Status error!
        let first_msg = match inbound.next().await {
            Some(Ok(m)) => m,
            Some(Err(e)) => return Err(e),
            None => return Err(Status::invalid_argument("stream closed before Register")),
        };

        let Some(proto::agent_message::Msg::Register(reg)) = first_msg.msg else {
            return Err(Status::failed_precondition(
                "first AgentMessage must be Register",
            ));
        };

        let (agent_rec, exec_recs, session) = validate_and_parse_register(&reg)?;
        let agent_id = agent_rec.id.clone();

        state
            .db
            .upsert_agent_registration(&agent_rec, &exec_recs)
            .await
            .map_err(|e| Status::internal(format!("db error on registration: {e}")))?;

        let (tx, rx) = mpsc::unbounded_channel::<proto::BackendMessage>();
        if let Ok(mut streams) = state.agent_streams.write() {
            streams.insert(agent_id.clone(), tx.clone());
        }

        // Send Welcome immediately
        let _ = tx.send(proto::BackendMessage {
            msg: Some(proto::backend_message::Msg::Welcome(proto::Welcome {
                protocol: 1,
                session: session.clone(),
            })),
        });

        // Reconcile running tasks declared in Register (§7.3): tasks the backend has on this
        // agent but the agent no longer runs are requeued; the ones it still runs keep their
        // executor busy.
        let running: Vec<(String, u32, String)> = reg
            .running
            .iter()
            .map(|rt| (rt.task.clone(), rt.attempt, rt.executor.clone()))
            .collect();
        state
            .db
            .reconcile_agent_running_tasks(&agent_id, &running)
            .await
            .map_err(|e| Status::internal(format!("db error reconciling running tasks: {e}")))?;

        for rt in &reg.running {
            if let Ok(Some(task)) = state.db.get_task(&rt.task).await {
                if task.current_attempt != rt.attempt || !task.status.is_active() {
                    let _ = tx.send(proto::BackendMessage {
                        msg: Some(proto::backend_message::Msg::Cancel(proto::Cancel {
                            task: rt.task.clone(),
                            attempt: rt.attempt,
                        })),
                    });
                }
            } else {
                let _ = tx.send(proto::BackendMessage {
                    msg: Some(proto::backend_message::Msg::Cancel(proto::Cancel {
                        task: rt.task.clone(),
                        attempt: rt.attempt,
                    })),
                });
            }
        }

        info!(
            agent = %agent_id,
            session = %session,
            executors = exec_recs.len(),
            "Agent registered on AgentGateway"
        );

        // Perform an initial metrics scrape and scheduler tick in background
        let state_init = state.clone();
        let agent_rec_clone = agent_rec.clone();
        tokio::spawn(async move {
            let _ = scrape_single_agent(&state_init, &agent_rec_clone).await;
            tick(&state_init).await;
        });

        let state_loop = state.clone();
        let session_loop = session.clone();
        let own_tx = tx.clone();
        tokio::spawn(async move {
            while let Some(item) = inbound.next().await {
                let msg = match item {
                    Ok(m) => m,
                    Err(e) => {
                        warn!(agent = %agent_id, error = %e, "Agent stream error");
                        break;
                    }
                };
                if let Err(e) = handle_agent_message(&state_loop, &agent_id, msg).await {
                    warn!(agent = %agent_id, error = %e, "Error handling AgentMessage");
                }
            }

            // Stream ended: mark agent offline and start reconnect_grace timer (§7.3)
            let marked_offline = state_loop
                .db
                .mark_agent_disconnected(&agent_id, &session_loop)
                .await
                .unwrap_or(false);

            // Only this stream's own sender: the agent may have reconnected meanwhile, and
            // removing the new stream's sender would drop every message to it.
            if let Ok(mut streams) = state_loop.agent_streams.write() {
                if streams
                    .get(&agent_id)
                    .is_some_and(|s| s.same_channel(&own_tx))
                {
                    streams.remove(&agent_id);
                }
            }
            drop(own_tx);

            if marked_offline {
                tick(&state_loop).await;

                let grace_s = state_loop.config.reconnect_grace_s;
                let state_grace = state_loop.clone();
                let agent_grace = agent_id.clone();
                let session_grace = session_loop.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs_f64(grace_s.max(0.01))).await;
                    if let Ok(affected) = state_grace
                        .db
                        .expire_disconnected_agent_tasks(&agent_grace, &session_grace)
                        .await
                    {
                        if !affected.is_empty() {
                            tick(&state_grace).await;
                        }
                    }
                });
            }
        });

        let out_stream = UnboundedReceiverStream::new(rx).map(Ok);
        Ok(Response::new(Box::pin(out_stream)))
    }
}

fn validate_and_parse_register(
    reg: &proto::Register,
) -> Result<(AgentRecord, Vec<ExecutorRecord>, String), Status> {
    if reg.protocol != 1 {
        return Err(Status::failed_precondition(format!(
            "unsupported protocol version {} (expected 1)",
            reg.protocol
        )));
    }
    if reg.agent.trim().is_empty() {
        return Err(Status::invalid_argument("agent id cannot be empty"));
    }

    // Validate all limits: any Requirement with kind == None (unknown field from newer proto)
    // MUST be rejected with FAILED_PRECONDITION (§5.1, §7.2).
    let mut limits = Vec::new();
    for lim in &reg.limits {
        let Some(stage_str) = proto_to_stage(lim.stage) else {
            return Err(Status::failed_precondition(
                "limit has unspecified or unknown stage",
            ));
        };
        if lim.requires.is_empty() {
            return Err(Status::failed_precondition(
                "limit.requires cannot be empty",
            ));
        }
        let mut reqs = Vec::with_capacity(lim.requires.len());
        for r in &lim.requires {
            let Some(kind) = &r.kind else {
                return Err(Status::failed_precondition(
                    "unknown or unset Requirement.kind in limit (refusing registration)",
                ));
            };
            match kind {
                proto::requirement::Kind::HostRamAvailableBytes(v) => {
                    reqs.push(RequirementDecl::HostRamAvailableBytes(*v));
                }
                proto::requirement::Kind::MaxConcurrentPerAgent(v) => {
                    if *v == 0 {
                        return Err(Status::failed_precondition(
                            "max_concurrent_per_agent must be >= 1",
                        ));
                    }
                    reqs.push(RequirementDecl::MaxConcurrentPerAgent(*v));
                }
                proto::requirement::Kind::ExecutorMemAvailableBytes(v) => {
                    reqs.push(RequirementDecl::ExecutorMemAvailableBytes(*v));
                }
            }
        }
        limits.push(LimitDecl {
            stage: stage_str.to_string(),
            above_px: lim.above_px,
            requires: reqs,
            above_references: lim.above_references,
            decode_mode: lim.decode_mode.clone().filter(|m| !m.is_empty()),
        });
    }

    let models: Vec<ModelDecl> = reg
        .models
        .iter()
        .map(|m| ModelDecl {
            id: m.id.clone(),
            rev: m.rev.clone(),
            dtype: m.dtype.clone(),
        })
        .collect();

    let now = now_s();
    let session = format!("sess_{}", ulid::Ulid::new().to_string().to_lowercase());

    let mut executors = Vec::with_capacity(reg.executors.len());
    for ex in &reg.executors {
        let mut stages = Vec::with_capacity(ex.stages.len());
        for sc in &ex.stages {
            let Some(st_name) = proto_to_stage(sc.stage) else {
                continue;
            };
            let modes: Vec<String> = sc
                .modes
                .iter()
                .filter_map(|m| match proto::DecodeMode::try_from(*m).ok()? {
                    proto::DecodeMode::Untiled => Some("untiled".to_string()),
                    proto::DecodeMode::Tiled => Some("tiled".to_string()),
                    proto::DecodeMode::Unspecified => None,
                })
                .collect();
            stages.push(StageCapDecl {
                stage: st_name.to_string(),
                models: sc.models.clone(),
                max_px: sc.max_px,
                modes,
                max_references: sc.max_references,
                reference_models: sc.reference_models.clone(),
            });
        }
        executors.push(ExecutorRecord {
            key: format!("{}:{}", reg.agent, ex.id),
            agent_id: reg.agent.clone(),
            id: ex.id.clone(),
            device: ex.device.clone(),
            mem_total_bytes: ex.mem_total_bytes,
            stages,
            state: ExecutorState::Idle,
            current_stage: None,
            current_model_id: None,
            current_model_rev: None,
            current_model_dtype: None,
            current_task_id: None,
            worker_pid: None,
            load_s: None,
            disabled_reason: None,
            lent: false,
            pending_lend: false,
            idle_since: Some(now),
            last_seen_at: now,
        });
    }

    let agent_rec = AgentRecord {
        id: reg.agent.clone(),
        agent_version: reg.agent_version.clone(),
        backend_kind: reg.backend_kind.clone(),
        metrics_url: reg.metrics_url.clone(),
        host_ram_total_bytes: reg.host_ram_total_bytes,
        models,
        limits,
        online: true,
        session: Some(session.clone()),
        connected_at: Some(now),
        last_seen_at: now,
    };

    Ok((agent_rec, executors, session))
}

async fn handle_agent_message(
    state: &Arc<AppState>,
    agent_id: &str,
    msg: proto::AgentMessage,
) -> Result<()> {
    let Some(inner) = msg.msg else {
        return Ok(());
    };

    match inner {
        proto::agent_message::Msg::Register(_) => {
            warn!(agent = %agent_id, "Ignoring duplicate Register on existing stream");
        }
        proto::agent_message::Msg::ExecutorState(es) => {
            let new_state = match proto::ExecutorStateKind::try_from(es.state)
                .unwrap_or(proto::ExecutorStateKind::ExecutorStateUnspecified)
            {
                proto::ExecutorStateKind::ExecutorStateIdle => ExecutorState::Idle,
                proto::ExecutorStateKind::ExecutorStateStarting => ExecutorState::Starting,
                proto::ExecutorStateKind::ExecutorStateReady => ExecutorState::Ready,
                proto::ExecutorStateKind::ExecutorStateBusy => ExecutorState::Busy,
                proto::ExecutorStateKind::ExecutorStateStopping => ExecutorState::Stopping,
                proto::ExecutorStateKind::ExecutorStateFailed => ExecutorState::Idle,
                proto::ExecutorStateKind::ExecutorStateUnspecified => ExecutorState::Idle,
            };
            let stage_opt = proto_to_stage(es.stage);
            let pid_opt = if es.pid > 0 { Some(es.pid) } else { None };
            let load_opt = if es.load_s > 0.0 {
                Some(es.load_s)
            } else {
                None
            };
            state
                .db
                .update_executor_state(
                    agent_id,
                    &es.executor,
                    new_state,
                    stage_opt,
                    pid_opt,
                    load_opt,
                )
                .await?;
            run_scheduler_tick(state).await?;
        }
        proto::agent_message::Msg::Progress(prog) => {
            let accepted = state
                .db
                .record_task_progress(
                    &prog.task,
                    prog.attempt,
                    prog.step,
                    prog.steps,
                    prog.s_per_step,
                    (!prog.sub_stage.is_empty()).then_some(prog.sub_stage.as_str()),
                    Some(&prog.stats),
                )
                .await?;
            if !accepted {
                // Task is no longer active for this attempt (e.g. cancelled); notify agent
                state.send_to_agent(
                    agent_id,
                    proto::BackendMessage {
                        msg: Some(proto::backend_message::Msg::Cancel(proto::Cancel {
                            task: prog.task,
                            attempt: prog.attempt,
                        })),
                    },
                );
            }
        }
        proto::agent_message::Msg::Done(done) => {
            let (task, attempt) = (done.task.clone(), done.attempt);
            if let Err(e) = handle_done_message(state, agent_id, done).await {
                // Fail the attempt rather than leave the task running and its executor busy:
                // the executor's next READY would be dropped as stale. A no-op once the task
                // is done (an error while finalizing the image).
                warn!(agent = %agent_id, task = %task, attempt, error = %e, "Error handling Done");
                state
                    .db
                    .fail_task_attempt(
                        &task,
                        attempt,
                        AttemptFailure::studio(
                            "upload_failed",
                            &format!("Studio could not record the result: {e:#}"),
                        ),
                    )
                    .await?;
            }
            run_scheduler_tick(state).await?;
        }
        proto::agent_message::Msg::Failed(failed) => {
            let class_str = failure_class_to_str(failed.failure_class);
            state
                .db
                .fail_task_attempt(
                    &failed.task,
                    failed.attempt,
                    AttemptFailure {
                        class: class_str,
                        detail: &failed.detail,
                        exit_code: failed.exit_code,
                        signal: failed.signal,
                        kernel_log_tail: &failed.kernel_log_tail,
                    },
                )
                .await?;
            run_scheduler_tick(state).await?;
        }
        proto::agent_message::Msg::Rejected(rej) => {
            warn!(
                agent = %agent_id,
                task = %rej.task,
                attempt = rej.attempt,
                reason = %rej.reason,
                "Agent rejected task assignment (protocol violation)"
            );
            state
                .db
                .handle_rejected_assignment(&rej.task, rej.attempt, &rej.reason)
                .await?;
            run_scheduler_tick(state).await?;
        }
    }

    Ok(())
}

async fn handle_done_message(
    state: &Arc<AppState>,
    agent_id: &str,
    done: proto::Done,
) -> Result<()> {
    // Validate mandatory Provenance (§3.2)
    let Some(prov_proto) = &done.provenance else {
        state
            .db
            .fail_task_attempt(
                &done.task,
                done.attempt,
                AttemptFailure::studio("python_error", "Done message missing mandatory Provenance"),
            )
            .await?;
        return Ok(());
    };
    let Some(model_proto) = &prov_proto.model else {
        state
            .db
            .fail_task_attempt(
                &done.task,
                done.attempt,
                AttemptFailure::studio("python_error", "Done Provenance missing model"),
            )
            .await?;
        return Ok(());
    };

    if prov_proto.agent.is_empty()
        || prov_proto.executor.is_empty()
        || prov_proto.device.is_empty()
        || prov_proto.backend_kind.is_empty()
        || model_proto.id.is_empty()
        || model_proto.rev.is_empty()
        || model_proto.dtype.is_empty()
        || prov_proto.torch_version.is_empty()
        || prov_proto.worker_version.is_empty()
    {
        state
            .db
            .fail_task_attempt(
                &done.task,
                done.attempt,
                AttemptFailure::studio("python_error", "Done Provenance has empty required fields"),
            )
            .await?;
        return Ok(());
    }

    let Some(task) = state.db.get_task(&done.task).await? else {
        return Ok(());
    };
    if task.current_attempt != done.attempt {
        // Stale attempt fenced out (§3, §4)
        return Ok(());
    }

    // Verify expected artifact was uploaded for (task, attempt)
    let expected_art_name = match task.stage.as_str() {
        "enhance" => "enhanced.json",
        "encode" => "embeds.safetensors",
        "denoise" => "latent.safetensors",
        "decode" => "image.png",
        _ => "",
    };

    let Some(uploaded_art) = state
        .db
        .get_artifact_for_task_attempt(&done.task, done.attempt, expected_art_name)
        .await?
    else {
        state
            .db
            .fail_task_attempt(
                &done.task,
                done.attempt,
                AttemptFailure::studio(
                    "upload_failed",
                    &format!("Done received before artifact {expected_art_name} was uploaded"),
                ),
            )
            .await?;
        return Ok(());
    };

    // Encode also uploads the latents of every reference Studio had none stored for; each one
    // was recorded on upload. Without them denoise could not run.
    if task.stage == "encode" {
        if let Some(job) = state.db.get_job(&task.job_id).await? {
            for (i, r) in job.references.iter().enumerate() {
                let stored = state
                    .db
                    .get_reference_latents(
                        &r.sha256,
                        &model_proto.id,
                        &model_proto.rev,
                        state.config.reference_res,
                    )
                    .await?;
                if stored.is_none() {
                    state
                        .db
                        .fail_task_attempt(
                            &done.task,
                            done.attempt,
                            AttemptFailure::studio(
                                "upload_failed",
                                &format!(
                                    "Done received before artifact ref{i}.safetensors was uploaded"
                                ),
                            ),
                        )
                        .await?;
                    return Ok(());
                }
            }
        }
    }

    let mut enhanced_info: Option<(String, String, String, String)> = None;
    let mut encode_model_info: Option<(String, String)> = None;

    if task.stage == "enhance" {
        let bytes = state.blobs.read_blob(&uploaded_art.sha256).await?;
        let parsed = validate_enhanced_json(&bytes)?;
        enhanced_info = Some((
            parsed.rewritten_prompt,
            parsed.wh_ratio,
            parsed.thinking,
            parsed.ratio_follow,
        ));
    } else if task.stage == "encode" {
        encode_model_info = Some((model_proto.id.clone(), model_proto.rev.clone()));
    }

    let prov_record = ProvenanceRecord {
        agent: if prov_proto.agent.is_empty() {
            agent_id.to_string()
        } else {
            prov_proto.agent.clone()
        },
        executor: prov_proto.executor.clone(),
        device: prov_proto.device.clone(),
        backend_kind: prov_proto.backend_kind.clone(),
        model: ModelDecl {
            id: model_proto.id.clone(),
            rev: model_proto.rev.clone(),
            dtype: model_proto.dtype.clone(),
        },
        torch_version: prov_proto.torch_version.clone(),
        worker_version: prov_proto.worker_version.clone(),
        timings: done.timings.clone(),
        peak_mem_bytes: done.peak_mem_bytes,
    };

    let completed = state
        .db
        .complete_task_attempt(
            &done.task,
            done.attempt,
            &prov_record,
            enhanced_info
                .as_ref()
                .map(|(p, r, t, f)| (p.as_str(), r.as_str(), t.as_str(), f.as_str())),
            encode_model_info
                .as_ref()
                .map(|(id, rev)| (id.as_str(), rev.as_str())),
        )
        .await?;

    if let Some((job_id, job_finished)) = completed {
        if job_finished {
            finalize_completed_job_image(
                state,
                &job_id,
                &done.task,
                done.attempt,
                &uploaded_art.sha256,
            )
            .await?;
        }
    }

    Ok(())
}

async fn finalize_completed_job_image(
    state: &Arc<AppState>,
    job_id: &str,
    decode_task_id: &str,
    decode_attempt: u32,
    raw_png_sha256: &str,
) -> Result<()> {
    let Some(job) = state.db.get_job(job_id).await? else {
        return Ok(());
    };
    let raw_png = state.blobs.read_blob(raw_png_sha256).await?;

    let metadata_json = json!({
        "job_id": job.id,
        "prompt": job.prompt,
        "enhanced_prompt": job.enhanced_prompt,
        "negative_prompt": job.negative_prompt,
        "width": job.width,
        "height": job.height,
        "steps": job.steps,
        "seed": job.seed,
        "true_cfg_scale": job.true_cfg_scale,
        "references": job.references,
        "provenance": job.provenance,
    });
    let metadata_str = ascii_json(&metadata_json)?;

    let tagged_png = Bytes::from(inject_png_text_chunk(
        &raw_png,
        "aries-image-studio",
        &metadata_str,
    )?);
    let writing = state.blobs.writing().await;
    let (new_sha, new_bytes) = state.blobs.store_blob(tagged_png.clone()).await?;

    // Update artifact record to point to the metadata-tagged PNG
    sqlx::query(
        "UPDATE artifacts SET sha256 = ?, bytes = ?, metadata_json = ?
         WHERE task_id = ? AND attempt = ? AND name = 'image.png'",
    )
    .bind(&new_sha)
    .bind(new_bytes as i64)
    .bind(&metadata_str)
    .bind(decode_task_id)
    .bind(decode_attempt as i64)
    .execute(state.db.pool())
    .await?;

    drop(writing);

    // Another job may have decoded the same bytes: delete only what nothing points at.
    if new_sha != raw_png_sha256 {
        let _collecting = state.blobs.collecting().await;
        if !state.db.blob_in_use(raw_png_sha256).await? {
            let _ = state.blobs.delete_blob(raw_png_sha256);
        }
    }

    // Thumbnails off the async runtime; the grids fall back to generating one on request.
    {
        let blobs = state.blobs.clone();
        let job_id = job_id.to_string();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = blobs.write_thumbnails(&job_id, &tagged_png) {
                warn!(job_id, error = %e, "thumbnail generation failed");
            }
        });
    }

    // Record e2e duration metric
    if let Some(finished) = job.finished_at {
        let e2e = (finished - job.created_at).max(0.0);
        let size_label = format!("{}x{}", job.width, job.height);
        state.metrics.observe_job_e2e(&size_label, job.enhance, e2e);
    }

    // GC this job's intermediate artifacts (§7.4); the periodic pass covers the rest.
    let _collecting = state.blobs.collecting().await;
    if let Ok(orphans) = state
        .db
        .gc_intermediate_artifacts(86400.0, false, Some(job_id))
        .await
    {
        for sha in orphans {
            let _ = state.blobs.delete_blob(&sha);
        }
    }

    Ok(())
}

fn failure_class_to_str(fc: i32) -> &'static str {
    match proto::FailureClass::try_from(fc).unwrap_or(proto::FailureClass::Unspecified) {
        proto::FailureClass::FailureGpuFault => "gpu_fault",
        proto::FailureClass::FailureEngineReset => "engine_reset",
        proto::FailureClass::FailureWedged => "wedged",
        proto::FailureClass::FailureOomKill => "oom_kill",
        proto::FailureClass::FailurePythonError => "python_error",
        proto::FailureClass::FailureDeviceMismatch => "device_mismatch",
        proto::FailureClass::FailureUploadFailed => "upload_failed",
        proto::FailureClass::FailureCancelled => "cancelled",
        proto::FailureClass::Unspecified => "python_error",
    }
}
