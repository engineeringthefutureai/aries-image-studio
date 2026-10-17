use crate::config::BackendConfig;
use crate::db::{
    now_s, AgentRecord, Database, ExecutorRecord, JobRecord, JobSubmitInput, ModelDecl,
    RequirementDecl, TaskRecord,
};
use crate::metrics::{AgentLiveMetrics, BackendMetricsRegistry};
use crate::proto;
use crate::storage::BlobStore;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use tracing::warn;

pub type AgentTx = mpsc::UnboundedSender<proto::BackendMessage>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeCapabilityInfo {
    pub width: u32,
    pub height: u32,
    pub px: u64,
    pub servable_now: bool,
    pub servable_ever: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageCapabilityInfo {
    pub stage: String,
    pub servable_now: bool,
    pub servable_ever: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitiesResponse {
    pub max_px_now: u64,
    pub max_px_ever: u64,
    pub enhance_now: bool,
    pub enhance_ever: bool,
    pub sizes: Vec<SizeCapabilityInfo>,
    pub stages: Vec<StageCapabilityInfo>,
}

pub struct AppState {
    pub config: BackendConfig,
    pub db: Database,
    pub blobs: BlobStore,
    pub metrics: BackendMetricsRegistry,
    pub live_metrics: RwLock<HashMap<String, AgentLiveMetrics>>,
    pub agent_streams: RwLock<HashMap<String, AgentTx>>,
    pub sched_lock: tokio::sync::Mutex<()>,
}

impl AppState {
    pub fn new(config: BackendConfig, db: Database, blobs: BlobStore) -> Arc<Self> {
        Arc::new(Self {
            config,
            db,
            blobs,
            metrics: BackendMetricsRegistry::new(),
            live_metrics: RwLock::new(HashMap::new()),
            agent_streams: RwLock::new(HashMap::new()),
            sched_lock: tokio::sync::Mutex::new(()),
        })
    }

    pub fn send_to_agent(&self, agent_id: &str, msg: proto::BackendMessage) -> bool {
        if let Ok(streams) = self.agent_streams.read() {
            if let Some(tx) = streams.get(agent_id) {
                return tx.send(msg).is_ok();
            }
        }
        false
    }
}

const KNOWN_EXECUTOR_WINDOW_S: f64 = 7.0 * 86400.0;

/// Whether an executor declares the decode mode a job asks for (`tiled` when the job sets
/// `tiling`). Executors that declare no modes are taken to support untiled only.
pub fn executor_supports_decode_mode(ex: &ExecutorRecord, tiled: bool) -> bool {
    let want = if tiled { "tiled" } else { "untiled" };
    match ex.stages.iter().find(|s| s.stage == "decode") {
        Some(cap) if cap.modes.is_empty() => !tiled,
        Some(cap) => cap.modes.iter().any(|m| m == want),
        None => false,
    }
}

pub fn executor_supports_stage_static(
    ex: &ExecutorRecord,
    agent: &AgentRecord,
    stage: &str,
    px: u64,
    required_model: Option<(&str, &str)>,
) -> Option<ModelDecl> {
    let cap = ex.stages.iter().find(|s| s.stage == stage)?;
    if matches!(stage, "denoise" | "decode") && cap.max_px > 0 && px > cap.max_px {
        return None;
    }
    if stage == "decode" && !cap.modes.is_empty() && !cap.modes.iter().any(|m| m == "untiled") {
        return None;
    }
    for model_id in &cap.models {
        if let Some(m) = agent.models.iter().find(|m| &m.id == model_id) {
            if let Some((req_id, req_rev)) = required_model {
                if m.id == req_id && m.rev == req_rev {
                    return Some(m.clone());
                }
            } else {
                return Some(m.clone());
            }
        }
    }
    None
}

/// Composer aspect presets, as `(label, width / height)`.
const ASPECT_PRESETS: [(&str, f64); 7] = [
    ("1:1", 1.0),
    ("4:3", 4.0 / 3.0),
    ("3:4", 3.0 / 4.0),
    ("3:2", 3.0 / 2.0),
    ("2:3", 2.0 / 3.0),
    ("16:9", 16.0 / 9.0),
    ("9:16", 9.0 / 16.0),
];

/// The preset ratio closest to `width`×`height` (½ and ¼ presets are rounded, so not exact).
pub fn aspect_label(width: u32, height: u32) -> &'static str {
    let r = (width as f64 / height.max(1) as f64).ln();
    ASPECT_PRESETS
        .iter()
        .min_by(|a, b| (a.1.ln() - r).abs().total_cmp(&(b.1.ln() - r).abs()))
        .map(|p| p.0)
        .unwrap_or("1:1")
}

/// The prompt rewriter's user message. It picks its own `wh_ratio` and frames the
/// description for it ("a square-format ...") unless the user states one, so state the
/// size the job will actually render at.
pub fn enhance_request(prompt: &str, width: u32, height: u32) -> String {
    format!("{}\n\nAspect ratio: {}", prompt.trim_end(), aspect_label(width, height))
}

/// Checks submit-time feasibility against all known executors seen in the last 7 days (§6.2).
/// Denoising steps a job may ask for (the UI's custom field allows the same range).
pub const MAX_STEPS: u32 = 100;

pub async fn check_submit_feasibility(
    db: &Database,
    input: &JobSubmitInput,
) -> Result<(), String> {
    if input.prompt.trim().is_empty() {
        return Err("prompt cannot be empty".to_string());
    }
    // Qwen-Image-2.1: VAE downsamples 16×, the DiT packs 2×2 patches; other sizes get floored.
    if input.width == 0 || input.height == 0 || input.width % 32 != 0 || input.height % 32 != 0 {
        return Err(format!(
            "invalid dimensions {}×{} (must be positive multiples of 32)",
            input.width, input.height
        ));
    }
    if input.steps == 0 || input.steps > MAX_STEPS {
        return Err(format!("invalid steps {} (must be 1–{MAX_STEPS})", input.steps));
    }

    let now = now_s();
    let agents: HashMap<String, AgentRecord> = db
        .list_agents()
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|a| now - a.last_seen_at <= KNOWN_EXECUTOR_WINDOW_S)
        .map(|a| (a.id.clone(), a))
        .collect();

    let executors: Vec<ExecutorRecord> = db
        .list_executors()
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|e| now - e.last_seen_at <= KNOWN_EXECUTOR_WINDOW_S && agents.contains_key(&e.agent_id))
        .collect();

    if executors.is_empty() {
        return Err("no known executor registered in the last 7 days".to_string());
    }

    let px = (input.width as u64) * (input.height as u64);
    if input.tiling && !executors.iter().any(|ex| executor_supports_decode_mode(ex, true)) {
        return Err("no executor declares tiled decode".to_string());
    }

    if input.enhance {
        let can_enhance = executors.iter().any(|ex| {
            agents
                .get(&ex.agent_id)
                .and_then(|ag| executor_supports_stage_static(ex, ag, "enhance", px, None))
                .is_some()
        });
        if !can_enhance {
            return Err("no known executor supports stage 'enhance'".to_string());
        }
    }

    for stage in ["encode", "denoise", "decode"] {
        let can_stage = executors.iter().any(|ex| {
            agents
                .get(&ex.agent_id)
                .and_then(|ag| executor_supports_stage_static(ex, ag, stage, px, None))
                .is_some()
        });
        if !can_stage {
            let max_stage_px = executors
                .iter()
                .flat_map(|ex| ex.stages.iter())
                .filter(|s| s.stage == stage)
                .map(|s| s.max_px)
                .max()
                .unwrap_or(0);
            if max_stage_px > 0 && px > max_stage_px {
                let side = (max_stage_px as f64).sqrt() as u32;
                return Err(format!(
                    "no known executor can {stage} {}×{} (max {}×{})",
                    input.width, input.height, side, side
                ));
            }
            return Err(format!(
                "no known executor can {stage} {}×{}",
                input.width, input.height
            ));
        }
    }

    // Ensure at least one common (model_id, model_rev) across encode, denoise, decode
    let mut candidate_models = Vec::new();
    for ex in &executors {
        if let Some(ag) = agents.get(&ex.agent_id) {
            if let Some(m) = executor_supports_stage_static(ex, ag, "encode", px, None) {
                if !candidate_models.contains(&m) {
                    candidate_models.push(m);
                }
            }
        }
    }

    let has_common_model = candidate_models.iter().any(|m| {
        let req = Some((m.id.as_str(), m.rev.as_str()));
        let ok_denoise = executors.iter().any(|ex| {
            agents
                .get(&ex.agent_id)
                .and_then(|ag| executor_supports_stage_static(ex, ag, "denoise", px, req))
                .is_some()
        });
        let ok_decode = executors.iter().any(|ex| {
            executor_supports_decode_mode(ex, input.tiling)
                && agents
                    .get(&ex.agent_id)
                    .and_then(|ag| executor_supports_stage_static(ex, ag, "decode", px, req))
                    .is_some()
        });
        ok_denoise && ok_decode
    });

    if !has_common_model {
        return Err("no compatible model@rev across encode, denoise, and decode".to_string());
    }

    Ok(())
}

/// Computes `/api/capabilities` for the UI composer (`now` vs `ever`).
pub async fn compute_capabilities(db: &Database) -> Result<CapabilitiesResponse> {
    let now = now_s();
    let agents: HashMap<String, AgentRecord> = db
        .list_agents()
        .await?
        .into_iter()
        .filter(|a| now - a.last_seen_at <= KNOWN_EXECUTOR_WINDOW_S)
        .map(|a| (a.id.clone(), a))
        .collect();

    let all_executors: Vec<ExecutorRecord> = db
        .list_executors()
        .await?
        .into_iter()
        .filter(|e| now - e.last_seen_at <= KNOWN_EXECUTOR_WINDOW_S && agents.contains_key(&e.agent_id))
        .collect();

    let online_executors: Vec<&ExecutorRecord> = all_executors
        .iter()
        .filter(|e| {
            agents.get(&e.agent_id).map(|a| a.online).unwrap_or(false)
                && !matches!(e.state.as_str(), "offline" | "lent" | "disabled")
                && !e.lent
                && e.disabled_reason.is_none()
        })
        .collect();

    let can_run_stage = |execs: &[&ExecutorRecord], stage: &str, px: u64| -> bool {
        execs.iter().any(|ex| {
            agents
                .get(&ex.agent_id)
                .and_then(|ag| executor_supports_stage_static(ex, ag, stage, px, None))
                .is_some()
        })
    };

    let all_refs: Vec<&ExecutorRecord> = all_executors.iter().collect();

    let mut max_px_ever = 0u64;
    let mut max_px_now = 0u64;
    for ex in &all_executors {
        let denoise_px = ex
            .stages
            .iter()
            .find(|s| s.stage == "denoise")
            .map(|s| s.max_px)
            .unwrap_or(0);
        let decode_px = ex
            .stages
            .iter()
            .find(|s| s.stage == "decode")
            .map(|s| s.max_px)
            .unwrap_or(0);
        let eff = denoise_px.min(decode_px);
        if eff > max_px_ever {
            max_px_ever = eff;
        }
    }
    for ex in &online_executors {
        let denoise_px = ex
            .stages
            .iter()
            .find(|s| s.stage == "denoise")
            .map(|s| s.max_px)
            .unwrap_or(0);
        let decode_px = ex
            .stages
            .iter()
            .find(|s| s.stage == "decode")
            .map(|s| s.max_px)
            .unwrap_or(0);
        let eff = denoise_px.min(decode_px);
        if eff > max_px_now {
            max_px_now = eff;
        }
    }

    let enhance_ever = can_run_stage(&all_refs, "enhance", 0);
    let enhance_now = can_run_stage(&online_executors, "enhance", 0);

    // The composer's full-size aspect presets (web/src/types.ts ASPECT_RATIOS).
    let standard_sizes = [
        (2048u32, 2048u32),
        (2400, 1792),
        (1792, 2400),
        (2528, 1696),
        (1696, 2528),
        (2752, 1536),
        (1536, 2752),
    ];

    let mut sizes = Vec::new();
    for (w, h) in standard_sizes {
        let px = (w as u64) * (h as u64);
        let ever = ["encode", "denoise", "decode"]
            .iter()
            .all(|s| can_run_stage(&all_refs, s, px));
        let now_ok = ["encode", "denoise", "decode"]
            .iter()
            .all(|s| can_run_stage(&online_executors, s, px));

        let reason = if now_ok {
            None
        } else if ever {
            Some(explain_unavailable_now(&all_executors, &agents, "decode", px))
        } else {
            Some(format!("no known executor supports {}×{}", w, h))
        };

        sizes.push(SizeCapabilityInfo {
            width: w,
            height: h,
            px,
            servable_now: now_ok,
            servable_ever: ever,
            reason,
        });
    }

    let mut stages = Vec::new();
    for stage in ["enhance", "encode", "denoise", "decode"] {
        let ever = can_run_stage(&all_refs, stage, 1024 * 1024);
        let now_ok = can_run_stage(&online_executors, stage, 1024 * 1024);
        let reason = if now_ok {
            None
        } else if ever {
            Some(explain_unavailable_now(
                &all_executors,
                &agents,
                stage,
                1024 * 1024,
            ))
        } else {
            Some(format!("no known executor supports {stage}"))
        };
        stages.push(StageCapabilityInfo {
            stage: stage.to_string(),
            servable_now: now_ok,
            servable_ever: ever,
            reason,
        });
    }

    Ok(CapabilitiesResponse {
        max_px_now,
        max_px_ever,
        enhance_now,
        enhance_ever,
        sizes,
        stages,
    })
}

fn explain_unavailable_now(
    all_executors: &[ExecutorRecord],
    agents: &HashMap<String, AgentRecord>,
    stage: &str,
    px: u64,
) -> String {
    let mut reasons = Vec::new();
    for ex in all_executors {
        let Some(ag) = agents.get(&ex.agent_id) else {
            continue;
        };
        if executor_supports_stage_static(ex, ag, stage, px, None).is_none() {
            continue;
        }
        if !ag.online || ex.state == "offline" {
            reasons.push(format!("executor {} offline", ex.id));
        } else if ex.state == "lent" || ex.lent {
            reasons.push(format!("executor {} lent", ex.id));
        } else if ex.state == "disabled" || ex.disabled_reason.is_some() {
            reasons.push(format!("executor {} disabled", ex.id));
        }
    }
    if reasons.is_empty() {
        format!("no executor online can {stage}")
    } else {
        reasons.join(", ")
    }
}

/// Checks live `limits` for a task on `(agent, executor)` (§5.1, §5.2).
/// Returns `Ok(())` if all limits are satisfied, or `Err(blocked_reason)` if blocked.
pub fn check_limits_for_task(
    agent: &AgentRecord,
    executor: &ExecutorRecord,
    stage: &str,
    px: u64,
    active_tasks_with_px: &[(TaskRecord, u64)],
    live_metrics: Option<&AgentLiveMetrics>,
    scrape_stale_s: f64,
    now: f64,
) -> Result<(), String> {
    for limit in &agent.limits {
        if limit.stage != stage || px <= limit.above_px {
            continue;
        }
        let needs_metrics = limit.requires.iter().any(|r| {
            matches!(
                r,
                RequirementDecl::HostRamAvailableBytes(_)
                    | RequirementDecl::ExecutorMemAvailableBytes(_)
            )
        });
        if needs_metrics {
            let Some(m) = live_metrics else {
                return Err(format!("waiting for metrics scrape from {}", agent.id));
            };
            if now - m.scraped_at > scrape_stale_s {
                return Err(format!(
                    "waiting for fresh metrics scrape from {} (stale {:.1}s)",
                    agent.id,
                    now - m.scraped_at
                ));
            }
        }

        for req in &limit.requires {
            match req {
                RequirementDecl::HostRamAvailableBytes(min_bytes) => {
                    let avail = live_metrics
                        .map(|m| m.host_ram_available_bytes)
                        .unwrap_or(0);
                    if avail < *min_bytes {
                        let avail_gib = (avail as f64) / (1024.0 * 1024.0 * 1024.0);
                        let req_gib = (*min_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
                        return Err(format!(
                            "waiting for host RAM ({avail_gib:.1} GiB available, {req_gib:.1} GiB required)"
                        ));
                    }
                }
                RequirementDecl::MaxConcurrentPerAgent(max_conc) => {
                    let running_matching = active_tasks_with_px
                        .iter()
                        .filter(|(t, t_px)| {
                            t.assigned_agent.as_deref() == Some(agent.id.as_str())
                                && t.stage == limit.stage
                                && *t_px > limit.above_px
                        })
                        .count() as u32;
                    if running_matching >= *max_conc {
                        return Err(format!(
                            "waiting for concurrent {} limit on {} (max {})",
                            limit.stage, agent.id, max_conc
                        ));
                    }
                }
                RequirementDecl::ExecutorMemAvailableBytes(min_bytes) => {
                    let avail = live_metrics
                        .and_then(|m| m.executors.get(&executor.id))
                        .map(|em| em.mem_available_bytes)
                        .unwrap_or(0);
                    if avail < *min_bytes {
                        return Err(format!("waiting for executor {} free memory", executor.id));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Runs one scheduler cycle (§6.3):
/// 1. Releases any `ready` executors with `pending_lend = true`.
/// 2. Feeds `ready(S)` resident workers first (Rule 1).
/// 3. Evaluates stickiness (`max_wait`) and `idle_grace` on remaining `ready` executors (Rules 2 & 3).
/// 4. Starts workers on `idle` executors (Rule 4), respecting `model@rev` affinity (Rule 5).
/// 5. Updates `blocked_reason` on any queued tasks that cannot currently run.
pub async fn run_scheduler_tick(state: &Arc<AppState>) -> Result<()> {
    let _guard = state.sched_lock.lock().await;
    let now = now_s();

    let agents: HashMap<String, AgentRecord> = state
        .db
        .list_agents()
        .await?
        .into_iter()
        .map(|a| (a.id.clone(), a))
        .collect();

    let mut executors = state.db.list_executors().await?;

    // Step 0: handle pending_lend on executors that just became `ready`
    for ex in &mut executors {
        if ex.pending_lend && ex.state == "ready" {
            if let Some((agent_id, exec_id, should_release)) =
                state.db.lend_executor(&ex.key).await?
            {
                if should_release {
                    state.send_to_agent(
                        &agent_id,
                        proto::BackendMessage {
                            msg: Some(proto::backend_message::Msg::Release(proto::Release {
                                executor: exec_id,
                            })),
                        },
                    );
                }
            }
            ex.state = "lent".to_string();
            ex.lent = true;
            ex.pending_lend = false;
        }
    }

    let queued_tasks = state.db.list_queued_tasks().await?;
    let active_tasks = state.db.list_active_tasks().await?;

    // Load jobs for queued and active tasks
    let mut jobs_cache: HashMap<String, JobRecord> = HashMap::new();
    for t in queued_tasks.iter().chain(active_tasks.iter()) {
        if !jobs_cache.contains_key(&t.job_id) {
            if let Some(j) = state.db.get_job(&t.job_id).await? {
                jobs_cache.insert(j.id.clone(), j);
            }
        }
    }

    let mut active_with_px: Vec<(TaskRecord, u64)> = active_tasks
        .into_iter()
        .map(|t| {
            let px = jobs_cache
                .get(&t.job_id)
                .map(|j| (j.width as u64) * (j.height as u64))
                .unwrap_or(0);
            (t, px)
        })
        .collect();

    let live_metrics_snap = state
        .live_metrics
        .read()
        .map(|g| g.clone())
        .unwrap_or_default();

    let mut assigned_task_ids: HashSet<String> = HashSet::new();
    let mut busy_executor_keys: HashSet<String> = HashSet::new();

    // Rule 1: Feed resident `ready(S)` workers first
    for task in &queued_tasks {
        let Some(job) = jobs_cache.get(&task.job_id) else {
            continue;
        };
        let px = (job.width as u64) * (job.height as u64);
        let req_model = if matches!(task.stage.as_str(), "denoise" | "decode") {
            match (&job.model_id, &job.model_rev) {
                (Some(id), Some(rev)) => Some((id.as_str(), rev.as_str())),
                _ => None,
            }
        } else {
            None
        };

        for ex in &mut executors {
            if busy_executor_keys.contains(&ex.key) || ex.state != "ready" || ex.lent || ex.pending_lend {
                continue;
            }
            if ex.current_stage.as_deref() != Some(task.stage.as_str()) {
                continue;
            }
            let Some(agent) = agents.get(&ex.agent_id) else {
                continue;
            };
            if !agent.online {
                continue;
            }
            if executor_supports_stage_static(ex, agent, &task.stage, px, req_model).is_none() {
                continue;
            }
            if task.stage == "decode" && !executor_supports_decode_mode(ex, job.tiling) {
                continue;
            }
            if let Some((req_id, req_rev)) = req_model {
                if let (Some(cur_id), Some(cur_rev)) = (&ex.current_model_id, &ex.current_model_rev)
                {
                    if cur_id != req_id || cur_rev != req_rev {
                        continue;
                    }
                }
            }
            let agent_metrics = live_metrics_snap.get(&agent.id);
            if check_limits_for_task(
                agent,
                ex,
                &task.stage,
                px,
                &active_with_px,
                agent_metrics,
                state.config.scrape_stale_s,
                now,
            )
            .is_err()
            {
                continue;
            }

            // Assign task to this resident worker. A failure here skips this task rather
            // than abandoning the whole tick (and every other executor with it).
            let dispatched = match dispatch_task_to_executor(state, task, job, ex).await {
                Ok(d) => d,
                Err(e) => {
                    warn!(task = %task.id, executor = %ex.key, error = %e, "Failed to assign task");
                    break;
                }
            };
            if dispatched {
                assigned_task_ids.insert(task.id.clone());
                busy_executor_keys.insert(ex.key.clone());
                ex.state = "busy".to_string();
                let mut assigned_clone = task.clone();
                assigned_clone.assigned_agent = Some(ex.agent_id.clone());
                assigned_clone.assigned_executor = Some(ex.id.clone());
                active_with_px.push((assigned_clone, px));
                break;
            }
        }
    }

    // Rule 4: Start a worker on an `idle` executor for the oldest runnable queued task,
    // preferring stages that have no resident (`ready`, `starting`, `busy`) worker anywhere.
    let mut stages_with_workers: HashSet<String> = executors
        .iter()
        .filter(|e| matches!(e.state.as_str(), "ready" | "starting" | "busy"))
        .filter_map(|e| e.current_stage.clone())
        .collect();

    // Sort remaining queued tasks so stages without resident workers are considered first, then by queued_at
    let mut remaining_tasks: Vec<&TaskRecord> = queued_tasks
        .iter()
        .filter(|t| !assigned_task_ids.contains(&t.id))
        .collect();

    remaining_tasks.sort_by(|a, b| {
        let a_has_worker = stages_with_workers.contains(&a.stage);
        let b_has_worker = stages_with_workers.contains(&b.stage);
        a_has_worker.cmp(&b_has_worker).then_with(|| {
            a.queued_at
                .unwrap_or(0.0)
                .partial_cmp(&b.queued_at.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    // Long stages (enhance, denoise) may run on several executors at once: a backlog gets
    // another worker unless the workers already starting or ready for that stage cover
    // every queued task. Short stages (encode, decode) get one worker at a time; it works
    // through its queue faster than a second one could load.
    let mut seen_per_stage: HashMap<String, usize> = HashMap::new();
    for task in &remaining_tasks {
        let seen = seen_per_stage.entry(task.stage.clone()).or_insert(0);
        let ahead_in_queue = *seen;
        *seen += 1;
        let already_served = if matches!(task.stage.as_str(), "enhance" | "denoise") {
            let free_workers = executors
                .iter()
                .filter(|e| {
                    matches!(e.state.as_str(), "starting" | "ready")
                        && e.current_stage.as_deref() == Some(task.stage.as_str())
                })
                .count();
            free_workers > ahead_in_queue
        } else {
            executors.iter().any(|e| {
                matches!(e.state.as_str(), "starting" | "ready" | "busy")
                    && e.current_stage.as_deref() == Some(task.stage.as_str())
            })
        };
        if already_served {
            continue;
        }
        let Some(job) = jobs_cache.get(&task.job_id) else {
            continue;
        };
        let px = (job.width as u64) * (job.height as u64);
        let req_model = if matches!(task.stage.as_str(), "denoise" | "decode") {
            match (&job.model_id, &job.model_rev) {
                (Some(id), Some(rev)) => Some((id.as_str(), rev.as_str())),
                _ => None,
            }
        } else {
            None
        };

        for ex in &mut executors {
            if busy_executor_keys.contains(&ex.key) || ex.state != "idle" || ex.lent || ex.pending_lend {
                continue;
            }
            let Some(agent) = agents.get(&ex.agent_id) else {
                continue;
            };
            if !agent.online {
                continue;
            }
            let Some(model) =
                executor_supports_stage_static(ex, agent, &task.stage, px, req_model)
            else {
                continue;
            };
            if task.stage == "decode" && !executor_supports_decode_mode(ex, job.tiling) {
                continue;
            }
            let agent_metrics = live_metrics_snap.get(&agent.id);
            if check_limits_for_task(
                agent,
                ex,
                &task.stage,
                px,
                &active_with_px,
                agent_metrics,
                state.config.scrape_stale_s,
                now,
            )
            .is_err()
            {
                continue;
            }

            // Send StartWorker
            state
                .db
                .mark_executor_starting(&ex.agent_id, &ex.id, &task.stage, &model)
                .await?;
            state.send_to_agent(
                &ex.agent_id,
                proto::BackendMessage {
                    msg: Some(proto::backend_message::Msg::StartWorker(
                        proto::StartWorker {
                            executor: ex.id.clone(),
                            stage: stage_to_proto(&task.stage) as i32,
                            model: Some(proto::ModelRef {
                                id: model.id.clone(),
                                rev: model.rev.clone(),
                                dtype: model.dtype.clone(),
                            }),
                        },
                    )),
                },
            );
            ex.state = "starting".to_string();
            ex.current_stage = Some(task.stage.clone());
            ex.current_model_id = Some(model.id);
            ex.current_model_rev = Some(model.rev);
            busy_executor_keys.insert(ex.key.clone());
            stages_with_workers.insert(task.stage.clone());
            break;
        }
    }

    // Rules 2 & 3 for `ready(S)` executors that got nothing above. A worker stays only while
    // a task of its own stage is queued. Otherwise, if another stage has queued work that no
    // free worker covers, the executor switches to it now; holding a loaded model while other
    // work waits only costs time. Idle grace (keeping a worker warm for a follow-up job)
    // applies only when no other work is waiting. `max_wait` still lets a long-waiting task
    // take over an executor whose own stage has queued work, so nothing starves.
    for idx in 0..executors.len() {
        let ex_snap = executors[idx].clone();
        if busy_executor_keys.contains(&ex_snap.key)
            || ex_snap.state != "ready"
            || ex_snap.lent
            || ex_snap.pending_lend
        {
            continue;
        }
        let Some(agent) = agents.get(&ex_snap.agent_id) else {
            continue;
        };
        if !agent.online {
            continue;
        }
        let cur_stage = ex_snap.current_stage.clone().unwrap_or_default();

        let has_queued_for_cur = queued_tasks
            .iter()
            .any(|t| !assigned_task_ids.contains(&t.id) && t.stage == cur_stage);

        let idle_duration = now - ex_snap.idle_since.unwrap_or(now);

        // Find if any other stage has a runnable task for this executor
        let mut switch_candidate: Option<(&TaskRecord, ModelDecl, f64)> = None;
        for task in &remaining_tasks {
            if assigned_task_ids.contains(&task.id) || task.stage == cur_stage {
                continue;
            }
            // Covered: other workers for that stage will take its queue. Free (starting or
            // ready) workers count for every stage; a busy one only for short stages, which
            // it gets through faster than this executor could load a model.
            let task_wait = now - task.queued_at.unwrap_or(now);
            let short_stage = matches!(task.stage.as_str(), "encode" | "decode");
            let covering = executors
                .iter()
                .filter(|other| {
                    other.key != ex_snap.key
                        && other.current_stage.as_deref() == Some(task.stage.as_str())
                        && (matches!(other.state.as_str(), "starting" | "ready")
                            || (short_stage && other.state == "busy"))
                })
                .count();
            let waiting = queued_tasks
                .iter()
                .filter(|t| !assigned_task_ids.contains(&t.id) && t.stage == task.stage)
                .count();
            if covering >= waiting && task_wait < state.config.max_wait_s {
                continue;
            }
            let Some(job) = jobs_cache.get(&task.job_id) else {
                continue;
            };
            let px = (job.width as u64) * (job.height as u64);
            let req_model = if matches!(task.stage.as_str(), "denoise" | "decode") {
                match (&job.model_id, &job.model_rev) {
                    (Some(id), Some(rev)) => Some((id.as_str(), rev.as_str())),
                    _ => None,
                }
            } else {
                None
            };
            let Some(model) =
                executor_supports_stage_static(&ex_snap, agent, &task.stage, px, req_model)
            else {
                continue;
            };
            let agent_metrics = live_metrics_snap.get(&agent.id);
            if check_limits_for_task(
                agent,
                &ex_snap,
                &task.stage,
                px,
                &active_with_px,
                agent_metrics,
                state.config.scrape_stale_s,
                now,
            )
            .is_err()
            {
                continue;
            }
            switch_candidate = Some((task, model, task_wait));
            break;
        }

        if let Some((target_task, model, task_wait)) = switch_candidate {
            let should_switch = !has_queued_for_cur || task_wait >= state.config.max_wait_s;

            if should_switch {
                state.metrics.record_executor_switch(
                    &ex_snap.agent_id,
                    &ex_snap.id,
                    &cur_stage,
                    &target_task.stage,
                );
                state
                    .db
                    .mark_executor_starting(&ex_snap.agent_id, &ex_snap.id, &target_task.stage, &model)
                    .await?;
                state.send_to_agent(
                    &ex_snap.agent_id,
                    proto::BackendMessage {
                        msg: Some(proto::backend_message::Msg::StartWorker(
                            proto::StartWorker {
                                executor: ex_snap.id.clone(),
                                stage: stage_to_proto(&target_task.stage) as i32,
                                model: Some(proto::ModelRef {
                                    id: model.id.clone(),
                                    rev: model.rev.clone(),
                                    dtype: model.dtype.clone(),
                                }),
                            },
                        )),
                    },
                );
                executors[idx].state = "starting".to_string();
                executors[idx].current_stage = Some(target_task.stage.clone());
                busy_executor_keys.insert(ex_snap.key.clone());
                continue;
            }
        }

        // Rule 3: nothing of its stage queued and nothing else to switch to: keep the worker
        // warm for idle_grace, then release it.
        if !has_queued_for_cur && idle_duration >= state.config.idle_grace_s {
            state
                .db
                .update_executor_state(
                    &ex_snap.agent_id,
                    &ex_snap.id,
                    "stopping",
                    Some(&cur_stage),
                    None,
                    None,
                )
                .await?;
            state.send_to_agent(
                &ex_snap.agent_id,
                proto::BackendMessage {
                    msg: Some(proto::backend_message::Msg::Release(proto::Release {
                        executor: ex_snap.id.clone(),
                    })),
                },
            );
            executors[idx].state = "stopping".to_string();
        }
    }

    // Step 5: Compute live `blocked_reason` for any still-queued tasks
    for task in &queued_tasks {
        if assigned_task_ids.contains(&task.id) {
            continue;
        }
        let Some(job) = jobs_cache.get(&task.job_id) else {
            continue;
        };
        let reason = diagnose_queued_task_block(
            task,
            job,
            &executors,
            &agents,
            &active_with_px,
            &live_metrics_snap,
            state.config.scrape_stale_s,
            now,
        );
        state
            .db
            .set_task_blocked_reason(&task.id, &job.id, reason.as_deref())
            .await?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn diagnose_queued_task_block(
    task: &TaskRecord,
    job: &JobRecord,
    executors: &[ExecutorRecord],
    agents: &HashMap<String, AgentRecord>,
    active_with_px: &[(TaskRecord, u64)],
    live_metrics: &HashMap<String, AgentLiveMetrics>,
    scrape_stale_s: f64,
    now: f64,
) -> Option<String> {
    let px = (job.width as u64) * (job.height as u64);
    let req_model = if matches!(task.stage.as_str(), "denoise" | "decode") {
        match (&job.model_id, &job.model_rev) {
            (Some(id), Some(rev)) => Some((id.as_str(), rev.as_str())),
            _ => None,
        }
    } else {
        None
    };

    let mut capable_executors = Vec::new();
    for ex in executors {
        let Some(ag) = agents.get(&ex.agent_id) else {
            continue;
        };
        if executor_supports_stage_static(ex, ag, &task.stage, px, req_model).is_some() {
            capable_executors.push((ex, ag));
        }
    }

    if capable_executors.is_empty() {
        if let Some((m_id, m_rev)) = req_model {
            return Some(format!(
                "no executor has model {}@{} for {}",
                m_id, m_rev, task.stage
            ));
        }
        return Some(format!(
            "no executor can {} {}×{}",
            task.stage, job.width, job.height
        ));
    }

    // Check if any capable executor is online and not lent/disabled
    let mut online_usable = Vec::new();
    let mut offline_or_unavailable_reasons = Vec::new();

    for (ex, ag) in &capable_executors {
        if !ag.online || ex.state == "offline" {
            offline_or_unavailable_reasons.push(format!("executor {} offline", ex.id));
        } else if ex.state == "lent" || ex.lent {
            offline_or_unavailable_reasons.push(format!("executor {} lent", ex.id));
        } else if ex.state == "disabled" || ex.disabled_reason.is_some() {
            let r = ex
                .disabled_reason
                .as_deref()
                .unwrap_or("disabled");
            offline_or_unavailable_reasons.push(format!("executor {} disabled ({})", ex.id, r));
        } else {
            online_usable.push((*ex, *ag));
        }
    }

    if online_usable.is_empty() {
        return Some(offline_or_unavailable_reasons.join("; "));
    }

    // Check live resource/concurrency limits on online usable executors
    let mut limit_reasons = Vec::new();
    for (ex, ag) in &online_usable {
        let m = live_metrics.get(&ag.id);
        if let Err(limit_err) = check_limits_for_task(
            ag,
            ex,
            &task.stage,
            px,
            active_with_px,
            m,
            scrape_stale_s,
            now,
        ) {
            limit_reasons.push(limit_err);
        }
    }

    if limit_reasons.len() == online_usable.len() {
        return limit_reasons.into_iter().next();
    }

    // Otherwise, an online usable executor is starting or busy with another task
    None
}

async fn dispatch_task_to_executor(
    state: &Arc<AppState>,
    task: &TaskRecord,
    job: &JobRecord,
    ex: &ExecutorRecord,
) -> Result<bool> {
    let Some((assigned_task, _, attempt)) = state
        .db
        .assign_task(&task.id, &ex.agent_id, &ex.id)
        .await?
    else {
        return Ok(false);
    };

    if let Some(q_at) = task.queued_at {
        let wait_s = (now_s() - q_at).max(0.0);
        state.metrics.observe_task_wait(&task.stage, wait_s);
    }

    let base_url = &state.config.artifact_base_url;
    let mut inputs = Vec::new();
    let mut uploads = Vec::new();

    let params = match assigned_task.stage.as_str() {
        "enhance" => {
            uploads.push(proto::ArtifactRef {
                name: "enhanced.json".to_string(),
                url: format!("{base_url}/agent/v1/artifacts/{}/{attempt}/enhanced.json", assigned_task.id),
                sha256: String::new(),
                bytes: 0,
            });
            Some(proto::assign::Params::Enhance(proto::EnhanceParams {
                prompt: enhance_request(&job.prompt, job.width, job.height),
            }))
        }
        "encode" => {
            uploads.push(proto::ArtifactRef {
                name: "embeds.safetensors".to_string(),
                url: format!(
                    "{base_url}/agent/v1/artifacts/{}/{attempt}/embeds.safetensors",
                    assigned_task.id
                ),
                sha256: String::new(),
                bytes: 0,
            });
            let (effective_prompt, prompt_source) = if let Some(ep) = &job.enhanced_prompt {
                (ep.clone(), "enhanced".to_string())
            } else {
                (job.prompt.clone(), "original".to_string())
            };
            let neg = if job.true_cfg_scale > 1.0 {
                job.negative_prompt.clone()
            } else {
                String::new()
            };
            Some(proto::assign::Params::Encode(proto::EncodeParams {
                prompt: effective_prompt,
                prompt_source,
                negative_prompt: neg,
                true_cfg_scale: job.true_cfg_scale,
            }))
        }
        "denoise" => {
            if let Some(embeds_art) = state
                .db
                .get_job_artifact_by_name(&job.id, "embeds.safetensors")
                .await?
            {
                inputs.push(proto::ArtifactRef {
                    name: "embeds.safetensors".to_string(),
                    url: format!(
                        "{base_url}/agent/v1/artifacts/{}/{}/embeds.safetensors",
                        embeds_art.task_id, embeds_art.attempt
                    ),
                    sha256: embeds_art.sha256,
                    bytes: embeds_art.bytes,
                });
            }
            uploads.push(proto::ArtifactRef {
                name: "latent.safetensors".to_string(),
                url: format!(
                    "{base_url}/agent/v1/artifacts/{}/{attempt}/latent.safetensors",
                    assigned_task.id
                ),
                sha256: String::new(),
                bytes: 0,
            });
            Some(proto::assign::Params::Denoise(proto::DenoiseParams {
                width: job.width,
                height: job.height,
                steps: job.steps,
                seed: job.seed,
                true_cfg_scale: job.true_cfg_scale,
            }))
        }
        "decode" => {
            if let Some(latent_art) = state
                .db
                .get_job_artifact_by_name(&job.id, "latent.safetensors")
                .await?
            {
                inputs.push(proto::ArtifactRef {
                    name: "latent.safetensors".to_string(),
                    url: format!(
                        "{base_url}/agent/v1/artifacts/{}/{}/latent.safetensors",
                        latent_art.task_id, latent_art.attempt
                    ),
                    sha256: latent_art.sha256,
                    bytes: latent_art.bytes,
                });
            }
            uploads.push(proto::ArtifactRef {
                name: "image.png".to_string(),
                url: format!(
                    "{base_url}/agent/v1/artifacts/{}/{attempt}/image.png",
                    assigned_task.id
                ),
                sha256: String::new(),
                bytes: 0,
            });
            Some(proto::assign::Params::Decode(proto::DecodeParams {
                mode: if job.tiling { proto::DecodeMode::Tiled } else { proto::DecodeMode::Untiled } as i32,
            }))
        }
        _ => None,
    };

    let assign_msg = proto::BackendMessage {
        msg: Some(proto::backend_message::Msg::Assign(proto::Assign {
            task: assigned_task.id,
            attempt,
            executor: ex.id.clone(),
            stage: stage_to_proto(&task.stage) as i32,
            inputs,
            uploads,
            params,
        })),
    };

    state.send_to_agent(&ex.agent_id, assign_msg);
    Ok(true)
}

pub fn stage_to_proto(stage: &str) -> proto::Stage {
    match stage {
        "enhance" => proto::Stage::Enhance,
        "encode" => proto::Stage::Encode,
        "denoise" => proto::Stage::Denoise,
        "decode" => proto::Stage::Decode,
        _ => proto::Stage::Unspecified,
    }
}

pub fn proto_to_stage(stage: i32) -> Option<&'static str> {
    match proto::Stage::try_from(stage).ok()? {
        proto::Stage::Enhance => Some("enhance"),
        proto::Stage::Encode => Some("encode"),
        proto::Stage::Denoise => Some("denoise"),
        proto::Stage::Decode => Some("decode"),
        proto::Stage::Unspecified => None,
    }
}
