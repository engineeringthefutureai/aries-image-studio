use crate::config::bearer_matches;
use crate::db::{is_valid_client_job_id, now_s, JobSubmitInput};
use crate::metrics::{ExecutorLiveMetrics, MetricsSnapshot};
use crate::proto;
use crate::scheduler::{
    check_submit_feasibility, compute_capabilities, run_scheduler_tick, AppState,
};
use crate::storage::{
    validate_embeds_safetensors, validate_enhanced_json, validate_latent_safetensors,
    validate_png_dimensions,
};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::Value;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};

pub fn build_http_router(state: Arc<AppState>) -> Router {
    let api_router = Router::new()
        .route("/api/jobs", post(submit_job).get(list_jobs))
        .route("/api/jobs/:id", get(get_job).delete(delete_job).patch(patch_job))
        .route("/api/jobs/:id/cancel", post(cancel_job))
        .route("/api/jobs/:id/retry", post(retry_job))
        .route("/api/images/:id", get(get_image))
        .route("/api/images/:id/thumb", get(get_image_thumb))
        .route("/api/system", get(get_system))
        .route("/api/capabilities", get(get_capabilities))
        .route("/api/preferences", get(get_preferences).put(put_preferences))
        .route("/api/executors/:id/lend", post(lend_executor))
        .route("/api/executors/:id/return", post(return_executor))
        .route("/api/executors/:id/disable", post(disable_executor))
        .route("/api/executors/:id/enable", post(enable_executor))
        .route(
            "/agent/v1/artifacts/:task_id/:attempt/:name",
            get(download_artifact).put(upload_artifact),
        )
        .route("/metrics", get(get_metrics))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    let static_dir = &state.config.static_dir;
    if static_dir.exists() {
        let index_file = static_dir.join("index.html");
        api_router.fallback_service(
            ServeDir::new(static_dir).not_found_service(ServeFile::new(index_file)),
        )
    } else {
        api_router
    }
}

#[derive(Debug, Deserialize)]
pub struct ListJobsQuery {
    pub changed_since: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct PatchJobBody {
    pub favorite: Option<bool>,
}

/// UI metadata and preferences are small JSON objects; anything bigger is a client bug.
const META_MAX_BYTES: usize = 4 * 1024;
const PREFERENCES_MAX_BYTES: usize = 16 * 1024;

fn is_small_object(v: &Value, max: usize) -> bool {
    v.is_object() && v.to_string().len() <= max
}

#[derive(Debug, Deserialize, Default)]
pub struct DisableExecutorBody {
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SystemStageQueue {
    pub stage: String,
    pub queued: usize,
    pub running: usize,
    pub blocked: usize,
    pub oldest_wait_s: f64,
    pub blocked_reasons: Vec<String>,
}

async fn submit_job(
    State(state): State<Arc<AppState>>,
    Json(input): Json<JobSubmitInput>,
) -> Response {
    if let Some(id) = &input.id {
        if !is_valid_client_job_id(id) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "id must be 1-64 characters of [A-Za-z0-9_-]" })),
            )
                .into_response();
        }
        // Idempotent resubmission: the job already exists, return it unchanged.
        if let Ok(Some(existing)) = state.db.get_job(id).await {
            return (StatusCode::OK, Json(existing)).into_response();
        }
    }
    if let Some(meta) = &input.meta {
        if !is_small_object(meta, META_MAX_BYTES) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "meta must be a JSON object of at most 4 KiB" })),
            )
                .into_response();
        }
    }
    if let Err(reason) = check_submit_feasibility(&state.db, &input).await {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "rejected": reason })),
        )
            .into_response();
    }

    let seed = input.seed.unwrap_or_else(|| {
        let t = (now_s() * 1_000_000.0) as u64;
        t ^ 0x5DEECE66Du64
    });

    match state.db.create_job(&input, seed).await {
        Ok(job) => {
            let _ = run_scheduler_tick(&state).await;
            let refreshed = state.db.get_job(&job.id).await.ok().flatten().unwrap_or(job);
            (StatusCode::CREATED, Json(refreshed)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn list_jobs(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ListJobsQuery>,
) -> Response {
    let version = state.db.current_version().await.unwrap_or(1);
    // Deletions since the client's version, so every client drops them too.
    let deleted = match q.changed_since {
        Some(since) => state.db.list_deleted_since(since).await.unwrap_or_default(),
        None => Vec::new(),
    };
    match state.db.list_jobs(q.changed_since).await {
        Ok(jobs) => (
            StatusCode::OK,
            Json(json!({
                "version": version,
                "jobs": jobs,
                "deleted": deleted,
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_job(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match state.db.get_job(&id).await {
        Ok(Some(job)) => (StatusCode::OK, Json(job)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "job not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn patch_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<PatchJobBody>,
) -> Response {
    let Some(favorite) = body.favorite else {
        return match state.db.get_job(&id).await {
            Ok(Some(job)) => (StatusCode::OK, Json(job)).into_response(),
            _ => (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))).into_response(),
        };
    };
    match state.db.set_job_favorite(&id, favorite).await {
        Ok(Some(job)) => (StatusCode::OK, Json(job)).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_preferences(State(state): State<Arc<AppState>>) -> Response {
    match state.db.get_preferences().await {
        Ok(prefs) => (StatusCode::OK, Json(prefs)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn put_preferences(State(state): State<Arc<AppState>>, Json(prefs): Json<Value>) -> Response {
    if !is_small_object(&prefs, PREFERENCES_MAX_BYTES) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "preferences must be a JSON object of at most 16 KiB" })),
        )
            .into_response();
    }
    match state.db.put_preferences(&prefs).await {
        Ok(()) => (StatusCode::OK, Json(prefs)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn delete_job(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    // Cancel first if running
    if let Ok(Some((Some((agent_id, task_id, attempt)), _))) = state.db.cancel_job(&id).await {
        state.send_to_agent(
            &agent_id,
            proto::BackendMessage {
                msg: Some(proto::backend_message::Msg::Cancel(proto::Cancel {
                    task: task_id,
                    attempt,
                })),
            },
        );
    }

    match state.db.delete_job(&id).await {
        Ok(Some(orphan_shas)) => {
            for sha in orphan_shas {
                let _ = state.blobs.delete_blob(&sha);
            }
            let thumb = state.blobs.thumb_path(&id);
            if thumb.exists() {
                let _ = std::fs::remove_file(thumb);
            }
            StatusCode::NO_CONTENT.into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "job not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn cancel_job(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match state.db.cancel_job(&id).await {
        Ok(Some((cancel_sig, job))) => {
            if let Some((agent_id, task_id, attempt)) = cancel_sig {
                state.send_to_agent(
                    &agent_id,
                    proto::BackendMessage {
                        msg: Some(proto::backend_message::Msg::Cancel(proto::Cancel {
                            task: task_id,
                            attempt,
                        })),
                    },
                );
            }
            let _ = run_scheduler_tick(&state).await;
            (StatusCode::OK, Json(job)).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "job not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn retry_job(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match state.db.retry_job(&id).await {
        Ok(Some(job)) => {
            let _ = run_scheduler_tick(&state).await;
            let refreshed = state.db.get_job(&job.id).await.ok().flatten().unwrap_or(job);
            (StatusCode::OK, Json(refreshed)).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "job not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_image(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let Ok(Some(art)) = state.db.get_job_artifact_by_name(&id, "image.png").await else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "image not found" })),
        )
            .into_response();
    };
    match state.blobs.get_blob(&art.sha256) {
        Ok(bytes) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "image/png".to_string()),
                (
                    header::CONTENT_DISPOSITION,
                    format!("inline; filename=\"{id}.png\""),
                ),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "image blob missing" })),
        )
            .into_response(),
    }
}

async fn get_image_thumb(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let thumb_path = state.blobs.thumb_path(&id);
    if let Ok(bytes) = std::fs::read(&thumb_path) {
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "image/webp")],
            bytes,
        )
            .into_response();
    }
    // Generate on the fly if missing
    if let Ok(Some(art)) = state.db.get_job_artifact_by_name(&id, "image.png").await {
        if let Ok(png_bytes) = state.blobs.get_blob(&art.sha256) {
            if let Ok(path) = state.blobs.write_thumbnail(&id, &png_bytes) {
                if let Ok(bytes) = std::fs::read(path) {
                    return (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "image/webp")],
                        bytes,
                    )
                        .into_response();
                }
            }
        }
    }
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": "thumbnail not found" })),
    )
        .into_response()
}

async fn get_capabilities(State(state): State<Arc<AppState>>) -> Response {
    match compute_capabilities(&state.db).await {
        Ok(caps) => (StatusCode::OK, Json(caps)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_system(State(state): State<Arc<AppState>>) -> Response {
    let now = now_s();
    let agents = state.db.list_agents().await.unwrap_or_default();
    let executors = state.db.list_executors().await.unwrap_or_default();
    let queued = state.db.list_queued_tasks().await.unwrap_or_default();
    let active = state.db.list_active_tasks().await.unwrap_or_default();
    let live_snap = state
        .live_metrics
        .read()
        .map(|g| g.clone())
        .unwrap_or_default();

    let mut agents_out = Vec::with_capacity(agents.len());
    for ag in &agents {
        let lm = live_snap.get(&ag.id);
        let scrape_age_s = lm.map(|m| (now - m.scraped_at).max(0.0));
        let scrape_stale = scrape_age_s
            .map(|age| age > state.config.scrape_stale_s)
            .unwrap_or(true);
        let host_ram_available_bytes = lm.map(|m| m.host_ram_available_bytes).unwrap_or(0);

        let ag_execs: Vec<_> = executors
            .iter()
            .filter(|e| e.agent_id == ag.id)
            .map(|e| {
                let em: ExecutorLiveMetrics = lm
                    .and_then(|m| m.executors.get(&e.id))
                    .cloned()
                    .unwrap_or_default();
                json!({
                    "key": e.key,
                    "id": e.id,
                    "agent_id": e.agent_id,
                    "device": e.device,
                    "mem_total_bytes": e.mem_total_bytes,
                    "mem_used_bytes": em.mem_used_bytes,
                    "mem_available_bytes": em.mem_available_bytes,
                    "overflow_host_bytes": em.overflow_host_bytes,
                    "stages": e.stages,
                    "state": e.state,
                    "current_stage": e.current_stage,
                    "current_model_id": e.current_model_id,
                    "current_model_rev": e.current_model_rev,
                    "current_task_id": e.current_task_id,
                    "worker_pid": e.worker_pid,
                    "load_s": e.load_s,
                    "disabled_reason": e.disabled_reason,
                    "lent": e.lent,
                    "pending_lend": e.pending_lend,
                })
            })
            .collect();

        agents_out.push(json!({
            "id": ag.id,
            "agent_version": ag.agent_version,
            "backend_kind": ag.backend_kind,
            "metrics_url": ag.metrics_url,
            "online": ag.online,
            "host_ram_total_bytes": ag.host_ram_total_bytes,
            "host_ram_available_bytes": host_ram_available_bytes,
            "scrape_age_s": scrape_age_s,
            "scrape_stale": scrape_stale,
            "models": ag.models,
            "limits": ag.limits,
            "executors": ag_execs,
        }));
    }

    let mut queues = Vec::new();
    for stage in ["enhance", "encode", "denoise", "decode"] {
        let stage_queued: Vec<_> = queued.iter().filter(|t| t.stage == stage).collect();
        let stage_running = active.iter().filter(|t| t.stage == stage).count();
        let stage_blocked = stage_queued
            .iter()
            .filter(|t| t.blocked_reason.is_some())
            .count();
        let oldest_wait_s = stage_queued
            .iter()
            .filter_map(|t| t.queued_at.map(|q| (now - q).max(0.0)))
            .fold(0.0f64, f64::max);
        let mut reasons = Vec::new();
        for t in &stage_queued {
            if let Some(r) = &t.blocked_reason {
                if !reasons.contains(r) {
                    reasons.push(r.clone());
                }
            }
        }
        queues.push(SystemStageQueue {
            stage: stage.to_string(),
            queued: stage_queued.len(),
            running: stage_running,
            blocked: stage_blocked,
            oldest_wait_s,
            blocked_reasons: reasons,
        });
    }

    (
        StatusCode::OK,
        Json(json!({
            "agents": agents_out,
            "queues": queues,
        })),
    )
        .into_response()
}

async fn lend_executor(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match state.db.lend_executor(&id).await {
        Ok(Some((agent_id, exec_id, should_release))) => {
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
            let _ = run_scheduler_tick(&state).await;
            let ex = state.db.find_executor(&id).await.ok().flatten();
            (StatusCode::OK, Json(ex)).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "executor not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn return_executor(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match state.db.return_executor(&id).await {
        Ok(Some(ex)) => {
            let _ = run_scheduler_tick(&state).await;
            (StatusCode::OK, Json(ex)).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "executor not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn disable_executor(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    body: Option<Json<DisableExecutorBody>>,
) -> Response {
    let reason = body
        .and_then(|b| b.0.reason)
        .unwrap_or_else(|| "operator disabled".to_string());
    match state.db.disable_executor(&id, &reason).await {
        Ok(Some((agent_id, exec_id, should_release))) => {
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
            let _ = run_scheduler_tick(&state).await;
            let ex = state.db.find_executor(&id).await.ok().flatten();
            (StatusCode::OK, Json(ex)).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "executor not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn enable_executor(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match state.db.enable_executor(&id).await {
        Ok(Some(ex)) => {
            let _ = run_scheduler_tick(&state).await;
            (StatusCode::OK, Json(ex)).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "executor not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

fn check_agent_http_auth(headers: &HeaderMap, expected_token: &str) -> bool {
    let Some(val) = headers.get(header::AUTHORIZATION) else {
        return false;
    };
    let Ok(s) = val.to_str() else {
        return false;
    };
    bearer_matches(s, expected_token)
}

async fn download_artifact(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path((task_id, attempt, name)): Path<(String, u32, String)>,
) -> Response {
    if !check_agent_http_auth(&headers, &state.config.agent_token) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "unauthorized" })),
        )
            .into_response();
    }
    let Ok(Some(art)) = state
        .db
        .get_artifact_for_task_attempt(&task_id, attempt, &name)
        .await
    else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "artifact not found" })),
        )
            .into_response();
    };
    match state.blobs.get_blob(&art.sha256) {
        Ok(bytes) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/octet-stream".to_string()),
                (
                    header::HeaderName::from_static("x-content-sha256"),
                    art.sha256,
                ),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "artifact blob missing" })),
        )
            .into_response(),
    }
}

async fn upload_artifact(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path((task_id, attempt, name)): Path<(String, u32, String)>,
    body: Bytes,
) -> Response {
    if !check_agent_http_auth(&headers, &state.config.agent_token) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "unauthorized" })),
        )
            .into_response();
    }

    // Verify SHA-256 if header provided
    let mut hasher = Sha256::new();
    hasher.update(&body);
    let computed_sha = hex::encode(hasher.finalize());
    if let Some(expected_hdr) = headers.get("x-content-sha256").and_then(|v| v.to_str().ok()) {
        if !expected_hdr.is_empty() && expected_hdr != computed_sha {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": format!("sha256 mismatch: header={expected_hdr}, computed={computed_sha}")
                })),
            )
                .into_response();
        }
    }

    let Ok(Some(task)) = state.db.get_task(&task_id).await else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "task not found" })),
        )
            .into_response();
    };
    if task.current_attempt != attempt
        || !matches!(task.status.as_str(), "assigned" | "running" | "uploading")
    {
        return (
            StatusCode::CONFLICT,
            Json(json!({
                "error": format!("stale or inactive attempt {attempt} for task {task_id}")
            })),
        )
            .into_response();
    }
    let Ok(Some(job)) = state.db.get_job(&task.job_id).await else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "job not found" })),
        )
            .into_response();
    };

    // Validate artifact payload before committing (§7.4)
    let mut meta_json: Option<String> = None;
    let kind = if name == "image.png" {
        "final"
    } else {
        "intermediate"
    };

    let validation_res: anyhow::Result<()> = match name.as_str() {
        "enhanced.json" => validate_enhanced_json(&body).map(|_| ()),
        "embeds.safetensors" => {
            validate_embeds_safetensors(&body, job.true_cfg_scale > 1.0).map(|hdr| {
                meta_json = serde_json::to_string(&hdr.metadata).ok();
            })
        }
        "latent.safetensors" => {
            validate_latent_safetensors(&body, Some(job.width), Some(job.height)).map(|hdr| {
                meta_json = serde_json::to_string(&hdr.metadata).ok();
            })
        }
        "image.png" => validate_png_dimensions(&body, Some(job.width), Some(job.height)).map(|_| ()),
        _ => Err(anyhow::anyhow!("unexpected artifact name '{name}'")),
    };

    if let Err(e) = validation_res {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response();
    }

    let (sha256, bytes_len) = match state.blobs.put_blob(&body) {
        Ok(res) => res,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };

    match state
        .db
        .insert_uploaded_artifact(
            &task_id,
            attempt,
            &name,
            &sha256,
            bytes_len,
            kind,
            meta_json.as_deref(),
        )
        .await
    {
        Ok(art) => (
            StatusCode::CREATED,
            Json(json!({
                "id": art.id,
                "sha256": art.sha256,
                "bytes": art.bytes,
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_metrics(State(state): State<Arc<AppState>>) -> Response {
    let now = now_s();
    let jobs_by_status = state.db.count_jobs_by_status().await.unwrap_or_default();
    let queued = state.db.list_queued_tasks().await.unwrap_or_default();
    let agents = state.db.list_agents().await.unwrap_or_default();
    let (inter_bytes, final_bytes) = state.db.artifact_bytes_by_kind().await.unwrap_or((0, 0));
    let live_snap = state
        .live_metrics
        .read()
        .map(|g| g.clone())
        .unwrap_or_default();


    let mut tasks_queued_by_stage = HashMap::new();
    let mut tasks_blocked = HashMap::new();
    for t in queued {
        *tasks_queued_by_stage.entry(t.stage.clone()).or_insert(0) += 1;
        if let Some(reason) = t.blocked_reason {
            *tasks_blocked.entry((t.stage, reason)).or_insert(0) += 1;
        }
    }

    let mut agents_connected = HashMap::new();
    let mut agent_scrape_age = HashMap::new();
    for a in agents {
        agents_connected.insert(a.id.clone(), a.online);
        if let Some(lm) = live_snap.get(&a.id) {
            agent_scrape_age.insert(a.id, (now - lm.scraped_at).max(0.0));
        }
    }

    let snap = MetricsSnapshot {
        jobs_by_status,
        tasks_queued_by_stage,
        tasks_blocked,
        agents_connected,
        agent_scrape_age,
        intermediate_bytes: inter_bytes,
        final_bytes,
    };

    let mut text = state.metrics.render(&snap);
    // Every agent's latest scrape, labelled by agent; a stale scrape (agent asleep or gone)
    // is left out rather than repeated, so a sleeping GPU box never looks like a live one.
    let fresh: Vec<(&str, &str)> = live_snap
        .iter()
        .filter(|(_, m)| now - m.scraped_at <= state.config.scrape_stale_s && !m.raw.is_empty())
        .map(|(id, m)| (id.as_str(), m.raw.as_str()))
        .collect();
    text.push_str(&crate::metrics::reexport_agent_metrics(&fresh));
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        text,
    )
        .into_response()
}
