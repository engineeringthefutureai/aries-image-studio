use crate::auth::{self, User};
use crate::cloud::normalize_enhancer;
use crate::config::bearer_matches;
use crate::db::{
    is_valid_client_job_id, now_s, JobPageQuery, JobStatus, JobSubmitInput, ReferenceInput,
    ReferenceLatents, UploadRecord,
};
use crate::legal;
use crate::metrics::{ExecutorLiveMetrics, MetricsSnapshot};
use crate::proto;
use crate::scheduler::{
    check_submit_feasibility, compute_capabilities, encode_cache_key, tick, AppState,
};
use crate::storage::{
    is_sha256_hex, probe_upload_image, upload_thumb_key, validate_embeds_safetensors,
    validate_enhanced_json, validate_latent_safetensors, validate_png_dimensions,
    validate_reference_latents, THUMB_SIZES,
};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{middleware, Extension, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tower::ServiceExt;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};

pub fn build_http_router(state: Arc<AppState>) -> Router {
    // The browser's API: a signed-in user on every request when sign-in is on (auth.rs).
    let user_api = Router::new()
        .route("/api/jobs", post(submit_job).get(list_jobs))
        .route(
            "/api/jobs/:id",
            get(get_job).delete(delete_job).patch(patch_job),
        )
        .route("/api/jobs/:id/cancel", post(cancel_job))
        .route("/api/jobs/:id/retry", post(retry_job))
        .route("/api/gallery/facets", get(get_gallery_facets))
        .route("/api/images/:id", get(get_image))
        .route("/api/images/:id/thumb", get(get_image_thumb))
        .route("/api/uploads", post(upload_image))
        .route("/api/uploads/:sha", get(get_upload))
        .route("/api/uploads/:sha/thumb", get(get_upload_thumb))
        .route("/api/system", get(get_system))
        .route("/api/capabilities", get(get_capabilities))
        .route(
            "/api/preferences",
            get(get_preferences).put(put_preferences),
        )
        .route("/api/executors/:id/lend", post(lend_executor))
        .route("/api/executors/:id/return", post(return_executor))
        .route("/api/executors/:id/disable", post(disable_executor))
        .route("/api/executors/:id/enable", post(enable_executor))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth::require_user,
        ));
    // Open: agents (their own token), metrics, health, the privacy policy and terms, and
    // signing in.
    let mut api_router = user_api
        .route(
            "/agent/v1/artifacts/:task_id/:attempt/:name",
            get(download_artifact).put(upload_artifact),
        )
        .route("/agent/v1/blobs/:sha", get(download_blob))
        .route("/metrics", get(get_metrics))
        .route("/healthz", get(|| async { "ok" }))
        .route("/privacy", get(legal::privacy))
        .route("/terms", get(legal::terms))
        .route("/api/me", get(auth::me))
        .route("/auth/login", get(auth::login))
        .route("/auth/callback", get(auth::callback))
        .route("/auth/link", get(auth::link))
        .route("/auth/logout", post(auth::logout))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024));
    // Any origin may call an open Studio; one that signs in is same-origin only.
    if state.auth.is_none() {
        api_router = api_router.layer(CorsLayer::permissive());
    }
    let api_router = api_router.with_state(state.clone());

    let static_dir = &state.config.static_dir;
    if static_dir.exists() {
        // The web app's own paths (`/gallery`, `/images/{id}`, ...) get index.html with 200, so
        // shared links open; unknown API paths stay 404.
        let serve =
            ServeDir::new(static_dir).fallback(ServeFile::new(static_dir.join("index.html")));
        api_router.fallback(move |req: axum::extract::Request| {
            let serve = serve.clone();
            async move {
                let path = req.uri().path();
                if path.starts_with("/api/") || path.starts_with("/agent/") {
                    return (StatusCode::NOT_FOUND, Json(json!({ "error": "not found" })))
                        .into_response();
                }
                match serve.oneshot(req).await {
                    Ok(res) => res.into_response(),
                    Err(never) => match never {},
                }
            }
        })
    } else {
        api_router
    }
}

#[derive(Debug, Deserialize)]
pub struct ListJobsQuery {
    pub changed_since: Option<i64>,
    /// Paged: at most this many jobs, newest first (with `before`, `status`, `favorite`,
    /// `q`, `aspect`). Without it and `changed_since`: every job.
    pub limit: Option<u32>,
    /// The `next` cursor of the previous page.
    pub before: Option<String>,
    /// `done`: finished images only.
    pub status: Option<String>,
    pub favorite: Option<bool>,
    /// Substring of the prompt, the enhanced prompt or the seed.
    pub q: Option<String>,
    /// `w:h`: images of this aspect ratio.
    pub aspect: Option<String>,
}

/// `POST /api/jobs/{id}/retry`, optional: retry with another enhancer (`builtin`, or a
/// cloud enhancer's id), e.g. after a cloud enhancer failed.
#[derive(Debug, Deserialize)]
pub struct RetryJobBody {
    pub enhancer: Option<String>,
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

/// Whether `user` may see job `id`. `false` also when there is no such job: someone
/// else's job is "not found", not "forbidden".
async fn may_see_job(state: &AppState, user: &User, id: &str) -> bool {
    match state.db.job_owner(id).await {
        Ok(Some(owner)) => user.scope().is_none_or(|me| owner.as_deref() == Some(me)),
        _ => false,
    }
}

fn not_found(what: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": format!("{what} not found") })),
    )
        .into_response()
}

/// Lending, disabling and enabling cards is for the owner (any user when sign-in is off).
fn owner_only(user: &User) -> Option<Response> {
    (!user.is_owner).then(|| {
        (
            StatusCode::FORBIDDEN,
            Json(json!({ "error": "only the owner can change the cards" })),
        )
            .into_response()
    })
}

async fn submit_job(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Json(mut input): Json<JobSubmitInput>,
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
            if !may_see_job(&state, &user, id).await {
                return (
                    StatusCode::CONFLICT,
                    Json(json!({ "error": "id is taken" })),
                )
                    .into_response();
            }
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
    input.enhancer = match normalize_enhancer(&state.cloud, input.enhancer.as_deref()) {
        Ok(e) => e.filter(|_| input.enhance),
        Err(e) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "rejected": e.to_string() })),
            )
                .into_response()
        }
    };
    if let Err(reason) = check_submit_feasibility(&state.db, &input).await {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "rejected": reason })),
        )
            .into_response();
    }

    let references = match resolve_references(&state, &user, &input.references).await {
        Ok(r) => r,
        Err(reason) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "rejected": reason })),
            )
                .into_response()
        }
    };

    let seed = input.seed.unwrap_or_else(|| {
        let t = (now_s() * 1_000_000.0) as u64;
        t ^ 0x5DEECE66Du64
    });

    match state
        .db
        .create_job(&input, seed, &references, user.scope())
        .await
    {
        Ok(job) => {
            tick(&state).await;
            let refreshed = state
                .db
                .get_job(&job.id)
                .await
                .ok()
                .flatten()
                .unwrap_or(job);
            (StatusCode::CREATED, Json(refreshed)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// Turns a job's references into upload hashes. An earlier result used as a reference
/// becomes an upload of its PNG (same blob), so every reference is stored the same way.
async fn resolve_references(
    state: &Arc<AppState>,
    user: &User,
    refs: &[ReferenceInput],
) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(refs.len());
    for (i, r) in refs.iter().enumerate() {
        match (&r.upload, &r.image) {
            (Some(sha), None) => match state.db.get_upload_for(sha, user.scope()).await {
                Ok(Some(up)) => out.push(up.sha256),
                Ok(None) => return Err(format!("reference {} not found; attach it again", i + 1)),
                Err(e) => return Err(e.to_string()),
            },
            (None, Some(job_id)) => {
                if !may_see_job(state, user, job_id).await {
                    return Err(format!("reference {}: job {job_id} not found", i + 1));
                }
                let art = match state.db.get_job_artifact_by_name(job_id, "image.png").await {
                    Ok(Some(a)) => a,
                    Ok(None) => {
                        return Err(format!("reference {}: job {job_id} has no image", i + 1))
                    }
                    Err(e) => return Err(e.to_string()),
                };
                let Ok(Some(job)) = state.db.get_job(job_id).await else {
                    return Err(format!("reference {}: job {job_id} not found", i + 1));
                };
                let up = UploadRecord {
                    sha256: art.sha256,
                    mime: "image/png".to_string(),
                    width: job.width,
                    height: job.height,
                    bytes: art.bytes,
                    created_at: now_s(),
                };
                match state.db.insert_upload(&up).await {
                    Ok(up) => {
                        if let Some(owner) = user.scope() {
                            state
                                .db
                                .add_upload_owner(&up.sha256, owner)
                                .await
                                .map_err(|e| e.to_string())?;
                        }
                        out.push(up.sha256)
                    }
                    Err(e) => return Err(e.to_string()),
                }
            }
            _ => {
                return Err(format!(
                    "reference {} needs exactly one of upload or image",
                    i + 1
                ))
            }
        }
    }
    Ok(out)
}

/// Reference images are photos and renders; anything larger is not worth carrying.
const UPLOAD_MAX_BYTES: usize = 32 * 1024 * 1024;

/// `POST /api/uploads`: an image to use as a reference (PNG, JPEG or WebP, the raw bytes as
/// the body). Stored by its SHA-256, so uploading the same image twice gives the same id.
async fn upload_image(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    body: Bytes,
) -> Response {
    if body.len() > UPLOAD_MAX_BYTES {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(
                json!({ "error": format!("images up to {} MB", UPLOAD_MAX_BYTES / (1024 * 1024)) }),
            ),
        )
            .into_response();
    }
    let (mime, width, height) = match probe_upload_image(&body) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    let _writing = state.blobs.writing().await;
    let (sha256, bytes) = match state.blobs.store_blob(body).await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };
    let rec = UploadRecord {
        sha256,
        mime: mime.to_string(),
        width,
        height,
        bytes,
        created_at: now_s(),
    };
    let inserted = match state.db.insert_upload(&rec).await {
        Ok(up) => match user.scope() {
            Some(owner) => state
                .db
                .add_upload_owner(&up.sha256, owner)
                .await
                .map(|_| up),
            None => Ok(up),
        },
        Err(e) => Err(e),
    };
    match inserted {
        Ok(up) => (
            StatusCode::CREATED,
            Json(json!({
                "id": up.sha256,
                "sha256": up.sha256,
                "width": up.width,
                "height": up.height,
                "mime": up.mime,
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

async fn get_upload(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(sha): Path<String>,
) -> Response {
    let Ok(Some(up)) = state.db.get_upload_for(&sha, user.scope()).await else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "upload not found" })),
        )
            .into_response();
    };
    match state.blobs.read_blob(&up.sha256).await {
        Ok(bytes) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, up.mime),
                (header::CACHE_CONTROL, immutable(&state).to_string()),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "upload blob missing" })),
        )
            .into_response(),
    }
}

/// JPEG thumbnail of an upload, like `/api/images/{id}/thumb`; made on first request.
async fn get_upload_thumb(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(sha): Path<String>,
    Query(query): Query<ThumbQuery>,
) -> Response {
    let size = query.size.unwrap_or(512);
    if !THUMB_SIZES.contains(&size) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("size must be one of {THUMB_SIZES:?}") })),
        )
            .into_response();
    }
    let Ok(Some(up)) = state.db.get_upload_for(&sha, user.scope()).await else {
        return not_found("upload");
    };
    let key = upload_thumb_key(&up.sha256);
    let found = match state.blobs.read_thumb(&key, size).await {
        Some(bytes) => Some(bytes),
        None => {
            let blobs = state.blobs.clone();
            tokio::task::spawn_blocking(move || {
                let image = blobs.get_blob(&up.sha256).ok()?;
                blobs.write_thumbnails(&key, &image).ok()?;
                std::fs::read(blobs.thumb_path(&key, size)).ok()
            })
            .await
            .ok()
            .flatten()
        }
    };
    match found {
        Some(bytes) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "image/jpeg"),
                (header::CACHE_CONTROL, immutable(&state)),
            ],
            bytes,
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "thumbnail not found" })),
        )
            .into_response(),
    }
}

async fn list_jobs(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Query(q): Query<ListJobsQuery>,
) -> Response {
    let version = state.db.current_version().await.unwrap_or(1);
    // Deletions since the client's version, so every client drops them too.
    let deleted = match q.changed_since {
        Some(since) => state
            .db
            .list_deleted_since(since, user.scope())
            .await
            .unwrap_or_default(),
        None => Vec::new(),
    };
    let result = match (q.changed_since, q.limit) {
        (None, Some(limit)) => {
            let ratio = q.aspect.as_deref().and_then(|a| {
                let (w, h) = a.split_once(':')?;
                let (w, h) = (w.trim().parse::<f64>().ok()?, h.trim().parse::<f64>().ok()?);
                (w > 0.0 && h > 0.0).then_some(w / h)
            });
            let page = JobPageQuery {
                limit,
                before: q.before.clone(),
                done_only: q.status.as_deref() == Some("done"),
                favorites_only: q.favorite == Some(true),
                search: q.q.clone(),
                ratio,
                owner: user.scope().map(str::to_string),
            };
            state.db.list_jobs_page(&page).await
        }
        (since, _) => state
            .db
            .list_jobs(since, user.scope())
            .await
            .map(|jobs| (jobs, None)),
    };
    match result {
        Ok((mut jobs, next)) => {
            // The enhancer's reasoning is most of a job's size: lists say whether there is
            // some (`has_thinking`), and `GET /api/jobs/{id}` has it.
            for job in &mut jobs {
                job.thinking = None;
            }
            (
                StatusCode::OK,
                Json(json!({
                    "version": version,
                    "jobs": jobs,
                    "deleted": deleted,
                    "next": next,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// Counts for the gallery's filters, over all finished images.
async fn get_gallery_facets(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
) -> Response {
    match state.db.gallery_facets(user.scope()).await {
        Ok(f) => (StatusCode::OK, Json(f)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_job(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(id): Path<String>,
) -> Response {
    if !may_see_job(&state, &user, &id).await {
        return not_found("job");
    }
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
    Extension(user): Extension<User>,
    Path(id): Path<String>,
    Json(body): Json<PatchJobBody>,
) -> Response {
    if !may_see_job(&state, &user, &id).await {
        return not_found("job");
    }
    let Some(favorite) = body.favorite else {
        return match state.db.get_job(&id).await {
            Ok(Some(job)) => (StatusCode::OK, Json(job)).into_response(),
            _ => (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "job not found" })),
            )
                .into_response(),
        };
    };
    match state.db.set_job_favorite(&id, favorite).await {
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

async fn get_preferences(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
) -> Response {
    match state.db.get_preferences(user.scope()).await {
        Ok(prefs) => (StatusCode::OK, Json(prefs)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn put_preferences(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Json(prefs): Json<Value>,
) -> Response {
    if !is_small_object(&prefs, PREFERENCES_MAX_BYTES) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "preferences must be a JSON object of at most 16 KiB" })),
        )
            .into_response();
    }
    match state.db.put_preferences(&prefs, user.scope()).await {
        Ok(()) => (StatusCode::OK, Json(prefs)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn delete_job(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(id): Path<String>,
) -> Response {
    if !may_see_job(&state, &user, &id).await {
        return not_found("job");
    }
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

    let collecting = state.blobs.collecting().await;
    match state.db.delete_job(&id).await {
        Ok(Some(orphan_shas)) => {
            for sha in orphan_shas {
                let _ = state.blobs.delete_blob(&sha);
            }
            drop(collecting);
            state.blobs.delete_thumbnails(&id);
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

async fn cancel_job(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(id): Path<String>,
) -> Response {
    if !may_see_job(&state, &user, &id).await {
        return not_found("job");
    }
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
            tick(&state).await;
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

async fn retry_job(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(id): Path<String>,
    body: Option<Json<RetryJobBody>>,
) -> Response {
    if !may_see_job(&state, &user, &id).await {
        return not_found("job");
    }
    // Only a failed or cancelled job can be retried; anything else would be a silent no-op
    // that a client could mistake for "queued again" (a finished image vanished that way).
    if let Ok(Some(job)) = state.db.get_job(&id).await {
        if !matches!(job.status, JobStatus::Failed | JobStatus::Cancelled) {
            return (
                StatusCode::CONFLICT,
                Json(json!({ "error": format!("only failed or cancelled jobs can be retried (job is {})", job.status) })),
            )
                .into_response();
        }
    }
    if let Some(Json(RetryJobBody {
        enhancer: Some(enhancer),
    })) = body
    {
        let enhancer = match normalize_enhancer(&state.cloud, Some(&enhancer)) {
            Ok(e) => e,
            Err(e) => {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(json!({ "rejected": e.to_string() })),
                )
                    .into_response()
            }
        };
        if let Err(e) = state.db.set_job_enhancer(&id, enhancer.as_deref()).await {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    }
    match state.db.retry_job(&id).await {
        Ok(Some(job)) => {
            tick(&state).await;
            let refreshed = state
                .db
                .get_job(&job.id)
                .await
                .ok()
                .flatten()
                .unwrap_or(job);
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

/// A finished job's image never changes (job ids are never reused), so browsers may keep
/// images and thumbnails for good instead of downloading them again on every remount.
const IMMUTABLE: &str = "public, max-age=31536000, immutable";
/// The same with sign-in: an account's images are its own, so no shared cache keeps them.
const IMMUTABLE_PRIVATE: &str = "private, max-age=31536000, immutable";

fn immutable(state: &AppState) -> &'static str {
    if state.auth.is_some() {
        IMMUTABLE_PRIVATE
    } else {
        IMMUTABLE
    }
}

async fn get_image(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(id): Path<String>,
) -> Response {
    if !may_see_job(&state, &user, &id).await {
        return not_found("image");
    }
    // The job turns `done` just before the PNG with its provenance chunk replaces the
    // decoder's: a request in between can find the plain PNG, or a blob deleted under it.
    for _ in 0..2 {
        let Ok(Some(art)) = state.db.get_job_artifact_by_name(&id, "image.png").await else {
            break;
        };
        let Ok(bytes) = state.blobs.read_blob(&art.sha256).await else {
            continue;
        };
        // Only the final PNG is cached for good.
        let cache = if art.metadata_json.is_some() {
            immutable(&state)
        } else {
            "no-store"
        };
        return (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "image/png".to_string()),
                (
                    header::CONTENT_DISPOSITION,
                    format!("inline; filename=\"{id}.png\""),
                ),
                (header::CACHE_CONTROL, cache.to_string()),
                (header::ETAG, format!("\"{}\"", art.sha256)),
            ],
            bytes,
        )
            .into_response();
    }
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": "image not found" })),
    )
        .into_response()
}

#[derive(Deserialize)]
struct ThumbQuery {
    size: Option<u32>,
}

/// JPEG thumbnail, `?size=` 512, 768 or 1024 (longer side; default 1024). Generated on request
/// when missing, e.g. for images from before thumbnails had these sizes.
async fn get_image_thumb(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(id): Path<String>,
    Query(query): Query<ThumbQuery>,
) -> Response {
    // The id becomes part of a file path below: anything but a job id (`..%2F`) is not found.
    if !is_valid_client_job_id(&id) || !may_see_job(&state, &user, &id).await {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "image not found" })),
        )
            .into_response();
    }
    let size = query.size.unwrap_or(1024);
    if !THUMB_SIZES.contains(&size) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("size must be one of {THUMB_SIZES:?}") })),
        )
            .into_response();
    }
    let found = match state.blobs.read_thumb(&id, size).await {
        Some(bytes) => Some(bytes),
        None => match state.db.get_job_artifact_by_name(&id, "image.png").await {
            Ok(Some(art)) => {
                let blobs = state.blobs.clone();
                let id = id.clone();
                tokio::task::spawn_blocking(move || {
                    let png = blobs.get_blob(&art.sha256).ok()?;
                    blobs.write_thumbnails(&id, &png).ok()?;
                    std::fs::read(blobs.thumb_path(&id, size)).ok()
                })
                .await
                .ok()
                .flatten()
            }
            _ => None,
        },
    };
    match found {
        Some(bytes) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "image/jpeg"),
                (header::CACHE_CONTROL, immutable(&state)),
            ],
            bytes,
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "thumbnail not found" })),
        )
            .into_response(),
    }
}

async fn get_capabilities(State(state): State<Arc<AppState>>) -> Response {
    match compute_capabilities(&state.db).await {
        Ok(mut caps) => {
            caps.cloud_enhancers = state.cloud.infos();
            (StatusCode::OK, Json(caps)).into_response()
        }
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

async fn lend_executor(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(id): Path<String>,
) -> Response {
    if let Some(denied) = owner_only(&user) {
        return denied;
    }
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
            tick(&state).await;
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

async fn return_executor(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(id): Path<String>,
) -> Response {
    if let Some(denied) = owner_only(&user) {
        return denied;
    }
    match state.db.return_executor(&id).await {
        Ok(Some(ex)) => {
            tick(&state).await;
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
    Extension(user): Extension<User>,
    Path(id): Path<String>,
    body: Option<Json<DisableExecutorBody>>,
) -> Response {
    if let Some(denied) = owner_only(&user) {
        return denied;
    }
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
            tick(&state).await;
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

async fn enable_executor(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(id): Path<String>,
) -> Response {
    if let Some(denied) = owner_only(&user) {
        return denied;
    }
    match state.db.enable_executor(&id).await {
        Ok(Some(ex)) => {
            tick(&state).await;
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
    match state.blobs.read_blob(&art.sha256).await {
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

/// Agents fetch reference images and stored reference latents by hash.
async fn download_blob(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(sha): Path<String>,
) -> Response {
    if !check_agent_http_auth(&headers, &state.config.agent_token) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "unauthorized" })),
        )
            .into_response();
    }
    if !is_sha256_hex(&sha) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "not a sha256" })),
        )
            .into_response();
    }
    match state.blobs.read_blob(&sha).await {
        Ok(bytes) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/octet-stream".to_string()),
                (header::HeaderName::from_static("x-content-sha256"), sha),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "blob not found" })),
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
    if let Some(expected_hdr) = headers
        .get("x-content-sha256")
        .and_then(|v| v.to_str().ok())
    {
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
    if task.current_attempt != attempt || !task.status.is_active() {
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
    let mut cache_key: Option<String> = None;
    let mut reference_latents: Option<(ReferenceLatents, String)> = None;
    let kind = if name == "image.png" {
        "final"
    } else {
        "intermediate"
    };

    let validation_res: anyhow::Result<()> = match name.as_str() {
        "enhanced.json" => validate_enhanced_json(&body).map(|_| ()),
        "embeds.safetensors" => {
            validate_embeds_safetensors(&body, job.true_cfg_scale > 1.0, job.references.len()).map(
                |hdr| {
                    meta_json = serde_json::to_string(&hdr.metadata).ok();
                    cache_key = Some(encode_cache_key(&job, state.config.reference_res));
                },
            )
        }
        n if n.starts_with("ref") && n.ends_with(".safetensors") => {
            // ref<i>.safetensors: latents of the job's i-th reference, stored for any later job.
            match n["ref".len()..n.len() - ".safetensors".len()]
                .parse::<usize>()
                .ok()
                .and_then(|i| job.references.get(i))
            {
                None => Err(anyhow::anyhow!("job has no reference for '{n}'")),
                Some(r) => validate_reference_latents(&body, &r.sha256, state.config.reference_res)
                    .map(|(hdr, width, height)| {
                        let meta = |k: &str| hdr.metadata.get(k).cloned().unwrap_or_default();
                        reference_latents = Some((
                            ReferenceLatents {
                                source_sha256: r.sha256.clone(),
                                model_id: meta("model"),
                                model_rev: meta("rev"),
                                reference_res: state.config.reference_res,
                                blob_sha256: String::new(),
                                bytes: 0,
                                width,
                                height,
                            },
                            meta("dtype"),
                        ));
                        meta_json = serde_json::to_string(&hdr.metadata).ok();
                    }),
            }
        }
        "latent.safetensors" => {
            validate_latent_safetensors(&body, Some(job.width), Some(job.height)).map(|hdr| {
                meta_json = serde_json::to_string(&hdr.metadata).ok();
            })
        }
        "image.png" => {
            validate_png_dimensions(&body, Some(job.width), Some(job.height)).map(|_| ())
        }
        _ => Err(anyhow::anyhow!("unexpected artifact name '{name}'")),
    };

    if let Err(e) = validation_res {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response();
    }

    let _writing = state.blobs.writing().await;
    let (sha256, bytes_len) = match state.blobs.store_blob(body).await {
        Ok(res) => res,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };

    let reference_latents = reference_latents.map(|(mut lat, dtype)| {
        lat.blob_sha256 = sha256.clone();
        lat.bytes = bytes_len;
        (lat, dtype)
    });
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
            cache_key.as_deref(),
            reference_latents
                .as_ref()
                .map(|(lat, dtype)| (lat, dtype.as_str())),
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
