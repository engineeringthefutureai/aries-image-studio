//! Cloud prompt enhancers (docs/CLOUD_ENHANCERS.md): Studio rewrites a job's prompt with a
//! provider's API (Anthropic, or Google's Gemini API) instead of an agent's enhance
//! worker. Off unless `ARIES_CLOUD_ENHANCERS` is on and the provider's key is set; the
//! built-in enhancer stays the default and the only one that works offline.
//!
//! A cloud enhance is an ordinary enhance task: assigned to the pseudo-agent `cloud` with
//! the provider as its executor, finished with the same `enhanced.json` fields and a
//! provenance record, failed (and retried once) like any other attempt.

use crate::db::{now_s, AttemptFailure, JobRecord, ModelDecl, ProvenanceRecord, TaskRecord};
use crate::scheduler::{aspect_label, enhance_request, tick, AppState};
use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Semaphore};
use tokio::time::Instant;
use tracing::{info, warn};

/// The `assigned_agent` of a cloud enhance task, and its provenance `agent`.
pub const CLOUD_AGENT: &str = "cloud";

/// Qwen-Image-2.1 takes up to 10 reference images, and both providers read that many.
pub const MAX_CLOUD_REFERENCES: u32 = 10;

/// The project's rewriting prompt (the worker's packaged copy); `ARIES_CLOUD_ENHANCER_PROMPT_FILE`
/// replaces it, e.g. with the system prompt shipped in the PE-T2I model directory.
const SYSTEM_PROMPT: &str = include_str!("../../backend/aries_worker/stages/system_prompt.txt");

/// Appended to the system prompt for a job with reference images: the request is then an
/// edit, written as an instruction like PE-I2I writes one (docs/CLOUD_ENHANCERS.md §3).
const REFERENCES_ADDENDUM: &str = r#"

## When images are attached

The user has attached reference images, in order: the first is "image 1", the second
"image 2", and so on. The request is then an edit or a composition built from them, and
these rules replace the observer's description above:

- Write an instruction for an image editing model, in one paragraph: what to change, then
  what to keep. Name every element the request touches precisely enough to find it in the
  images (position, colour, clothing), and say what it becomes.
- Then list what must stay as it is: identities and faces, poses, clothing, text, layout,
  style and anything else the request does not ask to change.
- With several images, refer to them as "image 1", "image 2", ... and say what comes from
  which.
- Keep the user's own text strings character for character.

Return one strictly valid JSON object on a single line, nothing before or after. Either
choose a new aspect ratio:

{"rewritten_prompt": "<the instruction>", "wh_ratio": "<e.g. 3:2>"}

or keep the aspect of one of the images:

{"rewritten_prompt": "<the instruction>", "ratio_follow": "<image1>"}"#;

/// Ratios the enhancer may choose (the worker's `VALID_RATIOS`).
const VALID_RATIOS: [&str; 7] = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"];

/// Reference images go out as JPEG, longest side at most this: enough to see what an edit
/// names, and a fraction of the tokens of the full image.
const REFERENCE_MAX_SIDE: u32 = 1024;

/// The reply limit when the job sets none: room for a reasoning model's thinking and the
/// JSON reply (400–800 tokens).
const DEFAULT_MAX_TOKENS: u32 = 8192;

/// Rate-limit (429) and overload answers are retried within the attempt this many times.
const BUSY_RETRIES: u32 = 4;

/// The longest a busy answer is waited out within an attempt.
const MAX_BUSY_WAIT: Duration = Duration::from_secs(120);

/// How long one request may take.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    Anthropic,
    Gemini,
}

/// One cloud enhancer: a provider's model, with that model's own limits. `Debug` leaves
/// the key out.
#[derive(Clone)]
pub struct ProviderConfig {
    /// What jobs name in `enhancer`: the model id (`claude-sonnet-5-5`, `gemini-3.5-flash-lite`).
    pub id: String,
    /// `anthropic` | `google`.
    pub provider: &'static str,
    pub api: Api,
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    /// Requests in flight at once.
    pub max_concurrent: usize,
    /// Requests and tokens per minute this model allows (0: no limit kept here).
    pub rpm: u32,
    pub tpm: u32,
    /// Gemini `thinkingLevel` (`high` | `minimal`); empty: the model's default.
    pub thinking_level: String,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("id", &self.id)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("max_concurrent", &self.max_concurrent)
            .field("rpm", &self.rpm)
            .field("tpm", &self.tpm)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Default)]
pub struct CloudConfig {
    /// Empty unless `ARIES_CLOUD_ENHANCERS` is on; then each model of a provider with a key.
    pub providers: Vec<ProviderConfig>,
    /// Replaces the packaged rewriting prompt.
    pub system_prompt: Option<String>,
}

impl CloudConfig {
    pub fn from_env() -> Self {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    /// Reads the settings through `var` (the environment, or a map in tests).
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Self {
        let get = |k: &str| {
            var(k)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let on = get("ARIES_CLOUD_ENHANCERS").is_some_and(|v| {
            matches!(v.to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes")
        });
        if !on {
            return Self::default();
        }
        let num = |k: &str, default: u32| get(k).and_then(|v| v.parse().ok()).unwrap_or(default);
        let max_concurrent = num("ARIES_CLOUD_ENHANCE_CONCURRENCY", 4).max(1) as usize;
        let mut providers = Vec::new();
        if let Some(api_key) = get("ANTHROPIC_API_KEY") {
            let model = get("ARIES_ANTHROPIC_MODEL").unwrap_or_else(|| "claude-sonnet-5-5".into());
            providers.push(ProviderConfig {
                id: model.clone(),
                provider: "anthropic",
                api: Api::Anthropic,
                api_key,
                model,
                base_url: get("ARIES_ANTHROPIC_BASE_URL")
                    .unwrap_or_else(|| "https://api.anthropic.com".into()),
                max_concurrent,
                rpm: num("ARIES_ANTHROPIC_RPM", 0),
                tpm: num("ARIES_ANTHROPIC_TPM", 0),
                thinking_level: String::new(),
            });
        }
        if let Some(api_key) = get("GEMINI_API_KEY") {
            let base_url = get("ARIES_GEMINI_BASE_URL")
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com".into());
            let thinking_level = get("ARIES_GEMINI_THINKING").unwrap_or_else(|| "high".into());
            let models = get("ARIES_GEMINI_MODELS").unwrap_or_else(|| GEMINI_MODELS.into());
            for (model, rpm, tpm) in parse_model_limits(&models) {
                providers.push(ProviderConfig {
                    id: model.clone(),
                    provider: "google",
                    api: Api::Gemini,
                    api_key: api_key.clone(),
                    model,
                    base_url: base_url.clone(),
                    max_concurrent,
                    rpm,
                    tpm,
                    thinking_level: thinking_level.clone(),
                });
            }
        }
        for p in &mut providers {
            p.base_url = p.base_url.trim_end_matches('/').to_string();
        }
        let system_prompt = get("ARIES_CLOUD_ENHANCER_PROMPT_FILE").and_then(|path| {
            match std::fs::read_to_string(&path) {
                Ok(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
                Ok(_) => None,
                Err(e) => {
                    warn!(path, error = %e, "ARIES_CLOUD_ENHANCER_PROMPT_FILE unreadable; using the packaged prompt");
                    None
                }
            }
        });
        Self {
            providers,
            system_prompt,
        }
    }
}

/// The Gemini models offered by default, each `model:rpm:tpm` with its free tier's limits:
/// Gemini 3.5 Flash-Lite (15 requests, 250K tokens a minute).
const GEMINI_MODELS: &str = "gemini-3.5-flash-lite:15:250000";

/// `model[:rpm[:tpm]],...` → `(model, rpm, tpm)`; a missing or bad limit is 0 (none kept).
fn parse_model_limits(list: &str) -> Vec<(String, u32, u32)> {
    list.split(',')
        .filter_map(|entry| {
            let mut parts = entry.trim().split(':');
            let model = parts.next()?.trim();
            if model.is_empty() {
                return None;
            }
            let mut limit = || {
                parts
                    .next()
                    .and_then(|n| n.trim().parse().ok())
                    .unwrap_or(0)
            };
            let rpm = limit();
            let tpm = limit();
            Some((model.to_string(), rpm, tpm))
        })
        .collect()
}

/// A cloud enhancer as `/api/capabilities` lists it.
#[derive(Debug, Clone, Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct CloudEnhancerInfo {
    pub id: String,
    /// `anthropic` | `google`.
    pub provider: String,
    /// For people: "Claude Sonnet 5.5", "Gemini 3.5 Flash Lite".
    pub label: String,
    pub model: String,
    pub max_references: u32,
}

/// A model id for people: `claude-sonnet-5-5` → "Claude Sonnet 5.5", `gemini-3.5-flash-lite`
/// → "Gemini 3.5 Flash Lite". Version numbers are joined with dots; a date suffix and `it` are dropped.
pub fn model_label(model: &str) -> String {
    let mut words: Vec<String> = Vec::new();
    let mut prev_numeric = false;
    for part in model.split(['-', '_']) {
        if part.is_empty()
            || part == "it"
            || (part.len() >= 8 && part.bytes().all(|b| b.is_ascii_digit()))
        {
            continue;
        }
        let numeric = part.bytes().all(|b| b.is_ascii_digit());
        if numeric && prev_numeric {
            if let Some(last) = words.last_mut() {
                last.push('.');
                last.push_str(part);
                continue;
            }
        }
        prev_numeric = numeric;
        let word = if part.bytes().any(|b| b.is_ascii_digit()) {
            part.to_ascii_uppercase()
        } else {
            let mut c = part.chars();
            c.next()
                .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
                .unwrap_or_default()
        };
        words.push(word);
    }
    words.join(" ")
}

struct Provider {
    cfg: ProviderConfig,
    slots: Semaphore,
    /// Requests sent in the last minute and their tokens, for `rpm` / `tpm`.
    window: Mutex<VecDeque<(Instant, u32)>>,
}

/// The configured providers, shared by every cloud enhance.
pub struct CloudEnhancers {
    client: reqwest::Client,
    providers: Vec<Provider>,
    system_prompt: String,
}

impl CloudEnhancers {
    pub fn new(cfg: &CloudConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_default();
        Self {
            client,
            providers: cfg
                .providers
                .iter()
                .map(|p| Provider {
                    cfg: p.clone(),
                    slots: Semaphore::new(p.max_concurrent),
                    window: Mutex::new(VecDeque::new()),
                })
                .collect(),
            system_prompt: cfg
                .system_prompt
                .clone()
                .unwrap_or_else(|| SYSTEM_PROMPT.trim().to_string()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Whether jobs may name `id` as their enhancer.
    pub fn has(&self, id: &str) -> bool {
        self.provider(id).is_some()
    }

    fn provider(&self, id: &str) -> Option<&Provider> {
        self.providers.iter().find(|p| p.cfg.id == id)
    }

    pub fn infos(&self) -> Vec<CloudEnhancerInfo> {
        self.providers
            .iter()
            .map(|p| CloudEnhancerInfo {
                id: p.cfg.id.clone(),
                provider: p.cfg.provider.to_string(),
                label: model_label(&p.cfg.model),
                model: p.cfg.model.clone(),
                max_references: MAX_CLOUD_REFERENCES,
            })
            .collect()
    }
}

impl Provider {
    /// Waits until a request of about `tokens` fits the last minute's `rpm` and `tpm`, and
    /// books it under the returned instant. Calls `on_wait` each time it has to wait.
    async fn book(&self, tokens: u32, mut on_wait: impl FnMut()) -> Instant {
        let minute = Duration::from_secs(60);
        loop {
            let wait = {
                let mut w = self.window.lock().await;
                let now = Instant::now();
                while w
                    .front()
                    .is_some_and(|(t, _)| now.duration_since(*t) >= minute)
                {
                    w.pop_front();
                }
                let used: u32 = w.iter().map(|(_, n)| n).sum();
                let rpm_ok = self.cfg.rpm == 0 || (w.len() as u32) < self.cfg.rpm;
                // A request larger than the whole budget still goes, alone.
                let tpm_ok = self.cfg.tpm == 0 || w.is_empty() || used + tokens <= self.cfg.tpm;
                if rpm_ok && tpm_ok {
                    w.push_back((now, tokens));
                    return now;
                }
                w.front()
                    .map(|(t, _)| minute.saturating_sub(now.duration_since(*t)))
                    .unwrap_or(minute)
            };
            on_wait();
            tokio::time::sleep(wait + Duration::from_millis(50)).await;
        }
    }

    /// Replaces a booking's estimate with what the provider counted.
    async fn settle(&self, booked_at: Instant, tokens: u32) {
        let mut w = self.window.lock().await;
        if let Some(entry) = w.iter_mut().find(|(t, _)| *t == booked_at) {
            entry.1 = tokens;
        }
    }
}

/// A reference image ready to send.
struct ImagePart {
    jpeg_b64: String,
    width: u32,
    height: u32,
}

/// What a provider answered.
#[derive(Debug, Default)]
struct Reply {
    text: String,
    thinking: String,
    /// The reply stopped at the token limit.
    truncated: bool,
    prompt_tokens: u32,
    output_tokens: u32,
    thought_tokens: u32,
}

/// Why an attempt failed: `class` goes to the attempt and job, `detail` is shown.
#[derive(Debug)]
struct CloudError {
    class: &'static str,
    detail: String,
    /// A rate-limit or overload answer: try again after this long.
    retry_after: Option<Duration>,
}

impl CloudError {
    fn new(class: &'static str, detail: impl Into<String>) -> Self {
        Self {
            class,
            detail: detail.into(),
            retry_after: None,
        }
    }
}

impl From<anyhow::Error> for CloudError {
    fn from(e: anyhow::Error) -> Self {
        Self::new("cloud_error", format!("{e:#}"))
    }
}

/// Runs one assigned cloud enhance attempt to the end: done with the rewritten prompt, or
/// failed (the scheduler retries it once). Then ticks the scheduler, so encode starts.
///
/// Boxed with `Send` spelled out: the scheduler tick spawns it and it awaits the tick, a
/// cycle the compiler cannot prove `Send` through opaque futures.
pub fn run_enhance(
    state: Arc<AppState>,
    task: TaskRecord,
    job: JobRecord,
    attempt: u32,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(run_enhance_inner(state, task, job, attempt))
}

async fn run_enhance_inner(state: Arc<AppState>, task: TaskRecord, job: JobRecord, attempt: u32) {
    // `assign_task` hands over the job row alone: references come with the full record.
    let job = match state.db.get_job(&job.id).await {
        Ok(Some(full)) => full,
        _ => job,
    };
    let provider_id = job.enhancer.clone().unwrap_or_default();
    let outcome = enhance(&state, &task, &job, attempt, &provider_id).await;
    let db = &state.db;
    let recorded = match outcome {
        Ok((enhanced, prov)) => {
            info!(job = %job.id, provider = %provider_id, "cloud enhance done");
            db.complete_task_attempt(
                &task.id,
                attempt,
                &prov,
                Some((
                    enhanced.rewritten_prompt.as_str(),
                    enhanced.wh_ratio.as_str(),
                    enhanced.thinking.as_str(),
                    enhanced.ratio_follow.as_str(),
                )),
                None,
            )
            .await
            .map(|_| ())
        }
        Err(e) => {
            warn!(job = %job.id, provider = %provider_id, class = e.class, detail = %e.detail, "cloud enhance failed");
            db.fail_task_attempt(
                &task.id,
                attempt,
                AttemptFailure::studio(e.class, &e.detail),
            )
            .await
            .map(|_| ())
        }
    };
    if let Err(e) = recorded {
        warn!(job = %job.id, error = %e, "could not record the cloud enhance result");
    }
    tick(&state).await;
}

async fn enhance(
    state: &Arc<AppState>,
    task: &TaskRecord,
    job: &JobRecord,
    attempt: u32,
    provider_id: &str,
) -> std::result::Result<(crate::storage::EnhancedJson, ProvenanceRecord), CloudError> {
    let cloud = &state.cloud;
    let Some(provider) = cloud.provider(provider_id) else {
        return Err(CloudError::new(
            "cloud_unavailable",
            format!("cloud enhancer '{provider_id}' is not enabled on this Studio"),
        ));
    };
    let started = now_s();
    let images = load_references(state, job).await?;
    let system = if images.is_empty() {
        cloud.system_prompt.clone()
    } else {
        format!("{}{REFERENCES_ADDENDUM}", cloud.system_prompt)
    };
    let user = enhance_request(&job.prompt, job.width, job.height);
    let max_tokens = job.enhance_max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let estimate = estimate_tokens(provider.cfg.api, &system, &user, &images, max_tokens);

    let _slot = provider
        .slots
        .acquire()
        .await
        .map_err(|_| CloudError::new("cloud_error", "shutting down"))?;
    let mut busy_retries = 0;
    let (reply, booked_at, waited_s) = loop {
        let mut waited = false;
        let wait_from = now_s();
        let booked_at = provider
            .book(estimate, || {
                if !waited {
                    waited = true;
                    let db = state.db.clone();
                    let task_id = task.id.clone();
                    tokio::spawn(async move {
                        let _ = db
                            .record_task_progress(
                                &task_id,
                                attempt,
                                0,
                                0,
                                0.0,
                                Some("waiting"),
                                None,
                            )
                            .await;
                    });
                }
            })
            .await;
        let waited_s = now_s() - wait_from;
        // Cancelled while it waited: nothing to send.
        if !task_still_current(state, &task.id, attempt).await {
            return Err(CloudError::new(
                "cancelled",
                "cancelled before the request was sent",
            ));
        }
        let _ = state
            .db
            .record_task_progress(&task.id, attempt, 0, 0, 0.0, Some("rewriting"), None)
            .await;
        let sent = match provider.cfg.api {
            Api::Anthropic => {
                call_anthropic(
                    &cloud.client,
                    &provider.cfg,
                    &system,
                    &user,
                    &images,
                    max_tokens,
                )
                .await
            }
            Api::Gemini => {
                call_gemini(
                    &cloud.client,
                    &provider.cfg,
                    &system,
                    &user,
                    &images,
                    max_tokens,
                )
                .await
            }
        };
        match sent {
            Ok(reply) => break (reply, booked_at, waited_s),
            Err(e) if e.retry_after.is_some() && busy_retries < BUSY_RETRIES => {
                busy_retries += 1;
                let pause = e.retry_after.unwrap_or_default();
                warn!(job = %job.id, provider = provider_id, detail = %e.detail, pause_s = pause.as_secs(), "provider busy; trying again");
                let _ = state
                    .db
                    .record_task_progress(&task.id, attempt, 0, 0, 0.0, Some("waiting"), None)
                    .await;
                tokio::time::sleep(pause).await;
            }
            Err(e) => return Err(e),
        }
    };
    provider
        .settle(booked_at, reply.prompt_tokens + reply.output_tokens)
        .await;

    let enhanced = parse_reply(&reply, job)?;
    let elapsed = now_s() - started;
    let request_s = (elapsed - waited_s).max(0.001);
    let mut timings: HashMap<String, f64> = HashMap::from([
        ("enhance_s".to_string(), elapsed),
        ("prompt_tokens".to_string(), reply.prompt_tokens as f64),
        ("generated_tokens".to_string(), reply.output_tokens as f64),
        ("max_new_tokens".to_string(), max_tokens as f64),
        (
            "generation_tps".to_string(),
            reply.output_tokens as f64 / request_s,
        ),
    ]);
    if reply.thought_tokens > 0 {
        timings.insert("thought_tokens".to_string(), reply.thought_tokens as f64);
        timings.insert(
            "reply_tokens".to_string(),
            reply.output_tokens.saturating_sub(reply.thought_tokens) as f64,
        );
    }
    if waited_s >= 0.5 {
        timings.insert("rate_wait_s".to_string(), waited_s);
    }
    let prov = ProvenanceRecord {
        agent: CLOUD_AGENT.to_string(),
        executor: provider.cfg.id.clone(),
        device: provider
            .cfg
            .base_url
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .to_string(),
        backend_kind: match provider.cfg.api {
            Api::Anthropic => "anthropic",
            Api::Gemini => "gemini",
        }
        .to_string(),
        model: ModelDecl {
            id: provider.cfg.model.clone(),
            rev: String::new(),
            dtype: String::new(),
        },
        torch_version: String::new(),
        worker_version: format!("aries-image-studio {}", env!("CARGO_PKG_VERSION")),
        timings,
        peak_mem_bytes: 0,
    };
    Ok((enhanced, prov))
}

async fn task_still_current(state: &AppState, task_id: &str, attempt: u32) -> bool {
    matches!(
        state.db.get_task(task_id).await,
        Ok(Some(t)) if t.current_attempt == attempt && t.status.is_active()
    )
}

/// The job's reference images as JPEG, longest side at most [`REFERENCE_MAX_SIDE`].
async fn load_references(state: &AppState, job: &JobRecord) -> Result<Vec<ImagePart>> {
    let mut parts = Vec::with_capacity(job.references.len());
    for r in &job.references {
        let bytes = state.blobs.read_blob(&r.sha256).await.with_context(|| {
            format!(
                "reference image {} unreadable",
                &r.sha256[..12.min(r.sha256.len())]
            )
        })?;
        let part = tokio::task::spawn_blocking(move || to_jpeg_part(&bytes)).await??;
        parts.push(part);
    }
    Ok(parts)
}

fn to_jpeg_part(bytes: &[u8]) -> Result<ImagePart> {
    let img = image::load_from_memory(bytes).context("reference image does not decode")?;
    let img = if img.width().max(img.height()) > REFERENCE_MAX_SIDE {
        img.resize(
            REFERENCE_MAX_SIDE,
            REFERENCE_MAX_SIDE,
            image::imageops::FilterType::Triangle,
        )
    } else {
        img
    };
    let rgb = img.to_rgb8();
    let mut out = std::io::Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 90).encode_image(&rgb)?;
    Ok(ImagePart {
        jpeg_b64: base64::engine::general_purpose::STANDARD.encode(out.into_inner()),
        width: rgb.width(),
        height: rgb.height(),
    })
}

/// About how many tokens a request counts against `tpm`: text at ~4 characters a token,
/// images as each provider counts them, and the reply at a quarter of its limit (most
/// replies stay well under it; [`Provider::settle`] puts in the real count afterwards).
fn estimate_tokens(
    api: Api,
    system: &str,
    user: &str,
    images: &[ImagePart],
    max_tokens: u32,
) -> u32 {
    let text = ((system.len() + user.len()) / 4) as u32;
    let image_tokens: u32 = images
        .iter()
        .map(|i| match api {
            // ~w·h/750 (Anthropic's vision docs).
            Api::Anthropic => i.width * i.height / 750,
            // 258 per 768-px tile (Gemini).
            Api::Gemini => i.width.div_ceil(768) * i.height.div_ceil(768) * 258,
        })
        .sum();
    text + image_tokens + max_tokens / 4
}

/// A provider's error answer as a [`CloudError`]: 429 and overload are "busy" and retried
/// after `Retry-After`, Gemini's `RetryInfo.retryDelay`, or 20 s. A wait longer than
/// [`MAX_BUSY_WAIT`] (a daily quota used up) fails at once instead.
fn http_error(
    provider: &str,
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> CloudError {
    let json = serde_json::from_str::<Value>(body).ok();
    let message = json
        .as_ref()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| body.chars().take(300).collect());
    let busy = matches!(status.as_u16(), 429 | 503 | 529);
    let class = match status.as_u16() {
        401 | 403 => "cloud_auth",
        429 => "cloud_rate_limited",
        _ => "cloud_error",
    };
    let header_s = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<f64>().ok());
    let gemini_s = json.as_ref().and_then(|v| {
        v["error"]["details"]
            .as_array()?
            .iter()
            .find_map(|d| d["retryDelay"].as_str())?
            .trim_end_matches('s')
            .parse::<f64>()
            .ok()
    });
    let wait = Duration::from_secs_f64(header_s.or(gemini_s).unwrap_or(20.0).max(1.0));
    CloudError {
        class,
        detail: format!("{provider}: HTTP {}: {message}", status.as_u16()),
        retry_after: (busy && wait <= MAX_BUSY_WAIT).then_some(wait),
    }
}

async fn post_json(
    cfg: &ProviderConfig,
    req: reqwest::RequestBuilder,
    body: &Value,
) -> std::result::Result<Value, CloudError> {
    let label = model_label(&cfg.model);
    let resp = req.json(body).send().await.map_err(|e| {
        // A dropped connection or a timeout is worth one more try.
        let transient = e.is_connect() || e.is_timeout();
        let mut err = CloudError::new(
            "cloud_error",
            format!("{label}: request failed: {}", e.without_url()),
        );
        if transient {
            err.retry_after = Some(Duration::from_secs(5));
        }
        err
    })?;
    let status = resp.status();
    let headers = resp.headers().clone();
    let text = resp.text().await.map_err(|e| {
        CloudError::new(
            "cloud_error",
            format!("{label}: reading the reply: {}", e.without_url()),
        )
    })?;
    if !status.is_success() {
        return Err(http_error(&label, status, &headers, &text));
    }
    serde_json::from_str(&text)
        .map_err(|_| CloudError::new("cloud_error", format!("{label}: the reply is not JSON")))
}

async fn call_anthropic(
    client: &reqwest::Client,
    cfg: &ProviderConfig,
    system: &str,
    user: &str,
    images: &[ImagePart],
    max_tokens: u32,
) -> std::result::Result<Reply, CloudError> {
    let mut content: Vec<Value> = images
        .iter()
        .map(|i| {
            json!({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": i.jpeg_b64},
            })
        })
        .collect();
    content.push(json!({"type": "text", "text": user}));
    let body = json!({
        "model": cfg.model,
        "max_tokens": max_tokens,
        // The system prompt is the same every time: cached, it costs a tenth on reads.
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": content}],
    });
    let req = client
        .post(format!("{}/v1/messages", cfg.base_url))
        .header("x-api-key", &cfg.api_key)
        .header("anthropic-version", "2023-06-01");
    let v = post_json(cfg, req, &body).await?;
    let mut reply = Reply::default();
    for block in v["content"].as_array().into_iter().flatten() {
        match block["type"].as_str() {
            Some("text") => reply
                .text
                .push_str(block["text"].as_str().unwrap_or_default()),
            Some("thinking") => reply
                .thinking
                .push_str(block["thinking"].as_str().unwrap_or_default()),
            _ => {}
        }
    }
    reply.truncated = v["stop_reason"] == "max_tokens";
    let usage = &v["usage"];
    let n = |k: &str| usage[k].as_u64().unwrap_or(0) as u32;
    reply.prompt_tokens =
        n("input_tokens") + n("cache_read_input_tokens") + n("cache_creation_input_tokens");
    reply.output_tokens = n("output_tokens");
    Ok(reply)
}

async fn call_gemini(
    client: &reqwest::Client,
    cfg: &ProviderConfig,
    system: &str,
    user: &str,
    images: &[ImagePart],
    max_tokens: u32,
) -> std::result::Result<Reply, CloudError> {
    let mut parts: Vec<Value> = images
        .iter()
        .map(|i| json!({"inlineData": {"mimeType": "image/jpeg", "data": i.jpeg_b64}}))
        .collect();
    parts.push(json!({"text": user}));
    let mut generation = json!({"maxOutputTokens": max_tokens});
    if !cfg.thinking_level.is_empty() {
        // Thought parts come back only when asked for.
        generation["thinkingConfig"] =
            json!({"thinkingLevel": cfg.thinking_level, "includeThoughts": true});
    }
    let body = json!({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation,
    });
    let req = client
        .post(format!(
            "{}/v1beta/models/{}:generateContent",
            cfg.base_url, cfg.model
        ))
        .header("x-goog-api-key", &cfg.api_key);
    let v = post_json(cfg, req, &body).await?;
    let candidate = &v["candidates"][0];
    if candidate.is_null() {
        let reason = v["promptFeedback"]["blockReason"]
            .as_str()
            .unwrap_or("no candidates");
        return Err(CloudError::new(
            "cloud_refused",
            format!("{}: no reply ({reason})", model_label(&cfg.model)),
        ));
    }
    let mut reply = Reply::default();
    for part in candidate["content"]["parts"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let text = part["text"].as_str().unwrap_or_default();
        if part["thought"].as_bool() == Some(true) {
            reply.thinking.push_str(text);
        } else {
            reply.text.push_str(text);
        }
    }
    reply.truncated = candidate["finishReason"] == "MAX_TOKENS";
    let usage = &v["usageMetadata"];
    let n = |k: &str| usage[k].as_u64().unwrap_or(0) as u32;
    reply.prompt_tokens = n("promptTokenCount");
    reply.thought_tokens = n("thoughtsTokenCount");
    reply.output_tokens = n("candidatesTokenCount") + reply.thought_tokens;
    Ok(reply)
}

/// The reply's JSON object as `enhanced.json` fields. A reply without a usable
/// `rewritten_prompt` fails the attempt: a cloud model that wrote something else is not
/// guessed at (the worker's lenient parser is for the local model's slips).
fn parse_reply(
    reply: &Reply,
    job: &JobRecord,
) -> std::result::Result<crate::storage::EnhancedJson, CloudError> {
    let object = extract_json_object(&reply.text).ok_or_else(|| {
        let why = if reply.truncated {
            "the reply ran out of tokens before its JSON (raise the enhancer token limit)"
        } else if reply.text.trim().is_empty() {
            "the reply was empty"
        } else {
            "the reply had no JSON object"
        };
        CloudError::new("bad_output", why)
    })?;
    let rewritten = object["rewritten_prompt"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CloudError::new("bad_output", "the reply's JSON has no rewritten_prompt"))?;
    let wh_ratio = object["wh_ratio"]
        .as_str()
        .map(str::trim)
        .filter(|r| VALID_RATIOS.contains(r))
        .unwrap_or_else(|| aspect_label(job.width, job.height));
    let refs = job.references.len();
    let ratio_follow = object["ratio_follow"]
        .as_str()
        .and_then(|s| {
            s.trim()
                .strip_prefix("<image")?
                .strip_suffix('>')?
                .parse::<usize>()
                .ok()
        })
        .filter(|n| (1..=refs).contains(n))
        .map(|n| format!("<image{n}>"))
        .unwrap_or_default();
    Ok(crate::storage::EnhancedJson {
        rewritten_prompt: rewritten.to_string(),
        wh_ratio: wh_ratio.to_string(),
        thinking: reply.thinking.trim().to_string(),
        ratio_follow,
    })
}

/// The first JSON object in `text` that has a `rewritten_prompt`, also inside a ```json
/// fence or after some prose.
fn extract_json_object(text: &str) -> Option<Value> {
    let mut from = 0;
    while let Some(start) = text[from..].find('{').map(|i| from + i) {
        let mut stream = serde_json::Deserializer::from_str(&text[start..]).into_iter::<Value>();
        if let Some(Ok(v)) = stream.next() {
            if v.get("rewritten_prompt").is_some() {
                return Some(v);
            }
        }
        from = start + 1;
    }
    None
}

/// Checks a submitted `enhancer` against what this Studio has: `None` and `builtin` mean
/// the agents' enhancer.
pub fn normalize_enhancer(
    cloud: &CloudEnhancers,
    enhancer: Option<&str>,
) -> Result<Option<String>> {
    match enhancer.map(str::trim) {
        None | Some("") | Some("builtin") => Ok(None),
        Some(id) if cloud.has(id) => Ok(Some(id.to_string())),
        Some(id) if cloud.is_empty() => {
            bail!("cloud enhancers are off on this Studio (enhancer '{id}')")
        }
        Some(id) => Err(anyhow!("unknown enhancer '{id}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_read_like_product_names() {
        assert_eq!(model_label("claude-sonnet-5-5"), "Claude Sonnet 5.5");
        assert_eq!(model_label("gemma-4-31b-it"), "Gemma 4 31B");
        assert_eq!(
            model_label("claude-sonnet-4-5-20250929"),
            "Claude Sonnet 4.5"
        );
    }

    #[test]
    fn off_unless_switched_on_and_keyed() {
        let vars = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.to_string())
            }
        };
        let keys_only =
            CloudConfig::from_vars(vars(&[("ANTHROPIC_API_KEY", "k"), ("GEMINI_API_KEY", "g")]));
        assert!(keys_only.providers.is_empty());
        let on = CloudConfig::from_vars(vars(&[
            ("ARIES_CLOUD_ENHANCERS", "on"),
            ("GEMINI_API_KEY", "g"),
        ]));
        let models: Vec<_> = on
            .providers
            .iter()
            .map(|p| (p.id.as_str(), p.provider, p.rpm, p.tpm))
            .collect();
        assert_eq!(models, [("gemini-3.5-flash-lite", "google", 15, 250_000)]);
        let gemini = &on.providers[0];
        assert!(
            !format!("{gemini:?}").contains("\"g\""),
            "Debug leaves the key out"
        );
    }

    #[test]
    fn busy_answers_wait_unless_the_quota_is_gone_for_long() {
        let none = reqwest::header::HeaderMap::new();
        let gemini = |delay: &str| {
            json!({"error": {"code": 429, "message": "quota", "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}
            ]}})
            .to_string()
        };
        let e = http_error(
            "Gemini 3.5 Flash Lite",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            &none,
            &gemini("37s"),
        );
        assert_eq!(e.retry_after, Some(Duration::from_secs(37)));
        assert_eq!(e.class, "cloud_rate_limited");
        let e = http_error(
            "Gemini 3.5 Flash Lite",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            &none,
            &gemini("43200s"),
        );
        assert_eq!(e.retry_after, None, "a daily quota fails now");
        assert_eq!(e.detail, "Gemini 3.5 Flash Lite: HTTP 429: quota");
        let e = http_error("x", reqwest::StatusCode::UNAUTHORIZED, &none, "{}");
        assert_eq!((e.class, e.retry_after), ("cloud_auth", None));
    }

    #[tokio::test(start_paused = true)]
    async fn the_minute_budget_holds_requests_back() {
        let cfg = CloudConfig::from_vars(|k| match k {
            "ARIES_CLOUD_ENHANCERS" => Some("on".into()),
            "GEMINI_API_KEY" => Some("g".into()),
            "ARIES_GEMINI_MODELS" => Some("m:0:1000".into()),
            _ => None,
        });
        let cloud = CloudEnhancers::new(&cfg);
        let p = cloud.provider("m").unwrap();
        let t0 = Instant::now();
        let mut waits = 0;
        p.book(600, || waits += 1).await;
        p.book(600, || waits += 1).await;
        assert_eq!(waits, 1);
        assert!(
            t0.elapsed() >= Duration::from_secs(60),
            "the second waits for the first to age out"
        );
        // Settled lower, the next one fits at once.
        let at = p.book(900, || waits += 1).await;
        p.settle(at, 100).await;
        p.book(500, || waits += 1).await;
        assert_eq!(waits, 2);
    }

    #[test]
    fn json_is_found_in_fences_and_prose() {
        let v = extract_json_object(
            "Sure!\n```json\n{\"rewritten_prompt\": \"a {red} cat\", \"wh_ratio\": \"3:2\"}\n```",
        )
        .unwrap();
        assert_eq!(v["rewritten_prompt"], "a {red} cat");
        assert!(extract_json_object("{\"a\": 1} no prompt here").is_none());
    }
}
