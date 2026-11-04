//! Studio as an MCP server (docs/INTEGRATIONS.md): `POST /mcp`, the Streamable HTTP
//! transport without sessions or server-sent events, each JSON-RPC request answered with
//! one JSON response. Its tools act as the token's account, through the same handlers as
//! the browser's API, so the rules (ownership, limits, validation) are the same.

use crate::api::{self, ListJobsQuery, PatchJobBody, ThumbQuery};
use crate::auth::{request_user, User};
use crate::db::{JobSubmitInput, ReferenceInput};
use crate::scheduler::AppState;
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Map, Value};
use std::sync::Arc;
use std::time::Duration;

/// Protocol versions this server speaks, newest first; a client asking for another gets
/// the newest.
const PROTOCOL_VERSIONS: [&str; 3] = ["2025-11-25", "2025-06-18", "2025-03-26"];
/// Longest a `get_jobs` call waits for its jobs to finish: MCP clients give up on a call
/// after about a minute.
const MAX_WAIT_S: u64 = 50;
/// Most jobs one `generate_images` call makes.
const MAX_COUNT: u64 = 4;
/// Most images one `find_images` call lists.
const MAX_FIND: u64 = 24;
/// Prompts in lists are cut to this many characters.
const LIST_PROMPT_CHARS: usize = 240;

/// The composer's aspect ratios at full size (`ASPECT_RATIOS` in web/src/types.ts).
const ASPECTS: [(&str, u32, u32); 7] = [
    ("1:1", 2048, 2048),
    ("4:3", 2400, 1792),
    ("3:4", 1792, 2400),
    ("3:2", 2528, 1696),
    ("2:3", 1696, 2528),
    ("16:9", 2752, 1536),
    ("9:16", 1536, 2752),
];
/// The composer's sizes (`SIZE_MULTIPLIERS` in web/src/types.ts).
const SIZES: [(&str, f64); 3] = [("full", 1.0), ("half", 0.5), ("quarter", 0.25)];
/// The composer's quality presets (`QUALITY_STEPS` in web/src/types.ts).
const QUALITIES: [(&str, u32); 3] = [("draft", 12), ("standard", 20), ("high", 40)];

const INSTRUCTIONS: &str = "Aries Image Studio makes images on the owner's own GPUs and keeps \
them in a gallery. Images take minutes, not seconds: generate_images starts jobs and returns \
at once; get_jobs with wait_seconds waits for them and shows the finished images. The cards \
may be asleep or busy with other work, so the first image can take longer. Search the \
gallery with find_images before making something that may already exist.";

/// `width × height` of an aspect at a size, multiples of 32 (`calculateDimensions` in
/// web/src/types.ts).
pub fn dimensions(aspect: &str, size: &str) -> Option<(u32, u32)> {
    let (_, w, h) = ASPECTS.iter().find(|(a, _, _)| *a == aspect)?;
    let (_, f) = SIZES.iter().find(|(s, _)| *s == size)?;
    let r = |v: u32| ((v as f64 * f / 32.0).round() as u32) * 32;
    Some((r(*w), r(*h)))
}

fn tools() -> Value {
    let aspects: Vec<&str> = ASPECTS.iter().map(|a| a.0).collect();
    let sizes: Vec<&str> = SIZES.iter().map(|s| s.0).collect();
    let qualities: Vec<&str> = QUALITIES.iter().map(|q| q.0).collect();
    json!([
        {
            "name": "generate_images",
            "title": "Generate images",
            "description": "Start making images from a prompt. Returns the new jobs' ids at once; \
                an image takes from about a minute (quarter size, draft) to several minutes \
                (full size, high), longer when other jobs are queued or the cards are asleep. \
                Follow up with get_jobs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "What to draw: subject, setting, style, light, composition. Any language; detailed prompts work best." },
                    "aspect": { "type": "string", "enum": aspects, "default": "1:1" },
                    "size": { "type": "string", "enum": sizes, "default": "full", "description": "full is about 4 MP (2048×2048 at 1:1); half and quarter are much faster, for drafts." },
                    "quality": { "type": "string", "enum": qualities, "default": "standard", "description": "Denoise steps: draft 12, standard 20, high 40." },
                    "count": { "type": "integer", "minimum": 1, "maximum": MAX_COUNT, "default": 1, "description": "Images of the same prompt, each with its own seed." },
                    "seed": { "type": "integer", "minimum": 0, "description": "Repeat an image exactly (with the same prompt and settings); with count, the images take seed, seed+1, ..." },
                    "enhance": { "type": "boolean", "default": false, "description": "Let Studio's prompt enhancer rewrite the prompt first (adds time). Leave off when the prompt is already detailed." },
                    "enhancer": { "type": "string", "description": "With enhance: 'local' (default, on the cards) or a cloud enhancer's id from studio_status." },
                    "negative_prompt": { "type": "string", "description": "What to keep out of the image." },
                    "references": { "type": "array", "items": { "type": "string" }, "maxItems": 4, "description": "Reference images, in order (the prompt calls them 'image 1', 'image 2', ...): ids of earlier images, or the hashes `aries upload` gives for files." }
                },
                "required": ["prompt"],
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": false, "destructiveHint": false, "idempotentHint": false, "openWorldHint": false }
        },
        {
            "name": "get_jobs",
            "title": "Check jobs",
            "description": "Status of jobs from generate_images: queued, enhancing, drawing (step and time left), decoding, done or failed. With wait_seconds, waits up to that long for them all to finish. Finished images come back as pictures (512 px) with their ids and links.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ids": { "type": "array", "items": { "type": "string" }, "minItems": 1, "maxItems": 16 },
                    "wait_seconds": { "type": "integer", "minimum": 0, "maximum": MAX_WAIT_S, "default": 0 },
                    "images": { "type": "boolean", "default": true, "description": "Include the finished images as pictures." }
                },
                "required": ["ids"],
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": true, "openWorldHint": false }
        },
        {
            "name": "find_images",
            "title": "Find images",
            "description": "The gallery's finished images: newest first, or ranked by a search (what the images show and their prompts' words, or a seed's digits), or by how alike they look to one image (like). Each result says how it matched.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Words or a description: 'rainy street at night', 'goblin'." },
                    "like": { "type": "string", "description": "An image id: images that look like it, most alike first." },
                    "favorites_only": { "type": "boolean", "default": false },
                    "limit": { "type": "integer", "minimum": 1, "maximum": MAX_FIND, "default": 8 },
                    "cursor": { "type": "string", "description": "next_cursor of the previous call, for more results." },
                    "images": { "type": "boolean", "default": true, "description": "Include each result as a small picture (512 px)." }
                },
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": true, "openWorldHint": false }
        },
        {
            "name": "view_image",
            "title": "View an image",
            "description": "One image, larger, with everything about it: prompt, enhanced prompt, seed, size, steps, references, links.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "size": { "type": "integer", "enum": [512, 768, 1024], "default": 1024, "description": "Longer side of the picture, px." }
                },
                "required": ["id"],
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": true, "openWorldHint": false }
        },
        {
            "name": "set_favorite",
            "title": "Favorite an image",
            "description": "Mark an image as a favorite, or unmark it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "favorite": { "type": "boolean", "default": true }
                },
                "required": ["id"],
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": false, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false }
        },
        {
            "name": "cancel_jobs",
            "title": "Cancel jobs",
            "description": "Stop jobs that have not finished. A finished image is kept.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ids": { "type": "array", "items": { "type": "string" }, "minItems": 1, "maxItems": 16 }
                },
                "required": ["ids"],
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": false, "destructiveHint": true, "idempotentHint": true, "openWorldHint": false }
        },
        {
            "name": "studio_status",
            "title": "Studio status",
            "description": "Whether the GPU cards are online, busy or asleep, how many jobs are waiting, the largest image Studio can make now, and the prompt enhancers there are.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false },
            "annotations": { "readOnlyHint": true, "openWorldHint": false }
        }
    ])
}

/// What a tool returns: its content blocks, and whether it failed.
struct ToolResult {
    content: Vec<Value>,
    is_error: bool,
}

impl ToolResult {
    fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![json!({ "type": "text", "text": text.into() })],
            is_error: false,
        }
    }

    fn error(text: impl Into<String>) -> Self {
        Self {
            is_error: true,
            ..Self::text(text)
        }
    }

    fn image(&mut self, jpeg: &[u8]) {
        self.content.push(json!({
            "type": "image",
            "data": STANDARD.encode(jpeg),
            "mimeType": "image/jpeg",
        }));
    }
}

/// A handler's answer: its JSON body, or the message of its error.
async fn body_json(res: Response) -> Result<Value, String> {
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024 * 1024)
        .await
        .map_err(|e| e.to_string())?;
    let v: Value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    if status.is_success() {
        Ok(v)
    } else {
        Err(v["rejected"]
            .as_str()
            .or(v["error"].as_str())
            .map(str::to_string)
            .unwrap_or_else(|| format!("Studio answered {status}")))
    }
}

async fn body_bytes(res: Response) -> Option<Bytes> {
    if !res.status().is_success() {
        return None;
    }
    axum::body::to_bytes(res.into_body(), 64 * 1024 * 1024)
        .await
        .ok()
}

/// Studio's address for links, as the request reached it.
fn base_url(state: &AppState, headers: &HeaderMap) -> String {
    if let Some(auth) = &state.auth {
        return auth.config.public_url(headers).to_string();
    }
    let host = headers
        .get("x-forwarded-host")
        .or(headers.get(header::HOST))
        .and_then(|h| h.to_str().ok())
        .unwrap_or("localhost");
    let proto = headers
        .get("x-forwarded-proto")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("http");
    format!("{proto}://{host}")
}

fn cut(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

/// Where a job is, in words: `drawing (step 7/20, about 40 s left)`.
fn job_state(job: &Value) -> String {
    let status = job["status"].as_str().unwrap_or("");
    match status {
        "done" => return "done".into(),
        "cancelled" => return "cancelled".into(),
        "failed" | "rejected" => {
            let detail = job["error_detail"]
                .as_str()
                .or(job["error_class"].as_str())
                .unwrap_or("no reason given");
            return format!("{status}: {}", cut(detail, 300));
        }
        _ => {}
    }
    let tasks = job["tasks"].as_array().cloned().unwrap_or_default();
    let active = tasks.iter().find(|t| {
        matches!(
            t["status"].as_str(),
            Some("assigned" | "running" | "uploading")
        )
    });
    match active {
        Some(t) => match t["stage"].as_str().unwrap_or("") {
            "enhance" => "enhancing the prompt".into(),
            "encode" => "encoding the prompt".into(),
            "denoise" => {
                let step = t["step"].as_u64().unwrap_or(0);
                let steps = t["steps"].as_u64().unwrap_or(0);
                let s_per_step = t["s_per_step"].as_f64().unwrap_or(0.0);
                let left = steps.saturating_sub(step) as f64 * s_per_step;
                if step > 0 && left > 0.0 {
                    format!(
                        "drawing (step {step}/{steps}, about {} s left)",
                        left.round()
                    )
                } else {
                    format!("drawing (step {step}/{steps})")
                }
            }
            "decode" => "decoding the image".into(),
            other => format!("running {other}"),
        },
        None => {
            let blocked = tasks
                .iter()
                .find_map(|t| t["blocked_reason"].as_str())
                .or(job["blocked_reason"].as_str());
            match blocked {
                Some(r) => format!("queued ({})", cut(r, 200)),
                None => "queued".into(),
            }
        }
    }
}

/// One job in a list: id, state, size and prompt.
fn job_line(job: &Value, base: &str) -> String {
    let id = job["id"].as_str().unwrap_or("");
    let mut line = format!(
        "- {id}: {} · {}×{} · seed {}",
        job_state(job),
        job["width"],
        job["height"],
        job["seed"]
    );
    if job["favorite"].as_bool() == Some(true) {
        line.push_str(" · favorite");
    }
    if job["status"] == "done" {
        line.push_str(&format!("\n  {base}/images/{id}"));
    }
    line.push_str(&format!(
        "\n  prompt: {}",
        cut(job["prompt"].as_str().unwrap_or(""), LIST_PROMPT_CHARS)
    ));
    line
}

fn str_arg<'a>(args: &'a Map<String, Value>, k: &str) -> Option<&'a str> {
    args.get(k)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn ids_arg(args: &Map<String, Value>) -> Result<Vec<String>, String> {
    let ids: Vec<String> = args
        .get("ids")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if ids.is_empty() || ids.len() > 16 {
        return Err("give 1 to 16 job ids in ids".into());
    }
    Ok(ids)
}

struct Ctx {
    state: Arc<AppState>,
    user: User,
    base: String,
}

impl Ctx {
    async fn job(&self, id: &str) -> Result<Value, String> {
        body_json(
            api::get_job(
                State(self.state.clone()),
                Extension(self.user.clone()),
                Path(id.to_string()),
            )
            .await,
        )
        .await
    }

    async fn thumb(&self, id: &str, size: u32) -> Option<Bytes> {
        body_bytes(
            api::get_image_thumb(
                State(self.state.clone()),
                Extension(self.user.clone()),
                Path(id.to_string()),
                Query(ThumbQuery { size: Some(size) }),
            )
            .await,
        )
        .await
    }

    async fn generate(&self, args: &Map<String, Value>) -> ToolResult {
        let Some(prompt) = str_arg(args, "prompt") else {
            return ToolResult::error("prompt is required");
        };
        let aspect = str_arg(args, "aspect").unwrap_or("1:1");
        let size = str_arg(args, "size").unwrap_or("full");
        let Some((width, height)) = dimensions(aspect, size) else {
            return ToolResult::error(format!(
                "aspect must be one of {:?} and size one of {:?}",
                ASPECTS.map(|a| a.0),
                SIZES.map(|s| s.0)
            ));
        };
        let quality = str_arg(args, "quality").unwrap_or("standard");
        let Some((_, steps)) = QUALITIES.iter().find(|(q, _)| *q == quality) else {
            return ToolResult::error(format!(
                "quality must be one of {:?}",
                QUALITIES.map(|q| q.0)
            ));
        };
        let count = args.get("count").and_then(Value::as_u64).unwrap_or(1);
        if !(1..=MAX_COUNT).contains(&count) {
            return ToolResult::error(format!("count must be 1 to {MAX_COUNT}"));
        }
        let seed = args.get("seed").and_then(Value::as_u64);
        let enhance = args
            .get("enhance")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let enhancer = str_arg(args, "enhancer")
            .filter(|_| enhance)
            .map(|e| if e == "local" { "builtin" } else { e })
            .map(str::to_string);
        let references: Vec<ReferenceInput> = args
            .get("references")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    // A hash is an upload (`aries upload`), anything else an image's id.
                    .map(|r| match crate::storage::is_sha256_hex(r) {
                        true => ReferenceInput {
                            upload: Some(r.to_string()),
                            image: None,
                        },
                        false => ReferenceInput {
                            upload: None,
                            image: Some(r.to_string()),
                        },
                    })
                    .collect()
            })
            .unwrap_or_default();
        let batch = ulid::Ulid::new().to_string();
        let mut made = Vec::new();
        for i in 0..count {
            // Same scheme as the web app: ULIDs, one batch, meta for its timeline.
            let id = if i == 0 {
                batch.clone()
            } else {
                ulid::Ulid::new().to_string()
            };
            let input = JobSubmitInput {
                id: Some(id),
                prompt: prompt.to_string(),
                negative_prompt: str_arg(args, "negative_prompt").unwrap_or("").to_string(),
                width,
                height,
                steps: *steps,
                seed: seed.map(|s| s.wrapping_add(i)),
                true_cfg_scale: 1.0,
                enhance,
                meta: Some(json!({
                    "batch_id": batch,
                    "aspect": aspect,
                    "size_multiplier": size,
                    "quality": quality,
                    "index": i,
                    "client": "mcp",
                })),
                tiling: false,
                references: references.clone(),
                enhance_max_tokens: None,
                enhancer: enhancer.clone(),
            };
            let res = api::submit_job(
                State(self.state.clone()),
                Extension(self.user.clone()),
                Json(input),
            )
            .await;
            match body_json(res).await {
                Ok(job) => made.push(job),
                Err(e) if made.is_empty() => return ToolResult::error(format!("Not started: {e}")),
                Err(e) => {
                    let ids: Vec<&str> = made.iter().filter_map(|j| j["id"].as_str()).collect();
                    return ToolResult::error(format!(
                        "Started {} of {count} ({}); the next was refused: {e}",
                        made.len(),
                        ids.join(", ")
                    ));
                }
            }
        }
        let lines: Vec<String> = made.iter().map(|j| job_line(j, &self.base)).collect();
        let ids: Vec<&str> = made.iter().filter_map(|j| j["id"].as_str()).collect();
        ToolResult::text(format!(
            "Started {} image{} at {width}×{height}, {steps} steps:\n{}\n\nNext: get_jobs with ids {:?} and wait_seconds {MAX_WAIT_S}.",
            made.len(),
            if made.len() == 1 { "" } else { "s" },
            lines.join("\n"),
            ids,
        ))
    }

    async fn get_jobs(&self, args: &Map<String, Value>) -> ToolResult {
        let ids = match ids_arg(args) {
            Ok(ids) => ids,
            Err(e) => return ToolResult::error(e),
        };
        let wait = args
            .get("wait_seconds")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(MAX_WAIT_S);
        let images = args.get("images").and_then(Value::as_bool).unwrap_or(true);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(wait);
        let jobs = loop {
            let mut jobs = Vec::with_capacity(ids.len());
            for id in &ids {
                jobs.push((id.clone(), self.job(id).await));
            }
            let open = jobs.iter().any(|(_, j)| {
                j.as_ref().is_ok_and(|j| {
                    !matches!(
                        j["status"].as_str(),
                        Some("done" | "failed" | "cancelled" | "rejected")
                    )
                })
            });
            if !open || tokio::time::Instant::now() >= deadline {
                break jobs;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        };
        let mut lines = Vec::new();
        let mut done = Vec::new();
        for (id, job) in &jobs {
            match job {
                Ok(j) => {
                    lines.push(job_line(j, &self.base));
                    if j["status"] == "done" {
                        done.push(id.clone());
                    }
                }
                Err(e) => lines.push(format!("- {id}: {e}")),
            }
        }
        let still = jobs
            .iter()
            .filter(|(_, j)| {
                j.as_ref().is_ok_and(|j| {
                    !matches!(
                        j["status"].as_str(),
                        Some("done" | "failed" | "cancelled" | "rejected")
                    )
                })
            })
            .count();
        let mut text = lines.join("\n");
        if still > 0 {
            text.push_str(&format!(
                "\n\n{still} not finished yet: call get_jobs again with wait_seconds {MAX_WAIT_S}."
            ));
        }
        let mut result = ToolResult::text(text);
        if images {
            for id in &done {
                if let Some(jpeg) = self.thumb(id, 512).await {
                    result
                        .content
                        .push(json!({ "type": "text", "text": format!("Image {id}:") }));
                    result.image(&jpeg);
                }
            }
        }
        result
    }

    async fn find(&self, args: &Map<String, Value>) -> ToolResult {
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(8)
            .clamp(1, MAX_FIND) as u32;
        let images = args.get("images").and_then(Value::as_bool).unwrap_or(true);
        let q = ListJobsQuery {
            changed_since: None,
            limit: Some(limit),
            before: str_arg(args, "cursor").map(str::to_string),
            status: Some("done".into()),
            favorite: args
                .get("favorites_only")
                .and_then(Value::as_bool)
                .filter(|f| *f),
            q: str_arg(args, "query").map(str::to_string),
            like: str_arg(args, "like").map(str::to_string),
            aspect: None,
            demo: None,
            batch: None,
        };
        let ranked = q.q.is_some() || q.like.is_some();
        let page = match body_json(
            api::list_jobs(
                State(self.state.clone()),
                Extension(self.user.clone()),
                Query(q),
            )
            .await,
        )
        .await
        {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };
        let jobs = page["jobs"].as_array().cloned().unwrap_or_default();
        if jobs.is_empty() {
            return ToolResult::text("No images found.");
        }
        let mut lines = Vec::new();
        for j in &jobs {
            let mut line = job_line(j, &self.base);
            if ranked {
                let mut how = Vec::new();
                if let Some(s) = j["search_strength"].as_str() {
                    how.push(format!("{s} look match"));
                }
                if let Some(w) = j["search_words"].as_str() {
                    let place = match w {
                        "enhanced" => "the enhanced prompt",
                        _ => "the prompt",
                    };
                    how.push(match (w, j["search_phrase"].as_bool() == Some(true)) {
                        ("seed", _) => "seed digits".to_string(),
                        (_, true) => format!("exact phrase in {place}"),
                        (_, false) => format!("words in {place}"),
                    });
                }
                if !how.is_empty() {
                    line.push_str(&format!("\n  matched: {}", how.join(", ")));
                }
            }
            lines.push(line);
        }
        let mut text = format!(
            "{} image{}{}:\n{}",
            jobs.len(),
            if jobs.len() == 1 { "" } else { "s" },
            if ranked {
                ", best match first"
            } else {
                ", newest first"
            },
            lines.join("\n")
        );
        if let Some(next) = page["next"].as_str() {
            text.push_str(&format!("\n\nMore: call again with cursor \"{next}\"."));
        }
        let mut result = ToolResult::text(text);
        if images {
            for j in &jobs {
                let Some(id) = j["id"].as_str() else { continue };
                if let Some(jpeg) = self.thumb(id, 512).await {
                    result
                        .content
                        .push(json!({ "type": "text", "text": format!("Image {id}:") }));
                    result.image(&jpeg);
                }
            }
        }
        result
    }

    async fn view(&self, args: &Map<String, Value>) -> ToolResult {
        let Some(id) = str_arg(args, "id") else {
            return ToolResult::error("id is required");
        };
        let size = args.get("size").and_then(Value::as_u64).unwrap_or(1024) as u32;
        let job = match self.job(id).await {
            Ok(j) => j,
            Err(e) => return ToolResult::error(e),
        };
        let mut text = format!(
            "{id}: {}\nSize {}×{}, {} steps, seed {}{}\nprompt: {}",
            job_state(&job),
            job["width"],
            job["height"],
            job["steps"],
            job["seed"],
            if job["favorite"].as_bool() == Some(true) {
                ", favorite"
            } else {
                ""
            },
            job["prompt"].as_str().unwrap_or("")
        );
        if let Some(n) = job["negative_prompt"].as_str().filter(|n| !n.is_empty()) {
            text.push_str(&format!("\nnegative prompt: {n}"));
        }
        if let Some(e) = job["enhanced_prompt"].as_str().filter(|e| !e.is_empty()) {
            text.push_str(&format!("\nenhanced prompt: {e}"));
        }
        let refs = job["references"].as_array().map(Vec::len).unwrap_or(0);
        if refs > 0 {
            text.push_str(&format!("\nreference images: {refs}"));
        }
        if job["status"] == "done" {
            text.push_str(&format!(
                "\npage: {base}/images/{id}\nfull PNG: {base}/api/images/{id}",
                base = self.base
            ));
        }
        let mut result = ToolResult::text(text);
        if job["status"] == "done" {
            match self.thumb(id, size).await {
                Some(jpeg) => result.image(&jpeg),
                None => {
                    return ToolResult::error(format!(
                        "size must be 512, 768 or 1024, or {id} has no image"
                    ))
                }
            }
        }
        result
    }

    async fn set_favorite(&self, args: &Map<String, Value>) -> ToolResult {
        let Some(id) = str_arg(args, "id") else {
            return ToolResult::error("id is required");
        };
        let favorite = args
            .get("favorite")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let res = api::patch_job(
            State(self.state.clone()),
            Extension(self.user.clone()),
            Path(id.to_string()),
            Json(PatchJobBody {
                favorite: Some(favorite),
                ..Default::default()
            }),
        )
        .await;
        match body_json(res).await {
            Ok(_) if favorite => ToolResult::text(format!("{id} is a favorite.")),
            Ok(_) => ToolResult::text(format!("{id} is no longer a favorite.")),
            Err(e) => ToolResult::error(e),
        }
    }

    async fn cancel(&self, args: &Map<String, Value>) -> ToolResult {
        let ids = match ids_arg(args) {
            Ok(ids) => ids,
            Err(e) => return ToolResult::error(e),
        };
        let mut lines = Vec::new();
        for id in &ids {
            let res = api::cancel_job(
                State(self.state.clone()),
                Extension(self.user.clone()),
                Path(id.clone()),
            )
            .await;
            lines.push(match body_json(res).await {
                Ok(job) => format!("- {id}: {}", job_state(&job)),
                Err(e) => format!("- {id}: {e}"),
            });
        }
        ToolResult::text(lines.join("\n"))
    }

    async fn status(&self) -> ToolResult {
        let system = match body_json(api::get_system(State(self.state.clone())).await).await {
            Ok(s) => s,
            Err(e) => return ToolResult::error(e),
        };
        let caps = body_json(api::get_capabilities(State(self.state.clone())).await)
            .await
            .unwrap_or(Value::Null);
        let mut lines = Vec::new();
        let agents = system["agents"].as_array().cloned().unwrap_or_default();
        if agents.is_empty() {
            lines.push("No GPU host has ever connected.".to_string());
        }
        for a in &agents {
            let cards: Vec<String> = a["executors"]
                .as_array()
                .map(|e| {
                    e.iter()
                        .map(|x| {
                            let state = x["state"].as_str().unwrap_or("?");
                            match x["current_stage"].as_str() {
                                Some(s) if state == "busy" => format!("busy ({s})"),
                                _ => state.to_string(),
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            lines.push(format!(
                "GPU host {}: {}; cards: {}",
                a["id"].as_str().unwrap_or("?"),
                if a["online"].as_bool() == Some(true) {
                    "online"
                } else {
                    "offline (asleep or down: jobs wait for it)"
                },
                if cards.is_empty() {
                    "none".to_string()
                } else {
                    cards.join(", ")
                }
            ));
        }
        let (mut queued, mut running) = (0, 0);
        for q in system["queues"].as_array().into_iter().flatten() {
            queued += q["queued"].as_u64().unwrap_or(0);
            running += q["running"].as_u64().unwrap_or(0);
        }
        lines.push(format!("Stages queued: {queued}, running: {running}."));
        if let Some(px) = caps["max_px_now"].as_u64().filter(|p| *p > 0) {
            lines.push(format!(
                "Largest image now: {:.1} MP{}.",
                px as f64 / 1e6,
                match caps["max_px_ever"].as_u64() {
                    Some(e) if e > px =>
                        format!(" ({:.1} MP when every card is free)", e as f64 / 1e6),
                    _ => String::new(),
                }
            ));
        }
        let mut enhancers = Vec::new();
        if caps["enhance_ever"].as_bool() == Some(true) {
            enhancers.push("local (on the cards)".to_string());
        }
        for c in caps["cloud_enhancers"].as_array().into_iter().flatten() {
            enhancers.push(format!(
                "{} ({})",
                c["id"].as_str().unwrap_or("?"),
                c["label"].as_str().unwrap_or("")
            ));
        }
        lines.push(if enhancers.is_empty() {
            "Prompt enhancers: none.".to_string()
        } else {
            format!("Prompt enhancers: {}.", enhancers.join(", "))
        });
        lines.push(format!("Gallery: {}/gallery", self.base));
        ToolResult::text(lines.join("\n"))
    }

    async fn call(&self, name: &str, args: &Map<String, Value>) -> Option<ToolResult> {
        Some(match name {
            "generate_images" => self.generate(args).await,
            "get_jobs" => self.get_jobs(args).await,
            "find_images" => self.find(args).await,
            "view_image" => self.view(args).await,
            "set_favorite" => self.set_favorite(args).await,
            "cancel_jobs" => self.cancel(args).await,
            "studio_status" => self.status().await,
            _ => return None,
        })
    }
}

fn rpc_result(id: &Value, result: Value) -> Response {
    Json(json!({ "jsonrpc": "2.0", "id": id, "result": result })).into_response()
}

fn rpc_error(id: &Value, code: i64, message: &str) -> Response {
    Json(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    }))
    .into_response()
}

/// `GET /mcp` (and `DELETE`): this server opens no event stream and keeps no sessions.
pub async fn mcp_other() -> Response {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        [(header::ALLOW, "POST")],
        "POST JSON-RPC messages here",
    )
        .into_response()
}

/// `POST /mcp`: one JSON-RPC message.
pub async fn mcp_post(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let user = match request_user(&state, &headers, &Method::POST).await {
        Ok((user, _)) => user,
        Err(res) => {
            let mut res = *res;
            if res.status() == StatusCode::UNAUTHORIZED {
                res.headers_mut().insert(
                    header::WWW_AUTHENTICATE,
                    header::HeaderValue::from_static("Bearer realm=\"aries\""),
                );
            }
            return res;
        }
    };
    let msg: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return rpc_error(&Value::Null, -32700, "parse error"),
    };
    if msg.is_array() {
        return rpc_error(&Value::Null, -32600, "batches are not supported");
    }
    // Notifications and responses (no `id`, or no `method`) need no answer.
    let (Some(id), Some(method)) = (msg.get("id"), msg["method"].as_str()) else {
        return StatusCode::ACCEPTED.into_response();
    };
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    match method {
        "initialize" => {
            let asked = params["protocolVersion"].as_str().unwrap_or("");
            let version = PROTOCOL_VERSIONS
                .iter()
                .find(|v| **v == asked)
                .unwrap_or(&PROTOCOL_VERSIONS[0]);
            rpc_result(
                id,
                json!({
                    "protocolVersion": version,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": {
                        "name": "aries-image-studio",
                        "title": "Aries Image Studio",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "instructions": INSTRUCTIONS,
                }),
            )
        }
        "ping" => rpc_result(id, json!({})),
        "tools/list" => rpc_result(id, json!({ "tools": tools() })),
        "tools/call" => {
            let Some(name) = params["name"].as_str() else {
                return rpc_error(id, -32602, "params.name is required");
            };
            let args = params["arguments"].as_object().cloned().unwrap_or_default();
            let ctx = Ctx {
                base: base_url(&state, &headers),
                state,
                user,
            };
            match ctx.call(name, &args).await {
                Some(r) => {
                    tracing::info!(tool = name, error = r.is_error, "MCP tool call");
                    rpc_result(id, json!({ "content": r.content, "isError": r.is_error }))
                }
                None => rpc_error(id, -32602, &format!("unknown tool {name}")),
            }
        }
        _ => rpc_error(id, -32601, &format!("method {method} not found")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The composer's sizes (`calculateDimensions` in web/src/types.ts).
    #[test]
    fn dimensions_match_the_composer() {
        assert_eq!(dimensions("1:1", "full"), Some((2048, 2048)));
        assert_eq!(dimensions("16:9", "half"), Some((1376, 768)));
        assert_eq!(dimensions("3:2", "quarter"), Some((640, 416)));
        assert_eq!(dimensions("5:1", "full"), None);
        assert_eq!(dimensions("1:1", "double"), None);
    }

    #[test]
    fn job_state_says_where_a_job_is() {
        let drawing = json!({ "status": "running", "tasks": [
            { "stage": "encode", "status": "done" },
            { "stage": "denoise", "status": "running", "step": 5, "steps": 20, "s_per_step": 2.0 }
        ]});
        assert_eq!(job_state(&drawing), "drawing (step 5/20, about 30 s left)");
        let blocked = json!({ "status": "submitted", "tasks": [
            { "stage": "denoise", "status": "queued", "blocked_reason": "no card free" }
        ]});
        assert_eq!(job_state(&blocked), "queued (no card free)");
        let failed = json!({ "status": "failed", "error_detail": "out of memory" });
        assert_eq!(job_state(&failed), "failed: out of memory");
    }
}
