use crate::cloud::{CloudEnhancerInfo, CloudEnhancers, CLOUD_AGENT};
use crate::config::BackendConfig;
use crate::db::{
    now_s, AgentRecord, Database, ExecutorRecord, ExecutorState, JobRecord, JobSubmitInput,
    ModelDecl, RequirementDecl, TaskRecord, TaskShape,
};
use crate::metrics::{AgentLiveMetrics, BackendMetricsRegistry};
use crate::proto;
use crate::storage::BlobStore;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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
    /// Most reference images a job may carry (0: none accepted).
    pub max_references_now: u32,
    pub max_references_ever: u32,
    pub enhance_now: bool,
    pub enhance_ever: bool,
    /// Most reference images Enhance reads (an image-aware enhancer, docs/ENHANCER_I2I.md);
    /// 0: Enhance is off for jobs with references.
    pub enhance_max_references_now: u32,
    pub enhance_max_references_ever: u32,
    pub sizes: Vec<SizeCapabilityInfo>,
    pub stages: Vec<StageCapabilityInfo>,
    /// Cloud prompt enhancers a job may name in `enhancer` (docs/CLOUD_ENHANCERS.md);
    /// empty unless switched on.
    #[serde(default)]
    pub cloud_enhancers: Vec<CloudEnhancerInfo>,
}

pub struct AppState {
    pub config: BackendConfig,
    pub db: Database,
    pub blobs: BlobStore,
    pub cloud: CloudEnhancers,
    pub metrics: BackendMetricsRegistry,
    pub live_metrics: RwLock<HashMap<String, AgentLiveMetrics>>,
    pub agent_streams: RwLock<HashMap<String, AgentTx>>,
    pub sched_lock: tokio::sync::Mutex<()>,
}

impl AppState {
    pub fn new(config: BackendConfig, db: Database, blobs: BlobStore) -> Arc<Self> {
        let cloud = CloudEnhancers::new(&config.cloud);
        Arc::new(Self {
            cloud,
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

/// The largest image an executor can both denoise and decode.
fn executor_max_px(ex: &ExecutorRecord) -> u64 {
    let max_px = |stage: &str| {
        ex.stages
            .iter()
            .find(|s| s.stage == stage)
            .map(|s| s.max_px)
            .unwrap_or(0)
    };
    max_px("denoise").min(max_px("decode"))
}

pub fn executor_supports_stage_static(
    ex: &ExecutorRecord,
    agent: &AgentRecord,
    stage: &str,
    px: u64,
    refs: u32,
    required_model: Option<(&str, &str)>,
) -> Option<ModelDecl> {
    let cap = ex.stages.iter().find(|s| s.stage == stage)?;
    if matches!(stage, "denoise" | "decode") && cap.max_px > 0 && px > cap.max_px {
        return None;
    }
    // Reference images go through encode and denoise, and enhance when the agent has an
    // image-aware enhancer; decode never sees them.
    if matches!(stage, "encode" | "denoise" | "enhance") && refs > cap.max_references {
        return None;
    }
    // `reference_models` take references: a task with references needs one of them, one
    // without prefers the others (enhance: PE-I2I vs PE-T2I).
    let takes_refs = |id: &String| cap.reference_models.contains(id);
    let models: Vec<&String> = if cap.reference_models.is_empty() {
        cap.models.iter().collect()
    } else if refs > 0 {
        cap.models.iter().filter(|id| takes_refs(id)).collect()
    } else {
        let (plain, with_refs): (Vec<&String>, Vec<&String>) =
            cap.models.iter().partition(|id| !takes_refs(id));
        plain.into_iter().chain(with_refs).collect()
    };
    for model_id in models {
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
    format!(
        "{}\n\nAspect ratio: {}",
        prompt.trim_end(),
        aspect_label(width, height)
    )
}

/// Checks submit-time feasibility against all known executors seen in the last 7 days (§6.2).
/// Denoising steps a job may ask for (the UI's custom field allows the same range).
pub const MAX_STEPS: u32 = 100;

/// The prompt rewriter's token limit a job may set: below 256 it cannot finish reasoning,
/// and 24576 (the PE-I2I model card generates with 24000) takes ~13 minutes at the int4
/// enhancers' ~31 tok/s.
pub const ENHANCE_TOKENS_RANGE: std::ops::RangeInclusive<u32> = 256..=24576;

/// Qwen-Image-2.1 takes up to 10 condition images; agents declare their own, lower limit.
pub const MAX_REFERENCES: usize = 10;

/// Most references an executor takes through both encode and denoise.
pub fn executor_max_references(ex: &ExecutorRecord) -> u32 {
    let cap = |stage: &str| {
        ex.stages
            .iter()
            .find(|s| s.stage == stage)
            .map(|s| s.max_references)
            .unwrap_or(0)
    };
    cap("encode").min(cap("denoise"))
}

/// Most references an executor's enhancer reads (0 without an image-aware enhancer).
pub fn executor_enhance_max_references(ex: &ExecutorRecord) -> u32 {
    ex.stages
        .iter()
        .find(|s| s.stage == "enhance" && !s.reference_models.is_empty())
        .map(|s| s.max_references)
        .unwrap_or(0)
}

/// Whether `ex`'s resident worker can take a task of `stage`: the same stage and, for
/// enhance, the model the task needs on this executor (PE-I2I with references, PE-T2I
/// without), so a text enhancer never gets a job with images or the other way round. An
/// enhancer whose model is unknown (re-announced after a reconnect) serves nothing; it is
/// released and started again with a known one.
fn resident_serves(
    ex: &ExecutorRecord,
    agent: Option<&AgentRecord>,
    stage: &str,
    px: u64,
    refs: u32,
) -> bool {
    if ex.current_stage.as_deref() != Some(stage) {
        return false;
    }
    if stage != "enhance" {
        return true;
    }
    let Some(agent) = agent else {
        return false;
    };
    match (
        executor_supports_stage_static(ex, agent, stage, px, refs, None),
        &ex.current_model_id,
    ) {
        (Some(m), Some(cur)) => &m.id == cur,
        (Some(_), None) | (None, _) => false,
    }
}

/// What a job's encode depends on (docs/REFERENCE_IMAGES.md §4): jobs of one send with the
/// same key share one encode. Seed, steps, size and tiling do not change the embeddings.
pub fn encode_cache_key(job: &JobRecord, reference_res: u32) -> String {
    let (prompt, source) = match &job.enhanced_prompt {
        Some(ep) => (ep.as_str(), "enhanced"),
        None => (job.prompt.as_str(), "original"),
    };
    let cfg = job.true_cfg_scale > 1.0;
    let refs: Vec<&str> = job.references.iter().map(|r| r.sha256.as_str()).collect();
    let canonical = serde_json::json!({
        "prompt": prompt,
        "prompt_source": source,
        "negative_prompt": if cfg { job.negative_prompt.as_str() } else { "" },
        "true_cfg_scale": if cfg { job.true_cfg_scale } else { 1.0 },
        "references": refs,
        "reference_res": if refs.is_empty() { 0 } else { reference_res },
    });
    hex::encode(Sha256::digest(canonical.to_string().as_bytes()))
}

pub async fn check_submit_feasibility(db: &Database, input: &JobSubmitInput) -> Result<(), String> {
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
        return Err(format!(
            "invalid steps {} (must be 1–{MAX_STEPS})",
            input.steps
        ));
    }
    if let Some(t) = input.enhance_max_tokens {
        if !ENHANCE_TOKENS_RANGE.contains(&t) {
            return Err(format!(
                "invalid enhance_max_tokens {t} (must be {}–{})",
                ENHANCE_TOKENS_RANGE.start(),
                ENHANCE_TOKENS_RANGE.end()
            ));
        }
    }
    let refs = input.references.len() as u32;
    if input.references.len() > MAX_REFERENCES {
        return Err(format!("at most {MAX_REFERENCES} reference images"));
    }
    if refs > 0 && input.true_cfg_scale > 1.0 {
        return Err("true_cfg_scale > 1 is not available with reference images".to_string());
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
        .filter(|e| {
            now - e.last_seen_at <= KNOWN_EXECUTOR_WINDOW_S && agents.contains_key(&e.agent_id)
        })
        .collect();

    if executors.is_empty() {
        return Err("no known executor registered in the last 7 days".to_string());
    }

    let px = (input.width as u64) * (input.height as u64);
    if input.tiling
        && !executors
            .iter()
            .any(|ex| executor_supports_decode_mode(ex, true))
    {
        return Err("no executor declares tiled decode".to_string());
    }
    if refs > 0 {
        let most = executors
            .iter()
            .map(executor_max_references)
            .max()
            .unwrap_or(0);
        if refs > most {
            return Err(if most == 0 {
                "no known executor accepts reference images".to_string()
            } else {
                format!("{refs} reference images, at most {most}")
            });
        }
    }

    // A cloud enhancer (checked by the API) reads text and images without an executor.
    if input.enhance && input.enhancer.is_none() {
        // With references only an image-aware enhancer will do: the text one could rewrite
        // away what an edit asks for.
        let can_enhance = executors.iter().any(|ex| {
            agents
                .get(&ex.agent_id)
                .and_then(|ag| executor_supports_stage_static(ex, ag, "enhance", px, refs, None))
                .is_some()
        });
        if !can_enhance {
            return Err(if refs > 0 {
                format!("no known executor can enhance with {refs} reference images")
            } else {
                "no known executor supports stage 'enhance'".to_string()
            });
        }
    }

    for stage in ["encode", "denoise", "decode"] {
        let can_stage = executors.iter().any(|ex| {
            agents
                .get(&ex.agent_id)
                .and_then(|ag| executor_supports_stage_static(ex, ag, stage, px, refs, None))
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
            if let Some(m) = executor_supports_stage_static(ex, ag, "encode", px, refs, None) {
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
                .and_then(|ag| executor_supports_stage_static(ex, ag, "denoise", px, refs, req))
                .is_some()
        });
        let ok_decode = executors.iter().any(|ex| {
            executor_supports_decode_mode(ex, input.tiling)
                && agents
                    .get(&ex.agent_id)
                    .and_then(|ag| executor_supports_stage_static(ex, ag, "decode", px, 0, req))
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
        .filter(|e| {
            now - e.last_seen_at <= KNOWN_EXECUTOR_WINDOW_S && agents.contains_key(&e.agent_id)
        })
        .collect();

    let online_executors: Vec<&ExecutorRecord> = all_executors
        .iter()
        .filter(|e| {
            agents.get(&e.agent_id).map(|a| a.online).unwrap_or(false)
                && !matches!(
                    e.state,
                    ExecutorState::Offline | ExecutorState::Lent | ExecutorState::Disabled
                )
                && !e.lent
                && e.disabled_reason.is_none()
        })
        .collect();

    let can_run_stage = |execs: &[&ExecutorRecord], stage: &str, px: u64| -> bool {
        execs.iter().any(|ex| {
            agents
                .get(&ex.agent_id)
                .and_then(|ag| executor_supports_stage_static(ex, ag, stage, px, 0, None))
                .is_some()
        })
    };

    let all_refs: Vec<&ExecutorRecord> = all_executors.iter().collect();

    let max_px_ever = all_executors.iter().map(executor_max_px).max().unwrap_or(0);
    let max_px_now = online_executors
        .iter()
        .map(|ex| executor_max_px(ex))
        .max()
        .unwrap_or(0);

    let max_references_ever = all_executors
        .iter()
        .map(executor_max_references)
        .max()
        .unwrap_or(0);
    let max_references_now = online_executors
        .iter()
        .map(|ex| executor_max_references(ex))
        .max()
        .unwrap_or(0);

    let enhance_ever = can_run_stage(&all_refs, "enhance", 0);
    let enhance_now = can_run_stage(&online_executors, "enhance", 0);
    let enhance_max_references_ever = all_executors
        .iter()
        .map(executor_enhance_max_references)
        .max()
        .unwrap_or(0);
    let enhance_max_references_now = online_executors
        .iter()
        .map(|ex| executor_enhance_max_references(ex))
        .max()
        .unwrap_or(0);

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
            Some(explain_unavailable_now(
                &all_executors,
                &agents,
                "decode",
                px,
            ))
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
        max_references_now,
        max_references_ever,
        enhance_now,
        enhance_ever,
        enhance_max_references_now,
        enhance_max_references_ever,
        sizes,
        stages,
        cloud_enhancers: Vec::new(),
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
        if executor_supports_stage_static(ex, ag, stage, px, 0, None).is_none() {
            continue;
        }
        if !ag.online || ex.state == ExecutorState::Offline {
            reasons.push(format!("executor {} offline", ex.id));
        } else if ex.state == ExecutorState::Lent || ex.lent {
            reasons.push(format!("executor {} lent", ex.id));
        } else if ex.state == ExecutorState::Disabled || ex.disabled_reason.is_some() {
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
#[allow(clippy::too_many_arguments)]
pub fn check_limits_for_task(
    agent: &AgentRecord,
    executor: &ExecutorRecord,
    stage: &str,
    shape: TaskShape,
    active_tasks_with_px: &[(TaskRecord, TaskShape)],
    live_metrics: Option<&AgentLiveMetrics>,
    scrape_stale_s: f64,
    now: f64,
) -> Result<(), String> {
    for limit in &agent.limits {
        if !limit.applies(stage, shape) {
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
                    // Every limit that carries this shares one count: the tasks that spill
                    // into host RAM (full-size decode, denoise with many references) take
                    // turns, whatever their stage. Side by side on a 30 GB host they ended
                    // in a global OOM (docs/HARDWARE.md F10).
                    let shares_slot = |t_stage: &str, t_shape: TaskShape| {
                        agent.limits.iter().any(|l| {
                            l.requires
                                .iter()
                                .any(|r| matches!(r, RequirementDecl::MaxConcurrentPerAgent(_)))
                                && l.applies(t_stage, t_shape)
                        })
                    };
                    let running_matching = active_tasks_with_px
                        .iter()
                        .filter(|(t, t_shape)| {
                            t.assigned_agent.as_deref() == Some(agent.id.as_str())
                                && shares_slot(&t.stage, *t_shape)
                        })
                        .count() as u32;
                    if running_matching >= *max_conc {
                        return Err(format!(
                            "waiting for a memory-heavy task on {} to finish (max {} at a time: full-size decode, denoise with many references)",
                            agent.id, max_conc
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

/// Runs a scheduler cycle where its failure has no caller to go to (after an API change, a
/// scrape, a reconnect): logs it, and the next tick tries again.
pub async fn tick(state: &Arc<AppState>) {
    if let Err(e) = run_scheduler_tick(state).await {
        warn!("scheduler tick failed: {e:#}");
    }
}

/// Runs one scheduler cycle (ARCHITECTURE_V3.md §6.3), each step a method of [`Tick`]:
/// 0. Executors with a lend pending are lent once their worker is `ready`; enhances by a
///    cloud enhancer start in Studio.
/// 1. Encodes whose inputs match another job's take its embeds, or wait for it.
/// 2. Resident `ready(S)` workers are fed first (rule 1).
/// 3. Idle executors start workers for the oldest runnable tasks (rule 2), with the model
///    revision encode used (rule 5).
/// 4. `ready` workers with nothing of their stage queued switch stage or, after idle grace,
///    are released (rules 3 and 4).
/// 5. Queued tasks that still cannot run get a live `blocked_reason`.
pub async fn run_scheduler_tick(state: &Arc<AppState>) -> Result<()> {
    let _guard = state.sched_lock.lock().await;
    let mut tick = Tick::load(state).await?;
    tick.lend_pending().await?;
    tick.start_cloud_enhances().await?;
    tick.share_encodes().await?;
    tick.feed_ready_workers().await;
    let order = tick.runnable_order();
    tick.start_idle_executors(&order).await?;
    tick.switch_or_release_ready(&order).await?;
    tick.update_blocked_reasons().await
}

/// One scheduler cycle: a snapshot of agents, executors, the queue and live metrics, and
/// what the cycle has decided so far.
struct Tick<'a> {
    state: &'a Arc<AppState>,
    now: f64,
    agents: HashMap<String, AgentRecord>,
    executors: Vec<ExecutorRecord>,
    queued: Vec<TaskRecord>,
    jobs: HashMap<String, JobRecord>,
    /// Running tasks and their shapes, plus those assigned this cycle (for `limits`).
    active: Vec<(TaskRecord, TaskShape)>,
    live_metrics: HashMap<String, AgentLiveMetrics>,
    /// Queued tasks this cycle has dealt with: assigned, completed from shared embeds, or
    /// waiting for an identical encode. They neither run again nor look blocked.
    taken: HashSet<String>,
    /// Executors this cycle gave a task, or a worker to start.
    claimed: HashSet<String>,
}

impl<'a> Tick<'a> {
    async fn load(state: &'a Arc<AppState>) -> Result<Self> {
        let db = &state.db;
        let now = now_s();
        let agents = db
            .list_agents()
            .await?
            .into_iter()
            .map(|a| (a.id.clone(), a))
            .collect();
        let executors = db.list_executors().await?;
        let queued = db.list_queued_tasks().await?;
        let active_tasks = db.list_active_tasks().await?;
        let job_ids: Vec<String> = queued
            .iter()
            .chain(&active_tasks)
            .map(|t| t.job_id.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let jobs: HashMap<String, JobRecord> = db
            .get_jobs(&job_ids)
            .await?
            .into_iter()
            .map(|j| (j.id.clone(), j))
            .collect();
        let active = active_tasks
            .into_iter()
            .map(|t| {
                let shape = jobs.get(&t.job_id).map(TaskShape::of).unwrap_or_default();
                (t, shape)
            })
            .collect();
        let live_metrics = state
            .live_metrics
            .read()
            .map(|g| g.clone())
            .unwrap_or_default();
        Ok(Self {
            state,
            now,
            agents,
            executors,
            queued,
            jobs,
            active,
            live_metrics,
            taken: HashSet::new(),
            claimed: HashSet::new(),
        })
    }

    /// Step 0: an executor with a lend pending is lent once its worker is `ready` (its task
    /// done), and the worker released.
    async fn lend_pending(&mut self) -> Result<()> {
        for ex in &mut self.executors {
            if !(ex.pending_lend && ex.state == ExecutorState::Ready) {
                continue;
            }
            if let Some((agent_id, executor, true)) = self.state.db.lend_executor(&ex.key).await? {
                self.state.send_to_agent(&agent_id, release_msg(executor));
            }
            ex.state = ExecutorState::Lent;
            ex.lent = true;
            ex.pending_lend = false;
        }
        Ok(())
    }

    /// Step 0b: the enhance of a job with a cloud enhancer runs in Studio itself
    /// (docs/CLOUD_ENHANCERS.md), never on an executor: assigned to the `cloud` pseudo-agent
    /// and its provider, and sent off; the provider's own limits queue it from there.
    async fn start_cloud_enhances(&mut self) -> Result<()> {
        for task in self.queued.iter().filter(|t| t.stage == "enhance") {
            let Some(job) = self.jobs.get(&task.job_id) else {
                continue;
            };
            let Some(provider) = job.enhancer.as_deref() else {
                continue;
            };
            self.taken.insert(task.id.clone());
            if let Some((task, job, attempt)) = self
                .state
                .db
                .assign_task(&task.id, CLOUD_AGENT, provider)
                .await?
            {
                tokio::spawn(crate::cloud::run_enhance(
                    self.state.clone(),
                    task,
                    job,
                    attempt,
                ));
            }
        }
        Ok(())
    }

    /// Step 1, one encode per send (docs/REFERENCE_IMAGES.md §4): a queued encode whose
    /// inputs match a finished one takes its embeds; one whose inputs match an encode that
    /// is running, or a queued one ahead of it, waits for that one instead of running too.
    async fn share_encodes(&mut self) -> Result<()> {
        let reference_res = self.state.config.reference_res;
        let mut encode_keys: HashSet<String> = self
            .active
            .iter()
            .filter(|(t, _)| t.stage == "encode")
            .filter_map(|(t, _)| self.jobs.get(&t.job_id))
            .map(|j| encode_cache_key(j, reference_res))
            .collect();
        let declared_models: HashSet<(String, String)> = self
            .agents
            .values()
            .flat_map(|a| a.models.iter().map(|m| (m.id.clone(), m.rev.clone())))
            .collect();
        for task in self.queued.iter().filter(|t| t.stage == "encode") {
            let Some(job) = self.jobs.get(&task.job_id) else {
                continue;
            };
            let key = encode_cache_key(job, reference_res);
            if let Some((source, model_id, model_rev)) =
                self.state.db.find_shared_embeds(&key).await?
            {
                if declared_models.contains(&(model_id, model_rev))
                    && self
                        .state
                        .db
                        .complete_task_from_shared(&task.id, &source, &key)
                        .await?
                {
                    self.taken.insert(task.id.clone());
                    continue;
                }
            }
            if !encode_keys.insert(key) {
                self.taken.insert(task.id.clone());
            }
        }
        Ok(())
    }

    /// The model `ex` would run `task` with, if it can take it now: its agent is online, it
    /// declares the stage at the job's size, references and decode mode, with the model
    /// revision encode used, and the agent's live limits allow it.
    fn can_run(
        &self,
        ex: &ExecutorRecord,
        task: &TaskRecord,
        job: &JobRecord,
    ) -> Option<ModelDecl> {
        let agent = self.agents.get(&ex.agent_id).filter(|a| a.online)?;
        let model = executor_supports_stage_static(
            ex,
            agent,
            &task.stage,
            job_px(job),
            job_refs(job),
            required_model(task, job),
        )?;
        if task.stage == "decode" && !executor_supports_decode_mode(ex, job.tiling) {
            return None;
        }
        check_limits_for_task(
            agent,
            ex,
            &task.stage,
            TaskShape::of(job),
            &self.active,
            self.live_metrics.get(&agent.id),
            self.state.config.scrape_stale_s,
            self.now,
        )
        .ok()?;
        Some(model)
    }

    /// Whether `ex` is free for this cycle's decisions and in `state`.
    fn available(&self, ex: &ExecutorRecord, state: ExecutorState) -> bool {
        !self.claimed.contains(&ex.key) && ex.state == state && !ex.lent && !ex.pending_lend
    }

    /// Step 2, rule 1: a resident `ready(S)` worker gets the oldest queued `S` task it can run.
    async fn feed_ready_workers(&mut self) {
        for t in 0..self.queued.len() {
            let task = &self.queued[t];
            if self.taken.contains(&task.id) {
                continue;
            }
            let Some(job) = self.jobs.get(&task.job_id) else {
                continue;
            };
            let req_model = required_model(task, job);
            for i in 0..self.executors.len() {
                let ex = &self.executors[i];
                if !self.available(ex, ExecutorState::Ready)
                    || !resident_serves(
                        ex,
                        self.agents.get(&ex.agent_id),
                        &task.stage,
                        job_px(job),
                        job_refs(job),
                    )
                {
                    continue;
                }
                // A resident worker keeps its model: only tasks for the revision it loaded.
                if let (Some((req_id, req_rev)), Some(cur_id), Some(cur_rev)) =
                    (req_model, &ex.current_model_id, &ex.current_model_rev)
                {
                    if cur_id != req_id || cur_rev != req_rev {
                        continue;
                    }
                }
                if self.can_run(ex, task, job).is_none() {
                    continue;
                }
                // A failure here skips this task rather than abandoning the whole tick (and
                // every other executor with it).
                match dispatch_task_to_executor(self.state, task, job, ex).await {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(e) => {
                        warn!(task = %task.id, executor = %ex.key, error = %e, "Failed to assign task");
                        break;
                    }
                }
                let mut assigned = task.clone();
                assigned.assigned_agent = Some(ex.agent_id.clone());
                assigned.assigned_executor = Some(ex.id.clone());
                self.taken.insert(task.id.clone());
                self.claimed.insert(ex.key.clone());
                self.active.push((assigned, TaskShape::of(job)));
                self.executors[i].state = ExecutorState::Busy;
                break;
            }
        }
    }

    /// The queued tasks not dealt with yet (indices into `queued`): stages with no worker
    /// (`ready`, `starting`, `busy`) anywhere first, then oldest first.
    fn runnable_order(&self) -> Vec<usize> {
        let stages_with_workers: HashSet<&str> = self
            .executors
            .iter()
            .filter(|e| {
                matches!(
                    e.state,
                    ExecutorState::Ready | ExecutorState::Starting | ExecutorState::Busy
                )
            })
            .filter_map(|e| e.current_stage.as_deref())
            .collect();
        let mut order: Vec<usize> = (0..self.queued.len())
            .filter(|&t| !self.taken.contains(&self.queued[t].id))
            .collect();
        order.sort_by(|&a, &b| {
            let (a, b) = (&self.queued[a], &self.queued[b]);
            let has_worker = |t: &TaskRecord| stages_with_workers.contains(t.stage.as_str());
            has_worker(a).cmp(&has_worker(b)).then_with(|| {
                a.queued_at
                    .unwrap_or(0.0)
                    .partial_cmp(&b.queued_at.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        order
    }

    /// Step 3, rule 2: an `idle` executor starts a worker for the first runnable task in
    /// `order` that no worker covers yet.
    ///
    /// Long stages (enhance, denoise) may run on several executors at once: a backlog gets
    /// another worker unless the workers already starting or ready for that stage cover
    /// every queued task. Short stages (encode, decode) get one worker at a time; it works
    /// through its queue faster than a second one could load. Enhance tasks queue per model
    /// (with or without references).
    async fn start_idle_executors(&mut self, order: &[usize]) -> Result<()> {
        let mut seen_per_lane: HashMap<(String, bool), usize> = HashMap::new();
        for &t in order {
            let task = &self.queued[t];
            let Some(job) = self.jobs.get(&task.job_id) else {
                continue;
            };
            let (px, refs) = (job_px(job), job_refs(job));
            let lane = (task.stage.clone(), task.stage == "enhance" && refs > 0);
            let seen = seen_per_lane.entry(lane).or_insert(0);
            let ahead_in_queue = *seen;
            *seen += 1;
            let serves = |e: &ExecutorRecord| {
                resident_serves(e, self.agents.get(&e.agent_id), &task.stage, px, refs)
            };
            let already_served = if matches!(task.stage.as_str(), "enhance" | "denoise") {
                let free_workers = self
                    .executors
                    .iter()
                    .filter(|e| {
                        matches!(e.state, ExecutorState::Starting | ExecutorState::Ready)
                            && serves(e)
                    })
                    .count();
                free_workers > ahead_in_queue
            } else {
                self.executors.iter().any(|e| {
                    matches!(
                        e.state,
                        ExecutorState::Starting | ExecutorState::Ready | ExecutorState::Busy
                    ) && serves(e)
                })
            };
            if already_served {
                continue;
            }
            for i in 0..self.executors.len() {
                let ex = &self.executors[i];
                if !self.available(ex, ExecutorState::Idle) {
                    continue;
                }
                let Some(model) = self.can_run(ex, task, job) else {
                    continue;
                };
                start_worker(self.state, &mut self.executors[i], &task.stage, model).await?;
                self.claimed.insert(self.executors[i].key.clone());
                break;
            }
        }
        Ok(())
    }

    /// Step 4, rules 3 and 4, for `ready(S)` executors that got nothing above. A worker stays
    /// only while a task of its own stage is queued. Otherwise, if another stage has queued
    /// work that no free worker covers, the executor switches to it now; holding a loaded
    /// model while other work waits only costs time. Idle grace (keeping a worker warm for a
    /// follow-up job) applies only when no other work is waiting. `max_wait` still lets a
    /// long-waiting task take over an executor whose own stage has queued work, so nothing
    /// starves.
    async fn switch_or_release_ready(&mut self, order: &[usize]) -> Result<()> {
        let max_wait = self.state.config.max_wait_s;
        for i in 0..self.executors.len() {
            let ex = &self.executors[i];
            if !self.available(ex, ExecutorState::Ready)
                || !self.agents.get(&ex.agent_id).is_some_and(|a| a.online)
            {
                continue;
            }
            // Whether a resident worker (stage, and enhancer model) could run `t`.
            let serves = |e: &ExecutorRecord, t: &TaskRecord| {
                self.jobs.get(&t.job_id).is_some_and(|job| {
                    resident_serves(
                        e,
                        self.agents.get(&e.agent_id),
                        &t.stage,
                        job_px(job),
                        job_refs(job),
                    )
                })
            };
            let has_queued_for_cur = self
                .queued
                .iter()
                .any(|t| !self.taken.contains(&t.id) && serves(ex, t));

            // The first runnable task of another stage (or enhance with the other enhancer: a
            // model switch) that the other workers do not cover.
            let mut switch_to: Option<(&str, ModelDecl, f64)> = None;
            for &t in order {
                let task = &self.queued[t];
                if self.taken.contains(&task.id) || serves(ex, task) {
                    continue;
                }
                // Covered: other workers for that stage will take its queue. Free (starting
                // or ready) workers count for every stage; a busy one only for short stages,
                // which it gets through faster than this executor could load a model.
                let task_wait = self.now - task.queued_at.unwrap_or(self.now);
                let short_stage = matches!(task.stage.as_str(), "encode" | "decode");
                let covering = self
                    .executors
                    .iter()
                    .filter(|other| {
                        other.key != ex.key
                            && serves(other, task)
                            && (matches!(
                                other.state,
                                ExecutorState::Starting | ExecutorState::Ready
                            ) || (short_stage && other.state == ExecutorState::Busy))
                    })
                    .count();
                let task_refs = self
                    .jobs
                    .get(&task.job_id)
                    .map(|j| !j.references.is_empty());
                let waiting = self
                    .queued
                    .iter()
                    .filter(|t| {
                        !self.taken.contains(&t.id)
                            && t.stage == task.stage
                            && (task.stage != "enhance"
                                || self.jobs.get(&t.job_id).map(|j| !j.references.is_empty())
                                    == task_refs)
                    })
                    .count();
                if covering >= waiting && task_wait < max_wait {
                    continue;
                }
                let Some(job) = self.jobs.get(&task.job_id) else {
                    continue;
                };
                let Some(model) = self.can_run(ex, task, job) else {
                    continue;
                };
                switch_to = Some((task.stage.as_str(), model, task_wait));
                break;
            }

            let cur_stage = ex.current_stage.clone().unwrap_or_default();
            let (agent_id, ex_id) = (ex.agent_id.clone(), ex.id.clone());
            let idle_duration = self.now - ex.idle_since.unwrap_or(self.now);
            if let Some((stage, model, task_wait)) = switch_to {
                if !has_queued_for_cur || task_wait >= max_wait {
                    let stage = stage.to_string();
                    self.state
                        .metrics
                        .record_executor_switch(&agent_id, &ex_id, &cur_stage, &stage);
                    start_worker(self.state, &mut self.executors[i], &stage, model).await?;
                    self.claimed.insert(self.executors[i].key.clone());
                    continue;
                }
            }

            // Rule 4: nothing of its stage queued and nothing else to switch to: keep the
            // worker warm for idle_grace, then release it.
            if !has_queued_for_cur && idle_duration >= self.state.config.idle_grace_s {
                self.state
                    .db
                    .update_executor_state(
                        &agent_id,
                        &ex_id,
                        ExecutorState::Stopping,
                        Some(&cur_stage),
                        None,
                        None,
                    )
                    .await?;
                self.state.send_to_agent(&agent_id, release_msg(ex_id));
                self.executors[i].state = ExecutorState::Stopping;
            }
        }
        Ok(())
    }

    /// Step 5: a live `blocked_reason` for every queued task this cycle could not place.
    async fn update_blocked_reasons(&self) -> Result<()> {
        for task in &self.queued {
            if self.taken.contains(&task.id) {
                continue;
            }
            let Some(job) = self.jobs.get(&task.job_id) else {
                continue;
            };
            let reason = diagnose_queued_task_block(
                task,
                job,
                &self.executors,
                &self.agents,
                &self.active,
                &self.live_metrics,
                self.state.config.scrape_stale_s,
                self.now,
            );
            // Most ticks change nothing: skip the write transaction then.
            if task.blocked_reason != reason {
                self.state
                    .db
                    .set_task_blocked_reason(&task.id, &job.id, reason.as_deref())
                    .await?;
            }
        }
        Ok(())
    }
}

fn job_px(job: &JobRecord) -> u64 {
    (job.width as u64) * (job.height as u64)
}

fn job_refs(job: &JobRecord) -> u32 {
    job.references.len() as u32
}

/// The `model@rev` denoise and decode must run with: the one the job's encode used (rule 5).
fn required_model<'j>(task: &TaskRecord, job: &'j JobRecord) -> Option<(&'j str, &'j str)> {
    if !matches!(task.stage.as_str(), "denoise" | "decode") {
        return None;
    }
    match (&job.model_id, &job.model_rev) {
        (Some(id), Some(rev)) => Some((id.as_str(), rev.as_str())),
        _ => None,
    }
}

/// Marks `ex` starting a `stage` worker with `model` and tells its agent to start it.
async fn start_worker(
    state: &AppState,
    ex: &mut ExecutorRecord,
    stage: &str,
    model: ModelDecl,
) -> Result<()> {
    state
        .db
        .mark_executor_starting(&ex.agent_id, &ex.id, stage, &model)
        .await?;
    state.send_to_agent(
        &ex.agent_id,
        proto::BackendMessage {
            msg: Some(proto::backend_message::Msg::StartWorker(
                proto::StartWorker {
                    executor: ex.id.clone(),
                    stage: stage_to_proto(stage) as i32,
                    model: Some(proto::ModelRef {
                        id: model.id.clone(),
                        rev: model.rev.clone(),
                        dtype: model.dtype.clone(),
                    }),
                },
            )),
        },
    );
    ex.state = ExecutorState::Starting;
    ex.current_stage = Some(stage.to_string());
    ex.current_model_id = Some(model.id);
    ex.current_model_rev = Some(model.rev);
    Ok(())
}

fn release_msg(executor: String) -> proto::BackendMessage {
    proto::BackendMessage {
        msg: Some(proto::backend_message::Msg::Release(proto::Release {
            executor,
        })),
    }
}

#[allow(clippy::too_many_arguments)]
fn diagnose_queued_task_block(
    task: &TaskRecord,
    job: &JobRecord,
    executors: &[ExecutorRecord],
    agents: &HashMap<String, AgentRecord>,
    active_with_px: &[(TaskRecord, TaskShape)],
    live_metrics: &HashMap<String, AgentLiveMetrics>,
    scrape_stale_s: f64,
    now: f64,
) -> Option<String> {
    let (px, refs) = (job_px(job), job_refs(job));
    let req_model = required_model(task, job);

    let mut capable_executors = Vec::new();
    for ex in executors {
        let Some(ag) = agents.get(&ex.agent_id) else {
            continue;
        };
        if executor_supports_stage_static(ex, ag, &task.stage, px, refs, req_model).is_some()
            && (task.stage != "decode" || executor_supports_decode_mode(ex, job.tiling))
        {
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
        if !ag.online || ex.state == ExecutorState::Offline {
            offline_or_unavailable_reasons.push(format!("executor {} offline", ex.id));
        } else if ex.state == ExecutorState::Lent || ex.lent {
            offline_or_unavailable_reasons.push(format!("executor {} lent", ex.id));
        } else if ex.state == ExecutorState::Disabled || ex.disabled_reason.is_some() {
            let r = ex.disabled_reason.as_deref().unwrap_or("disabled");
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
            TaskShape::of(job),
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
    let base_url = &state.config.artifact_base_url;
    // Denoise reads each reference's stored latents, for the model the job was encoded with.
    // Looked up before assigning, so a missing one leaves the task queued rather than stuck.
    let mut reference_inputs = Vec::new();
    if task.stage == "denoise" {
        let (model_id, model_rev) = (
            job.model_id.clone().unwrap_or_default(),
            job.model_rev.clone().unwrap_or_default(),
        );
        for (i, r) in job.references.iter().enumerate() {
            let Some(lat) = state
                .db
                .get_reference_latents(&r.sha256, &model_id, &model_rev, state.config.reference_res)
                .await?
            else {
                anyhow::bail!("no stored latents for reference {i} of job {}", job.id);
            };
            reference_inputs.push(proto::ArtifactRef {
                name: format!("ref{i}.safetensors"),
                url: format!("{base_url}/agent/v1/blobs/{}", lat.blob_sha256),
                sha256: lat.blob_sha256,
                bytes: lat.bytes,
            });
        }
    }

    let Some((assigned_task, _, attempt)) =
        state.db.assign_task(&task.id, &ex.agent_id, &ex.id).await?
    else {
        return Ok(false);
    };

    if let Some(q_at) = task.queued_at {
        let wait_s = (now_s() - q_at).max(0.0);
        state.metrics.observe_task_wait(&task.stage, wait_s);
    }

    let mut inputs = Vec::new();
    let mut uploads = Vec::new();

    let params = match assigned_task.stage.as_str() {
        "enhance" => {
            uploads.push(proto::ArtifactRef {
                name: "enhanced.json".to_string(),
                url: format!(
                    "{base_url}/agent/v1/artifacts/{}/{attempt}/enhanced.json",
                    assigned_task.id
                ),
                sha256: String::new(),
                bytes: 0,
            });
            // An image-aware enhancer reads the references' pixels, like encode.
            let mut references = Vec::with_capacity(job.references.len());
            for (i, r) in job.references.iter().enumerate() {
                let bytes = state
                    .db
                    .get_upload(&r.sha256)
                    .await?
                    .map(|u| u.bytes)
                    .unwrap_or(0);
                inputs.push(proto::ArtifactRef {
                    name: format!("ref{i}.image"),
                    url: format!("{base_url}/agent/v1/blobs/{}", r.sha256),
                    sha256: r.sha256.clone(),
                    bytes,
                });
                references.push(proto::Reference {
                    sha256: r.sha256.clone(),
                });
            }
            Some(proto::assign::Params::Enhance(proto::EnhanceParams {
                prompt: enhance_request(&job.prompt, job.width, job.height),
                max_new_tokens: job.enhance_max_tokens.unwrap_or(0),
                references,
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
            // Every reference's pixels go in; latents come back only for the ones Studio has
            // none stored for (with this worker's model and the configured resolution).
            let reference_res = state.config.reference_res;
            let model = (
                ex.current_model_id.clone().unwrap_or_default(),
                ex.current_model_rev.clone().unwrap_or_default(),
            );
            let mut references = Vec::with_capacity(job.references.len());
            for (i, r) in job.references.iter().enumerate() {
                let bytes = state
                    .db
                    .get_upload(&r.sha256)
                    .await?
                    .map(|u| u.bytes)
                    .unwrap_or(0);
                inputs.push(proto::ArtifactRef {
                    name: format!("ref{i}.image"),
                    url: format!("{base_url}/agent/v1/blobs/{}", r.sha256),
                    sha256: r.sha256.clone(),
                    bytes,
                });
                let stored = state
                    .db
                    .get_reference_latents(&r.sha256, &model.0, &model.1, reference_res)
                    .await?;
                if stored.is_none() {
                    uploads.push(proto::ArtifactRef {
                        name: format!("ref{i}.safetensors"),
                        url: format!(
                            "{base_url}/agent/v1/artifacts/{}/{attempt}/ref{i}.safetensors",
                            assigned_task.id
                        ),
                        sha256: String::new(),
                        bytes: 0,
                    });
                }
                references.push(proto::Reference {
                    sha256: r.sha256.clone(),
                });
            }
            Some(proto::assign::Params::Encode(proto::EncodeParams {
                prompt: effective_prompt,
                prompt_source,
                negative_prompt: neg,
                true_cfg_scale: job.true_cfg_scale,
                reference_res: if references.is_empty() {
                    0
                } else {
                    reference_res
                },
                references,
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
            inputs.append(&mut reference_inputs);
            Some(proto::assign::Params::Denoise(proto::DenoiseParams {
                width: job.width,
                height: job.height,
                steps: job.steps,
                seed: job.seed,
                true_cfg_scale: job.true_cfg_scale,
                references: job.references.len() as u32,
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
                mode: if job.tiling {
                    proto::DecodeMode::Tiled
                } else {
                    proto::DecodeMode::Untiled
                } as i32,
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
