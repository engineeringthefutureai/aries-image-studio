use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutorLiveMetrics {
    pub mem_used_bytes: u64,
    pub mem_total_bytes: u64,
    /// total - used
    pub mem_available_bytes: u64,
    pub overflow_host_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentLiveMetrics {
    pub host_ram_available_bytes: u64,
    pub executors: HashMap<String, ExecutorLiveMetrics>,
    pub scraped_at: f64,
    /// The scrape as received, re-exported (labelled by agent) on the backend's /metrics.
    #[serde(skip)]
    pub raw: String,
}

/// Parses Prometheus exposition format text from an agent's `/metrics` endpoint. The
/// names are pinned in proto/aries/agent/v1/agent_metrics.json.
pub fn parse_agent_prometheus_text(text: &str, now_s: f64) -> AgentLiveMetrics {
    let mut out = AgentLiveMetrics {
        host_ram_available_bytes: 0,
        executors: HashMap::new(),
        scraped_at: now_s,
        raw: text.to_string(),
    };

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((metric_part, val_part)) = split_metric_and_value(line) else {
            continue;
        };
        let Ok(val_f64) = val_part.parse::<f64>() else {
            continue;
        };
        let (name, labels) = parse_name_and_labels(metric_part);
        match name {
            "aries_host_ram_available_bytes" => {
                if val_f64 >= 0.0 {
                    out.host_ram_available_bytes = val_f64 as u64;
                }
            }
            "aries_executor_mem_used_bytes" => {
                if let Some(exec_id) = labels.get("executor") {
                    let entry = out.executors.entry(exec_id.clone()).or_default();
                    if val_f64 >= 0.0 {
                        entry.mem_used_bytes = val_f64 as u64;
                    }
                }
            }
            "aries_executor_mem_total_bytes" => {
                if let Some(exec_id) = labels.get("executor") {
                    let entry = out.executors.entry(exec_id.clone()).or_default();
                    if val_f64 >= 0.0 {
                        entry.mem_total_bytes = val_f64 as u64;
                    }
                }
            }
            "aries_executor_mem_overflow_bytes" => {
                if let Some(exec_id) = labels.get("executor") {
                    let entry = out.executors.entry(exec_id.clone()).or_default();
                    if val_f64 >= 0.0 {
                        entry.overflow_host_bytes = val_f64 as u64;
                    }
                }
            }
            _ => {}
        }
    }

    for ex in out.executors.values_mut() {
        ex.mem_available_bytes = ex.mem_total_bytes.saturating_sub(ex.mem_used_bytes);
    }
    out
}

fn split_metric_and_value(line: &str) -> Option<(&str, &str)> {
    // Find the first whitespace after closing '}' (or after metric name if no labels)
    let search_start = line.find('}').map(|i| i + 1).unwrap_or(0);
    let rest = &line[search_start..];
    let ws_rel = rest.find(char::is_whitespace)?;
    let split_idx = search_start + ws_rel;
    let metric_part = line[..split_idx].trim();
    let val_rest = line[split_idx..].trim();
    // Value is the first token in val_rest (ignoring optional timestamp)
    let val_token = val_rest.split_whitespace().next()?;
    Some((metric_part, val_token))
}

fn parse_name_and_labels(metric_part: &str) -> (&str, HashMap<String, String>) {
    let mut labels = HashMap::new();
    let Some(open_idx) = metric_part.find('{') else {
        return (metric_part, labels);
    };
    let name = &metric_part[..open_idx];
    let Some(close_idx) = metric_part.rfind('}') else {
        return (name, labels);
    };
    let inner = &metric_part[open_idx + 1..close_idx];
    for pair in inner.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        if let Some((k, v)) = pair.split_once('=') {
            let k = k.trim();
            let v = v.trim().trim_matches('"');
            labels.insert(k.to_string(), v.to_string());
        }
    }
    (name, labels)
}

/// Bucket bounds of the preview timings: a few ms on a fast CPU, maybe a second on a slow one.
pub const PREVIEW_BUCKETS: [f64; 10] = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0];
/// Bucket bounds of `aries_step_preview_bytes`.
pub const PREVIEW_BYTES_BUCKETS: [f64; 8] = [5e3, 1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6];

/// A Prometheus histogram's state: cumulative bucket counts, sum and count.
#[derive(Debug, Default, Clone)]
pub struct Histogram {
    buckets: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn observe(&mut self, bounds: &[f64], v: f64) {
        self.buckets.resize(bounds.len(), 0);
        for (b, le) in self.buckets.iter_mut().zip(bounds) {
            if v <= *le {
                *b += 1;
            }
        }
        self.sum += v;
        self.count += 1;
    }

    fn render(&self, out: &mut String, name: &str, labels: &str, bounds: &[f64]) {
        for (b, le) in self.buckets.iter().zip(bounds) {
            out.push_str(&format!("{name}_bucket{{{labels},le=\"{le}\"}} {b}\n"));
        }
        out.push_str(&format!(
            "{name}_bucket{{{labels},le=\"+Inf\"}} {}\n",
            self.count
        ));
        out.push_str(&format!("{name}_sum{{{labels}}} {:.4}\n", self.sum));
        out.push_str(&format!("{name}_count{{{labels}}} {}\n", self.count));
    }
}

#[derive(Debug, Default, Clone)]
pub struct BackendMetricsRegistry {
    inner: Arc<RwLock<BackendMetricsInner>>,
}

#[derive(Debug, Default)]
struct BackendMetricsInner {
    pub executor_switches: HashMap<(String, String, String, String), u64>,
    pub job_e2e_sum: HashMap<(String, String), f64>,
    pub job_e2e_count: HashMap<(String, String), u64>,
    pub task_wait_sum: HashMap<String, f64>,
    pub task_wait_count: HashMap<String, u64>,
    /// Preview renders from a latent, by size (seconds, `PREVIEW_BUCKETS`).
    pub preview: HashMap<String, Histogram>,
    /// Per-step previews from denoise workers, by size: time to make (`PREVIEW_BUCKETS`)
    /// and JPEG size (`PREVIEW_BYTES_BUCKETS`).
    pub step_preview: HashMap<String, (Histogram, Histogram)>,
    /// Search embeddings (search.rs), by kind (`image` or `query`); seconds, `PREVIEW_BUCKETS`.
    pub search_embed: HashMap<String, Histogram>,
    /// Images with an embedding in the search index.
    pub search_indexed: Option<u64>,
}

#[derive(Debug, Default)]
pub struct MetricsSnapshot {
    pub jobs_by_status: HashMap<String, u64>,
    pub tasks_queued_by_stage: HashMap<String, u64>,
    pub tasks_blocked: HashMap<(String, String), u64>,
    pub agents_connected: HashMap<String, bool>,
    pub agent_scrape_age: HashMap<String, f64>,
    pub intermediate_bytes: u64,
    pub final_bytes: u64,
}

impl BackendMetricsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_executor_switch(
        &self,
        agent: &str,
        executor: &str,
        from_stage: &str,
        to_stage: &str,
    ) {
        if let Ok(mut g) = self.inner.write() {
            let key = (
                agent.to_string(),
                executor.to_string(),
                from_stage.to_string(),
                to_stage.to_string(),
            );
            *g.executor_switches.entry(key).or_insert(0) += 1;
        }
    }

    pub fn observe_job_e2e(&self, size: &str, enhanced: bool, seconds: f64) {
        if let Ok(mut g) = self.inner.write() {
            let key = (
                size.to_string(),
                if enhanced { "true" } else { "false" }.to_string(),
            );
            *g.job_e2e_sum.entry(key.clone()).or_insert(0.0) += seconds;
            *g.job_e2e_count.entry(key).or_insert(0) += 1;
        }
    }

    pub fn observe_task_wait(&self, stage: &str, seconds: f64) {
        if let Ok(mut g) = self.inner.write() {
            *g.task_wait_sum.entry(stage.to_string()).or_insert(0.0) += seconds;
            *g.task_wait_count.entry(stage.to_string()).or_insert(0) += 1;
        }
    }

    /// One rough preview rendered from a latent (preview.rs): reading it, the linear map and
    /// the JPEG, timed on whatever CPU Studio runs on.
    pub fn observe_preview(&self, size: &str, seconds: f64) {
        if let Ok(mut g) = self.inner.write() {
            g.preview
                .entry(size.to_string())
                .or_default()
                .observe(&PREVIEW_BUCKETS, seconds);
        }
    }

    /// One per-step preview from a denoise worker: the worker's time to make it and its
    /// size in bytes.
    /// One search embedding: `kind` is `image` (a thumbnail) or `query` (a search's text).
    pub fn observe_search_embed(&self, kind: &str, seconds: f64) {
        if let Ok(mut g) = self.inner.write() {
            g.search_embed
                .entry(kind.to_string())
                .or_default()
                .observe(&PREVIEW_BUCKETS, seconds);
        }
    }

    pub fn set_search_indexed(&self, images: u64) {
        if let Ok(mut g) = self.inner.write() {
            g.search_indexed = Some(images);
        }
    }

    pub fn observe_step_preview(&self, size: &str, seconds: f64, bytes: usize) {
        if let Ok(mut g) = self.inner.write() {
            let (t, b) = g.step_preview.entry(size.to_string()).or_default();
            t.observe(&PREVIEW_BUCKETS, seconds);
            b.observe(&PREVIEW_BYTES_BUCKETS, bytes as f64);
        }
    }

    pub fn render(&self, snap: &MetricsSnapshot) -> String {
        let mut out = String::new();

        out.push_str("# HELP aries_jobs_total Total jobs by status\n");
        out.push_str("# TYPE aries_jobs_total gauge\n");
        for status in [
            "submitted",
            "running",
            "done",
            "failed",
            "cancelled",
            "rejected",
        ] {
            let count = snap.jobs_by_status.get(status).copied().unwrap_or(0);
            out.push_str(&format!(
                "aries_jobs_total{{status=\"{status}\"}} {count}\n"
            ));
        }

        out.push_str("# HELP aries_tasks_queued Number of queued tasks per stage\n");
        out.push_str("# TYPE aries_tasks_queued gauge\n");
        for stage in ["enhance", "encode", "denoise", "decode"] {
            let count = snap.tasks_queued_by_stage.get(stage).copied().unwrap_or(0);
            out.push_str(&format!(
                "aries_tasks_queued{{stage=\"{stage}\"}} {count}\n"
            ));
        }

        out.push_str("# HELP aries_tasks_blocked Number of blocked tasks by stage and reason\n");
        out.push_str("# TYPE aries_tasks_blocked gauge\n");
        for ((stage, reason), count) in &snap.tasks_blocked {
            let safe_reason = reason.replace('\\', "\\\\").replace('"', "\\\"");
            out.push_str(&format!(
                "aries_tasks_blocked{{stage=\"{stage}\",reason=\"{safe_reason}\"}} {count}\n"
            ));
        }

        out.push_str("# HELP aries_agent_connected Whether each known agent is connected\n");
        out.push_str("# TYPE aries_agent_connected gauge\n");
        for (agent, connected) in &snap.agents_connected {
            let val = if *connected { 1 } else { 0 };
            out.push_str(&format!(
                "aries_agent_connected{{agent=\"{agent}\"}} {val}\n"
            ));
        }

        out.push_str(
            "# HELP aries_agent_scrape_age_seconds Seconds since last successful /metrics scrape\n",
        );
        out.push_str("# TYPE aries_agent_scrape_age_seconds gauge\n");
        for (agent, age) in &snap.agent_scrape_age {
            out.push_str(&format!(
                "aries_agent_scrape_age_seconds{{agent=\"{agent}\"}} {age:.3}\n"
            ));
        }

        out.push_str("# HELP aries_artifacts_bytes Stored artifact bytes by kind\n");
        out.push_str("# TYPE aries_artifacts_bytes gauge\n");
        out.push_str(&format!(
            "aries_artifacts_bytes{{kind=\"intermediate\"}} {}\n",
            snap.intermediate_bytes
        ));
        out.push_str(&format!(
            "aries_artifacts_bytes{{kind=\"final\"}} {}\n",
            snap.final_bytes
        ));

        if let Ok(g) = self.inner.read() {
            out.push_str("# HELP aries_executor_switch_total Executor stage switches\n");
            out.push_str("# TYPE aries_executor_switch_total counter\n");
            for ((agent, executor, from_stage, to_stage), count) in &g.executor_switches {
                out.push_str(&format!(
                    "aries_executor_switch_total{{agent=\"{agent}\",executor=\"{executor}\",from_stage=\"{from_stage}\",to_stage=\"{to_stage}\"}} {count}\n"
                ));
            }

            out.push_str("# HELP aries_job_e2e_seconds End-to-end job duration seconds\n");
            out.push_str("# TYPE aries_job_e2e_seconds summary\n");
            for ((size, enhanced), sum) in &g.job_e2e_sum {
                let count = g
                    .job_e2e_count
                    .get(&(size.clone(), enhanced.clone()))
                    .copied()
                    .unwrap_or(0);
                out.push_str(&format!(
                    "aries_job_e2e_seconds_sum{{size=\"{size}\",enhanced=\"{enhanced}\"}} {sum:.3}\n"
                ));
                out.push_str(&format!(
                    "aries_job_e2e_seconds_count{{size=\"{size}\",enhanced=\"{enhanced}\"}} {count}\n"
                ));
            }

            out.push_str(
                "# HELP aries_preview_seconds Time to render a preview from a latent, in Studio\n",
            );
            out.push_str("# TYPE aries_preview_seconds histogram\n");
            for (size, h) in &g.preview {
                let labels = format!("size=\"{size}\"");
                h.render(&mut out, "aries_preview_seconds", &labels, &PREVIEW_BUCKETS);
            }
            out.push_str(
                "# HELP aries_step_preview_seconds Time a denoise worker took to make a per-step preview\n",
            );
            out.push_str("# TYPE aries_step_preview_seconds histogram\n");
            for (size, (t, _)) in &g.step_preview {
                let labels = format!("size=\"{size}\"");
                t.render(
                    &mut out,
                    "aries_step_preview_seconds",
                    &labels,
                    &PREVIEW_BUCKETS,
                );
            }
            out.push_str("# HELP aries_step_preview_bytes Size of per-step preview JPEGs\n");
            out.push_str("# TYPE aries_step_preview_bytes histogram\n");
            for (size, (_, b)) in &g.step_preview {
                let labels = format!("size=\"{size}\"");
                b.render(
                    &mut out,
                    "aries_step_preview_bytes",
                    &labels,
                    &PREVIEW_BYTES_BUCKETS,
                );
            }

            out.push_str(
                "# HELP aries_search_embed_seconds Time to embed an image or a search query, in Studio\n",
            );
            out.push_str("# TYPE aries_search_embed_seconds histogram\n");
            for (kind, h) in &g.search_embed {
                let labels = format!("kind=\"{kind}\"");
                h.render(
                    &mut out,
                    "aries_search_embed_seconds",
                    &labels,
                    &PREVIEW_BUCKETS,
                );
            }
            if let Some(n) = g.search_indexed {
                out.push_str(
                    "# HELP aries_search_indexed_images Images with an embedding in the search index\n",
                );
                out.push_str("# TYPE aries_search_indexed_images gauge\n");
                out.push_str(&format!("aries_search_indexed_images {n}\n"));
            }

            out.push_str("# HELP aries_task_wait_seconds Time spent queued before assignment\n");
            out.push_str("# TYPE aries_task_wait_seconds summary\n");
            for (stage, sum) in &g.task_wait_sum {
                let count = g.task_wait_count.get(stage).copied().unwrap_or(0);
                out.push_str(&format!(
                    "aries_task_wait_seconds_sum{{stage=\"{stage}\"}} {sum:.3}\n"
                ));
                out.push_str(&format!(
                    "aries_task_wait_seconds_count{{stage=\"{stage}\"}} {count}\n"
                ));
            }
        }

        out
    }
}

/// Agent families the backend does not re-export: it exports its own per-agent version.
const NOT_REEXPORTED: &[&str] = &["aries_agent_connected"];

/// Merges agents' scrapes into one exposition: each family once (one HELP and TYPE), every
/// sample with an `agent` label added. Families named in `NOT_REEXPORTED` are skipped.
pub fn reexport_agent_metrics(scrapes: &[(&str, &str)]) -> String {
    // family -> (HELP/TYPE lines, samples), in first-seen order
    let mut order: Vec<String> = Vec::new();
    let mut families: HashMap<String, (Vec<String>, Vec<String>)> = HashMap::new();
    for (agent, text) in scrapes {
        let agent_label = agent.replace('\\', "\\\\").replace('"', "\\\"");
        let mut current = String::new();
        for line in text.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("# ") {
                let mut parts = rest.splitn(3, ' ');
                let (kind, name) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
                if kind != "HELP" && kind != "TYPE" {
                    continue;
                }
                current = name.to_string();
                let entry = families.entry(current.clone()).or_insert_with(|| {
                    order.push(current.clone());
                    (Vec::new(), Vec::new())
                });
                if !entry.0.iter().any(|l| l == line) && entry.0.len() < 2 {
                    entry.0.push(line.to_string());
                }
                continue;
            }
            if line.starts_with('#') || current.is_empty() {
                continue;
            }
            let labelled = match line.find('{') {
                Some(i) if line[i + 1..].starts_with('}') => {
                    format!(
                        "{}{{agent=\"{agent_label}\"}}{}",
                        &line[..i],
                        &line[i + 2..]
                    )
                }
                Some(i) => format!("{}{{agent=\"{agent_label}\",{}", &line[..i], &line[i + 1..]),
                None => match line.find(' ') {
                    Some(i) => format!("{}{{agent=\"{agent_label}\"}}{}", &line[..i], &line[i..]),
                    None => continue,
                },
            };
            if let Some(entry) = families.get_mut(&current) {
                entry.1.push(labelled);
            }
        }
    }
    let mut out = String::new();
    for name in order {
        if NOT_REEXPORTED.contains(&name.as_str()) {
            continue;
        }
        let (meta, samples) = &families[&name];
        for l in meta.iter().chain(samples) {
            out.push_str(l);
            out.push('\n');
        }
    }
    out
}
