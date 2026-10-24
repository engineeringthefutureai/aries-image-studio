use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelDecl {
    pub id: String,
    pub rev: String,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageCapDecl {
    pub stage: String,
    pub models: Vec<String>,
    pub max_px: u64,
    #[serde(default)]
    pub modes: Vec<String>,
    /// Most reference images per task (0 = none).
    #[serde(default)]
    pub max_references: u32,
    /// Which of `models` take references (enhance: PE-I2I); empty: all of them.
    #[serde(default)]
    pub reference_models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum RequirementDecl {
    #[serde(rename = "host_ram_available_bytes")]
    HostRamAvailableBytes(u64),
    #[serde(rename = "max_concurrent_per_agent")]
    MaxConcurrentPerAgent(u32),
    #[serde(rename = "executor_mem_available_bytes")]
    ExecutorMemAvailableBytes(u64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitDecl {
    pub stage: String,
    pub above_px: u64,
    pub requires: Vec<RequirementDecl>,
    /// Set: the limit applies only to tasks with more reference images than this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub above_references: Option<u32>,
    /// Set: the limit applies only to decodes in this mode (`untiled` | `tiled`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_mode: Option<String>,
}

/// What the limits look at in a task: pixels, reference images and, for decode, tiling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TaskShape {
    pub px: u64,
    pub refs: u32,
    pub tiled: bool,
}

impl TaskShape {
    pub fn of(job: &JobRecord) -> Self {
        Self {
            px: (job.width as u64) * (job.height as u64),
            refs: job.references.len() as u32,
            tiled: job.tiling,
        }
    }
}

/// A page of `GET /api/jobs`: newest first, `limit` jobs after the `before` cursor.
#[derive(Debug, Clone, Default)]
pub struct JobPageQuery {
    pub limit: u32,
    pub before: Option<String>,
    /// Finished images only (the gallery).
    pub done_only: bool,
    pub favorites_only: bool,
    /// Substring of the prompt, the enhanced prompt or the seed.
    pub search: Option<String>,
    /// Width / height, within 0.01.
    pub ratio: Option<f64>,
}

/// The opaque cursor of the page after `job`.
pub fn job_cursor(job: &JobRecord) -> String {
    format!("{}_{}", job.created_at, job.id)
}

/// The intermediate artifact a stage takes from the stage before it (the dispatcher's
/// inputs in `scheduler.rs`).
fn stage_input_artifact(stage: &str) -> Option<&'static str> {
    match stage {
        "denoise" => Some("embeds.safetensors"),
        "decode" => Some("latent.safetensors"),
        _ => None,
    }
}

fn parse_job_cursor(cursor: &str) -> Option<(f64, String)> {
    let (created_at, id) = cursor.split_once('_')?;
    Some((created_at.parse().ok()?, id.to_string()))
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct GalleryFacets {
    pub total: u64,
    pub favorites: u64,
    pub sizes: Vec<SizeCount>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SizeCount {
    pub width: u32,
    pub height: u32,
    pub count: u64,
}

/// What [`Database::load_jobs`] binds to its condition's `?`.
enum JobFilter {
    Version(i64),
    /// A JSON array of job ids, for `json_each`.
    Ids(String),
}

impl JobFilter {
    fn bind<'q>(
        &self,
        q: sqlx::query::Query<'q, sqlx::Sqlite, sqlx::sqlite::SqliteArguments<'q>>,
    ) -> sqlx::query::Query<'q, sqlx::Sqlite, sqlx::sqlite::SqliteArguments<'q>> {
        match self {
            Self::Version(v) => q.bind(*v),
            Self::Ids(ids) => q.bind(ids.clone()),
        }
    }
}

impl LimitDecl {
    /// Whether this limit covers a task of `stage` with this shape.
    pub fn applies(&self, stage: &str, shape: TaskShape) -> bool {
        let mode = if shape.tiled { "tiled" } else { "untiled" };
        self.stage == stage
            && shape.px > self.above_px
            && self.above_references.is_none_or(|r| shape.refs > r)
            && (stage != "decode" || self.decode_mode.as_deref().is_none_or(|m| m == mode))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub agent: String,
    pub executor: String,
    pub device: String,
    pub backend_kind: String,
    pub model: ModelDecl,
    pub torch_version: String,
    pub worker_version: String,
    #[serde(default)]
    pub timings: HashMap<String, f64>,
    #[serde(default)]
    pub peak_mem_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSubmitInput {
    /// Client-chosen id (the SPA uses a ULID); resubmitting the same id is idempotent.
    #[serde(default)]
    pub id: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    #[serde(default = "default_steps")]
    pub steps: u32,
    pub seed: Option<u64>,
    #[serde(default = "default_cfg")]
    pub true_cfg_scale: f32,
    #[serde(default)]
    pub enhance: bool,
    /// UI-owned metadata stored with the job (a JSON object, at most 4 KiB).
    #[serde(default)]
    pub meta: Option<serde_json::Value>,
    /// Decode with VAE tiling: faster and lighter, but can leave seams (HARDWARE.md §2).
    #[serde(default)]
    pub tiling: bool,
    /// Reference images in prompt order ("image 1" first): uploads or earlier results.
    #[serde(default)]
    pub references: Vec<ReferenceInput>,
    /// The prompt rewriter's token limit (reasoning and reply); unset: the worker's 2048.
    #[serde(default)]
    pub enhance_max_tokens: Option<u32>,
    /// Who rewrites the prompt: a cloud enhancer's id, its model (`claude-sonnet-5-5`;
    /// docs/CLOUD_ENHANCERS.md), or unset / `builtin` for the agents' enhancer.
    #[serde(default)]
    pub enhancer: Option<String>,
}

/// One reference of a submitted job: an upload (`POST /api/uploads`) or a job's image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceInput {
    #[serde(default)]
    pub upload: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
}

/// A reference image as stored with a job, and as listed by the API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobReference {
    pub sha256: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadRecord {
    pub sha256: String,
    pub mime: String,
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
    pub created_at: f64,
}

/// Stored VAE latents of one reference image (docs/REFERENCE_IMAGES.md §4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceLatents {
    pub source_sha256: String,
    pub model_id: String,
    pub model_rev: String,
    pub reference_res: u32,
    pub blob_sha256: String,
    pub bytes: u64,
    pub width: u32,
    pub height: u32,
}

// The UI's Standard preset.
fn default_steps() -> u32 {
    20
}

// 1.0 = no classifier-free guidance: one transformer pass per step (HARDWARE.md §1).
fn default_cfg() -> f32 {
    1.0
}

/// A client-supplied job id must be short and URL/path safe.
pub fn is_valid_client_job_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// A status stored as TEXT: the same string in SQLite, in the API's JSON and here.
macro_rules! text_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident = $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type)]
        pub enum $name {
            $(#[serde(rename = $text)] #[sqlx(rename = $text)] $variant),+
        }

        impl $name {
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text),+
                }
            }
        }

        impl std::str::FromStr for $name {
            type Err = anyhow::Error;
            fn from_str(s: &str) -> Result<Self> {
                match s {
                    $($text => Ok(Self::$variant),)+
                    other => Err(anyhow::anyhow!("unknown {} {other:?}", stringify!($name))),
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

text_enum! {
    /// ARCHITECTURE_V3.md §4.
    JobStatus {
        Submitted = "submitted",
        Running = "running",
        Done = "done",
        Failed = "failed",
        Cancelled = "cancelled",
        Rejected = "rejected",
    }
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Done | Self::Failed | Self::Cancelled | Self::Rejected
        )
    }
}

text_enum! {
    /// ARCHITECTURE_V3.md §4; `blocked` is `queued` with a reason.
    TaskStatus {
        Waiting = "waiting",
        Queued = "queued",
        Assigned = "assigned",
        Running = "running",
        Uploading = "uploading",
        Done = "done",
        Failed = "failed",
        Cancelled = "cancelled",
    }
}

impl TaskStatus {
    /// On an executor: assigned, running or uploading.
    pub fn is_active(self) -> bool {
        matches!(self, Self::Assigned | Self::Running | Self::Uploading)
    }
}

text_enum! {
    /// ARCHITECTURE_V3.md §6.1.
    ExecutorState {
        Offline = "offline",
        Idle = "idle",
        Starting = "starting",
        Ready = "ready",
        Busy = "busy",
        Stopping = "stopping",
        Lent = "lent",
        Disabled = "disabled",
    }
}

/// Why an attempt failed: the agent's `Failed` report, or a failure Studio decided itself.
#[derive(Debug, Clone, Copy)]
pub struct AttemptFailure<'a> {
    pub class: &'a str,
    pub detail: &'a str,
    pub exit_code: i32,
    pub signal: i32,
    pub kernel_log_tail: &'a str,
}

impl<'a> AttemptFailure<'a> {
    /// A failure Studio decided (no worker exit to report).
    pub fn studio(class: &'a str, detail: &'a str) -> Self {
        Self {
            class,
            detail,
            exit_code: -1,
            signal: 0,
            kernel_log_tail: "",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub status: JobStatus,
    pub prompt: String,
    pub enhanced_prompt: Option<String>,
    pub thinking: Option<String>,
    /// Whether the job has enhancer reasoning; job lists leave `thinking` out (it is large)
    /// and the viewer fetches it from `GET /api/jobs/{id}`.
    #[serde(default)]
    pub has_thinking: bool,
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub seed: u64,
    pub true_cfg_scale: f32,
    pub enhance: bool,
    /// The prompt rewriter's token limit; unset: the worker's default.
    pub enhance_max_tokens: Option<u32>,
    /// The cloud enhancer that rewrites the prompt (docs/CLOUD_ENHANCERS.md); unset: the
    /// agents' built-in enhancer.
    #[serde(default)]
    pub enhancer: Option<String>,
    /// Decode with VAE tiling (off by default).
    pub tiling: bool,
    pub wh_ratio: Option<String>,
    /// PE-I2I's aspect answer: `<imageN>` to keep that reference's aspect. Stored, not used yet.
    pub ratio_follow: Option<String>,
    pub model_id: Option<String>,
    pub model_rev: Option<String>,
    pub error_class: Option<String>,
    pub error_detail: Option<String>,
    pub blocked_reason: Option<String>,
    /// UI-owned metadata (turn id, aspect label, size and quality preset), opaque here.
    pub meta: serde_json::Value,
    pub favorite: bool,
    /// Reference images in prompt order.
    #[serde(default)]
    pub references: Vec<JobReference>,
    pub version: i64,
    pub created_at: f64,
    pub started_at: Option<f64>,
    pub finished_at: Option<f64>,
    #[serde(default)]
    pub tasks: Vec<TaskRecord>,
    #[serde(default)]
    pub provenance: HashMap<String, ProvenanceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    pub id: String,
    pub job_id: String,
    pub seq: i32,
    pub stage: String,
    pub status: TaskStatus,
    pub attempts: u32,
    pub current_attempt: u32,
    pub assigned_agent: Option<String>,
    pub assigned_executor: Option<String>,
    pub blocked_reason: Option<String>,
    pub step: u32,
    pub steps: u32,
    pub s_per_step: f64,
    /// enhance: `prefill` | `thinking` | `rewriting` while running
    pub sub_stage: Option<String>,
    /// enhance: live token stats from the worker (generated_tokens, generation_tps, ...)
    pub stats: Option<HashMap<String, f64>>,
    pub error_class: Option<String>,
    pub error_detail: Option<String>,
    pub queued_at: Option<f64>,
    pub started_at: Option<f64>,
    pub finished_at: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub id: String,
    pub job_id: String,
    pub task_id: String,
    pub attempt: u32,
    pub name: String,
    pub sha256: String,
    pub bytes: u64,
    pub kind: String,
    pub metadata_json: Option<String>,
    pub created_at: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRecord {
    pub id: String,
    pub agent_version: String,
    pub backend_kind: String,
    pub metrics_url: String,
    pub host_ram_total_bytes: u64,
    pub models: Vec<ModelDecl>,
    pub limits: Vec<LimitDecl>,
    pub online: bool,
    pub session: Option<String>,
    pub connected_at: Option<f64>,
    pub last_seen_at: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorRecord {
    pub key: String, // "<agent_id>:<executor_id>"
    pub agent_id: String,
    pub id: String, // "<executor_id>" e.g. "xpu0"
    pub device: String,
    pub mem_total_bytes: u64,
    pub stages: Vec<StageCapDecl>,
    pub state: ExecutorState,
    pub current_stage: Option<String>,
    pub current_model_id: Option<String>,
    pub current_model_rev: Option<String>,
    pub current_model_dtype: Option<String>,
    pub current_task_id: Option<String>,
    pub worker_pid: Option<u32>,
    pub load_s: Option<f64>,
    pub disabled_reason: Option<String>,
    pub lent: bool,
    pub pending_lend: bool,
    pub idle_since: Option<f64>,
    pub last_seen_at: f64,
}

#[derive(Debug, Clone)]
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    pub async fn connect(db_path: impl AsRef<Path>) -> Result<Self> {
        let path_str = db_path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow!("invalid database path"))?;
        if let Some(parent) = db_path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let opts = SqliteConnectOptions::from_str(&format!("sqlite://{path_str}"))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(std::time::Duration::from_secs(10))
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await
            .context("failed to connect to SQLite")?;

        let db = Self { pool };
        db.init_schema().await?;
        Ok(db)
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    async fn init_schema(&self) -> Result<()> {
        let queries = [
            "CREATE TABLE IF NOT EXISTS global_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL DEFAULT 1
            );",
            "INSERT OR IGNORE INTO global_state (id, version) VALUES (1, 1);",
            "CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                prompt TEXT NOT NULL,
                enhanced_prompt TEXT,
                thinking TEXT,
                negative_prompt TEXT NOT NULL DEFAULT '',
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                steps INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                true_cfg_scale REAL NOT NULL DEFAULT 1.0,
                enhance INTEGER NOT NULL DEFAULT 0,
                wh_ratio TEXT,
                model_id TEXT,
                model_rev TEXT,
                error_class TEXT,
                error_detail TEXT,
                blocked_reason TEXT,
                version INTEGER NOT NULL,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL
            );",
            "CREATE INDEX IF NOT EXISTS idx_jobs_version ON jobs(version);",
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);",
            "CREATE TABLE IF NOT EXISTS deleted_jobs (
                id TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            );",
            "CREATE INDEX IF NOT EXISTS idx_deleted_jobs_version ON deleted_jobs(version);",
            "CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                json TEXT NOT NULL
            );",
            "CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                current_attempt INTEGER NOT NULL DEFAULT 0,
                assigned_agent TEXT,
                assigned_executor TEXT,
                blocked_reason TEXT,
                step INTEGER NOT NULL DEFAULT 0,
                steps INTEGER NOT NULL DEFAULT 0,
                s_per_step REAL NOT NULL DEFAULT 0.0,
                sub_stage TEXT,
                stats_json TEXT,
                error_class TEXT,
                error_detail TEXT,
                queued_at REAL,
                started_at REAL,
                finished_at REAL
            );",
            "CREATE INDEX IF NOT EXISTS idx_tasks_job_id ON tasks(job_id, seq);",
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, stage);",
            "CREATE TABLE IF NOT EXISTS task_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                stage TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                agent TEXT NOT NULL,
                executor TEXT NOT NULL,
                status TEXT NOT NULL,
                device TEXT,
                backend_kind TEXT,
                model_id TEXT,
                model_rev TEXT,
                model_dtype TEXT,
                torch_version TEXT,
                worker_version TEXT,
                timings_json TEXT,
                peak_mem_bytes INTEGER,
                failure_class TEXT,
                failure_detail TEXT,
                exit_code INTEGER,
                signal INTEGER,
                kernel_log_tail TEXT,
                started_at REAL NOT NULL,
                finished_at REAL,
                UNIQUE(task_id, attempt)
            );",
            "CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                attempt INTEGER NOT NULL,
                name TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                kind TEXT NOT NULL,
                metadata_json TEXT,
                created_at REAL NOT NULL,
                UNIQUE(task_id, attempt, name)
            );",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_job ON artifacts(job_id);",
            "CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                agent_version TEXT NOT NULL,
                backend_kind TEXT NOT NULL,
                metrics_url TEXT NOT NULL,
                host_ram_total_bytes INTEGER NOT NULL,
                models_json TEXT NOT NULL,
                limits_json TEXT NOT NULL,
                online INTEGER NOT NULL DEFAULT 0,
                session TEXT,
                connected_at REAL,
                last_seen_at REAL NOT NULL
            );",
            "CREATE TABLE IF NOT EXISTS uploads (
                sha256 TEXT PRIMARY KEY,
                mime TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                bytes INTEGER NOT NULL,
                created_at REAL NOT NULL
            );",
            "CREATE TABLE IF NOT EXISTS job_references (
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                idx INTEGER NOT NULL,
                sha256 TEXT NOT NULL REFERENCES uploads(sha256),
                PRIMARY KEY (job_id, idx)
            );",
            "CREATE INDEX IF NOT EXISTS idx_job_references_sha ON job_references(sha256);",
            "CREATE TABLE IF NOT EXISTS reference_latents (
                source_sha256 TEXT NOT NULL REFERENCES uploads(sha256) ON DELETE CASCADE,
                model_id TEXT NOT NULL,
                model_rev TEXT NOT NULL,
                reference_res INTEGER NOT NULL,
                blob_sha256 TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                dtype TEXT,
                created_at REAL NOT NULL,
                PRIMARY KEY (source_sha256, model_id, model_rev, reference_res)
            );",
            "CREATE TABLE IF NOT EXISTS executors (
                key TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                executor_id TEXT NOT NULL,
                device TEXT NOT NULL,
                mem_total_bytes INTEGER NOT NULL,
                stages_json TEXT NOT NULL,
                state TEXT NOT NULL,
                current_stage TEXT,
                current_model_id TEXT,
                current_model_rev TEXT,
                current_model_dtype TEXT,
                current_task_id TEXT,
                worker_pid INTEGER,
                load_s REAL,
                disabled_reason TEXT,
                lent INTEGER NOT NULL DEFAULT 0,
                pending_lend INTEGER NOT NULL DEFAULT 0,
                idle_since REAL,
                last_seen_at REAL NOT NULL
            );",
        ];
        for q in queries {
            sqlx::query(q).execute(&self.pool).await?;
        }
        // Columns added after the first V3 schema.
        self.ensure_column("jobs", "thinking", "TEXT").await?;
        self.ensure_column("jobs", "meta", "TEXT NOT NULL DEFAULT '{}'")
            .await?;
        self.ensure_column("jobs", "favorite", "INTEGER NOT NULL DEFAULT 0")
            .await?;
        self.ensure_column("jobs", "tiling", "INTEGER NOT NULL DEFAULT 0")
            .await?;
        self.ensure_column("jobs", "enhance_max_tokens", "INTEGER")
            .await?;
        self.ensure_column("jobs", "ratio_follow", "TEXT").await?;
        self.ensure_column("jobs", "enhancer", "TEXT").await?;
        self.ensure_column("tasks", "sub_stage", "TEXT").await?;
        self.ensure_column("tasks", "stats_json", "TEXT").await?;
        // Encode inputs, so jobs of one send share one encode (scheduler.rs `encode_cache_key`).
        self.ensure_column("artifacts", "cache_key", "TEXT").await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_artifacts_cache_key ON artifacts(cache_key)")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn ensure_column(&self, table: &str, column: &str, decl: &str) -> Result<()> {
        let rows = sqlx::query(&format!("PRAGMA table_info({table})"))
            .fetch_all(&self.pool)
            .await?;
        if !rows.iter().any(|r| r.get::<String, _>("name") == column) {
            sqlx::query(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"))
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }

    async fn next_version_tx(tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>) -> Result<i64> {
        sqlx::query("UPDATE global_state SET version = version + 1 WHERE id = 1")
            .execute(&mut **tx)
            .await?;
        let row = sqlx::query("SELECT version FROM global_state WHERE id = 1")
            .fetch_one(&mut **tx)
            .await?;
        Ok(row.get::<i64, _>("version"))
    }

    pub async fn current_version(&self) -> Result<i64> {
        let row = sqlx::query("SELECT version FROM global_state WHERE id = 1")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.get::<i64, _>("version"))
    }

    /// Registers or updates an agent and its executors on `Register`.
    pub async fn upsert_agent_registration(
        &self,
        agent: &AgentRecord,
        executors: &[ExecutorRecord],
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let now = now_s();
        let models_json = serde_json::to_string(&agent.models)?;
        let limits_json = serde_json::to_string(&agent.limits)?;

        sqlx::query(
            "INSERT INTO agents (
                id, agent_version, backend_kind, metrics_url, host_ram_total_bytes,
                models_json, limits_json, online, session, connected_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                agent_version = excluded.agent_version,
                backend_kind = excluded.backend_kind,
                metrics_url = excluded.metrics_url,
                host_ram_total_bytes = excluded.host_ram_total_bytes,
                models_json = excluded.models_json,
                limits_json = excluded.limits_json,
                online = 1,
                session = excluded.session,
                connected_at = excluded.connected_at,
                last_seen_at = excluded.last_seen_at",
        )
        .bind(&agent.id)
        .bind(&agent.agent_version)
        .bind(&agent.backend_kind)
        .bind(&agent.metrics_url)
        .bind(agent.host_ram_total_bytes as i64)
        .bind(&models_json)
        .bind(&limits_json)
        .bind(&agent.session)
        .bind(now)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        for ex in executors {
            let stages_json = serde_json::to_string(&ex.stages)?;
            // Preserve `disabled` and `lent` state across agent reconnects (§6.1, §7.5)!
            let existing = sqlx::query(
                "SELECT state, disabled_reason, lent, pending_lend FROM executors WHERE key = ?",
            )
            .bind(&ex.key)
            .fetch_optional(&mut *tx)
            .await?;

            let (init_state, disabled_reason, lent, pending_lend) = if let Some(row) = existing {
                let prev_state: ExecutorState = row.get("state");
                let prev_reason: Option<String> = row.get("disabled_reason");
                let prev_lent: i64 = row.get("lent");
                let prev_pending_lend: i64 = row.get("pending_lend");
                if prev_state == ExecutorState::Disabled || prev_reason.is_some() {
                    (ExecutorState::Disabled, prev_reason, prev_lent != 0, false)
                } else if prev_lent != 0 || prev_state == ExecutorState::Lent {
                    (ExecutorState::Lent, None, true, false)
                } else {
                    (ex.state, None, false, prev_pending_lend != 0)
                }
            } else {
                (
                    ex.state,
                    ex.disabled_reason.clone(),
                    ex.lent,
                    ex.pending_lend,
                )
            };

            let idle_since = if matches!(init_state, ExecutorState::Idle | ExecutorState::Ready) {
                Some(now)
            } else {
                None
            };

            sqlx::query(
                "INSERT INTO executors (
                    key, agent_id, executor_id, device, mem_total_bytes, stages_json,
                    state, current_stage, current_model_id, current_model_rev, current_model_dtype,
                    current_task_id, worker_pid, load_s, disabled_reason, lent, pending_lend,
                    idle_since, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    device = excluded.device,
                    mem_total_bytes = excluded.mem_total_bytes,
                    stages_json = excluded.stages_json,
                    state = excluded.state,
                    current_stage = NULL,
                    current_model_id = NULL,
                    current_model_rev = NULL,
                    current_model_dtype = NULL,
                    current_task_id = NULL,
                    worker_pid = NULL,
                    disabled_reason = excluded.disabled_reason,
                    lent = excluded.lent,
                    pending_lend = excluded.pending_lend,
                    idle_since = COALESCE(executors.idle_since, excluded.idle_since),
                    last_seen_at = excluded.last_seen_at",
            )
            .bind(&ex.key)
            .bind(&ex.agent_id)
            .bind(&ex.id)
            .bind(&ex.device)
            .bind(ex.mem_total_bytes as i64)
            .bind(&stages_json)
            .bind(init_state)
            .bind(&disabled_reason)
            .bind(if lent { 1i64 } else { 0i64 })
            .bind(if pending_lend { 1i64 } else { 0i64 })
            .bind(idle_since)
            .bind(now)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    pub async fn mark_agent_disconnected(&self, agent_id: &str, session: &str) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        let now = now_s();
        // Only mark offline if session still matches (haven't already reconnected on a new stream)
        let res = sqlx::query(
            "UPDATE agents SET online = 0, last_seen_at = ? WHERE id = ? AND session = ?",
        )
        .bind(now)
        .bind(agent_id)
        .bind(session)
        .execute(&mut *tx)
        .await?;
        if res.rows_affected() == 0 {
            tx.commit().await?;
            return Ok(false);
        }

        // Preserve disabled/lent state indicator via disabled_reason / lent column, set state to offline unless disabled
        sqlx::query(
            "UPDATE executors
             SET state = CASE WHEN state = 'disabled' THEN 'disabled' ELSE 'offline' END,
                 worker_pid = NULL,
                 last_seen_at = ?
             WHERE agent_id = ?",
        )
        .bind(now)
        .bind(agent_id)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(true)
    }

    /// Reclaims in-flight tasks for an agent that stayed offline past `reconnect_grace` (§7.3)
    /// since `session` ended. A timer from an earlier session does nothing: the agent came
    /// back since, and a later disconnect has its own timer.
    pub async fn expire_disconnected_agent_tasks(
        &self,
        agent_id: &str,
        session: &str,
    ) -> Result<Vec<String>> {
        self.reclaim_agent_tasks(agent_id, &[], Some(session)).await
    }

    /// On (re)registration: reclaims tasks the backend has on this agent that the agent no
    /// longer reports in `Register.running` (it restarted, or its result was lost), and marks
    /// the executors of the tasks it does report busy again.
    pub async fn reconcile_agent_running_tasks(
        &self,
        agent_id: &str,
        running: &[(String, u32, String)],
    ) -> Result<Vec<String>> {
        let keep: Vec<(String, u32)> = running.iter().map(|(t, a, _)| (t.clone(), *a)).collect();
        let affected = self.reclaim_agent_tasks(agent_id, &keep, None).await?;
        let now = now_s();
        for (task_id, attempt, executor_id) in running {
            sqlx::query(
                "UPDATE executors SET state = 'busy', current_task_id = ?, idle_since = NULL, last_seen_at = ?
                 WHERE key = ? AND state != 'disabled'
                   AND EXISTS (SELECT 1 FROM tasks WHERE id = ? AND current_attempt = ?
                               AND assigned_agent = ? AND assigned_executor = ?
                               AND status IN ('assigned', 'running', 'uploading'))",
            )
            .bind(task_id)
            .bind(now)
            .bind(format!("{agent_id}:{executor_id}"))
            .bind(task_id)
            .bind(*attempt as i64)
            .bind(agent_id)
            .bind(executor_id)
            .execute(&self.pool)
            .await?;
        }
        Ok(affected)
    }

    /// Requeues (or fails, on the last attempt) every in-flight task assigned to `agent_id`
    /// except the `(task, attempt)` pairs in `keep`. With `require_offline`, does nothing
    /// if the agent has reconnected in the meantime.
    async fn reclaim_agent_tasks(
        &self,
        agent_id: &str,
        keep: &[(String, u32)],
        offline_session: Option<&str>,
    ) -> Result<Vec<String>> {
        let mut tx = self.pool.begin().await?;
        if let Some(session) = offline_session {
            let row = sqlx::query("SELECT online, session FROM agents WHERE id = ?")
                .bind(agent_id)
                .fetch_optional(&mut *tx)
                .await?;
            if let Some(r) = row {
                let current: Option<String> = r.get("session");
                if r.get::<i64, _>("online") != 0 || current.as_deref() != Some(session) {
                    tx.commit().await?;
                    return Ok(Vec::new());
                }
            }
        }

        let task_rows = sqlx::query(
            "SELECT id, job_id, current_attempt, attempts FROM tasks
             WHERE assigned_agent = ? AND status IN ('assigned', 'running', 'uploading')",
        )
        .bind(agent_id)
        .fetch_all(&mut *tx)
        .await?;

        let now = now_s();
        let mut affected_jobs = Vec::new();
        for tr in task_rows {
            let task_id: String = tr.get("id");
            let job_id: String = tr.get("job_id");
            let attempt: i64 = tr.get("current_attempt");
            let attempts: i64 = tr.get("attempts");
            if keep
                .iter()
                .any(|(t, a)| *t == task_id && *a as i64 == attempt)
            {
                continue;
            }

            sqlx::query(
                "UPDATE task_attempts
                 SET status = 'lost', failure_class = 'lost', failure_detail = 'agent vanished', finished_at = ?
                 WHERE task_id = ? AND attempt = ?",
            )
            .bind(now)
            .bind(&task_id)
            .bind(attempt)
            .execute(&mut *tx)
            .await?;

            let ver = Self::next_version_tx(&mut tx).await?;
            if attempts >= 2 {
                sqlx::query(
                    "UPDATE tasks
                     SET status = 'failed', assigned_agent = NULL, assigned_executor = NULL,
                         error_class = 'lost', error_detail = 'agent vanished', finished_at = ?
                     WHERE id = ?",
                )
                .bind(now)
                .bind(&task_id)
                .execute(&mut *tx)
                .await?;

                sqlx::query(
                    "UPDATE jobs
                     SET status = 'failed', error_class = 'lost', error_detail = 'agent vanished',
                         version = ?, finished_at = ?
                     WHERE id = ?",
                )
                .bind(ver)
                .bind(now)
                .bind(&job_id)
                .execute(&mut *tx)
                .await?;
            } else {
                sqlx::query(
                    "UPDATE tasks
                     SET status = 'queued', assigned_agent = NULL, assigned_executor = NULL,
                         step = 0, s_per_step = 0.0, queued_at = COALESCE(queued_at, ?)
                     WHERE id = ?",
                )
                .bind(now)
                .bind(&task_id)
                .execute(&mut *tx)
                .await?;

                sqlx::query("UPDATE jobs SET version = ? WHERE id = ?")
                    .bind(ver)
                    .bind(&job_id)
                    .execute(&mut *tx)
                    .await?;
            }
            affected_jobs.push(job_id);
        }

        if offline_session.is_some() {
            sqlx::query(
                "UPDATE executors
                 SET current_task_id = NULL, current_stage = NULL, current_model_id = NULL,
                     current_model_rev = NULL, current_model_dtype = NULL
                 WHERE agent_id = ?",
            )
            .bind(agent_id)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(affected_jobs)
    }

    pub async fn list_agents(&self) -> Result<Vec<AgentRecord>> {
        let rows = sqlx::query("SELECT * FROM agents ORDER BY id ASC")
            .fetch_all(&self.pool)
            .await?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let models: Vec<ModelDecl> =
                serde_json::from_str(r.get::<&str, _>("models_json")).unwrap_or_default();
            let limits: Vec<LimitDecl> =
                serde_json::from_str(r.get::<&str, _>("limits_json")).unwrap_or_default();
            out.push(AgentRecord {
                id: r.get("id"),
                agent_version: r.get("agent_version"),
                backend_kind: r.get("backend_kind"),
                metrics_url: r.get("metrics_url"),
                host_ram_total_bytes: r.get::<i64, _>("host_ram_total_bytes") as u64,
                models,
                limits,
                online: r.get::<i64, _>("online") != 0,
                session: r.get("session"),
                connected_at: r.get("connected_at"),
                last_seen_at: r.get("last_seen_at"),
            });
        }
        Ok(out)
    }

    pub async fn list_executors(&self) -> Result<Vec<ExecutorRecord>> {
        let rows = sqlx::query("SELECT * FROM executors ORDER BY agent_id ASC, executor_id ASC")
            .fetch_all(&self.pool)
            .await?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(Self::row_to_executor(&r));
        }
        Ok(out)
    }

    pub async fn find_executor(&self, id_or_key: &str) -> Result<Option<ExecutorRecord>> {
        let row = sqlx::query(
            "SELECT * FROM executors WHERE key = ? OR executor_id = ? ORDER BY last_seen_at DESC LIMIT 1",
        )
        .bind(id_or_key)
        .bind(id_or_key)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.as_ref().map(Self::row_to_executor))
    }

    fn row_to_executor(r: &sqlx::sqlite::SqliteRow) -> ExecutorRecord {
        let stages: Vec<StageCapDecl> =
            serde_json::from_str(r.get::<&str, _>("stages_json")).unwrap_or_default();
        ExecutorRecord {
            key: r.get("key"),
            agent_id: r.get("agent_id"),
            id: r.get("executor_id"),
            device: r.get("device"),
            mem_total_bytes: r.get::<i64, _>("mem_total_bytes") as u64,
            stages,
            state: r.get("state"),
            current_stage: r.get("current_stage"),
            current_model_id: r.get("current_model_id"),
            current_model_rev: r.get("current_model_rev"),
            current_model_dtype: r.get("current_model_dtype"),
            current_task_id: r.get("current_task_id"),
            worker_pid: r.get::<Option<i64>, _>("worker_pid").map(|v| v as u32),
            load_s: r.get("load_s"),
            disabled_reason: r.get("disabled_reason"),
            lent: r.get::<i64, _>("lent") != 0,
            pending_lend: r.get::<i64, _>("pending_lend") != 0,
            idle_since: r.get("idle_since"),
            last_seen_at: r.get("last_seen_at"),
        }
    }

    /// Creates a new job and its 3 or 4 tasks (`enhance`?, `encode`, `denoise`, `decode`).
    /// `references` are upload hashes in prompt order, already checked to exist.
    pub async fn create_job(
        &self,
        input: &JobSubmitInput,
        seed: u64,
        references: &[String],
    ) -> Result<JobRecord> {
        let mut tx = self.pool.begin().await?;
        let now = now_s();
        let ver = Self::next_version_tx(&mut tx).await?;
        let job_id = match &input.id {
            Some(id) => id.clone(),
            None => format!("job_{}", ulid::Ulid::new().to_string().to_lowercase()),
        };

        sqlx::query(
            "INSERT INTO jobs (
                id, status, prompt, negative_prompt, width, height, steps, seed,
                true_cfg_scale, enhance, meta, tiling, version, created_at, enhance_max_tokens,
                enhancer
            ) VALUES (?, 'submitted', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&job_id)
        .bind(&input.prompt)
        .bind(&input.negative_prompt)
        .bind(input.width as i64)
        .bind(input.height as i64)
        .bind(input.steps as i64)
        .bind(seed as i64)
        .bind(input.true_cfg_scale as f64)
        .bind(if input.enhance { 1i64 } else { 0i64 })
        .bind(
            input
                .meta
                .as_ref()
                .map(|m| m.to_string())
                .unwrap_or_else(|| "{}".to_string()),
        )
        .bind(input.tiling as i64)
        .bind(ver)
        .bind(now)
        .bind(
            input
                .enhance_max_tokens
                .filter(|_| input.enhance)
                .map(|t| t as i64),
        )
        .bind(input.enhancer.as_deref().filter(|_| input.enhance))
        .execute(&mut *tx)
        .await?;

        for (idx, sha) in references.iter().enumerate() {
            sqlx::query("INSERT INTO job_references (job_id, idx, sha256) VALUES (?, ?, ?)")
                .bind(&job_id)
                .bind(idx as i64)
                .bind(sha)
                .execute(&mut *tx)
                .await?;
        }

        let mut stages = Vec::new();
        if input.enhance {
            stages.push("enhance");
        }
        stages.extend(["encode", "denoise", "decode"]);

        for (seq, stage) in stages.iter().enumerate() {
            let task_id = format!("task_{}", ulid::Ulid::new().to_string().to_lowercase());
            let is_first = seq == 0;
            let status = if is_first { "queued" } else { "waiting" };
            let queued_at = if is_first { Some(now) } else { None };
            let steps = if *stage == "denoise" { input.steps } else { 0 };
            sqlx::query(
                "INSERT INTO tasks (
                    id, job_id, seq, stage, status, steps, queued_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&task_id)
            .bind(&job_id)
            .bind(seq as i32)
            .bind(*stage)
            .bind(status)
            .bind(steps as i64)
            .bind(queued_at)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        self.get_job(&job_id)
            .await?
            .ok_or_else(|| anyhow!("newly created job missing"))
    }

    pub async fn get_job(&self, job_id: &str) -> Result<Option<JobRecord>> {
        let row = sqlx::query("SELECT * FROM jobs WHERE id = ?")
            .bind(job_id)
            .fetch_optional(&self.pool)
            .await?;
        let Some(r) = row else {
            return Ok(None);
        };
        let mut job = Self::row_to_job(&r);
        job.tasks = self.list_tasks_for_job(job_id).await?;
        job.provenance = self.get_job_provenance(job_id).await?;
        job.references = self.list_job_references(job_id).await?;
        Ok(Some(job))
    }

    /// Jobs (all, or changed since a version) with their tasks and provenance.
    pub async fn list_jobs(&self, changed_since: Option<i64>) -> Result<Vec<JobRecord>> {
        self.load_jobs(
            "j.version > ?",
            JobFilter::Version(changed_since.unwrap_or(i64::MIN)),
        )
        .await
    }

    /// The jobs with these ids that exist, in one batch (the scheduler tick's jobs).
    pub async fn get_jobs(&self, ids: &[String]) -> Result<Vec<JobRecord>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids = serde_json::to_string(ids)?;
        self.load_jobs(
            "j.id IN (SELECT value FROM json_each(?))",
            JobFilter::Ids(ids),
        )
        .await
    }

    /// Jobs matching `cond` (on `jobs j`, with one `?` for `filter`) with their tasks,
    /// provenance and references, in four queries rather than four per job.
    async fn load_jobs(&self, cond: &str, filter: JobFilter) -> Result<Vec<JobRecord>> {
        let jobs_sql = format!("SELECT j.* FROM jobs j WHERE {cond} ORDER BY j.created_at DESC");
        let rows = filter
            .bind(sqlx::query(&jobs_sql))
            .fetch_all(&self.pool)
            .await?;
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let tasks_sql = format!(
            "SELECT t.* FROM tasks t JOIN jobs j ON t.job_id = j.id
             WHERE {cond} ORDER BY t.job_id, t.seq ASC"
        );
        let task_rows = filter
            .bind(sqlx::query(&tasks_sql))
            .fetch_all(&self.pool)
            .await?;
        let attempts_sql = format!(
            "SELECT a.* FROM task_attempts a JOIN jobs j ON a.job_id = j.id
             WHERE {cond} AND a.status = 'done' ORDER BY a.attempt ASC"
        );
        let attempt_rows = filter
            .bind(sqlx::query(&attempts_sql))
            .fetch_all(&self.pool)
            .await?;

        let mut tasks: HashMap<String, Vec<TaskRecord>> = HashMap::new();
        for r in &task_rows {
            let t = Self::row_to_task(r);
            tasks.entry(t.job_id.clone()).or_default().push(t);
        }
        let references_sql = format!(
            "SELECT r.job_id, u.sha256, u.width, u.height FROM job_references r
             JOIN jobs j ON r.job_id = j.id JOIN uploads u ON r.sha256 = u.sha256
             WHERE {cond} ORDER BY r.job_id, r.idx"
        );
        let reference_rows = filter
            .bind(sqlx::query(&references_sql))
            .fetch_all(&self.pool)
            .await?;
        let mut references: HashMap<String, Vec<JobReference>> = HashMap::new();
        for r in &reference_rows {
            references
                .entry(r.get("job_id"))
                .or_default()
                .push(Self::row_to_job_reference(r));
        }
        let mut provenance: HashMap<String, HashMap<String, ProvenanceRecord>> = HashMap::new();
        for r in &attempt_rows {
            let job_id: String = r.get("job_id");
            let stage: String = r.get("stage");
            provenance
                .entry(job_id)
                .or_default()
                .insert(stage, Self::row_to_provenance(r));
        }

        Ok(rows
            .iter()
            .map(|r| {
                let mut job = Self::row_to_job(r);
                job.tasks = tasks.remove(&job.id).unwrap_or_default();
                job.provenance = provenance.remove(&job.id).unwrap_or_default();
                job.references = references.remove(&job.id).unwrap_or_default();
                job
            })
            .collect())
    }

    /// One page of jobs, newest first, after `q.before` and matching its filters. Returns the
    /// page and the cursor of the next one (`None`: this was the last).
    pub async fn list_jobs_page(
        &self,
        q: &JobPageQuery,
    ) -> Result<(Vec<JobRecord>, Option<String>)> {
        let mut sql = sqlx::QueryBuilder::<sqlx::Sqlite>::new("SELECT id FROM jobs WHERE 1 = 1");
        if q.done_only {
            sql.push(" AND status = 'done'");
        }
        if q.favorites_only {
            sql.push(" AND favorite = 1");
        }
        if let Some(text) = q.search.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
            let like = format!(
                "%{}%",
                text.replace('\\', "\\\\")
                    .replace('%', "\\%")
                    .replace('_', "\\_")
            );
            sql.push(" AND (prompt LIKE ")
                .push_bind(like.clone())
                .push(" ESCAPE '\\'");
            sql.push(" OR enhanced_prompt LIKE ")
                .push_bind(like.clone())
                .push(" ESCAPE '\\'");
            sql.push(" OR CAST(seed AS TEXT) LIKE ")
                .push_bind(like)
                .push(" ESCAPE '\\')");
        }
        if let Some(ratio) = q.ratio {
            sql.push(" AND abs(CAST(width AS REAL) / height - ")
                .push_bind(ratio)
                .push(") < 0.01");
        }
        if let Some((created_at, id)) = q.before.as_deref().and_then(parse_job_cursor) {
            sql.push(" AND (created_at < ")
                .push_bind(created_at)
                .push(" OR (created_at = ")
                .push_bind(created_at)
                .push(" AND id < ")
                .push_bind(id)
                .push("))");
        }
        let limit = q.limit.clamp(1, 200);
        sql.push(" ORDER BY created_at DESC, id DESC LIMIT ")
            .push_bind(limit as i64 + 1);
        let mut ids: Vec<String> = sql
            .build()
            .fetch_all(&self.pool)
            .await?
            .iter()
            .map(|r| r.get("id"))
            .collect();
        let more = ids.len() > limit as usize;
        ids.truncate(limit as usize);
        let mut jobs = self.get_jobs(&ids).await?;
        jobs.sort_by(|a, b| {
            b.created_at
                .total_cmp(&a.created_at)
                .then_with(|| b.id.cmp(&a.id))
        });
        let next = if more {
            jobs.last().map(job_cursor)
        } else {
            None
        };
        Ok((jobs, next))
    }

    /// Counts for the gallery's filters: finished images in all, favorites, and per size.
    pub async fn gallery_facets(&self) -> Result<GalleryFacets> {
        let rows = sqlx::query(
            "SELECT width, height, COUNT(*) AS n, SUM(favorite) AS favorites
             FROM jobs WHERE status = 'done' GROUP BY width, height",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut facets = GalleryFacets::default();
        for r in rows {
            let count = r.get::<i64, _>("n") as u64;
            facets.total += count;
            facets.favorites += r.get::<Option<i64>, _>("favorites").unwrap_or(0) as u64;
            facets.sizes.push(SizeCount {
                width: r.get::<i64, _>("width") as u32,
                height: r.get::<i64, _>("height") as u32,
                count,
            });
        }
        Ok(facets)
    }

    pub async fn list_job_references(&self, job_id: &str) -> Result<Vec<JobReference>> {
        let rows = sqlx::query(
            "SELECT u.sha256, u.width, u.height FROM job_references r
             JOIN uploads u ON r.sha256 = u.sha256 WHERE r.job_id = ? ORDER BY r.idx",
        )
        .bind(job_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(Self::row_to_job_reference).collect())
    }

    fn row_to_job_reference(r: &sqlx::sqlite::SqliteRow) -> JobReference {
        JobReference {
            sha256: r.get("sha256"),
            width: r.get::<i64, _>("width") as u32,
            height: r.get::<i64, _>("height") as u32,
        }
    }

    /// Records an uploaded image (its bytes are in the blob store under `sha256`). Uploading
    /// the same image again keeps the first record.
    pub async fn insert_upload(&self, up: &UploadRecord) -> Result<UploadRecord> {
        sqlx::query(
            "INSERT OR IGNORE INTO uploads (sha256, mime, width, height, bytes, created_at)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&up.sha256)
        .bind(&up.mime)
        .bind(up.width as i64)
        .bind(up.height as i64)
        .bind(up.bytes as i64)
        .bind(up.created_at)
        .execute(&self.pool)
        .await?;
        self.get_upload(&up.sha256)
            .await?
            .ok_or_else(|| anyhow!("upload {} missing after insert", up.sha256))
    }

    pub async fn get_upload(&self, sha256: &str) -> Result<Option<UploadRecord>> {
        let row = sqlx::query("SELECT * FROM uploads WHERE sha256 = ?")
            .bind(sha256)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| UploadRecord {
            sha256: r.get("sha256"),
            mime: r.get("mime"),
            width: r.get::<i64, _>("width") as u32,
            height: r.get::<i64, _>("height") as u32,
            bytes: r.get::<i64, _>("bytes") as u64,
            created_at: r.get("created_at"),
        }))
    }

    /// Stored latents of a reference for a model and resolution, if any.
    pub async fn get_reference_latents(
        &self,
        source_sha256: &str,
        model_id: &str,
        model_rev: &str,
        reference_res: u32,
    ) -> Result<Option<ReferenceLatents>> {
        let row = sqlx::query(
            "SELECT * FROM reference_latents
             WHERE source_sha256 = ? AND model_id = ? AND model_rev = ? AND reference_res = ?",
        )
        .bind(source_sha256)
        .bind(model_id)
        .bind(model_rev)
        .bind(reference_res as i64)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| ReferenceLatents {
            source_sha256: r.get("source_sha256"),
            model_id: r.get("model_id"),
            model_rev: r.get("model_rev"),
            reference_res: r.get::<i64, _>("reference_res") as u32,
            blob_sha256: r.get("blob_sha256"),
            bytes: r.get::<i64, _>("bytes") as u64,
            width: r.get::<i64, _>("width") as u32,
            height: r.get::<i64, _>("height") as u32,
        }))
    }

    async fn upsert_reference_latents_tx(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        lat: &ReferenceLatents,
        dtype: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO reference_latents (
                source_sha256, model_id, model_rev, reference_res, blob_sha256, bytes,
                width, height, dtype, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_sha256, model_id, model_rev, reference_res) DO UPDATE SET
                blob_sha256 = excluded.blob_sha256, bytes = excluded.bytes,
                width = excluded.width, height = excluded.height, dtype = excluded.dtype",
        )
        .bind(&lat.source_sha256)
        .bind(&lat.model_id)
        .bind(&lat.model_rev)
        .bind(lat.reference_res as i64)
        .bind(&lat.blob_sha256)
        .bind(lat.bytes as i64)
        .bind(lat.width as i64)
        .bind(lat.height as i64)
        .bind(dtype)
        .bind(now_s())
        .execute(&mut **tx)
        .await?;
        Ok(())
    }

    /// Whether any record still points at blob `sha256`. Hold [`BlobStore::collecting`]
    /// from this check until the file is gone.
    pub async fn blob_in_use(&self, sha256: &str) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        let used = Self::blob_in_use_tx(&mut tx, sha256).await?;
        tx.commit().await?;
        Ok(used)
    }

    /// Whether any record still points at blob `sha256`.
    async fn blob_in_use_tx(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        sha256: &str,
    ) -> Result<bool> {
        let row = sqlx::query(
            "SELECT 1 FROM artifacts WHERE sha256 = ?1
             UNION ALL SELECT 1 FROM uploads WHERE sha256 = ?1
             UNION ALL SELECT 1 FROM reference_latents WHERE blob_sha256 = ?1
             LIMIT 1",
        )
        .bind(sha256)
        .fetch_optional(&mut **tx)
        .await?;
        Ok(row.is_some())
    }

    /// Deletes uploads older than `max_age_s` that no job refers to, with their stored
    /// latents. Returns the blobs nothing refers to any more.
    pub async fn gc_uploads(&self, max_age_s: f64) -> Result<Vec<String>> {
        let mut tx = self.pool.begin().await?;
        let rows = sqlx::query(
            "SELECT u.sha256 FROM uploads u
             WHERE u.created_at < ?
               AND NOT EXISTS (SELECT 1 FROM job_references r WHERE r.sha256 = u.sha256)",
        )
        .bind(now_s() - max_age_s)
        .fetch_all(&mut *tx)
        .await?;
        let mut candidates = Vec::new();
        for r in rows {
            let sha: String = r.get("sha256");
            let latents =
                sqlx::query("SELECT blob_sha256 FROM reference_latents WHERE source_sha256 = ?")
                    .bind(&sha)
                    .fetch_all(&mut *tx)
                    .await?;
            candidates.extend(latents.iter().map(|l| l.get::<String, _>("blob_sha256")));
            sqlx::query("DELETE FROM reference_latents WHERE source_sha256 = ?")
                .bind(&sha)
                .execute(&mut *tx)
                .await?;
            sqlx::query("DELETE FROM uploads WHERE sha256 = ?")
                .bind(&sha)
                .execute(&mut *tx)
                .await?;
            candidates.push(sha);
        }
        let mut orphans = Vec::new();
        for sha in candidates {
            if !Self::blob_in_use_tx(&mut tx, &sha).await? && !orphans.contains(&sha) {
                orphans.push(sha);
            }
        }
        tx.commit().await?;
        Ok(orphans)
    }

    /// Ids of jobs deleted after `version`.
    pub async fn list_deleted_since(&self, version: i64) -> Result<Vec<String>> {
        let rows = sqlx::query("SELECT id FROM deleted_jobs WHERE version > ? ORDER BY version")
            .bind(version)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.iter().map(|r| r.get::<String, _>("id")).collect())
    }

    pub async fn set_job_favorite(
        &self,
        job_id: &str,
        favorite: bool,
    ) -> Result<Option<JobRecord>> {
        let mut tx = self.pool.begin().await?;
        let ver = Self::next_version_tx(&mut tx).await?;
        let res = sqlx::query("UPDATE jobs SET favorite = ?, version = ? WHERE id = ?")
            .bind(favorite as i64)
            .bind(ver)
            .bind(job_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        if res.rows_affected() == 0 {
            return Ok(None);
        }
        self.get_job(job_id).await
    }

    /// The UI's preferences (composer settings, theme): one JSON document, `{}` if unset.
    pub async fn get_preferences(&self) -> Result<serde_json::Value> {
        let row = sqlx::query("SELECT json FROM preferences WHERE id = 1")
            .fetch_optional(&self.pool)
            .await?;
        Ok(row
            .and_then(|r| serde_json::from_str(&r.get::<String, _>("json")).ok())
            .unwrap_or_else(|| serde_json::json!({})))
    }

    pub async fn put_preferences(&self, prefs: &serde_json::Value) -> Result<()> {
        sqlx::query("INSERT INTO preferences (id, json) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET json = excluded.json")
            .bind(prefs.to_string())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn count_jobs_by_status(&self) -> Result<HashMap<String, u64>> {
        let rows = sqlx::query("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .iter()
            .map(|r| {
                (
                    r.get::<String, _>("status"),
                    r.get::<i64, _>("n").max(0) as u64,
                )
            })
            .collect())
    }

    pub async fn list_tasks_for_job(&self, job_id: &str) -> Result<Vec<TaskRecord>> {
        let rows = sqlx::query("SELECT * FROM tasks WHERE job_id = ? ORDER BY seq ASC")
            .bind(job_id)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.iter().map(Self::row_to_task).collect())
    }

    pub async fn get_task(&self, task_id: &str) -> Result<Option<TaskRecord>> {
        let row = sqlx::query("SELECT * FROM tasks WHERE id = ?")
            .bind(task_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.as_ref().map(Self::row_to_task))
    }

    pub async fn list_queued_tasks(&self) -> Result<Vec<TaskRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM tasks WHERE status = 'queued' ORDER BY queued_at ASC, id ASC",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(Self::row_to_task).collect())
    }

    pub async fn list_active_tasks(&self) -> Result<Vec<TaskRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM tasks WHERE status IN ('assigned', 'running', 'uploading') ORDER BY started_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(Self::row_to_task).collect())
    }

    pub async fn get_job_provenance(
        &self,
        job_id: &str,
    ) -> Result<HashMap<String, ProvenanceRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM task_attempts WHERE job_id = ? AND status = 'done' ORDER BY attempt ASC",
        )
        .bind(job_id)
        .fetch_all(&self.pool)
        .await?;
        let mut map = HashMap::new();
        for r in rows {
            let stage: String = r.get("stage");
            map.insert(stage, Self::row_to_provenance(&r));
        }
        Ok(map)
    }

    fn row_to_provenance(r: &sqlx::sqlite::SqliteRow) -> ProvenanceRecord {
        {
            let timings: HashMap<String, f64> = r
                .get::<Option<String>, _>("timings_json")
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
            ProvenanceRecord {
                agent: r.get("agent"),
                executor: r.get("executor"),
                device: r.get::<Option<String>, _>("device").unwrap_or_default(),
                backend_kind: r
                    .get::<Option<String>, _>("backend_kind")
                    .unwrap_or_default(),
                model: ModelDecl {
                    id: r.get::<Option<String>, _>("model_id").unwrap_or_default(),
                    rev: r.get::<Option<String>, _>("model_rev").unwrap_or_default(),
                    dtype: r
                        .get::<Option<String>, _>("model_dtype")
                        .unwrap_or_default(),
                },
                torch_version: r
                    .get::<Option<String>, _>("torch_version")
                    .unwrap_or_default(),
                worker_version: r
                    .get::<Option<String>, _>("worker_version")
                    .unwrap_or_default(),
                timings,
                peak_mem_bytes: r.get::<Option<i64>, _>("peak_mem_bytes").unwrap_or(0) as u64,
            }
        }
    }

    fn row_to_job(r: &sqlx::sqlite::SqliteRow) -> JobRecord {
        JobRecord {
            id: r.get("id"),
            status: r.get("status"),
            prompt: r.get("prompt"),
            enhanced_prompt: r.get("enhanced_prompt"),
            thinking: r.get("thinking"),
            has_thinking: r
                .get::<Option<String>, _>("thinking")
                .is_some_and(|t| !t.is_empty()),
            negative_prompt: r.get("negative_prompt"),
            width: r.get::<i64, _>("width") as u32,
            height: r.get::<i64, _>("height") as u32,
            steps: r.get::<i64, _>("steps") as u32,
            seed: r.get::<i64, _>("seed") as u64,
            true_cfg_scale: r.get::<f64, _>("true_cfg_scale") as f32,
            enhance: r.get::<i64, _>("enhance") != 0,
            enhance_max_tokens: r
                .get::<Option<i64>, _>("enhance_max_tokens")
                .map(|t| t as u32),
            enhancer: r.get("enhancer"),
            tiling: r.get::<Option<i64>, _>("tiling").unwrap_or(0) != 0,
            wh_ratio: r.get("wh_ratio"),
            ratio_follow: r.get("ratio_follow"),
            model_id: r.get("model_id"),
            model_rev: r.get("model_rev"),
            error_class: r.get("error_class"),
            error_detail: r.get("error_detail"),
            blocked_reason: r.get("blocked_reason"),
            meta: r
                .get::<Option<String>, _>("meta")
                .and_then(|m| serde_json::from_str(&m).ok())
                .unwrap_or_else(|| serde_json::json!({})),
            favorite: r.get::<Option<i64>, _>("favorite").unwrap_or(0) != 0,
            references: Vec::new(),
            version: r.get("version"),
            created_at: r.get("created_at"),
            started_at: r.get("started_at"),
            finished_at: r.get("finished_at"),
            tasks: Vec::new(),
            provenance: HashMap::new(),
        }
    }

    fn row_to_task(r: &sqlx::sqlite::SqliteRow) -> TaskRecord {
        TaskRecord {
            id: r.get("id"),
            job_id: r.get("job_id"),
            seq: r.get("seq"),
            stage: r.get("stage"),
            status: r.get("status"),
            attempts: r.get::<i64, _>("attempts") as u32,
            current_attempt: r.get::<i64, _>("current_attempt") as u32,
            assigned_agent: r.get("assigned_agent"),
            assigned_executor: r.get("assigned_executor"),
            blocked_reason: r.get("blocked_reason"),
            step: r.get::<i64, _>("step") as u32,
            steps: r.get::<i64, _>("steps") as u32,
            s_per_step: r.get("s_per_step"),
            sub_stage: r.get("sub_stage"),
            stats: r
                .get::<Option<String>, _>("stats_json")
                .and_then(|j| serde_json::from_str(&j).ok()),
            error_class: r.get("error_class"),
            error_detail: r.get("error_detail"),
            queued_at: r.get("queued_at"),
            started_at: r.get("started_at"),
            finished_at: r.get("finished_at"),
        }
    }

    /// Sets or clears `blocked_reason` on a queued task and its job (bumping job version if changed).
    pub async fn set_task_blocked_reason(
        &self,
        task_id: &str,
        job_id: &str,
        reason: Option<&str>,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query("SELECT blocked_reason FROM tasks WHERE id = ?")
            .bind(task_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(r) = row else {
            tx.commit().await?;
            return Ok(());
        };
        let current: Option<String> = r.get("blocked_reason");
        if current.as_deref() == reason {
            tx.commit().await?;
            return Ok(());
        }
        let ver = Self::next_version_tx(&mut tx).await?;
        sqlx::query("UPDATE tasks SET blocked_reason = ? WHERE id = ?")
            .bind(reason)
            .bind(task_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("UPDATE jobs SET blocked_reason = ?, version = ? WHERE id = ?")
            .bind(reason)
            .bind(ver)
            .bind(job_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Marks an executor as `starting(stage)` when `StartWorker` is sent.
    pub async fn mark_executor_starting(
        &self,
        agent_id: &str,
        executor_id: &str,
        stage: &str,
        model: &ModelDecl,
    ) -> Result<()> {
        let key = format!("{agent_id}:{executor_id}");
        let now = now_s();
        sqlx::query(
            "UPDATE executors
             SET state = 'starting', current_stage = ?, current_model_id = ?, current_model_rev = ?,
                 current_model_dtype = ?, idle_since = NULL, last_seen_at = ?
             WHERE key = ?",
        )
        .bind(stage)
        .bind(&model.id)
        .bind(&model.rev)
        .bind(&model.dtype)
        .bind(now)
        .bind(&key)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Updates executor state from an `ExecutorState` stream message.
    pub async fn update_executor_state(
        &self,
        agent_id: &str,
        executor_id: &str,
        new_state: ExecutorState,
        stage: Option<&str>,
        pid: Option<u32>,
        load_s: Option<f64>,
    ) -> Result<()> {
        let key = format!("{agent_id}:{executor_id}");
        let now = now_s();
        let mut tx = self.pool.begin().await?;
        let existing = sqlx::query(
            "SELECT state, lent, pending_lend, disabled_reason, current_task_id FROM executors WHERE key = ?",
        )
        .bind(&key)
        .fetch_optional(&mut *tx)
        .await?;

        let Some(row) = existing else {
            tx.commit().await?;
            return Ok(());
        };
        let prev_state: ExecutorState = row.get("state");
        let current_task: Option<String> = row.get("current_task_id");
        // The agent reports READY right after Done. By then the backend may already have
        // assigned the next task (state busy, current_task_id set); that READY is stale and
        // must not reopen the executor for a second assignment.
        if new_state == ExecutorState::Ready
            && prev_state == ExecutorState::Busy
            && current_task.is_some()
        {
            tx.commit().await?;
            return Ok(());
        }
        let disabled_reason: Option<String> = row.get("disabled_reason");
        let lent: i64 = row.get("lent");
        let pending_lend: i64 = row.get("pending_lend");

        if prev_state == ExecutorState::Disabled || disabled_reason.is_some() {
            tx.commit().await?;
            return Ok(());
        }

        let effective_state =
            if new_state == ExecutorState::Idle && (lent != 0 || pending_lend != 0) {
                ExecutorState::Lent
            } else {
                new_state
            };
        let effective_lent = if effective_state == ExecutorState::Lent {
            1i64
        } else {
            lent
        };
        let effective_pending_lend = if effective_state == ExecutorState::Lent {
            0i64
        } else {
            pending_lend
        };
        let idle_since = if matches!(effective_state, ExecutorState::Idle | ExecutorState::Ready) {
            Some(now)
        } else {
            None
        };

        if matches!(effective_state, ExecutorState::Idle | ExecutorState::Lent) {
            sqlx::query(
                "UPDATE executors
                 SET state = ?, current_stage = NULL, current_model_id = NULL, current_model_rev = NULL,
                     current_model_dtype = NULL, current_task_id = NULL, worker_pid = NULL,
                     lent = ?, pending_lend = ?, idle_since = ?, last_seen_at = ?
                 WHERE key = ?",
            )
            .bind(effective_state)
            .bind(effective_lent)
            .bind(effective_pending_lend)
            .bind(idle_since)
            .bind(now)
            .bind(&key)
            .execute(&mut *tx)
            .await?;
        } else {
            sqlx::query(
                "UPDATE executors
                 SET state = ?,
                     current_stage = COALESCE(?, current_stage),
                     worker_pid = COALESCE(?, worker_pid),
                     load_s = CASE WHEN ? > 0 THEN ? ELSE load_s END,
                     idle_since = ?,
                     last_seen_at = ?
                 WHERE key = ?",
            )
            .bind(effective_state)
            .bind(stage)
            .bind(pid.map(|p| p as i64))
            .bind(load_s.unwrap_or(0.0))
            .bind(load_s)
            .bind(idle_since)
            .bind(now)
            .bind(&key)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Assigns a queued task to an executor in a single transaction, creating a new attempt.
    pub async fn assign_task(
        &self,
        task_id: &str,
        agent_id: &str,
        executor_id: &str,
    ) -> Result<Option<(TaskRecord, JobRecord, u32)>> {
        let mut tx = self.pool.begin().await?;
        let now = now_s();
        let t_row = sqlx::query("SELECT * FROM tasks WHERE id = ? AND status = 'queued'")
            .bind(task_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(tr) = t_row else {
            tx.commit().await?;
            return Ok(None);
        };
        let mut task = Self::row_to_task(&tr);
        let j_row = sqlx::query("SELECT * FROM jobs WHERE id = ?")
            .bind(&task.job_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(jr) = j_row else {
            tx.commit().await?;
            return Ok(None);
        };
        let job = Self::row_to_job(&jr);
        if job.status == JobStatus::Cancelled || job.status == JobStatus::Failed {
            tx.commit().await?;
            return Ok(None);
        }

        // `attempts` is the retry budget (reset by retry, given back on a rejection);
        // attempt numbers only grow, so every attempt keeps its own row and artifacts.
        let attempts = task.attempts + 1;
        let attempt = task.current_attempt + 1;
        let ver = Self::next_version_tx(&mut tx).await?;

        sqlx::query(
            "UPDATE tasks
             SET status = 'assigned', attempts = ?, current_attempt = ?,
                 assigned_agent = ?, assigned_executor = ?, blocked_reason = NULL,
                 step = 0, s_per_step = 0.0, sub_stage = NULL, stats_json = NULL, started_at = ?
             WHERE id = ?",
        )
        .bind(attempts as i64)
        .bind(attempt as i64)
        .bind(agent_id)
        .bind(executor_id)
        .bind(now)
        .bind(task_id)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "INSERT INTO task_attempts (
                task_id, job_id, stage, attempt, agent, executor, status, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'assigned', ?)",
        )
        .bind(task_id)
        .bind(&task.job_id)
        .bind(&task.stage)
        .bind(attempt as i64)
        .bind(agent_id)
        .bind(executor_id)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "UPDATE jobs
             SET status = 'running', blocked_reason = NULL,
                 started_at = COALESCE(started_at, ?), version = ?
             WHERE id = ?",
        )
        .bind(now)
        .bind(ver)
        .bind(&task.job_id)
        .execute(&mut *tx)
        .await?;

        let key = format!("{agent_id}:{executor_id}");
        sqlx::query(
            "UPDATE executors
             SET state = 'busy', current_task_id = ?, idle_since = NULL, last_seen_at = ?
             WHERE key = ?",
        )
        .bind(task_id)
        .bind(now)
        .bind(&key)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        task.status = TaskStatus::Assigned;
        task.attempts = attempts;
        task.current_attempt = attempt;
        task.assigned_agent = Some(agent_id.to_string());
        task.assigned_executor = Some(executor_id.to_string());
        task.blocked_reason = None;
        task.started_at = Some(now);
        Ok(Some((task, job, attempt)))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn record_task_progress(
        &self,
        task_id: &str,
        attempt: u32,
        step: u32,
        steps: u32,
        s_per_step: f64,
        sub_stage: Option<&str>,
        stats: Option<&HashMap<String, f64>>,
    ) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query("SELECT job_id, current_attempt, status FROM tasks WHERE id = ?")
            .bind(task_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(r) = row else {
            tx.commit().await?;
            return Ok(false);
        };
        let current_attempt = r.get::<i64, _>("current_attempt") as u32;
        let status: TaskStatus = r.get("status");
        if current_attempt != attempt || !status.is_active() {
            tx.commit().await?;
            return Ok(false);
        }
        let job_id: String = r.get("job_id");
        let ver = Self::next_version_tx(&mut tx).await?;

        let stats_json = match stats {
            Some(m) if !m.is_empty() => Some(serde_json::to_string(m)?),
            _ => None,
        };
        sqlx::query(
            "UPDATE tasks SET status = 'running', step = ?, steps = ?, s_per_step = ?,
                 sub_stage = COALESCE(?, sub_stage), stats_json = COALESCE(?, stats_json)
             WHERE id = ?",
        )
        .bind(step as i64)
        .bind(steps as i64)
        .bind(s_per_step)
        .bind(sub_stage)
        .bind(stats_json)
        .bind(task_id)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "UPDATE task_attempts SET status = 'running' WHERE task_id = ? AND attempt = ?",
        )
        .bind(task_id)
        .bind(attempt as i64)
        .execute(&mut *tx)
        .await?;

        sqlx::query("UPDATE jobs SET version = ? WHERE id = ?")
            .bind(ver)
            .bind(&job_id)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn insert_uploaded_artifact(
        &self,
        task_id: &str,
        attempt: u32,
        name: &str,
        sha256: &str,
        bytes: u64,
        kind: &str,
        metadata_json: Option<&str>,
        cache_key: Option<&str>,
        reference_latents: Option<(&ReferenceLatents, &str)>,
    ) -> Result<ArtifactRecord> {
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query("SELECT job_id, current_attempt, status FROM tasks WHERE id = ?")
            .bind(task_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(r) = row else {
            bail!("unknown task {task_id}");
        };
        let current_attempt = r.get::<i64, _>("current_attempt") as u32;
        let status: TaskStatus = r.get("status");
        if current_attempt != attempt || !status.is_active() {
            bail!("stale or inactive attempt {attempt} for task {task_id} (status={status}, current={current_attempt})");
        }
        let job_id: String = r.get("job_id");
        let now = now_s();
        let id = format!("art_{}", ulid::Ulid::new().to_string().to_lowercase());

        sqlx::query("UPDATE tasks SET status = 'uploading' WHERE id = ?")
            .bind(task_id)
            .execute(&mut *tx)
            .await?;

        sqlx::query(
            "INSERT INTO artifacts (
                id, job_id, task_id, attempt, name, sha256, bytes, kind, metadata_json, created_at,
                cache_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, attempt, name) DO UPDATE SET
                sha256 = excluded.sha256,
                bytes = excluded.bytes,
                metadata_json = excluded.metadata_json,
                created_at = excluded.created_at,
                cache_key = excluded.cache_key",
        )
        .bind(&id)
        .bind(&job_id)
        .bind(task_id)
        .bind(attempt as i64)
        .bind(name)
        .bind(sha256)
        .bind(bytes as i64)
        .bind(kind)
        .bind(metadata_json)
        .bind(now)
        .bind(cache_key)
        .execute(&mut *tx)
        .await?;

        // Stored for later jobs only once the attempt fence above has passed.
        if let Some((lat, dtype)) = reference_latents {
            Self::upsert_reference_latents_tx(&mut tx, lat, dtype).await?;
        }

        tx.commit().await?;
        Ok(ArtifactRecord {
            id,
            job_id,
            task_id: task_id.to_string(),
            attempt,
            name: name.to_string(),
            sha256: sha256.to_string(),
            bytes,
            kind: kind.to_string(),
            metadata_json: metadata_json.map(|s| s.to_string()),
            created_at: now,
        })
    }

    pub async fn get_artifact_for_task_attempt(
        &self,
        task_id: &str,
        attempt: u32,
        name: &str,
    ) -> Result<Option<ArtifactRecord>> {
        let row =
            sqlx::query("SELECT * FROM artifacts WHERE task_id = ? AND attempt = ? AND name = ?")
                .bind(task_id)
                .bind(attempt as i64)
                .bind(name)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.map(|r| ArtifactRecord {
            id: r.get("id"),
            job_id: r.get("job_id"),
            task_id: r.get("task_id"),
            attempt: r.get::<i64, _>("attempt") as u32,
            name: r.get("name"),
            sha256: r.get("sha256"),
            bytes: r.get::<i64, _>("bytes") as u64,
            kind: r.get("kind"),
            metadata_json: r.get("metadata_json"),
            created_at: r.get("created_at"),
        }))
    }

    pub async fn get_job_artifact_by_name(
        &self,
        job_id: &str,
        name: &str,
    ) -> Result<Option<ArtifactRecord>> {
        let row = sqlx::query(
            "SELECT a.* FROM artifacts a
             JOIN tasks t ON a.task_id = t.id AND a.attempt = t.current_attempt
             WHERE a.job_id = ? AND a.name = ? AND t.status = 'done'
             ORDER BY a.created_at DESC LIMIT 1",
        )
        .bind(job_id)
        .bind(name)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| ArtifactRecord {
            id: r.get("id"),
            job_id: r.get("job_id"),
            task_id: r.get("task_id"),
            attempt: r.get::<i64, _>("attempt") as u32,
            name: r.get("name"),
            sha256: r.get("sha256"),
            bytes: r.get::<i64, _>("bytes") as u64,
            kind: r.get("kind"),
            metadata_json: r.get("metadata_json"),
            created_at: r.get("created_at"),
        }))
    }

    /// Completes a task attempt with mandatory `ProvenanceRecord`, advancing the job to the next task or `done`.
    /// Returns `Ok(Some((job_id, job_finished)))` if accepted, or `Ok(None)` if fenced out (stale attempt).
    pub async fn complete_task_attempt(
        &self,
        task_id: &str,
        attempt: u32,
        prov: &ProvenanceRecord,
        enhanced_prompt: Option<(&str, &str, &str, &str)>,
        encode_model: Option<(&str, &str)>,
    ) -> Result<Option<(String, bool)>> {
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query("SELECT * FROM tasks WHERE id = ?")
            .bind(task_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(tr) = row else {
            tx.commit().await?;
            return Ok(None);
        };
        let task = Self::row_to_task(&tr);
        if task.current_attempt != attempt || !task.status.is_active() {
            tx.commit().await?;
            return Ok(None);
        }

        let now = now_s();
        let ver = Self::next_version_tx(&mut tx).await?;
        let timings_json = serde_json::to_string(&prov.timings)?;

        sqlx::query(
            "UPDATE task_attempts
             SET status = 'done', device = ?, backend_kind = ?, model_id = ?, model_rev = ?,
                 model_dtype = ?, torch_version = ?, worker_version = ?, timings_json = ?,
                 peak_mem_bytes = ?, finished_at = ?
             WHERE task_id = ? AND attempt = ?",
        )
        .bind(&prov.device)
        .bind(&prov.backend_kind)
        .bind(&prov.model.id)
        .bind(&prov.model.rev)
        .bind(&prov.model.dtype)
        .bind(&prov.torch_version)
        .bind(&prov.worker_version)
        .bind(&timings_json)
        .bind(prov.peak_mem_bytes as i64)
        .bind(now)
        .bind(task_id)
        .bind(attempt as i64)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "UPDATE tasks
             SET status = 'done', step = steps, blocked_reason = NULL, sub_stage = NULL, finished_at = ?
             WHERE id = ?",
        )
        .bind(now)
        .bind(task_id)
        .execute(&mut *tx)
        .await?;

        if let Some((rewritten, wh_ratio, thinking, ratio_follow)) = enhanced_prompt {
            sqlx::query(
                "UPDATE jobs SET enhanced_prompt = ?, wh_ratio = ?, thinking = ?, ratio_follow = ? WHERE id = ?",
            )
                .bind(rewritten)
                .bind(wh_ratio)
                .bind((!thinking.is_empty()).then_some(thinking))
                .bind((!ratio_follow.is_empty()).then_some(ratio_follow))
                .bind(&task.job_id)
                .execute(&mut *tx)
                .await?;
        }
        if let Some((m_id, m_rev)) = encode_model {
            sqlx::query("UPDATE jobs SET model_id = ?, model_rev = ? WHERE id = ?")
                .bind(m_id)
                .bind(m_rev)
                .bind(&task.job_id)
                .execute(&mut *tx)
                .await?;
        }

        let job_finished = Self::advance_job_tx(&mut tx, &task, now, ver).await?;

        // Mark the executor the task was assigned to ready (provenance names are agent-reported)
        let key = match (&task.assigned_agent, &task.assigned_executor) {
            (Some(a), Some(e)) => format!("{a}:{e}"),
            _ => format!("{}:{}", prov.agent, prov.executor),
        };
        sqlx::query(
            "UPDATE executors
             SET current_task_id = NULL,
                 state = CASE WHEN state = 'busy' THEN 'ready' ELSE state END,
                 idle_since = ?,
                 last_seen_at = ?
             WHERE key = ? AND current_task_id = ?",
        )
        .bind(now)
        .bind(now)
        .bind(&key)
        .bind(task_id)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(Some((task.job_id, job_finished)))
    }

    /// After `task` is done: queue the job's next task, or mark the job done (returns true).
    async fn advance_job_tx(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        task: &TaskRecord,
        now: f64,
        ver: i64,
    ) -> Result<bool> {
        let next_row =
            sqlx::query("SELECT id FROM tasks WHERE job_id = ? AND seq = ? AND status = 'waiting'")
                .bind(&task.job_id)
                .bind(task.seq + 1)
                .fetch_optional(&mut **tx)
                .await?;

        if let Some(nr) = next_row {
            let next_task_id: String = nr.get("id");
            sqlx::query("UPDATE tasks SET status = 'queued', queued_at = ? WHERE id = ?")
                .bind(now)
                .bind(&next_task_id)
                .execute(&mut **tx)
                .await?;
            sqlx::query("UPDATE jobs SET version = ?, blocked_reason = NULL WHERE id = ?")
                .bind(ver)
                .bind(&task.job_id)
                .execute(&mut **tx)
                .await?;
            Ok(false)
        } else {
            sqlx::query(
                "UPDATE jobs SET status = 'done', blocked_reason = NULL, finished_at = ?, version = ? WHERE id = ?",
            )
            .bind(now)
            .bind(ver)
            .bind(&task.job_id)
            .execute(&mut **tx)
            .await?;
            Ok(true)
        }
    }

    /// Finished embeds with these encode inputs (`cache_key`), from a done encode whose
    /// model is known: `(artifact, model_id, model_rev)`.
    pub async fn find_shared_embeds(
        &self,
        cache_key: &str,
    ) -> Result<Option<(ArtifactRecord, String, String)>> {
        let row = sqlx::query(
            "SELECT a.*, ta.model_id AS src_model_id, ta.model_rev AS src_model_rev
             FROM artifacts a
             JOIN tasks t ON a.task_id = t.id AND a.attempt = t.current_attempt
             JOIN task_attempts ta ON ta.task_id = a.task_id AND ta.attempt = a.attempt
             WHERE a.cache_key = ? AND a.name = 'embeds.safetensors' AND t.status = 'done'
               AND ta.model_id IS NOT NULL AND ta.model_rev IS NOT NULL
             ORDER BY a.created_at DESC LIMIT 1",
        )
        .bind(cache_key)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| {
            (
                ArtifactRecord {
                    id: r.get("id"),
                    job_id: r.get("job_id"),
                    task_id: r.get("task_id"),
                    attempt: r.get::<i64, _>("attempt") as u32,
                    name: r.get("name"),
                    sha256: r.get("sha256"),
                    bytes: r.get::<i64, _>("bytes") as u64,
                    kind: r.get("kind"),
                    metadata_json: r.get("metadata_json"),
                    created_at: r.get("created_at"),
                },
                r.get("src_model_id"),
                r.get("src_model_rev"),
            )
        }))
    }

    /// Completes a queued encode task with another job's embeds (same `cache_key`) instead of
    /// running it: one encode per send (docs/REFERENCE_IMAGES.md §4). The attempt copies the
    /// source attempt's provenance, timings and peak memory, with `"shared": 1` added to the
    /// timings. Returns false if the task is no longer queued.
    pub async fn complete_task_from_shared(
        &self,
        task_id: &str,
        source: &ArtifactRecord,
        cache_key: &str,
    ) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        let Some(tr) = sqlx::query("SELECT * FROM tasks WHERE id = ? AND status = 'queued'")
            .bind(task_id)
            .fetch_optional(&mut *tx)
            .await?
        else {
            tx.commit().await?;
            return Ok(false);
        };
        let task = Self::row_to_task(&tr);
        let Some(src) =
            sqlx::query("SELECT * FROM task_attempts WHERE task_id = ? AND attempt = ?")
                .bind(&source.task_id)
                .bind(source.attempt as i64)
                .fetch_optional(&mut *tx)
                .await?
        else {
            tx.commit().await?;
            return Ok(false);
        };
        let now = now_s();
        let ver = Self::next_version_tx(&mut tx).await?;
        let attempt = task.current_attempt + 1;
        let mut timings: HashMap<String, f64> = src
            .get::<Option<String>, _>("timings_json")
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        timings.insert("shared".to_string(), 1.0);

        sqlx::query(
            "INSERT INTO task_attempts (
                task_id, job_id, stage, attempt, agent, executor, status, device, backend_kind,
                model_id, model_rev, model_dtype, torch_version, worker_version, timings_json,
                peak_mem_bytes, started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'done', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(task_id)
        .bind(&task.job_id)
        .bind(&task.stage)
        .bind(attempt as i64)
        .bind(src.get::<String, _>("agent"))
        .bind(src.get::<String, _>("executor"))
        .bind(src.get::<Option<String>, _>("device"))
        .bind(src.get::<Option<String>, _>("backend_kind"))
        .bind(src.get::<Option<String>, _>("model_id"))
        .bind(src.get::<Option<String>, _>("model_rev"))
        .bind(src.get::<Option<String>, _>("model_dtype"))
        .bind(src.get::<Option<String>, _>("torch_version"))
        .bind(src.get::<Option<String>, _>("worker_version"))
        .bind(serde_json::to_string(&timings)?)
        .bind(src.get::<Option<i64>, _>("peak_mem_bytes").unwrap_or(0))
        .bind(now)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "INSERT INTO artifacts (
                id, job_id, task_id, attempt, name, sha256, bytes, kind, metadata_json, created_at,
                cache_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'intermediate', ?, ?, ?)",
        )
        .bind(format!(
            "art_{}",
            ulid::Ulid::new().to_string().to_lowercase()
        ))
        .bind(&task.job_id)
        .bind(task_id)
        .bind(attempt as i64)
        .bind(&source.name)
        .bind(&source.sha256)
        .bind(source.bytes as i64)
        .bind(&source.metadata_json)
        .bind(now)
        .bind(cache_key)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "UPDATE tasks
             SET status = 'done', attempts = ?, current_attempt = ?, blocked_reason = NULL,
                 sub_stage = NULL, started_at = ?, finished_at = ?
             WHERE id = ?",
        )
        .bind((task.attempts + 1) as i64)
        .bind(attempt as i64)
        .bind(now)
        .bind(now)
        .bind(task_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "UPDATE jobs SET status = 'running', model_id = ?, model_rev = ?,
                 started_at = COALESCE(started_at, ?) WHERE id = ?",
        )
        .bind(src.get::<Option<String>, _>("model_id"))
        .bind(src.get::<Option<String>, _>("model_rev"))
        .bind(now)
        .bind(&task.job_id)
        .execute(&mut *tx)
        .await?;
        Self::advance_job_tx(&mut tx, &task, now, ver).await?;
        tx.commit().await?;
        Ok(true)
    }

    /// Handles a failed attempt (`Failed` message).
    /// Retries if `attempts < 2` (and not cancelled); fails the task and job if `attempts >= 2`.
    /// Disables the executor persistently if `failure_class` is `wedged`, `gpu_fault`, or `device_mismatch`.
    pub async fn fail_task_attempt(
        &self,
        task_id: &str,
        attempt: u32,
        failure: AttemptFailure<'_>,
    ) -> Result<Option<String>> {
        let AttemptFailure {
            class: failure_class,
            detail,
            exit_code,
            signal,
            kernel_log_tail,
        } = failure;
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query("SELECT * FROM tasks WHERE id = ?")
            .bind(task_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(tr) = row else {
            tx.commit().await?;
            return Ok(None);
        };
        let task = Self::row_to_task(&tr);
        if task.current_attempt != attempt || !task.status.is_active() {
            tx.commit().await?;
            return Ok(None);
        }

        let now = now_s();
        let ver = Self::next_version_tx(&mut tx).await?;

        sqlx::query(
            "UPDATE task_attempts
             SET status = 'failed', failure_class = ?, failure_detail = ?,
                 exit_code = ?, signal = ?, kernel_log_tail = ?, finished_at = ?
             WHERE task_id = ? AND attempt = ?",
        )
        .bind(failure_class)
        .bind(detail)
        .bind(exit_code as i64)
        .bind(signal as i64)
        .bind(kernel_log_tail)
        .bind(now)
        .bind(task_id)
        .bind(attempt as i64)
        .execute(&mut *tx)
        .await?;

        if let (Some(agent_id), Some(executor_id)) = (&task.assigned_agent, &task.assigned_executor)
        {
            let key = format!("{agent_id}:{executor_id}");
            if matches!(failure_class, "wedged" | "gpu_fault" | "device_mismatch") {
                let reason = format!("{failure_class}: {detail}");
                sqlx::query(
                    "UPDATE executors
                     SET state = 'disabled', disabled_reason = ?, current_task_id = NULL,
                         current_stage = NULL, worker_pid = NULL, last_seen_at = ?
                     WHERE key = ?",
                )
                .bind(&reason)
                .bind(now)
                .bind(&key)
                .execute(&mut *tx)
                .await?;
            } else {
                sqlx::query(
                    "UPDATE executors
                     SET current_task_id = NULL, last_seen_at = ?
                     WHERE key = ?",
                )
                .bind(now)
                .bind(&key)
                .execute(&mut *tx)
                .await?;
            }
        }

        if failure_class == "cancelled" {
            // A user cancel has already ended the task (and never gets here), so this is a
            // cancel from the agent's side: end the whole job, as `cancel_job` does, rather
            // than leave it running with nothing left to run.
            sqlx::query(
                "UPDATE tasks
                 SET status = 'cancelled', assigned_agent = NULL, assigned_executor = NULL,
                     blocked_reason = NULL, finished_at = COALESCE(finished_at, ?)
                 WHERE job_id = ? AND status NOT IN ('done', 'failed', 'cancelled')",
            )
            .bind(now)
            .bind(&task.job_id)
            .execute(&mut *tx)
            .await?;

            sqlx::query(
                "UPDATE jobs
                 SET status = 'cancelled', blocked_reason = NULL, finished_at = ?, version = ?
                 WHERE id = ? AND status NOT IN ('done', 'failed', 'cancelled')",
            )
            .bind(now)
            .bind(ver)
            .bind(&task.job_id)
            .execute(&mut *tx)
            .await?;
        } else if task.attempts >= 2 {
            sqlx::query(
                "UPDATE tasks
                 SET status = 'failed', assigned_agent = NULL, assigned_executor = NULL,
                     error_class = ?, error_detail = ?, finished_at = ?
                 WHERE id = ?",
            )
            .bind(failure_class)
            .bind(detail)
            .bind(now)
            .bind(task_id)
            .execute(&mut *tx)
            .await?;

            sqlx::query(
                "UPDATE jobs
                 SET status = 'failed', error_class = ?, error_detail = ?,
                     blocked_reason = NULL, finished_at = ?, version = ?
                 WHERE id = ?",
            )
            .bind(failure_class)
            .bind(detail)
            .bind(now)
            .bind(ver)
            .bind(&task.job_id)
            .execute(&mut *tx)
            .await?;
        } else {
            // Re-queue for retry (attempt 2)
            sqlx::query(
                "UPDATE tasks
                 SET status = 'queued', assigned_agent = NULL, assigned_executor = NULL,
                     step = 0, s_per_step = 0.0, queued_at = ?
                 WHERE id = ?",
            )
            .bind(now)
            .bind(task_id)
            .execute(&mut *tx)
            .await?;

            sqlx::query("UPDATE jobs SET version = ? WHERE id = ?")
                .bind(ver)
                .bind(&task.job_id)
                .execute(&mut *tx)
                .await?;
        }

        tx.commit().await?;
        Ok(Some(task.job_id))
    }

    /// Handles `Rejected` from an agent (§6.4): re-queues the task without burning the user's retry budget
    /// (decrements `attempts` back) and marks the executor `disabled("protocol_violation: ...")`.
    pub async fn handle_rejected_assignment(
        &self,
        task_id: &str,
        attempt: u32,
        reason: &str,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query("SELECT * FROM tasks WHERE id = ?")
            .bind(task_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(tr) = row else {
            tx.commit().await?;
            return Ok(());
        };
        let task = Self::row_to_task(&tr);
        // A rejection for an attempt that is no longer current (or a task that already
        // moved on) is stale: requeueing it would rewind a finished task.
        if task.current_attempt != attempt || !task.status.is_active() {
            tx.commit().await?;
            return Ok(());
        }
        let now = now_s();
        let ver = Self::next_version_tx(&mut tx).await?;

        sqlx::query(
            "UPDATE task_attempts
             SET status = 'rejected', failure_detail = ?, finished_at = ?
             WHERE task_id = ? AND attempt = ?",
        )
        .bind(reason)
        .bind(now)
        .bind(task_id)
        .bind(attempt as i64)
        .execute(&mut *tx)
        .await?;

        if let (Some(agent_id), Some(executor_id)) = (&task.assigned_agent, &task.assigned_executor)
        {
            let key = format!("{agent_id}:{executor_id}");
            let dis_reason = format!("protocol_violation: {reason}");
            sqlx::query(
                "UPDATE executors
                 SET state = 'disabled', disabled_reason = ?, current_task_id = NULL, last_seen_at = ?
                 WHERE key = ?",
            )
            .bind(&dis_reason)
            .bind(now)
            .bind(&key)
            .execute(&mut *tx)
            .await?;
        }

        let new_attempts = task.attempts.saturating_sub(1);
        sqlx::query(
            "UPDATE tasks
             SET status = 'queued', attempts = ?, assigned_agent = NULL, assigned_executor = NULL,
                 queued_at = ?
             WHERE id = ?",
        )
        .bind(new_attempts as i64)
        .bind(now)
        .bind(task_id)
        .execute(&mut *tx)
        .await?;

        sqlx::query("UPDATE jobs SET version = ? WHERE id = ?")
            .bind(ver)
            .bind(&task.job_id)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;
        Ok(())
    }

    /// Cancels a job and any active/queued tasks. Returns `(agent_id, task_id, attempt)` if an in-flight task needs `Cancel`.
    pub async fn cancel_job(
        &self,
        job_id: &str,
    ) -> Result<Option<(Option<(String, String, u32)>, JobRecord)>> {
        let mut tx = self.pool.begin().await?;
        let j_row = sqlx::query("SELECT * FROM jobs WHERE id = ?")
            .bind(job_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(jr) = j_row else {
            tx.commit().await?;
            return Ok(None);
        };
        let job = Self::row_to_job(&jr);
        if matches!(
            job.status,
            JobStatus::Done | JobStatus::Failed | JobStatus::Cancelled
        ) {
            tx.commit().await?;
            return Ok(self.get_job(job_id).await?.map(|full| (None, full)));
        }

        let now = now_s();
        let ver = Self::next_version_tx(&mut tx).await?;

        let active_row = sqlx::query(
            "SELECT id, assigned_agent, assigned_executor, current_attempt FROM tasks
             WHERE job_id = ? AND status IN ('assigned', 'running', 'uploading')",
        )
        .bind(job_id)
        .fetch_optional(&mut *tx)
        .await?;

        let mut cancel_signal = None;
        if let Some(ar) = active_row {
            let t_id: String = ar.get("id");
            let ag: Option<String> = ar.get("assigned_agent");
            let ex: Option<String> = ar.get("assigned_executor");
            let att = ar.get::<i64, _>("current_attempt") as u32;
            if let Some(agent_id) = ag {
                if let Some(executor_id) = ex {
                    let key = format!("{agent_id}:{executor_id}");
                    sqlx::query("UPDATE executors SET current_task_id = NULL WHERE key = ?")
                        .bind(&key)
                        .execute(&mut *tx)
                        .await?;
                }
                cancel_signal = Some((agent_id, t_id, att));
            }
        }

        sqlx::query(
            "UPDATE tasks
             SET status = CASE WHEN status = 'done' THEN 'done' ELSE 'cancelled' END,
                 assigned_agent = NULL, assigned_executor = NULL, blocked_reason = NULL,
                 finished_at = COALESCE(finished_at, ?)
             WHERE job_id = ?",
        )
        .bind(now)
        .bind(job_id)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "UPDATE jobs
             SET status = 'cancelled', blocked_reason = NULL, finished_at = ?, version = ?
             WHERE id = ?",
        )
        .bind(now)
        .bind(ver)
        .bind(job_id)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        // Deleted since the commit: still return the signal, the agent has to stop the task.
        let full = match self.get_job(job_id).await? {
            Some(full) => full,
            None => JobRecord {
                status: JobStatus::Cancelled,
                finished_at: Some(now),
                ..job
            },
        };
        Ok(Some((cancel_signal, full)))
    }

    /// Changes who rewrites an Enhance job's prompt (`None`: the built-in enhancer), for a
    /// retry with another enhancer. Only a failed or cancelled job changes.
    pub async fn set_job_enhancer(&self, job_id: &str, enhancer: Option<&str>) -> Result<bool> {
        let done = sqlx::query(
            "UPDATE jobs SET enhancer = ? WHERE id = ? AND enhance = 1 AND status IN ('failed', 'cancelled')",
        )
        .bind(enhancer)
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected() == 1)
    }

    /// Requeues the in-flight tasks of cloud enhancers (docs/CLOUD_ENHANCERS.md): they ran in
    /// a Studio process that has ended. Called at startup.
    pub async fn reclaim_cloud_tasks(&self) -> Result<Vec<String>> {
        self.reclaim_agent_tasks(crate::cloud::CLOUD_AGENT, &[], None)
            .await
    }

    /// Retries a failed or cancelled job from its first incomplete task (preserving completed stage artifacts!).
    pub async fn retry_job(&self, job_id: &str) -> Result<Option<JobRecord>> {
        let mut tx = self.pool.begin().await?;
        let j_row = sqlx::query("SELECT * FROM jobs WHERE id = ?")
            .bind(job_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(jr) = j_row else {
            tx.commit().await?;
            return Ok(None);
        };
        let job = Self::row_to_job(&jr);
        if !matches!(job.status, JobStatus::Failed | JobStatus::Cancelled) {
            tx.commit().await?;
            return self.get_job(job_id).await;
        }

        let now = now_s();
        let ver = Self::next_version_tx(&mut tx).await?;
        let task_rows = sqlx::query("SELECT * FROM tasks WHERE job_id = ? ORDER BY seq ASC")
            .bind(job_id)
            .fetch_all(&mut *tx)
            .await?;

        let tasks: Vec<TaskRecord> = task_rows.iter().map(Self::row_to_task).collect();
        // The first task to run: the first unfinished one, moved back past finished stages
        // whose output it needs and the intermediate cleanup has since deleted.
        let mut start = tasks
            .iter()
            .position(|t| t.status != TaskStatus::Done)
            .unwrap_or(tasks.len());
        while start > 0 && start < tasks.len() {
            let Some(input) = stage_input_artifact(&tasks[start].stage) else {
                break;
            };
            if Self::job_artifact_present_tx(&mut tx, job_id, input).await? {
                break;
            }
            start -= 1;
        }

        for (i, t) in tasks.iter().enumerate().skip(start) {
            let new_status = if i == start {
                TaskStatus::Queued
            } else {
                TaskStatus::Waiting
            };
            let queued_at = if new_status == TaskStatus::Queued {
                Some(now)
            } else {
                None
            };
            sqlx::query(
                "UPDATE tasks
                 SET status = ?, attempts = 0, assigned_agent = NULL, assigned_executor = NULL,
                     blocked_reason = NULL, error_class = NULL, error_detail = NULL,
                     step = 0, s_per_step = 0.0, queued_at = ?, started_at = NULL, finished_at = NULL
                 WHERE id = ?",
            )
            .bind(new_status)
            .bind(queued_at)
            .bind(&t.id)
            .execute(&mut *tx)
            .await?;
        }

        sqlx::query(
            "UPDATE jobs
             SET status = 'submitted', error_class = NULL, error_detail = NULL,
                 blocked_reason = NULL, finished_at = NULL, version = ?
             WHERE id = ?",
        )
        .bind(ver)
        .bind(job_id)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        self.get_job(job_id).await
    }

    /// Whether the job still has `name` from a finished task's current attempt (what the
    /// dispatcher would send the next stage).
    async fn job_artifact_present_tx(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        job_id: &str,
        name: &str,
    ) -> Result<bool> {
        let row = sqlx::query(
            "SELECT 1 FROM artifacts a
             JOIN tasks t ON a.task_id = t.id AND a.attempt = t.current_attempt
             WHERE a.job_id = ? AND a.name = ? AND t.status = 'done'
             LIMIT 1",
        )
        .bind(job_id)
        .bind(name)
        .fetch_optional(&mut **tx)
        .await?;
        Ok(row.is_some())
    }

    /// Deletes a job and returns the SHA-256 hashes of blobs that are no longer referenced by any remaining artifact.
    pub async fn delete_job(&self, job_id: &str) -> Result<Option<Vec<String>>> {
        let mut tx = self.pool.begin().await?;
        let exists = sqlx::query("SELECT id FROM jobs WHERE id = ?")
            .bind(job_id)
            .fetch_optional(&mut *tx)
            .await?;
        if exists.is_none() {
            tx.commit().await?;
            return Ok(None);
        }

        let sha_rows = sqlx::query("SELECT DISTINCT sha256 FROM artifacts WHERE job_id = ?")
            .bind(job_id)
            .fetch_all(&mut *tx)
            .await?;
        let candidate_shas: Vec<String> = sha_rows.iter().map(|r| r.get("sha256")).collect();

        let ver = Self::next_version_tx(&mut tx).await?;
        // Tombstone, so clients polling `changed_since` learn about the deletion.
        sqlx::query("INSERT OR REPLACE INTO deleted_jobs (id, version) VALUES (?, ?)")
            .bind(job_id)
            .bind(ver)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM jobs WHERE id = ?")
            .bind(job_id)
            .execute(&mut *tx)
            .await?;

        let mut orphan_shas = Vec::new();
        for sha in candidate_shas {
            if !Self::blob_in_use_tx(&mut tx, &sha).await? {
                orphan_shas.push(sha);
            }
        }

        tx.commit().await?;
        Ok(Some(orphan_shas))
    }

    /// Operator control: lend an executor (`POST /api/executors/{id}/lend`).
    /// Returns `(agent_id, executor_id, should_send_release)`.
    pub async fn lend_executor(&self, id_or_key: &str) -> Result<Option<(String, String, bool)>> {
        let Some(ex) = self.find_executor(id_or_key).await? else {
            return Ok(None);
        };
        let now = now_s();
        if ex.state == ExecutorState::Busy {
            // Set pending_lend; scheduler will not assign further work, and when task finishes it releases
            sqlx::query("UPDATE executors SET pending_lend = 1, last_seen_at = ? WHERE key = ?")
                .bind(now)
                .bind(&ex.key)
                .execute(&self.pool)
                .await?;
            Ok(Some((ex.agent_id, ex.id, false)))
        } else {
            let should_release = matches!(ex.state, ExecutorState::Ready | ExecutorState::Starting);
            sqlx::query(
                "UPDATE executors
                 SET state = 'lent', lent = 1, pending_lend = 0, current_stage = NULL,
                     current_model_id = NULL, current_model_rev = NULL, current_model_dtype = NULL,
                     worker_pid = NULL, last_seen_at = ?
                 WHERE key = ?",
            )
            .bind(now)
            .bind(&ex.key)
            .execute(&self.pool)
            .await?;
            Ok(Some((ex.agent_id, ex.id, should_release)))
        }
    }

    pub async fn return_executor(&self, id_or_key: &str) -> Result<Option<ExecutorRecord>> {
        let Some(ex) = self.find_executor(id_or_key).await? else {
            return Ok(None);
        };
        let now = now_s();
        sqlx::query(
            "UPDATE executors
             SET state = CASE WHEN state = 'lent' THEN 'idle' ELSE state END,
                 lent = 0, pending_lend = 0, idle_since = ?, last_seen_at = ?
             WHERE key = ?",
        )
        .bind(now)
        .bind(now)
        .bind(&ex.key)
        .execute(&self.pool)
        .await?;
        self.find_executor(&ex.key).await
    }

    pub async fn disable_executor(
        &self,
        id_or_key: &str,
        reason: &str,
    ) -> Result<Option<(String, String, bool)>> {
        let Some(ex) = self.find_executor(id_or_key).await? else {
            return Ok(None);
        };
        let now = now_s();
        let should_release = matches!(
            ex.state,
            ExecutorState::Ready | ExecutorState::Starting | ExecutorState::Busy
        );
        sqlx::query(
            "UPDATE executors
             SET state = 'disabled', disabled_reason = ?, current_stage = NULL,
                 current_model_id = NULL, current_model_rev = NULL, current_model_dtype = NULL,
                 worker_pid = NULL, last_seen_at = ?
             WHERE key = ?",
        )
        .bind(reason)
        .bind(now)
        .bind(&ex.key)
        .execute(&self.pool)
        .await?;
        Ok(Some((ex.agent_id, ex.id, should_release)))
    }

    pub async fn enable_executor(&self, id_or_key: &str) -> Result<Option<ExecutorRecord>> {
        let Some(ex) = self.find_executor(id_or_key).await? else {
            return Ok(None);
        };
        let now = now_s();
        sqlx::query(
            "UPDATE executors
             SET state = 'idle', disabled_reason = NULL, idle_since = ?, last_seen_at = ?
             WHERE key = ?",
        )
        .bind(now)
        .bind(now)
        .bind(&ex.key)
        .execute(&self.pool)
        .await?;
        self.find_executor(&ex.key).await
    }

    /// Garbage-collects intermediate artifacts for completed jobs, or failed/cancelled jobs older than `failed_retention_s`,
    /// or all non-active intermediate artifacts when `emergency_disk_pressure` is true.
    /// `only_job` limits it to one job (its own intermediates once it is done).
    /// Returns SHA-256 hashes of blobs that are no longer referenced in `artifacts`.
    pub async fn gc_intermediate_artifacts(
        &self,
        failed_retention_s: f64,
        emergency_disk_pressure: bool,
        only_job: Option<&str>,
    ) -> Result<Vec<String>> {
        let mut tx = self.pool.begin().await?;
        let cutoff = now_s() - failed_retention_s;
        let rows = if emergency_disk_pressure {
            sqlx::query(
                "SELECT a.id, a.sha256 FROM artifacts a
                 JOIN jobs j ON a.job_id = j.id
                 WHERE a.kind = 'intermediate' AND j.status IN ('done', 'failed', 'cancelled')
                   AND (?1 IS NULL OR j.id = ?1)",
            )
            .bind(only_job)
            .fetch_all(&mut *tx)
            .await?
        } else {
            sqlx::query(
                "SELECT a.id, a.sha256 FROM artifacts a
                 JOIN jobs j ON a.job_id = j.id
                 WHERE a.kind = 'intermediate'
                   AND (j.status = 'done' OR (j.status IN ('failed', 'cancelled') AND j.finished_at < ?1))
                   AND (?2 IS NULL OR j.id = ?2)",
            )
            .bind(cutoff)
            .bind(only_job)
            .fetch_all(&mut *tx)
            .await?
        };

        if rows.is_empty() {
            tx.commit().await?;
            return Ok(Vec::new());
        }

        let mut candidate_shas = Vec::new();
        for r in rows {
            let id: String = r.get("id");
            let sha: String = r.get("sha256");
            sqlx::query("DELETE FROM artifacts WHERE id = ?")
                .bind(&id)
                .execute(&mut *tx)
                .await?;
            candidate_shas.push(sha);
        }

        let mut orphan_shas = Vec::new();
        for sha in candidate_shas {
            if !Self::blob_in_use_tx(&mut tx, &sha).await? && !orphan_shas.contains(&sha) {
                orphan_shas.push(sha);
            }
        }

        tx.commit().await?;
        Ok(orphan_shas)
    }

    pub async fn artifact_bytes_by_kind(&self) -> Result<(u64, u64)> {
        let rows = sqlx::query(
            "SELECT kind, COALESCE(SUM(bytes), 0) AS total_bytes FROM artifacts GROUP BY kind",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut inter = 0u64;
        let mut fin = 0u64;
        for r in rows {
            let kind: String = r.get("kind");
            let bytes = r.get::<i64, _>("total_bytes").max(0) as u64;
            if kind == "intermediate" {
                inter = bytes;
            } else if kind == "final" {
                fin = bytes;
            }
        }
        Ok((inter, fin))
    }
}
