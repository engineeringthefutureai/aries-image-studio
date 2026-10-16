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
}

fn default_steps() -> u32 {
    30
}

// 1.0 = no classifier-free guidance: one transformer pass per step (HARDWARE.md §1).
fn default_cfg() -> f32 {
    1.0
}

/// A client-supplied job id must be short and URL/path safe.
pub fn is_valid_client_job_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub status: String,
    pub prompt: String,
    pub enhanced_prompt: Option<String>,
    pub thinking: Option<String>,
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub seed: u64,
    pub true_cfg_scale: f32,
    pub enhance: bool,
    pub wh_ratio: Option<String>,
    pub model_id: Option<String>,
    pub model_rev: Option<String>,
    pub error_class: Option<String>,
    pub error_detail: Option<String>,
    pub blocked_reason: Option<String>,
    /// UI-owned metadata (turn id, aspect label, size and quality preset), opaque here.
    pub meta: serde_json::Value,
    pub favorite: bool,
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
    pub status: String,
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
    pub state: String,
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
        self.ensure_column("jobs", "meta", "TEXT NOT NULL DEFAULT '{}'").await?;
        self.ensure_column("jobs", "favorite", "INTEGER NOT NULL DEFAULT 0").await?;
        self.ensure_column("tasks", "sub_stage", "TEXT").await?;
        self.ensure_column("tasks", "stats_json", "TEXT").await?;
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

    async fn next_version_tx(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    ) -> Result<i64> {
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
                let prev_state: String = row.get("state");
                let prev_reason: Option<String> = row.get("disabled_reason");
                let prev_lent: i64 = row.get("lent");
                let prev_pending_lend: i64 = row.get("pending_lend");
                if prev_state == "disabled" || prev_reason.is_some() {
                    ("disabled".to_string(), prev_reason, prev_lent != 0, false)
                } else if prev_lent != 0 || prev_state == "lent" {
                    ("lent".to_string(), None, true, false)
                } else {
                    (ex.state.clone(), None, false, prev_pending_lend != 0)
                }
            } else {
                (ex.state.clone(), ex.disabled_reason.clone(), ex.lent, ex.pending_lend)
            };

            let idle_since = if init_state == "idle" || init_state == "ready" {
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
            .bind(&init_state)
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

    /// Reclaims in-flight tasks for an agent that stayed offline past `reconnect_grace` (§7.3).
    pub async fn expire_disconnected_agent_tasks(&self, agent_id: &str) -> Result<Vec<String>> {
        self.reclaim_agent_tasks(agent_id, &[], true).await
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
        let affected = self.reclaim_agent_tasks(agent_id, &keep, false).await?;
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
        require_offline: bool,
    ) -> Result<Vec<String>> {
        let mut tx = self.pool.begin().await?;
        if require_offline {
            let row = sqlx::query("SELECT online FROM agents WHERE id = ?")
                .bind(agent_id)
                .fetch_optional(&mut *tx)
                .await?;
            if let Some(r) = row {
                if r.get::<i64, _>("online") != 0 {
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
            if keep.iter().any(|(t, a)| *t == task_id && *a as i64 == attempt) {
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

        if require_offline {
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
    pub async fn create_job(&self, input: &JobSubmitInput, seed: u64) -> Result<JobRecord> {
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
                true_cfg_scale, enhance, meta, version, created_at
            ) VALUES (?, 'submitted', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        .bind(ver)
        .bind(now)
        .execute(&mut *tx)
        .await?;

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
        Ok(Some(job))
    }

    /// Jobs (all, or changed since a version) with their tasks and provenance, in three
    /// queries rather than two per job.
    pub async fn list_jobs(&self, changed_since: Option<i64>) -> Result<Vec<JobRecord>> {
        let since = changed_since.unwrap_or(i64::MIN);
        let rows = sqlx::query("SELECT * FROM jobs WHERE version > ? ORDER BY created_at DESC")
            .bind(since)
            .fetch_all(&self.pool)
            .await?;
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let task_rows = sqlx::query(
            "SELECT t.* FROM tasks t JOIN jobs j ON t.job_id = j.id
             WHERE j.version > ? ORDER BY t.job_id, t.seq ASC",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;
        let attempt_rows = sqlx::query(
            "SELECT a.* FROM task_attempts a JOIN jobs j ON a.job_id = j.id
             WHERE j.version > ? AND a.status = 'done' ORDER BY a.attempt ASC",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        let mut tasks: HashMap<String, Vec<TaskRecord>> = HashMap::new();
        for r in &task_rows {
            let t = Self::row_to_task(r);
            tasks.entry(t.job_id.clone()).or_default().push(t);
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
                job
            })
            .collect())
    }

    /// Ids of jobs deleted after `version`.
    pub async fn list_deleted_since(&self, version: i64) -> Result<Vec<String>> {
        let rows = sqlx::query("SELECT id FROM deleted_jobs WHERE version > ? ORDER BY version")
            .bind(version)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.iter().map(|r| r.get::<String, _>("id")).collect())
    }

    pub async fn set_job_favorite(&self, job_id: &str, favorite: bool) -> Result<Option<JobRecord>> {
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
            .map(|r| (r.get::<String, _>("status"), r.get::<i64, _>("n").max(0) as u64))
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
                peak_mem_bytes: r
                    .get::<Option<i64>, _>("peak_mem_bytes")
                    .unwrap_or(0) as u64,
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
            negative_prompt: r.get("negative_prompt"),
            width: r.get::<i64, _>("width") as u32,
            height: r.get::<i64, _>("height") as u32,
            steps: r.get::<i64, _>("steps") as u32,
            seed: r.get::<i64, _>("seed") as u64,
            true_cfg_scale: r.get::<f64, _>("true_cfg_scale") as f32,
            enhance: r.get::<i64, _>("enhance") != 0,
            wh_ratio: r.get("wh_ratio"),
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
        new_state: &str,
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
        let prev_state: String = row.get("state");
        let current_task: Option<String> = row.get("current_task_id");
        // The agent reports READY right after Done. By then the backend may already have
        // assigned the next task (state busy, current_task_id set); that READY is stale and
        // must not reopen the executor for a second assignment.
        if new_state == "ready" && prev_state == "busy" && current_task.is_some() {
            tx.commit().await?;
            return Ok(());
        }
        let disabled_reason: Option<String> = row.get("disabled_reason");
        let lent: i64 = row.get("lent");
        let pending_lend: i64 = row.get("pending_lend");

        if prev_state == "disabled" || disabled_reason.is_some() {
            tx.commit().await?;
            return Ok(());
        }

        let effective_state = if new_state == "idle" && (lent != 0 || pending_lend != 0) {
            "lent"
        } else {
            new_state
        };
        let effective_lent = if effective_state == "lent" { 1i64 } else { lent };
        let effective_pending_lend = if effective_state == "lent" {
            0i64
        } else {
            pending_lend
        };
        let idle_since = if effective_state == "idle" || effective_state == "ready" {
            Some(now)
        } else {
            None
        };

        if effective_state == "idle" || effective_state == "lent" {
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
        if job.status == "cancelled" || job.status == "failed" {
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
        task.status = "assigned".to_string();
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
        let row = sqlx::query(
            "SELECT job_id, current_attempt, status FROM tasks WHERE id = ?",
        )
        .bind(task_id)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(r) = row else {
            tx.commit().await?;
            return Ok(false);
        };
        let current_attempt = r.get::<i64, _>("current_attempt") as u32;
        let status: String = r.get("status");
        if current_attempt != attempt
            || !matches!(status.as_str(), "assigned" | "running" | "uploading")
        {
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

        sqlx::query("UPDATE task_attempts SET status = 'running' WHERE task_id = ? AND attempt = ?")
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

    pub async fn insert_uploaded_artifact(
        &self,
        task_id: &str,
        attempt: u32,
        name: &str,
        sha256: &str,
        bytes: u64,
        kind: &str,
        metadata_json: Option<&str>,
    ) -> Result<ArtifactRecord> {
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query(
            "SELECT job_id, current_attempt, status FROM tasks WHERE id = ?",
        )
        .bind(task_id)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(r) = row else {
            bail!("unknown task {task_id}");
        };
        let current_attempt = r.get::<i64, _>("current_attempt") as u32;
        let status: String = r.get("status");
        if current_attempt != attempt
            || !matches!(status.as_str(), "assigned" | "running" | "uploading")
        {
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
                id, job_id, task_id, attempt, name, sha256, bytes, kind, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, attempt, name) DO UPDATE SET
                sha256 = excluded.sha256,
                bytes = excluded.bytes,
                metadata_json = excluded.metadata_json,
                created_at = excluded.created_at",
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
        .execute(&mut *tx)
        .await?;

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
        let row = sqlx::query(
            "SELECT * FROM artifacts WHERE task_id = ? AND attempt = ? AND name = ?",
        )
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
        enhanced_prompt: Option<(&str, &str, &str)>,
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
        if task.current_attempt != attempt
            || !matches!(task.status.as_str(), "assigned" | "running" | "uploading")
        {
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

        if let Some((rewritten, wh_ratio, thinking)) = enhanced_prompt {
            sqlx::query("UPDATE jobs SET enhanced_prompt = ?, wh_ratio = ?, thinking = ? WHERE id = ?")
                .bind(rewritten)
                .bind(wh_ratio)
                .bind((!thinking.is_empty()).then_some(thinking))
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

        // Advance next waiting task or mark job done
        let next_row = sqlx::query(
            "SELECT id FROM tasks WHERE job_id = ? AND seq = ? AND status = 'waiting'",
        )
        .bind(&task.job_id)
        .bind(task.seq + 1)
        .fetch_optional(&mut *tx)
        .await?;

        let job_finished = if let Some(nr) = next_row {
            let next_task_id: String = nr.get("id");
            sqlx::query("UPDATE tasks SET status = 'queued', queued_at = ? WHERE id = ?")
                .bind(now)
                .bind(&next_task_id)
                .execute(&mut *tx)
                .await?;
            sqlx::query("UPDATE jobs SET version = ?, blocked_reason = NULL WHERE id = ?")
                .bind(ver)
                .bind(&task.job_id)
                .execute(&mut *tx)
                .await?;
            false
        } else {
            sqlx::query(
                "UPDATE jobs SET status = 'done', blocked_reason = NULL, finished_at = ?, version = ? WHERE id = ?",
            )
            .bind(now)
            .bind(ver)
            .bind(&task.job_id)
            .execute(&mut *tx)
            .await?;
            true
        };

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

    /// Handles a failed attempt (`Failed` message).
    /// Retries if `attempts < 2` (and not cancelled); fails the task and job if `attempts >= 2`.
    /// Disables the executor persistently if `failure_class` is `wedged`, `gpu_fault`, or `device_mismatch`.
    pub async fn fail_task_attempt(
        &self,
        task_id: &str,
        attempt: u32,
        failure_class: &str,
        detail: &str,
        exit_code: i32,
        signal: i32,
        kernel_log_tail: &str,
    ) -> Result<Option<String>> {
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
        if task.current_attempt != attempt
            || !matches!(task.status.as_str(), "assigned" | "running" | "uploading")
        {
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
            if matches!(
                failure_class,
                "wedged" | "gpu_fault" | "device_mismatch"
            ) {
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
            sqlx::query(
                "UPDATE tasks
                 SET status = 'cancelled', assigned_agent = NULL, assigned_executor = NULL, finished_at = ?
                 WHERE id = ?",
            )
            .bind(now)
            .bind(task_id)
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
        if task.current_attempt != attempt
            || !matches!(task.status.as_str(), "assigned" | "running" | "uploading")
        {
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
    pub async fn cancel_job(&self, job_id: &str) -> Result<Option<(Option<(String, String, u32)>, JobRecord)>> {
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
        if matches!(job.status.as_str(), "done" | "failed" | "cancelled") {
            tx.commit().await?;
            let full = self.get_job(job_id).await?.unwrap();
            return Ok(Some((None, full)));
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
        let full = self.get_job(job_id).await?.unwrap();
        Ok(Some((cancel_signal, full)))
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
        if !matches!(job.status.as_str(), "failed" | "cancelled") {
            tx.commit().await?;
            return self.get_job(job_id).await;
        }

        let now = now_s();
        let ver = Self::next_version_tx(&mut tx).await?;
        let task_rows = sqlx::query("SELECT * FROM tasks WHERE job_id = ? ORDER BY seq ASC")
            .bind(job_id)
            .fetch_all(&mut *tx)
            .await?;

        let mut queued_one = false;
        for r in task_rows {
            let t = Self::row_to_task(&r);
            if t.status == "done" {
                continue;
            }
            let new_status = if !queued_one {
                queued_one = true;
                "queued"
            } else {
                "waiting"
            };
            let queued_at = if new_status == "queued" {
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

    /// Deletes a job and returns the SHA-256 hashes of blobs that are no longer referenced by any remaining artifact.
    pub async fn delete_job(&self, job_id: &str) -> Result< Option<Vec<String>> > {
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
            let still_used = sqlx::query("SELECT 1 FROM artifacts WHERE sha256 = ? LIMIT 1")
                .bind(&sha)
                .fetch_optional(&mut *tx)
                .await?;
            if still_used.is_none() {
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
        if ex.state == "busy" {
            // Set pending_lend; scheduler will not assign further work, and when task finishes it releases
            sqlx::query(
                "UPDATE executors SET pending_lend = 1, last_seen_at = ? WHERE key = ?",
            )
            .bind(now)
            .bind(&ex.key)
            .execute(&self.pool)
            .await?;
            Ok(Some((ex.agent_id, ex.id, false)))
        } else {
            let should_release = matches!(ex.state.as_str(), "ready" | "starting");
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
        let should_release = matches!(ex.state.as_str(), "ready" | "starting" | "busy");
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
    /// Returns SHA-256 hashes of blobs that are no longer referenced in `artifacts`.
    pub async fn gc_intermediate_artifacts(
        &self,
        failed_retention_s: f64,
        emergency_disk_pressure: bool,
    ) -> Result<Vec<String>> {
        let mut tx = self.pool.begin().await?;
        let cutoff = now_s() - failed_retention_s;
        let rows = if emergency_disk_pressure {
            sqlx::query(
                "SELECT a.id, a.sha256 FROM artifacts a
                 JOIN jobs j ON a.job_id = j.id
                 WHERE a.kind = 'intermediate' AND j.status IN ('done', 'failed', 'cancelled')",
            )
            .fetch_all(&mut *tx)
            .await?
        } else {
            sqlx::query(
                "SELECT a.id, a.sha256 FROM artifacts a
                 JOIN jobs j ON a.job_id = j.id
                 WHERE a.kind = 'intermediate'
                   AND (j.status = 'done' OR (j.status IN ('failed', 'cancelled') AND j.finished_at < ?))",
            )
            .bind(cutoff)
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
            let still_used = sqlx::query("SELECT 1 FROM artifacts WHERE sha256 = ? LIMIT 1")
                .bind(&sha)
                .fetch_optional(&mut *tx)
                .await?;
            if still_used.is_none() && !orphan_shas.contains(&sha) {
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
