"""SQLite job store and state machine for the manager (ARCHITECTURE.md §3).

Every transition is one transaction and bumps a global `version` counter that is
stamped on the job, so clients can poll for changes. Live progress updates do not
bump the version.
"""
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from aries_manager.storage import get_job_paths, run_garbage_collection

logger = logging.getLogger("aries_manager.db")

STAGES = ("enhance", "encode", "denoise", "decode")

ACTIVE_STATUS = {
    "enhance": "enhancing",
    "encode": "encoding",
    "denoise": "denoising",
    "decode": "decoding",
}

NEXT_STAGE = {
    "enhance": "encode",
    "encode": "denoise",
    "denoise": "decode",
    "decode": "stored",
}

# A stage that fails this many times marks the job failed (no crash loops).
MAX_STAGE_ATTEMPTS = 2

TERMINAL_STATUSES = ("stored", "acknowledged", "collected", "cancelled", "failed")

SCHEMA = """
CREATE TABLE IF NOT EXISTS global_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    sub_stage TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    prompt TEXT NOT NULL,
    enhanced_prompt TEXT,
    thinking TEXT,
    aspect TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    enhance INTEGER NOT NULL DEFAULT 0,
    decode_mode TEXT NOT NULL DEFAULT 'untiled',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    error TEXT,
    crash_class TEXT,
    card INTEGER,
    worker_pid INTEGER,
    step INTEGER DEFAULT 0,
    s_per_step REAL,
    eta_s REAL,
    timings TEXT,
    peak_vram INTEGER,
    sha256 TEXT,
    enhance_stats TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_version ON jobs(version);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_stage ON jobs(stage);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);

CREATE TABLE IF NOT EXISTS stage_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    stage TEXT NOT NULL,
    card INTEGER NOT NULL,
    pid INTEGER NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    exit_code INTEGER,
    jobs_processed INTEGER DEFAULT 0,
    load_s REAL,
    exit_reason TEXT,
    queue_depths TEXT
);

CREATE TABLE IF NOT EXISTS cards (
    card INTEGER PRIMARY KEY,
    state TEXT NOT NULL,
    current_stage TEXT,
    current_worker_pid INTEGER,
    unavailable_reason TEXT,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Columns added after the first schema; added in place to databases created before them.
ADDED_COLUMNS = {
    "jobs": {
        "sub_stage": "TEXT",
        "enhanced_prompt": "TEXT",
        "thinking": "TEXT",
        "enhance_stats": "TEXT",
    },
    "stage_runs": {
        "jobs_processed": "INTEGER DEFAULT 0",
        "exit_reason": "TEXT",
        "queue_depths": "TEXT",
    },
}


def _load_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def _job_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Row -> API dict: decode JSON columns and backfill the enhanced prompt from disk."""
    job = dict(row)
    job["timings"] = _load_json(job.get("timings"))
    job["enhance_stats"] = _load_json(job.get("enhance_stats"))

    if job.get("enhance") and not job.get("enhanced_prompt"):
        enhanced = Path(get_job_paths(job["id"])["enhanced_json"])
        if enhanced.is_file():
            try:
                data = json.loads(enhanced.read_text())
                job["enhanced_prompt"] = data.get("rewritten_prompt")
                job["thinking"] = data.get("thinking")
            except (OSError, ValueError):
                pass

    if job.get("status") == "enhancing" and not job.get("sub_stage"):
        job["sub_stage"] = "prefill"
    return job


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """New connection in WAL mode with a row factory."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # PRAGMAs must run in autocommit mode, outside any transaction.
    conn.isolation_level = None
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.isolation_level = "DEFERRED"
    return conn


def init_db(db_path: Path, num_cards: int = 2) -> None:
    """Create tables and indexes, add missing columns, seed the version and card rows."""
    conn = get_db_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        with conn:
            for table, columns in ADDED_COLUMNS.items():
                existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table});")}
                for name, decl in columns.items():
                    if name not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl};")
            conn.execute("INSERT OR IGNORE INTO global_state (id, version) VALUES (1, 1);")
            now = time.time()
            for card_idx in range(num_cards):
                conn.execute(
                    "INSERT OR IGNORE INTO cards (card, state, updated_at) VALUES (?, 'idle', ?);",
                    (card_idx, now),
                )
        logger.info("Initialized manager database at %s", db_path)
    finally:
        conn.close()


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = get_db_connection(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as conn, conn:
            yield conn

    @staticmethod
    def _bump_version(conn: sqlite3.Connection) -> int:
        row = conn.execute("UPDATE global_state SET version = version + 1 WHERE id = 1 RETURNING version;").fetchone()
        return row["version"]

    @staticmethod
    def _status(conn: sqlite3.Connection, job_id: str) -> Optional[str]:
        row = conn.execute("SELECT status FROM jobs WHERE id = ?;", (job_id,)).fetchone()
        return row["status"] if row else None

    # --- Reads ---

    def get_current_version(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT version FROM global_state WHERE id = 1;").fetchone()
            return row["version"] if row else 0

    def get_jobs_changed_since(self, since_version: int) -> Tuple[int, List[Dict[str, Any]]]:
        """(current version, jobs whose version is greater than `since_version`)."""
        with self._connect() as conn:
            row = conn.execute("SELECT version FROM global_state WHERE id = 1;").fetchone()
            rows = conn.execute(
                "SELECT * FROM jobs WHERE version > ? ORDER BY version ASC;", (since_version,)
            ).fetchall()
            return (row["version"] if row else 0), [_job_dict(r) for r in rows]

    def get_current_jobs(self, include_collected: bool = False) -> List[Dict[str, Any]]:
        """Jobs in flight or stored awaiting pickup (or all jobs with `include_collected`)."""
        query = "SELECT * FROM jobs"
        if not include_collected:
            query += " WHERE status NOT IN ('collected', 'acknowledged')"
        with self._connect() as conn:
            return [_job_dict(r) for r in conn.execute(query + " ORDER BY created_at ASC;")]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?;", (job_id,)).fetchone()
            return _job_dict(row) if row else None

    def find_active_job_for_pid(self, pid: int) -> Optional[str]:
        """Id of the job a worker process was running when it died, if any."""
        placeholders = ",".join("?" * len(ACTIVE_STATUS))
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT id FROM jobs WHERE worker_pid = ? AND status IN ({placeholders});",
                (pid, *ACTIVE_STATUS.values()),
            ).fetchone()
            return row["id"] if row else None

    def get_queue_depths(self) -> Dict[str, int]:
        depths = {stage: 0 for stage in STAGES}
        with self._connect() as conn:
            for row in conn.execute(
                "SELECT stage, COUNT(*) AS cnt FROM jobs WHERE status LIKE '%\\_queued' ESCAPE '\\' GROUP BY stage;"
            ):
                depths[row["stage"]] = row["cnt"]
        return depths

    def count_jobs_by_status(self) -> Dict[str, int]:
        with self._connect() as conn:
            return {
                r["status"]: r["cnt"]
                for r in conn.execute("SELECT status, COUNT(*) AS cnt FROM jobs GROUP BY status;")
            }

    def get_oldest_waiting_time(self, stage: str) -> Optional[float]:
        """Age in seconds of the oldest queued job for a stage."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(created_at) AS oldest FROM jobs WHERE status = ?;", (f"{stage}_queued",)
            ).fetchone()
            return time.time() - row["oldest"] if row and row["oldest"] is not None else None

    def get_recent_stage_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            runs = [dict(r) for r in conn.execute(
                "SELECT * FROM stage_runs ORDER BY started_at DESC LIMIT ?;", (limit,)
            )]
        for run in runs:
            run["queue_depths"] = _load_json(run.get("queue_depths"))
        return runs

    # --- Transitions ---

    def submit_jobs(self, job_specs: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """Insert jobs; ids come from the client, so resubmitting the same id is a no-op."""
        now = time.time()
        with self._transaction() as conn:
            version = self._bump_version(conn)
            for spec in job_specs:
                stage = "enhance" if spec.get("enhance") else "encode"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs (
                        id, version, status, stage, attempts, prompt, aspect,
                        width, height, steps, seed, enhance, decode_mode,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        spec["id"], version, f"{stage}_queued", stage, spec["prompt"], spec.get("aspect", "1:1"),
                        spec["width"], spec["height"], spec.get("steps", 40), spec.get("seed", 42),
                        1 if spec.get("enhance") else 0, spec.get("decode_mode", "untiled"), now, now,
                    ),
                )
        return version, [spec["id"] for spec in job_specs]

    def claim_oldest_queued_job(self, stage: str, card: int, pid: int) -> Optional[Dict[str, Any]]:
        """Atomically move the oldest queued job of `stage` to its active status."""
        now = time.time()
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1;",
                (f"{stage}_queued",),
            ).fetchone()
            if not row:
                return None
            job_id = row["id"]
            conn.execute(
                """
                UPDATE jobs SET
                    version = ?, status = ?,
                    sub_stage = ?,
                    card = ?, worker_pid = ?,
                    started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ?;
                """,
                (
                    self._bump_version(conn), ACTIVE_STATUS[stage],
                    "prefill" if stage == "enhance" else None,
                    card, pid, now, now, job_id,
                ),
            )
            return _job_dict(conn.execute("SELECT * FROM jobs WHERE id = ?;", (job_id,)).fetchone())

    def record_progress(
        self,
        job_id: str,
        step: Optional[int] = None,
        steps: Optional[int] = None,
        s_per_step: Optional[float] = None,
        sub_stage: Optional[str] = None,
        enhance_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store live progress. Fields left as None keep their value; the version is not bumped."""
        eta_s = None
        if step is not None and steps is not None and s_per_step:
            eta_s = round(max(0, steps - step) * s_per_step, 1)
        stats_json = json.dumps(enhance_stats) if enhance_stats is not None else None
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    step = COALESCE(?, step),
                    s_per_step = COALESCE(?, s_per_step),
                    eta_s = COALESCE(?, eta_s),
                    sub_stage = COALESCE(?, sub_stage),
                    enhance_stats = COALESCE(?, enhance_stats),
                    updated_at = ?
                WHERE id = ?;
                """,
                (step, s_per_step, eta_s, sub_stage, stats_json, time.time(), job_id),
            )

    def advance_stage_success(
        self,
        job_id: str,
        stage: str,
        timings: Dict[str, Any],
        peak_vram: int,
        sha256: Optional[str] = None,
        enhanced_prompt: Optional[str] = None,
        thinking: Optional[str] = None,
    ) -> Optional[int]:
        """Move a job to the next stage's queue (or `stored`). Returns None if the job is
        no longer running this stage, e.g. it was cancelled while the worker finished."""
        if stage not in NEXT_STAGE:
            raise ValueError(f"Unknown stage: {stage}")
        next_stage = NEXT_STAGE[stage]
        next_status = "stored" if next_stage == "stored" else f"{next_stage}_queued"
        now = time.time()

        with self._transaction() as conn:
            row = conn.execute(
                "SELECT status, stage, timings, created_at FROM jobs WHERE id = ?;", (job_id,)
            ).fetchone()
            if not row or row["status"] in TERMINAL_STATUSES or row["stage"] != stage:
                return None
            merged = _load_json(row["timings"]) or {}
            merged.update(timings)
            if next_status == "stored":
                merged["total_s"] = round(now - row["created_at"], 3)  # submit -> stored

            version = self._bump_version(conn)
            conn.execute(
                """
                UPDATE jobs SET
                    version = ?, status = ?, stage = ?,
                    sub_stage = NULL, attempts = 0,
                    card = CASE WHEN ? = 'stored' THEN card ELSE NULL END,
                    worker_pid = NULL,
                    step = 0, s_per_step = NULL, eta_s = NULL,
                    timings = ?,
                    peak_vram = MAX(COALESCE(peak_vram, 0), ?),
                    sha256 = COALESCE(?, sha256),
                    enhanced_prompt = COALESCE(?, enhanced_prompt),
                    thinking = COALESCE(?, thinking),
                    completed_at = CASE WHEN ? = 'stored' THEN ? ELSE completed_at END,
                    updated_at = ?
                WHERE id = ?;
                """,
                (
                    version, next_status, next_stage,
                    next_status,
                    json.dumps(merged), peak_vram, sha256, enhanced_prompt, thinking,
                    next_status, now, now, job_id,
                ),
            )
            return version

    def record_stage_failure(
        self,
        job_id: str,
        stage: str,
        error_msg: str,
        crash_class: Optional[str] = None,
    ) -> Optional[int]:
        """Requeue a job whose stage failed, or mark it failed after MAX_STAGE_ATTEMPTS.
        Returns None if the job is no longer running this stage (e.g. cancelled)."""
        now = time.time()
        with self._transaction() as conn:
            row = conn.execute("SELECT status, stage, attempts FROM jobs WHERE id = ?;", (job_id,)).fetchone()
            if not row or row["status"] in TERMINAL_STATUSES or row["stage"] != stage:
                return None
            attempts = row["attempts"] + 1
            failed = attempts >= MAX_STAGE_ATTEMPTS

            version = self._bump_version(conn)
            conn.execute(
                """
                UPDATE jobs SET
                    version = ?, status = ?, attempts = ?,
                    card = CASE WHEN ? THEN card ELSE NULL END,
                    worker_pid = NULL,
                    step = 0, s_per_step = NULL, eta_s = NULL,
                    error = ?, crash_class = ?,
                    completed_at = CASE WHEN ? THEN ? ELSE completed_at END,
                    updated_at = ?
                WHERE id = ?;
                """,
                (
                    version, "failed" if failed else f"{stage}_queued", attempts,
                    failed, error_msg, crash_class, failed, now, now, job_id,
                ),
            )
            return version

    def cancel_job(self, job_id: str) -> Optional[int]:
        """Cancel a queued or running job. Returns None if it is unknown or already finished."""
        now = time.time()
        with self._transaction() as conn:
            status = self._status(conn, job_id)
            if status is None or status in TERMINAL_STATUSES:
                return None
            version = self._bump_version(conn)
            conn.execute(
                """
                UPDATE jobs SET version = ?, status = 'cancelled', worker_pid = NULL,
                    completed_at = ?, updated_at = ?
                WHERE id = ?;
                """,
                (version, now, now, job_id),
            )
            return version

    def acknowledge_job(self, job_id: str) -> Optional[int]:
        """stored -> acknowledged, once the client has the image."""
        with self._transaction() as conn:
            if self._status(conn, job_id) != "stored":
                return None
            version = self._bump_version(conn)
            conn.execute(
                "UPDATE jobs SET version = ?, status = 'acknowledged', updated_at = ? WHERE id = ?;",
                (version, time.time(), job_id),
            )
            return version

    def delete_failed_jobs(self, hard_delete: bool = True) -> List[str]:
        """Delete (or mark collected) every failed job. Returns their ids."""
        with self._transaction() as conn:
            job_ids = [r["id"] for r in conn.execute("SELECT id FROM jobs WHERE status = 'failed';")]
            if not job_ids:
                return []
            version = self._bump_version(conn)
            if hard_delete:
                conn.execute("DELETE FROM jobs WHERE status = 'failed';")
            else:
                conn.execute(
                    "UPDATE jobs SET version = ?, status = 'collected', updated_at = ? WHERE status = 'failed';",
                    (version, time.time()),
                )
            return job_ids

    def delete_job(self, job_id: str, hard_delete: bool = True) -> Optional[int]:
        """Delete (or mark collected) a finished job. Raises ValueError for active jobs."""
        with self._transaction() as conn:
            status = self._status(conn, job_id)
            if status is None:
                return None
            if status not in TERMINAL_STATUSES:
                raise ValueError(f"Cannot delete job in active state '{status}'. Cancel it first.")
            version = self._bump_version(conn)
            if hard_delete:
                conn.execute("DELETE FROM jobs WHERE id = ?;", (job_id,))
            else:
                conn.execute(
                    "UPDATE jobs SET version = ?, status = 'collected', updated_at = ? WHERE id = ?;",
                    (version, time.time(), job_id),
                )
            return version

    def startup_reconcile(self) -> Dict[str, Any]:
        """ARCHITECTURE.md §8: on manager start, jobs left in an active status go back to
        their queue with attempts unchanged (the manager failed, not the job)."""
        now = time.time()
        with self._transaction() as conn:
            version = self._bump_version(conn)
            reconciled = []
            for row in conn.execute("SELECT id, stage FROM jobs WHERE status IN (?, ?, ?, ?);", tuple(ACTIVE_STATUS.values())).fetchall():
                conn.execute(
                    """
                    UPDATE jobs SET version = ?, status = ?, worker_pid = NULL, card = NULL,
                        step = 0, s_per_step = NULL, eta_s = NULL, updated_at = ?
                    WHERE id = ?;
                    """,
                    (version, f"{row['stage']}_queued", now, row["id"]),
                )
                reconciled.append(row["id"])
            conn.execute(
                """
                UPDATE cards SET state = 'idle', current_stage = NULL, current_worker_pid = NULL, updated_at = ?
                WHERE state = 'running';
                """,
                (now,),
            )
        logger.info("Startup reconciliation: %d in-flight jobs requeued.", len(reconciled))
        return {"reconciled_jobs": reconciled}

    def run_gc(self) -> Dict[str, int]:
        with self._transaction() as conn:
            return run_garbage_collection(conn)

    def record_stage_run(
        self,
        stage: str,
        card: int,
        pid: int,
        started_at: float,
        ended_at: float,
        exit_code: int,
        jobs_processed: int = 0,
        load_s: Optional[float] = None,
        job_id: Optional[str] = None,
        exit_reason: Optional[str] = None,
        queue_depths: Optional[Dict[str, int]] = None,
    ) -> None:
        """Append one worker process run to the `stage_runs` audit table (ARCHITECTURE.md §5)."""
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO stage_runs (
                    job_id, stage, card, pid, started_at, ended_at,
                    exit_code, jobs_processed, load_s, exit_reason, queue_depths
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    job_id, stage, card, pid, started_at, ended_at, exit_code, jobs_processed,
                    load_s, exit_reason, json.dumps(queue_depths) if queue_depths is not None else None,
                ),
            )
