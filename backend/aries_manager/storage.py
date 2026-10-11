"""Per-job files and garbage collection for the manager (ARCHITECTURE.md §6)."""
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import psutil

from aries_manager import config

logger = logging.getLogger("aries_manager.storage")


def get_job_dir(job_id: str) -> Path:
    return config.JOBS_DIR / job_id


def get_job_paths(job_id: str) -> Dict[str, str]:
    jdir = get_job_dir(job_id)
    return {
        "job_dir": str(jdir),
        "request_json": str(jdir / "request.json"),
        "enhanced_json": str(jdir / "enhanced.json"),
        "embeds_pt": str(jdir / "embeds.pt"),
        "latent_pt": str(jdir / "latent.pt"),
        "image_png": str(jdir / "image.png"),
    }


def init_job_dir(job_id: str, request_data: Dict[str, Any]) -> Dict[str, str]:
    """Create job directory and save request.json atomically."""
    jdir = get_job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    req_file = jdir / "request.json"
    tmp_file = req_file.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(request_data, indent=2))
    tmp_file.replace(req_file)

    return get_job_paths(job_id)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_disk_watermark() -> Tuple[bool, float, int]:
    """
    Check if disk usage exceeds the watermark ratio.
    Returns (is_above_watermark, current_usage_ratio, available_bytes).
    """
    target_path = config.DATA_DIR
    while not target_path.exists() and target_path.parent != target_path:
        target_path = target_path.parent

    usage = psutil.disk_usage(str(target_path))
    used_ratio = usage.used / usage.total
    is_above = used_ratio >= config.DISK_WATERMARK_RATIO
    return is_above, used_ratio, usage.free


def get_storage_bytes() -> int:
    """Total size of all per-job files."""
    if not config.JOBS_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in config.JOBS_DIR.rglob("*") if p.is_file())


def delete_job_files_for_acknowledged(job_id: str) -> None:
    """Once the client has the image, drop latent.pt, image.png and enhanced.json."""
    paths = get_job_paths(job_id)
    for key in ("latent_pt", "image_png", "enhanced_json"):
        p = Path(paths[key])
        if p.exists():
            try:
                p.unlink()
                logger.debug("Deleted %s for acknowledged job %s", key, job_id)
            except Exception as e:
                logger.warning("Failed to delete %s for job %s: %s", key, job_id, e)


def cleanup_job_directory(job_id: str) -> None:
    """Completely remove directory and all files for a job."""
    job_dir = get_job_dir(job_id)
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.debug("Removed storage directory for job %s", job_id)
        except Exception as e:
            logger.warning("Failed to remove storage directory for job %s: %s", job_id, e)


def run_garbage_collection(db_conn: Any) -> Dict[str, int]:
    """
    Periodic cleanup, run inside the caller's transaction:
    1. Stored jobs never fetched within TTL_DAYS lose their files and are collected.
    2. Acknowledged jobs lose their files and are collected.
    3. Collected rows older than AUDIT_RETENTION_DAYS are deleted.
    """
    now = time.time()
    ttl_seconds = config.TTL_DAYS * 86400
    retention_seconds = config.AUDIT_RETENTION_DAYS * 86400
    stats = {"expired": 0, "collected": 0, "purged": 0}

    # 1. Expire stored jobs never fetched within the TTL
    try:
        cur = db_conn.execute(
            "SELECT id FROM jobs WHERE status = 'stored' AND (completed_at < ? OR (completed_at IS NULL AND created_at < ?));",
            (now - ttl_seconds, now - ttl_seconds),
        )
        expired_jobs = [r["id"] for r in cur.fetchall()]
        for jid in expired_jobs:
            delete_job_files_for_acknowledged(jid)
            db_conn.execute(
                "UPDATE jobs SET status = 'collected', updated_at = ? WHERE id = ?;",
                (now, jid),
            )
            stats["expired"] += 1
    except Exception as e:
        logger.warning("GC error during expiration check: %s", e)

    # 2. Mark acknowledged jobs as collected once files are removed
    try:
        cur = db_conn.execute("SELECT id FROM jobs WHERE status = 'acknowledged';")
        ack_jobs = [r["id"] for r in cur.fetchall()]
        for jid in ack_jobs:
            delete_job_files_for_acknowledged(jid)
            db_conn.execute(
                "UPDATE jobs SET status = 'collected', updated_at = ? WHERE id = ?;",
                (now, jid),
            )
            stats["collected"] += 1
    except Exception as e:
        logger.warning("GC error during acknowledgment collection: %s", e)

    # 3. Purge collected rows older than retention days
    try:
        cur = db_conn.execute(
            "DELETE FROM jobs WHERE status = 'collected' AND updated_at < ?;",
            (now - retention_seconds,),
        )
        stats["purged"] = cur.rowcount
    except Exception as e:
        logger.warning("GC error during row purge: %s", e)

    return stats
