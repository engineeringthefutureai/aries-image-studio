"""HTTP API and background loop of the manager (ARCHITECTURE.md §7)."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from aries_manager import __version__
from aries_manager.config import (
    DB_PATH,
    DISK_WATERMARK_RATIO,
    GC_INTERVAL_SECONDS,
    NUM_CARDS,
    SOCKET_PATH,
)
from aries_manager.db import Database, init_db
from aries_manager.hardware import get_host_available_ram_bytes
from aries_manager.metrics import HOST_AVAILABLE_BYTES, JOBS_BY_STATE, QUEUE_DEPTH, STORAGE_BYTES
from aries_manager.scheduler import CardScheduler
from aries_manager.server import ManagerSocketServer
from aries_manager.storage import (
    check_disk_watermark,
    cleanup_job_directory,
    delete_job_files_for_acknowledged,
    get_job_paths,
    get_storage_bytes,
    init_job_dir,
)
from aries_manager.ulid import generate_ulid

logger = logging.getLogger("aries_manager.api")


class JobSubmission(BaseModel):
    id: Optional[str] = None  # client-supplied ULID makes resubmission idempotent
    prompt: str
    aspect: str = "1:1"
    width: int = 2048
    height: int = 2048
    steps: int = 40
    seed: int = 42
    enhance: bool = False
    decode_mode: str = "untiled"


db = Database(DB_PATH)
scheduler = CardScheduler(db, num_cards=NUM_CARDS)
socket_server = ManagerSocketServer(db, scheduler, socket_path=SOCKET_PATH)


def _update_metrics(include_storage: bool) -> None:
    for stage, depth in db.get_queue_depths().items():
        QUEUE_DEPTH.labels(stage=stage).set(depth)
    for state, count in db.count_jobs_by_status().items():
        JOBS_BY_STATE.labels(state=state).set(count)
    HOST_AVAILABLE_BYTES.set(get_host_available_ram_bytes())
    if include_storage:
        STORAGE_BYTES.set(get_storage_bytes())


async def _scheduler_loop() -> None:
    """Once a second: give idle cards work, refresh gauges, and periodically collect garbage."""
    last_gc = 0.0
    while True:
        try:
            for card_idx in range(scheduler.num_cards):
                if scheduler.card_states[card_idx] == "idle":
                    stage = scheduler.pick_next_stage(card_idx)
                    if stage:
                        await scheduler.spawn_stage_worker(stage, card_idx)

            run_gc = time.monotonic() - last_gc > GC_INTERVAL_SECONDS
            if run_gc:
                db.run_gc()
                last_gc = time.monotonic()
            _update_metrics(include_storage=run_gc)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error in scheduler loop: %s", e)

        await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(DB_PATH, num_cards=NUM_CARDS)
    db.startup_reconcile()
    await socket_server.start()
    loop_task = asyncio.create_task(_scheduler_loop())
    logger.info("Aries manager %s started.", __version__)

    yield

    loop_task.cancel()
    await socket_server.stop()
    logger.info("Aries manager shut down.")


app = FastAPI(title="Aries Image Studio Manager", version=__version__, lifespan=lifespan)


def _not_found(detail: str = "Job not found") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


# --- Jobs ---

@app.post("/jobs", status_code=status.HTTP_201_CREATED)
async def submit_jobs(payload: Union[JobSubmission, List[JobSubmission]]):
    """Submit one job or a list of jobs."""
    is_above, ratio, _ = check_disk_watermark()
    if is_above:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Disk watermark exceeded: {ratio * 100:.1f}% used (limit {DISK_WATERMARK_RATIO * 100:.0f}%)",
        )

    specs = []
    for item in payload if isinstance(payload, list) else [payload]:
        spec = item.model_dump()
        spec["id"] = item.id or generate_ulid()
        init_job_dir(spec["id"], spec)
        specs.append(spec)

    version, job_ids = db.submit_jobs(specs)
    return {"version": version, "job_ids": job_ids}


@app.get("/jobs")
async def get_jobs(all: bool = False):
    """Jobs in flight or stored awaiting pickup; `?all=true` includes collected ones."""
    return db.get_current_jobs(include_collected=all)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise _not_found()
    return job


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued or running job; a running worker stops at its next step boundary."""
    await socket_server.cancel_job(job_id)
    version = db.cancel_job(job_id)
    if version is None:
        raise _not_found("Job not found or already finished")
    return {"version": version, "status": "cancelled"}


@app.get("/jobs/{job_id}/image")
async def get_job_image(job_id: str):
    """The finished PNG, with its SHA-256 in `X-Content-SHA256`. Idempotent until acknowledged."""
    job = db.get_job(job_id)
    if not job:
        raise _not_found()
    img_path = Path(get_job_paths(job_id)["image_png"])
    if not img_path.exists():
        raise _not_found("Image not yet available")
    return FileResponse(str(img_path), media_type="image/png", headers={"X-Content-SHA256": job.get("sha256") or ""})


@app.delete("/jobs/{job_id}/image")
async def acknowledge_job_image(job_id: str):
    """Acknowledge receipt of the image; the manager then deletes its copy."""
    version = db.acknowledge_job(job_id)
    if version is None:
        raise _not_found("Job not found or not in stored state")
    delete_job_files_for_acknowledged(job_id)
    return {"version": version, "status": "acknowledged"}


@app.delete("/jobs/failed")
async def delete_failed_jobs(hard: bool = True):
    """Delete every failed job and its files."""
    job_ids = db.delete_failed_jobs(hard_delete=hard)
    for job_id in job_ids:
        cleanup_job_directory(job_id)
    return {"status": "deleted", "deleted_jobs": job_ids, "count": len(job_ids)}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, hard: bool = True):
    """Acknowledge a stored job, or delete a finished one. Active jobs must be cancelled first."""
    job = db.get_job(job_id)
    if not job:
        raise _not_found()

    if job["status"] == "stored":
        version = db.acknowledge_job(job_id)
        delete_job_files_for_acknowledged(job_id)
        return {"version": version, "status": "acknowledged", "job_id": job_id}

    try:
        version = db.delete_job(job_id, hard_delete=hard)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    cleanup_job_directory(job_id)
    return {"version": version, "status": "deleted", "job_id": job_id}


# --- Status and telemetry ---

@app.get("/status")
async def get_status():
    """Cards, VRAM, queue depth per stage and host RAM."""
    return scheduler.get_status()


@app.get("/health")
async def get_health():
    """Status plus an overall state: `error` if every card is unavailable, `busy` if any runs."""
    status_data = scheduler.get_status()
    cards = status_data["cards"]
    if cards and all(c["state"] == "unavailable" for c in cards):
        state = "error"
    elif any(c["state"] == "running" for c in cards):
        state = "busy"
    else:
        state = "ready"
    return {"status": "ok", "state": state, **status_data}


@app.get("/stage_runs")
async def get_stage_runs(limit: int = Query(default=50, ge=1, le=200)):
    """Recent stage worker runs: why each exited, jobs processed, queue depths at exit."""
    return db.get_recent_stage_runs(limit=limit)


@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
