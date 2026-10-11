"""Unix socket server for stage workers: the JSON-lines protocol of ARCHITECTURE.md §5."""
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from aries_manager.config import HEARTBEAT_TIMEOUT_SECONDS, SOCKET_PATH
from aries_manager.db import Database
from aries_manager.storage import get_job_paths, compute_sha256
from aries_manager.scheduler import CardScheduler
from aries_manager.metrics import (
    MODEL_LOAD_SECONDS,
    WORKER_EVICTIONS_TOTAL,
    EVICTION_QUEUE_DEPTH,
    EVICTION_JOBS_PROCESSED,
    ENHANCE_PREFILL_TPS,
    ENHANCE_GENERATION_TPS,
    ENHANCE_TOKENS_TOTAL,
)

logger = logging.getLogger("aries_manager.server")


class WorkerSession:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.stage: Optional[str] = None
        self.card: Optional[int] = None
        self.pid: Optional[int] = None
        self.last_heartbeat: float = time.time()
        self.current_job_id: Optional[str] = None
        self.jobs_processed: int = 0
        self.is_active = True

    async def send_action(self, data: Dict[str, Any]) -> None:
        line = json.dumps(data) + "\n"
        self.writer.write(line.encode("utf-8"))
        await self.writer.drain()


class ManagerSocketServer:
    def __init__(self, db: Database, scheduler: CardScheduler, socket_path: Path = SOCKET_PATH):
        self.db = db
        self.scheduler = scheduler
        self.socket_path = socket_path
        self.server: Optional[asyncio.Server] = None
        self.sessions: Dict[int, WorkerSession] = {}  # pid -> session
        self.running = False
        # Worker events; "heartbeat" needs no handler (any message refreshes the heartbeat).
        self._handlers = {
            "hello": self._on_hello,
            "claim": self._on_claim,
            "progress": self._on_progress,
            "done": self._on_done,
            "error": self._on_error,
        }

    async def start(self) -> None:
        """Start listening on Unix domain socket."""
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()

        self.server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
        )
        self.running = True
        logger.info("Manager socket server listening at %s", self.socket_path)
        asyncio.create_task(self._heartbeat_checker_loop())

    async def stop(self) -> None:
        """Close socket server and disconnect workers."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)
        logger.info("Manager socket server stopped.")

    async def cancel_job(self, job_id: str) -> bool:
        """Send cancel action to worker currently running job_id."""
        for session in self.sessions.values():
            if session.current_job_id == job_id:
                logger.info("Sending cancel action for job %s to worker PID %d", job_id, session.pid)
                await session.send_action({"action": "cancel", "id": job_id})
                return True
        return False

    async def _heartbeat_checker_loop(self) -> None:
        """Drop the connection of any worker that has gone silent."""
        while self.running:
            try:
                await asyncio.sleep(5.0)
                now = time.time()
                for pid, session in list(self.sessions.items()):
                    if now - session.last_heartbeat > HEARTBEAT_TIMEOUT_SECONDS:
                        logger.warning(
                            "Worker PID %d (stage %s) silent for >%.0fs; closing.",
                            pid, session.stage, HEARTBEAT_TIMEOUT_SECONDS,
                        )
                        session.is_active = False
                        try:
                            session.writer.close()
                            await session.writer.wait_closed()
                        except Exception:
                            pass
                        self.sessions.pop(pid, None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat monitor error: %s", e)

    async def _evict(self, session: WorkerSession, kind: str, reason: str) -> None:
        """Tell a worker to exit so its card can switch stage, recording why."""
        stage, card = session.stage, session.card
        logger.info(
            "Worker PID %d (stage '%s', card %s, processed %d jobs) told to exit: %s (%s)",
            session.pid, stage, card, session.jobs_processed, kind, reason,
        )
        depths = self.db.get_queue_depths()
        self.scheduler.set_card_eviction(card, f"{kind}: {reason}", depths)
        WORKER_EVICTIONS_TOTAL.labels(stage=stage, card=str(card), reason=kind).inc()
        EVICTION_JOBS_PROCESSED.labels(stage=stage, reason=kind).observe(session.jobs_processed)
        for target_stage, depth in depths.items():
            EVICTION_QUEUE_DEPTH.labels(evicted_stage=stage, target_stage=target_stage, reason=kind).observe(depth)
        await session.send_action({"action": "exit"})

    async def _on_hello(self, session: WorkerSession, msg: Dict[str, Any]) -> None:
        session.stage = msg.get("stage")
        session.card = msg.get("card")
        session.pid = msg.get("pid")
        load_s = msg.get("load_s", 0.0)
        self.sessions[session.pid] = session
        MODEL_LOAD_SECONDS.labels(stage=session.stage).observe(load_s)
        self.scheduler.card_load_s[session.card] = load_s
        self.scheduler.update_card_vram(session.card, msg.get("vram"), session.pid)
        logger.info(
            "Worker PID %d connected: stage '%s' on card %d (load %.2fs)",
            session.pid, session.stage, session.card, load_s,
        )

    async def _on_claim(self, session: WorkerSession, msg: Dict[str, Any]) -> None:
        stage, card = session.stage, session.card

        # Stickiness: a worker that has done some work yields once another stage waits too long.
        if session.jobs_processed > 0:
            should_exit, reason = self.scheduler.should_worker_exit_for_max_wait(stage)
            if should_exit:
                await self._evict(session, "max_wait", reason)
                return

        job = self.db.claim_oldest_queued_job(stage, card, session.pid)
        if job:
            session.current_job_id = job["id"]
            self.scheduler.set_card_job(card, job["id"])
            await session.send_action({
                "action": "job",
                "id": job["id"],
                "paths": get_job_paths(job["id"]),
                "params": job,
            })
            return

        # Nothing of this stage left: exit only if this card is needed for queued work.
        needed, reason = self.scheduler.is_card_needed_by_waiting_stages(card, stage)
        if needed:
            await self._evict(session, "card_needed", reason)
        else:
            await session.send_action({"action": "none"})

    async def _on_progress(self, session: WorkerSession, msg: Dict[str, Any]) -> None:
        sub_stage = msg.get("sub_stage")
        self.db.record_progress(
            msg.get("id"),
            step=msg.get("step"),
            steps=msg.get("steps"),
            s_per_step=msg.get("s_per_step"),
            sub_stage=sub_stage,
            enhance_stats=msg.get("enhance_stats"),
        )
        if sub_stage:
            self.scheduler.set_card_sub_stage(session.card, sub_stage)

    async def _on_done(self, session: WorkerSession, msg: Dict[str, Any]) -> None:
        job_id = msg.get("id")
        timings = msg.get("timings", {})
        peak_vram = msg.get("peak_vram", 0)
        stage = session.stage
        paths = get_job_paths(job_id)

        sha256 = enhanced_prompt = thinking = None
        if stage == "enhance":
            try:
                data = json.loads(Path(paths["enhanced_json"]).read_text())
                enhanced_prompt = data.get("rewritten_prompt")
                thinking = data.get("thinking")
            except (OSError, ValueError) as e:
                logger.warning("Failed to read enhanced.json for %s: %s", job_id, e)
            self._observe_enhance_metrics(timings)
        elif stage == "decode":
            img_path = Path(paths["image_png"])
            if img_path.exists():
                sha256 = compute_sha256(img_path)

        version = self.db.advance_stage_success(
            job_id, stage, timings, peak_vram,
            sha256=sha256, enhanced_prompt=enhanced_prompt, thinking=thinking,
        )
        if version is None:
            logger.info("Ignoring completion of %s for job %s: job is no longer in that stage", stage, job_id)
        session.jobs_processed += 1
        self.scheduler.inc_card_jobs_processed(session.card)
        if peak_vram > 0:
            self.scheduler.update_card_vram(session.card, {"peak": peak_vram}, session.pid)
        session.current_job_id = None
        self.scheduler.set_card_job(session.card, None)

    async def _on_error(self, session: WorkerSession, msg: Dict[str, Any]) -> None:
        job_id = msg.get("id")
        self.db.record_stage_failure(job_id, session.stage, msg.get("message", "Unknown error"))
        session.current_job_id = None
        self.scheduler.set_card_job(session.card, None)

    @staticmethod
    def _observe_enhance_metrics(timings: Dict[str, Any]) -> None:
        if timings.get("prefill_tps", 0) > 0:
            ENHANCE_PREFILL_TPS.observe(timings["prefill_tps"])
        if timings.get("generation_tps", 0) > 0:
            ENHANCE_GENERATION_TPS.observe(timings["generation_tps"])
        for kind in ("prompt", "thought", "reply"):
            count = timings.get(f"{kind}_tokens", 0)
            if count > 0:
                ENHANCE_TOKENS_TOTAL.labels(type=kind).inc(count)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        session = WorkerSession(reader, writer)
        try:
            while self.running and session.is_active:
                line = await reader.readline()
                if not line:
                    break

                text = line.decode("utf-8").strip()
                if not text:
                    continue

                msg = json.loads(text)
                event = msg.get("event")
                session.last_heartbeat = time.time()  # any message counts as a heartbeat
                if "vram" in msg and session.card is not None:
                    self.scheduler.update_card_vram(session.card, msg["vram"], session.pid)

                handler = self._handlers.get(event)
                if handler:
                    await handler(session, msg)

        except Exception as e:
            logger.warning("Error in worker session (PID %s): %s", session.pid, e)
        finally:
            if session.card is not None:
                self.scheduler.set_card_job(session.card, None)
            if session.pid in self.sessions:
                self.sessions.pop(session.pid, None)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
