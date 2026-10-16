"""Unix socket client for stage workers: the JSON-lines protocol of ARCHITECTURE.md §5."""
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from aries_worker.stages.loader import get_pci_address, get_versions, get_visible_devices, get_vram_stats

logger = logging.getLogger("aries_worker.client")

# Stages whose runners report progress and honour cancellation mid-job.
INTERRUPTIBLE_STAGES = {"enhance", "denoise"}
HEARTBEAT_INTERVAL_S = 5.0


class WorkerSocketClient:
    def __init__(
        self,
        stage: str,
        socket_path: str,
        runner: Any,
        idle_grace: float = 3600.0,
    ):
        self.stage = stage
        self.socket_path = socket_path
        self.runner = runner
        self.idle_grace = idle_grace

        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.running = False
        self.cancelled_jobs = set()
        self.should_exit = False
        self.msg_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self, max_retries: int = 20, retry_delay: float = 0.5) -> None:
        """Connect to the agent's Unix domain socket."""
        for attempt in range(max_retries):
            try:
                self.reader, self.writer = await asyncio.open_unix_connection(self.socket_path)
                logger.info("Connected to agent socket at %s", self.socket_path)
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
                if attempt == max_retries - 1:
                    logger.error("Failed to connect to agent socket after %d attempts: %s", max_retries, e)
                    raise
                await asyncio.sleep(retry_delay)

    async def send_event(self, data: Dict[str, Any]) -> None:
        """Send a JSON-lines message to the agent."""
        if not self.writer:
            return
        line = json.dumps(data) + "\n"
        self.writer.write(line.encode("utf-8"))
        await self.writer.drain()

    async def _heartbeat_loop(self) -> None:
        """Send heartbeat with VRAM usage every 5s."""
        while self.running:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
                if self.running:
                    await self.send_event({
                        "event": "heartbeat",
                        "vram": get_vram_stats(),
                    })
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Heartbeat error: %s", e)
                break

    async def _listen_loop(self) -> None:
        """Read incoming JSON-lines messages from the agent."""
        while self.running and self.reader:
            try:
                line = await self.reader.readline()
                if not line:
                    logger.info("Agent closed socket connection.")
                    self.should_exit = True
                    await self.msg_queue.put({"action": "exit"})
                    break

                text = line.decode("utf-8").strip()
                if not text:
                    continue

                msg = json.loads(text)
                action = msg.get("action")

                if action == "cancel":
                    job_id = msg.get("id")
                    logger.info("Received cancel request for job %s", job_id)
                    self.cancelled_jobs.add(job_id)
                elif action == "exit":
                    logger.info("Received exit request from agent.")
                    self.should_exit = True
                    await self.msg_queue.put(msg)
                else:
                    await self.msg_queue.put(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Error reading from agent socket: %s", e)
                self.should_exit = True
                await self.msg_queue.put({"action": "exit"})
                break

    async def run(self) -> None:
        """Main lifecycle loop of the stage worker."""
        await self.connect()
        self.running = True
        visible_devices = get_visible_devices()

        await self.send_event({
            "event": "hello",
            "stage": self.stage,
            "pid": os.getpid(),
            "pci": get_pci_address(),
            "devices": visible_devices,
            "versions": get_versions(),
            "load_s": getattr(self.runner, "load_s", 0.0),
            "vram": get_vram_stats(),
        })

        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        listen_task = asyncio.create_task(self._listen_loop())

        try:
            while self.running and not self.should_exit:
                # Ask the agent for the next job
                await self.send_event({"event": "claim"})

                # Wait for the agent's response (job, none, exit)
                msg = await self.msg_queue.get()
                action = msg.get("action")

                if action == "exit":
                    logger.info("Exiting on agent instruction.")
                    break

                elif action == "none":
                    # Start idle grace wait
                    logger.debug("No jobs available for stage %s; waiting idle grace (%.1fs)...", self.stage, self.idle_grace)
                    idle_start = time.monotonic()
                    job_received = False

                    while (time.monotonic() - idle_start) < self.idle_grace and not self.should_exit:
                        # Wait for either exit, next claim, or timeout
                        try:
                            check_msg = await asyncio.wait_for(self.msg_queue.get(), timeout=1.0)
                            if check_msg.get("action") == "exit":
                                self.should_exit = True
                                break
                            elif check_msg.get("action") == "job":
                                msg = check_msg
                                action = "job"
                                job_received = True
                                break
                        except asyncio.TimeoutError:
                            # Periodic claim poll during idle grace
                            await self.send_event({"event": "claim"})

                    if not job_received:
                        if not self.should_exit:
                            logger.info("Idle grace (%.1fs) expired with no new work; exiting.", self.idle_grace)
                        break

                if action == "job":
                    await self._run_job(msg)

        finally:
            self.running = False
            heartbeat_task.cancel()
            listen_task.cancel()
            if self.writer:
                try:
                    self.writer.close()
                    await self.writer.wait_closed()
                except Exception:
                    pass
            logger.info("Stage worker (stage=%s) finished.", self.stage)

    async def _run_job(self, msg: Dict[str, Any]) -> None:
        """Run one claimed job in a thread so heartbeats and cancels keep flowing."""
        job_id = msg.get("id")
        paths = msg.get("paths", {})
        params = msg.get("params", {})
        logger.info("Claimed job %s for stage %s", job_id, self.stage)
        loop = asyncio.get_running_loop()

        def progress_cb(
            step: Optional[int] = None,
            steps: Optional[int] = None,
            sps: Optional[float] = None,
            sub_stage: Optional[str] = None,
            enhance_stats: Optional[Dict[str, Any]] = None,
        ) -> None:
            payload: Dict[str, Any] = {"event": "progress", "id": job_id, "vram": get_vram_stats()}
            for key, value in (
                ("step", step),
                ("steps", steps),
                ("s_per_step", sps),
                ("sub_stage", sub_stage),
                ("enhance_stats", enhance_stats),
            ):
                if value is not None:
                    payload[key] = value
            loop.call_soon_threadsafe(asyncio.create_task, self.send_event(payload))

        def cancel_check() -> bool:
            return job_id in self.cancelled_jobs

        extra = (progress_cb, cancel_check) if self.stage in INTERRUPTIBLE_STAGES else ()
        try:
            timings, peak_vram = await asyncio.to_thread(self.runner.run_job, job_id, paths, params, *extra)
            if cancel_check():
                raise RuntimeError("cancelled")
            await self.send_event({
                "event": "done",
                "id": job_id,
                "timings": timings,
                "peak_vram": peak_vram,
                "vram": get_vram_stats(),
            })
        except Exception as e:
            if cancel_check():
                logger.info("Job %s cancelled.", job_id)
                message = "cancelled"
            else:
                logger.exception("Error executing stage %s on job %s: %s", self.stage, job_id, e)
                message = str(e)
            await self.send_event({"event": "error", "id": job_id, "message": message})
        finally:
            self.cancelled_jobs.discard(job_id)
