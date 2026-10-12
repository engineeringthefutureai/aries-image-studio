"""Card scheduler and worker subprocess executor for Aries Manager."""
import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

from aries_manager.config import (
    DEFAULT_CARD_VRAM_BYTES,
    DEFAULT_DECODE_RAM_FLOOR_BYTES,
    ENHANCE_IDLE_GRACE_SECONDS,
    GiB,
    GPU_NAME,
    IDLE_GRACE_SECONDS,
    MAX_WAIT_SECONDS,
    NUM_CARDS,
    SOCKET_PATH,
)
from aries_manager.db import Database
from aries_manager.hardware import (
    get_pci_address_map,
    get_host_available_ram_bytes,
    check_for_wedged_card,
    get_vram_overflow_bytes,
)
from aries_manager.metrics import (
    CARD_STATE,
    STAGE_RUNS_TOTAL,
    WORKER_CRASHES_TOTAL,
    VRAM_OVERFLOW_BYTES,
)

logger = logging.getLogger("aries_manager.scheduler")

CARD_STATES = ("idle", "running", "unavailable")
EXIT_CODE_OOM_KILL = 137
OOM_RAM_FLOOR_STEP_BYTES = 2 * GiB
# Backoff before respawning on a card whose worker died before connecting
# (bad model path, broken runtime): doubles per failure, capped.
STARTUP_BACKOFF_BASE_S = 5.0
STARTUP_BACKOFF_MAX_S = 300.0


def _empty_vram() -> Dict[str, int]:
    return {"allocated": 0, "reserved": 0, "peak": 0, "total": DEFAULT_CARD_VRAM_BYTES, "overflow": 0}


class CardScheduler:
    """Tracks each card's state and spawns one stage worker per free card (ARCHITECTURE.md §4)."""

    def __init__(self, db: Database, num_cards: int = NUM_CARDS):
        self.db = db
        self.num_cards = num_cards
        self.pci_map = get_pci_address_map(num_cards)
        self.decode_ram_floor = DEFAULT_DECODE_RAM_FLOOR_BYTES

        # Per-card runtime state, keyed by card index. States: CARD_STATES.
        self.card_states: Dict[int, str] = {i: "idle" for i in range(num_cards)}
        self.card_stages: Dict[int, Optional[str]] = {i: None for i in range(num_cards)}
        self.card_procs: Dict[int, Optional[asyncio.subprocess.Process]] = {i: None for i in range(num_cards)}
        self.card_pids: Dict[int, Optional[int]] = {i: None for i in range(num_cards)}
        self.card_jobs: Dict[int, Optional[str]] = {i: None for i in range(num_cards)}
        self.card_sub_stages: Dict[int, Optional[str]] = {i: None for i in range(num_cards)}
        self.card_reasons: Dict[int, Optional[str]] = {i: None for i in range(num_cards)}
        self.card_exit_reasons: Dict[int, Optional[str]] = {i: None for i in range(num_cards)}
        self.card_eviction_depths: Dict[int, Optional[Dict[str, int]]] = {i: None for i in range(num_cards)}
        self.card_jobs_processed: Dict[int, int] = {i: 0 for i in range(num_cards)}
        self.card_load_s: Dict[int, Optional[float]] = {i: None for i in range(num_cards)}
        self.card_startup_failures: Dict[int, int] = {i: 0 for i in range(num_cards)}
        self.card_retry_at: Dict[int, float] = {i: 0.0 for i in range(num_cards)}
        self.card_vram: Dict[int, Dict[str, int]] = {i: _empty_vram() for i in range(num_cards)}

        # At most one decode at a time: an untiled 2048² decode spills ~2 GiB into host RAM.
        self.untiled_decode_running = False

        self._update_metrics()

    def set_card_job(self, card_idx: Optional[int], job_id: Optional[str]) -> None:
        """Update currently executing job ID for a card executor."""
        if card_idx is not None and 0 <= card_idx < self.num_cards:
            self.card_jobs[card_idx] = job_id
            if job_id is None:
                self.card_sub_stages[card_idx] = None

    def set_card_sub_stage(self, card_idx: Optional[int], sub_stage: Optional[str]) -> None:
        """Update active sub_stage (e.g. prefill, thinking, rewriting) for a card executor."""
        if card_idx is not None and 0 <= card_idx < self.num_cards:
            self.card_sub_stages[card_idx] = sub_stage

    def update_card_vram(
        self,
        card_idx: Optional[int],
        vram_data: Optional[Dict[str, int]],
        pid: Optional[int] = None,
    ) -> None:
        """Update VRAM telemetry from worker handshake, heartbeat, progress, or done event."""
        if card_idx is None or not (0 <= card_idx < self.num_cards):
            return
        curr = self.card_vram[card_idx]
        if vram_data:
            curr["allocated"] = vram_data.get("allocated", curr["allocated"])
            curr["reserved"] = vram_data.get("reserved", curr["reserved"])
            curr["peak"] = max(curr["peak"], vram_data.get("peak", 0))
            curr["total"] = vram_data.get("total") or curr["total"]
        if pid:
            overflow = get_vram_overflow_bytes(pid)
            curr["overflow"] = overflow
            VRAM_OVERFLOW_BYTES.labels(card=str(card_idx)).set(overflow)

    def inc_card_jobs_processed(self, card_idx: Optional[int]) -> None:
        """Increment count of processed jobs for active worker on card."""
        if card_idx is not None and 0 <= card_idx < self.num_cards:
            self.card_jobs_processed[card_idx] += 1

    def set_card_eviction(
        self,
        card_idx: Optional[int],
        reason: Optional[str],
        depths: Optional[Dict[str, int]] = None,
    ) -> None:
        """Record eviction reason and queue depths snapshot for a card worker."""
        if card_idx is not None and 0 <= card_idx < self.num_cards:
            self.card_exit_reasons[card_idx] = reason
            self.card_eviction_depths[card_idx] = depths

    def _update_metrics(self) -> None:
        for card_idx in range(self.num_cards):
            st = self.card_states[card_idx]
            for candidate in CARD_STATES:
                CARD_STATE.labels(card=str(card_idx), state=candidate).set(1 if st == candidate else 0)

    def get_status(self) -> Dict[str, Any]:
        """Return runtime status of all cards and active workers including VRAM telemetry."""
        cards_info = []
        devices_info = []
        for c in range(self.num_cards):
            vram = self.card_vram[c]
            pci = self.pci_map.get(c, "")
            cards_info.append({
                "card": c,
                "pci": pci,
                "state": self.card_states[c],
                "stage": self.card_stages[c],
                "sub_stage": self.card_sub_stages.get(c),
                "pid": self.card_pids[c],
                "job_id": self.card_jobs[c],
                "unavailable_reason": self.card_reasons[c],
                "vram_used": vram["allocated"],
                "vram_reserved": vram["reserved"],
                "vram_total": vram["total"],
                "vram_peak": vram["peak"],
                "vram_overflow": vram["overflow"],
            })
            devices_info.append({
                "index": c,
                "name": f"{GPU_NAME} (GPU {c})",
                "pci": pci,
                "vram_used": vram["allocated"],
                "vram_total": vram["total"],
                "vram_peak_last_job": vram["peak"],
                "allocated": vram["allocated"],
                "reserved": vram["reserved"],
                "peak": vram["peak"],
                "overflow": vram["overflow"],
            })
        return {
            "cards": cards_info,
            "devices": devices_info,
            "decode_ram_floor_bytes": self.decode_ram_floor,
            "host_available_ram_bytes": get_host_available_ram_bytes(),
            "queue_depths": self.db.get_queue_depths(),
        }

    def mark_card_unavailable(self, card_idx: int, reason: str) -> None:
        """Mark card unavailable after critical hardware fault."""
        self.card_states[card_idx] = "unavailable"
        self.card_reasons[card_idx] = reason
        self._update_metrics()
        logger.error("Card %d marked UNAVAILABLE: %s", card_idx, reason)

    def can_stage_run_on_card(self, stage: str, card_idx: int, depths: Optional[Dict[str, int]] = None) -> bool:
        """Evaluate stage constraints for a card per ARCHITECTURE.md §4."""
        if self.card_states[card_idx] != "idle":
            return False

        if stage == "decode":
            host_ram = get_host_available_ram_bytes()
            if host_ram < self.decode_ram_floor:
                logger.debug("Cannot run decode: host RAM %d < floor %d", host_ram, self.decode_ram_floor)
                return False
            if self.untiled_decode_running:
                logger.debug("Cannot run decode: another untiled 2048² decode is already running")
                return False
            return True

        elif stage == "encode":
            # Limit: at most one encode worker at a time across cards
            if any(s == "encode" for s in self.card_stages.values()):
                logger.debug("Cannot run encode: another encode worker is already running")
                return False
            return True

        elif stage in ("enhance", "denoise"):
            # Don't spawn another worker if workers already starting on other cards
            # (no job claimed yet) can cover everything queued for this stage.
            workers_ready = sum(
                1 for c in range(self.num_cards)
                if c != card_idx and self.card_stages[c] == stage and self.card_jobs[c] is None
            )
            if depths is None:
                depths = self.db.get_queue_depths()
            if depths.get(stage, 0) <= workers_ready:
                logger.debug(
                    "Card %d skipping %s: %d queued <= %d workers ready/spawning",
                    card_idx, stage, depths.get(stage, 0), workers_ready,
                )
                return False
            return True

        return False

    def pick_next_stage(self, card_idx: int) -> Optional[str]:
        """
        Pick next stage for an idle card per ARCHITECTURE.md §4:
        1. Candidate stages = stages with queued jobs whose constraints this card meets.
        2. Pick stage whose oldest waiting job is oldest.
        """
        if time.monotonic() < self.card_retry_at[card_idx]:
            return None

        depths = self.db.get_queue_depths()
        candidates = []

        for stage, depth in depths.items():
            if depth > 0 and self.can_stage_run_on_card(stage, card_idx, depths=depths):
                waiting_time = self.db.get_oldest_waiting_time(stage) or 0.0
                candidates.append((stage, waiting_time))

        if not candidates:
            return None

        # Sort by waiting time descending (oldest waiting job is first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen_stage = candidates[0][0]
        logger.info(
            "Scheduler selected stage '%s' for Card %d (waiting time %.1fs)",
            chosen_stage, card_idx, candidates[0][1]
        )
        return chosen_stage

    def should_worker_exit_for_max_wait(self, current_stage: str) -> Tuple[bool, Optional[str]]:
        """
        ARCHITECTURE.md §4 Stickiness:
        A worker keeps claiming jobs of its stage unless another stage's oldest job
        has waited longer than max_wait (default 10 min) and longer than current stage.
        Returns (should_exit, reason).
        """
        depths = self.db.get_queue_depths()
        current_waiting = self.db.get_oldest_waiting_time(current_stage) or 0.0
        for other_stage, depth in depths.items():
            if other_stage != current_stage and depth > 0:
                waiting_time = self.db.get_oldest_waiting_time(other_stage) or 0.0
                if waiting_time >= MAX_WAIT_SECONDS and waiting_time > current_waiting:
                    reason = (
                        f"stage '{other_stage}' has waited {waiting_time:.1f}s >= max_wait {MAX_WAIT_SECONDS:.1f}s "
                        f"(current stage '{current_stage}' oldest {current_waiting:.1f}s)"
                    )
                    logger.info("Worker on '%s' should yield: %s", current_stage, reason)
                    return True, reason
        return False, None

    def is_card_needed_by_waiting_stages(self, card_idx: Optional[int], current_stage: str) -> Tuple[bool, Optional[str]]:
        """
        Check if any queued stage actually needs this card right now.
        Returns (is_needed, reason).
        Returns False if:
        - No other stage is waiting in the queue.
        - Another card is already idle and can run the waiting stage.
        - Another card is already running the waiting stage and covers the depth.
        """
        if card_idx is None:
            return False, None

        depths = self.db.get_queue_depths()
        waiting_stages = [s for s, cnt in depths.items() if s != current_stage and cnt > 0]
        if not waiting_stages:
            return False, None

        # If another card is idle and can pick up the waiting stage, this card is NOT needed.
        for other_card in range(self.num_cards):
            if other_card != card_idx and self.card_states[other_card] == "idle":
                if any(self.can_stage_run_on_card(s, other_card, depths=depths) for s in waiting_stages):
                    return False, None

        # If another card is already running the waiting stage and can drain the queue, this card is NOT needed.
        for s in waiting_stages:
            workers_on_stage = sum(
                1 for c in range(self.num_cards)
                if c != card_idx and self.card_stages[c] == s
            )
            if depths[s] > workers_on_stage:
                reason = f"queued stage '{s}' has depth {depths[s]} > {workers_on_stage} active workers on other cards"
                return True, reason

        return False, None

    async def spawn_stage_worker(self, stage: str, card_idx: int) -> asyncio.subprocess.Process:
        """Spawn a dedicated stage worker subprocess for a card."""
        self.card_states[card_idx] = "running"
        self.card_stages[card_idx] = stage
        self.card_jobs_processed[card_idx] = 0
        self.card_exit_reasons[card_idx] = None
        self.card_eviction_depths[card_idx] = None
        # Peak VRAM is per worker run: kept after exit, reset when the next one starts.
        self.card_vram[card_idx]["peak"] = 0
        self._update_metrics()

        env = os.environ.copy()
        env["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{card_idx}"
        env["ZE_ENABLE_PCI_ID_DEVICE_ORDER"] = "1"

        idle_grace = ENHANCE_IDLE_GRACE_SECONDS if stage == "enhance" else IDLE_GRACE_SECONDS

        cmd = [
            sys.executable,
            "-m",
            "aries_worker",
            "--stage",
            stage,
            "--card",
            str(card_idx),
            "--socket",
            str(SOCKET_PATH),
            "--idle-grace",
            str(idle_grace),
        ]

        logger.info("Spawning stage worker on card %d: %s", card_idx, " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self.card_procs[card_idx] = proc
        self.card_pids[card_idx] = proc.pid
        if stage == "decode":
            self.untiled_decode_running = True

        asyncio.create_task(self._monitor_worker_process(stage, card_idx, proc))
        return proc

    async def _stream_pipe(self, pipe: Optional[asyncio.StreamReader], prefix: str, is_stderr: bool = False) -> bytes:
        """Stream a subprocess pipe line-by-line to manager logs and collect bytes."""
        if not pipe:
            return b""
        collected = []
        while True:
            line = await pipe.readline()
            if not line:
                break
            collected.append(line)
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                if is_stderr:
                    logger.warning("%s %s", prefix, text)
                else:
                    logger.info("%s %s", prefix, text)
        return b"".join(collected)

    async def _monitor_worker_process(
        self,
        stage: str,
        card_idx: int,
        proc: asyncio.subprocess.Process,
    ) -> None:
        """Monitor worker process exit, stream logs, and handle crash classifications."""
        pid = proc.pid
        t0 = time.time()
        prefix = f"[worker:{stage}:gpu{card_idx}:pid{pid}]"
        _, stderr = await asyncio.gather(
            self._stream_pipe(proc.stdout, prefix, is_stderr=False),
            self._stream_pipe(proc.stderr, prefix, is_stderr=True),
        )
        exit_code = await proc.wait()
        duration = time.time() - t0

        logger.info(
            "Worker process %d (stage=%s, card=%d) exited with code %s after %.1fs",
            pid, stage, card_idx, exit_code, duration
        )

        if stage == "decode":
            self.untiled_decode_running = False

        self.card_procs[card_idx] = None
        self.card_pids[card_idx] = None
        self.card_stages[card_idx] = None
        self.set_card_job(card_idx, None)

        STAGE_RUNS_TOTAL.labels(stage=stage, card=str(card_idx), exit=str(exit_code)).inc()

        job_id = None
        crash_class = None
        if exit_code != 0:
            # Classify the crash (ARCHITECTURE.md §8).
            crash_class = "python_error"
            if exit_code == EXIT_CODE_OOM_KILL:
                crash_class = "oom_kill"
                self.decode_ram_floor += OOM_RAM_FLOOR_STEP_BYTES
                logger.warning(
                    "Host OOM kill (exit %d): raising decode_ram_floor to %d bytes for this session",
                    exit_code, self.decode_ram_floor,
                )

            wedged_msg = check_for_wedged_card(self.pci_map.get(card_idx, ""))
            if wedged_msg:
                crash_class = "wedged"
                self.mark_card_unavailable(card_idx, wedged_msg)

            WORKER_CRASHES_TOTAL.labels(
                stage=stage, card=str(card_idx), exit=str(exit_code), crash_class=crash_class,
            ).inc()

            # Requeue (or fail) the job the dead worker was running.
            job_id = self.db.find_active_job_for_pid(pid)
            if job_id:
                err_text = f"Worker died with exit code {exit_code}: {stderr.decode('utf-8', errors='replace')[-200:]}"
                self.db.record_stage_failure(job_id, stage, err_text, crash_class)
                logger.warning("Reconciled job %s after worker %d died", job_id, pid)

        if self.card_states[card_idx] != "unavailable":
            self.card_states[card_idx] = "idle"

        # A worker sends `hello` once its model is loaded; dying before that means it
        # cannot start at all, so back off instead of respawning every second.
        if exit_code != 0 and self.card_load_s.get(card_idx) is None:
            failures = self.card_startup_failures[card_idx] = self.card_startup_failures[card_idx] + 1
            delay = min(STARTUP_BACKOFF_MAX_S, STARTUP_BACKOFF_BASE_S * 2 ** (failures - 1))
            self.card_retry_at[card_idx] = time.monotonic() + delay
            logger.error(
                "Worker for stage '%s' on card %d failed to start (%d in a row); retrying in %.0fs",
                stage, card_idx, failures, delay,
            )
        else:
            self.card_startup_failures[card_idx] = 0
            self.card_retry_at[card_idx] = 0.0

        # Record the run in the stage_runs audit table (ARCHITECTURE.md §5).
        exit_reason = self.card_exit_reasons.get(card_idx) or crash_class or "idle_grace"
        queue_depths = self.card_eviction_depths.get(card_idx)
        if queue_depths is None:
            try:
                queue_depths = self.db.get_queue_depths()
            except Exception:
                queue_depths = None

        jobs_processed = self.card_jobs_processed.get(card_idx, 0)

        try:
            self.db.record_stage_run(
                stage=stage,
                card=card_idx,
                pid=pid,
                started_at=t0,
                ended_at=time.time(),
                exit_code=exit_code,
                jobs_processed=jobs_processed,
                load_s=self.card_load_s.get(card_idx),
                job_id=job_id,
                exit_reason=exit_reason,
                queue_depths=queue_depths,
            )
        except Exception as e:
            logger.warning("Failed to record stage run in database: %s", e)
        finally:
            self.card_exit_reasons[card_idx] = None
            self.card_eviction_depths[card_idx] = None
            self.card_jobs_processed[card_idx] = 0
            self.card_load_s[card_idx] = None
            self.card_vram[card_idx]["allocated"] = 0
            self.card_vram[card_idx]["reserved"] = 0
            self.card_vram[card_idx]["overflow"] = 0
            VRAM_OVERFLOW_BYTES.labels(card=str(card_idx)).set(0)

        self._update_metrics()
