"""Thin agent implementation (Architecture V3 §5-§10).

Dials the Rust backend over gRPC (`AgentGateway.Connect`), declares static capabilities
and live resource limits, supervises one stage worker subprocess per executor over the
existing Unix-socket JSON-lines protocol, transfers task inputs/outputs over HTTP, and
exposes `/metrics` for live scraping.
"""
import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import grpc
import httpx

from aries.agent.v1 import agent_pb2, agent_pb2_grpc
from aries_agent.config import AgentConfig, ExecutorConfig, ModelConfig
from aries_agent.hardware import (
    classify_worker_crash,
    get_host_ram_available_bytes,
    get_kernel_log_tail,
    get_process_rss_bytes,
    get_vram_overflow_bytes,
)
from aries_agent.metrics import AgentMetrics

logger = logging.getLogger("aries_agent")

STAGE_ENUM_TO_NAME = {
    agent_pb2.STAGE_ENHANCE: "enhance",
    agent_pb2.STAGE_ENCODE: "encode",
    agent_pb2.STAGE_DENOISE: "denoise",
    agent_pb2.STAGE_DECODE: "decode",
}

DECODE_MODE_TO_NAME = {
    agent_pb2.DECODE_MODE_UNTILED: "untiled",
    agent_pb2.DECODE_MODE_TILED: "tiled",
}
DECODE_NAME_TO_MODE = {v: k for k, v in DECODE_MODE_TO_NAME.items()}

FAILURE_CLASS_TO_NAME = {
    agent_pb2.FAILURE_CLASS_UNSPECIFIED: "unspecified",
    agent_pb2.FAILURE_GPU_FAULT: "gpu_fault",
    agent_pb2.FAILURE_ENGINE_RESET: "engine_reset",
    agent_pb2.FAILURE_WEDGED: "wedged",
    agent_pb2.FAILURE_OOM_KILL: "oom_kill",
    agent_pb2.FAILURE_PYTHON_ERROR: "python_error",
    agent_pb2.FAILURE_DEVICE_MISMATCH: "device_mismatch",
    agent_pb2.FAILURE_UPLOAD_FAILED: "upload_failed",
    agent_pb2.FAILURE_CANCELLED: "cancelled",
}

ARTIFACT_PATH_KEYS = {
    "enhanced.json": "enhanced_json",
    "embeds.safetensors": "embeds_safetensors",
    "latent.safetensors": "latent_safetensors",
    "image.png": "image_png",
}


def worker_preexec(oom_score_adj: int):
    """Runs in the worker between fork and exec: raise its oom_score_adj so a global OOM kill
    (or the kernel's thrash protection) takes a worker, not the desktop session. Raising it
    needs no privilege."""

    def apply() -> None:
        try:
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write(str(oom_score_adj))
        except OSError:
            pass

    return apply


def compute_file_sha256(path: Path) -> Tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


@dataclass
class ActiveTask:
    assign: agent_pb2.Assign
    scratch_dir: Path
    input_paths: Dict[str, str] = field(default_factory=dict)
    output_paths: Dict[str, Path] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    cancelled: bool = False
    dispatched_to_worker: bool = False
    completed: bool = False


class ExecutorRuntime:
    """Supervises at most one stage worker process on one GPU executor."""

    def __init__(
        self,
        cfg: ExecutorConfig,
        agent_cfg: AgentConfig,
        metrics: AgentMetrics,
        emit_msg: Callable[[agent_pb2.AgentMessage], None],
        on_worker_count_change: Callable[[], None],
        http_client_factory: Callable[[], httpx.AsyncClient],
    ) -> None:
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        self.metrics = metrics
        self.emit_msg = emit_msg
        self.on_worker_count_change = on_worker_count_change
        self.http_client_factory = http_client_factory

        self.state: int = agent_pb2.EXECUTOR_STATE_IDLE
        self.stage: int = agent_pb2.STAGE_UNSPECIFIED
        self.model: Optional[agent_pb2.ModelRef] = None
        self.process: Optional[asyncio.subprocess.Process] = None
        self.pid: int = 0
        self.load_s: float = 0.0
        self.versions: Dict[str, str] = {}
        self.worker_writer: Optional[asyncio.StreamWriter] = None

        self.active_task: Optional[ActiveTask] = None
        self.release_requested: bool = False
        # Set while a finished task's outputs upload: a release then waits for the upload.
        self.finishing: bool = False
        self.socket_path = self.agent_cfg.scratch_dir / f"{self.cfg.id}.sock"
        self.unix_server: Optional[asyncio.Server] = None
        self._monitor_task: Optional[asyncio.Task] = None
        # Why the agent killed the current worker itself; reported instead of the exit code.
        self.kill_detail: str = ""

    @property
    def is_worker_alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def worker_ram_bytes(self) -> int:
        """Host RAM held by the worker: resident pages plus driver spill out of VRAM."""
        if not self.is_worker_alive:
            return 0
        return get_process_rss_bytes(self.pid) + get_vram_overflow_bytes(self.pid)

    def kill_for_host_ram(self, detail: str) -> None:
        """SIGKILL the worker; `_monitor_process` reports it as `oom_kill` with `detail`."""
        proc = self.process
        if proc is None or proc.returncode is not None:
            return
        logger.warning("Killing worker PID %s on %s: %s", proc.pid, self.cfg.id, detail)
        self.kill_detail = detail
        self.metrics.ram_guard_kills_total.labels(
            executor=self.cfg.id, stage=STAGE_ENUM_TO_NAME.get(self.stage, "unknown")
        ).inc()
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    def set_state(self, kind: int, detail: str = "") -> None:
        self.state = kind
        state_name = {
            agent_pb2.EXECUTOR_STATE_IDLE: "idle",
            agent_pb2.EXECUTOR_STATE_STARTING: "starting",
            agent_pb2.EXECUTOR_STATE_READY: "ready",
            agent_pb2.EXECUTOR_STATE_BUSY: "busy",
            agent_pb2.EXECUTOR_STATE_STOPPING: "stopping",
            agent_pb2.EXECUTOR_STATE_FAILED: "failed",
        }.get(kind, "idle")
        self.metrics.set_executor_state(self.cfg.id, state_name)
        self.emit_msg(
            agent_pb2.AgentMessage(
                executor_state=agent_pb2.ExecutorState(
                    executor=self.cfg.id,
                    state=kind,
                    stage=self.stage,
                    pid=self.pid,
                    load_s=self.load_s if kind == agent_pb2.EXECUTOR_STATE_READY else 0.0,
                    detail=detail,
                )
            )
        )

    def announce_state(self) -> None:
        """Re-send the current executor state (after a reconnect the backend starts from idle)."""
        if self.state != agent_pb2.EXECUTOR_STATE_IDLE:
            self.set_state(self.state)

    async def start_socket_server(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)
        self.unix_server = await asyncio.start_unix_server(
            self._handle_worker_conn,
            path=str(self.socket_path),
        )

    async def stop_socket_server(self) -> None:
        if self.unix_server:
            self.unix_server.close()
            await self.unix_server.wait_closed()
            self.unix_server = None
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)

    def _find_model_config(self, model_id: str) -> Optional[ModelConfig]:
        for m in self.agent_cfg.models:
            if m.id == model_id:
                return m
        return None

    async def start_worker(self, cmd: agent_pb2.StartWorker) -> None:
        """Spawn a fresh stage worker process on this executor (§10.3)."""
        stage_name = STAGE_ENUM_TO_NAME.get(cmd.stage)
        if not stage_name:
            self.set_state(agent_pb2.EXECUTOR_STATE_FAILED, detail=f"unknown stage {cmd.stage}")
            self.set_state(agent_pb2.EXECUTOR_STATE_IDLE)
            return

        cmd_model_id = cmd.model.id if cmd.model else None
        cur_model_id = self.model.id if self.model else None
        if (
            self.is_worker_alive
            and self.stage == cmd.stage
            and (cmd_model_id is None or cur_model_id == cmd_model_id)
            and self.state in (agent_pb2.EXECUTOR_STATE_STARTING, agent_pb2.EXECUTOR_STATE_READY)
        ):
            logger.info(
                "Executor %s already has worker alive for stage '%s' (state=%s); ignoring duplicate StartWorker",
                self.cfg.id,
                stage_name,
                self.state,
            )
            # The backend marked the executor `starting` when it sent this; tell it where we are.
            self.set_state(self.state)
            return

        if self.state != agent_pb2.EXECUTOR_STATE_IDLE or self.is_worker_alive:
            await self.kill_worker(reason="restart_before_new_worker", emit_idle=False)

        self.stage = cmd.stage
        self.model = cmd.model
        self.load_s = 0.0
        self.pid = 0
        self.release_requested = False
        self.kill_detail = ""

        await self.start_socket_server()
        self.set_state(agent_pb2.EXECUTOR_STATE_STARTING)

        env = os.environ.copy()
        env["ZE_ENABLE_PCI_ID_DEVICE_ORDER"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["ARIES_DECODE_SPLIT_ABOVE_PX"] = str(self.agent_cfg.decode_split_above_px)
        if self.agent_cfg.backend_kind == "xpu":
            env["ONEAPI_DEVICE_SELECTOR"] = self.cfg.selector
        elif self.agent_cfg.backend_kind == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = self.cfg.selector
        elif self.agent_cfg.backend_kind == "rocm":
            env["HIP_VISIBLE_DEVICES"] = self.cfg.selector

        if self.cfg.device and "ARIES_VISIBLE_DEVICES" not in env:
            env["ARIES_DEVICE_PCI"] = self.cfg.device

        for m in self.agent_cfg.models:
            # MODEL_DTYPE is this stage's model: the enhancer's for enhance, else the main one's.
            # The default enhancer is the text one; StartWorker.model picks PE-I2I below.
            if m.id == self.agent_cfg.enhancer_i2i_model_id:
                continue
            if "PE" in m.id or m.id.endswith("-PE-T2I"):
                env["ENHANCER_MODEL_PATH"] = m.path
                env["ENHANCER_MODEL_ID"] = m.id
                env["ENHANCER_MODEL_REV"] = m.rev
                if stage_name == "enhance":
                    env["MODEL_DTYPE"] = m.dtype
            else:
                env["MODEL_PATH"] = m.path
                env["MODEL_ID"] = m.id
                env["MODEL_REV"] = m.rev
                if stage_name != "enhance":
                    env["MODEL_DTYPE"] = m.dtype

        if cmd.model and cmd.model.id:
            mc = self._find_model_config(cmd.model.id)
            if mc:
                if stage_name == "enhance":
                    env["ENHANCER_MODEL_PATH"] = mc.path
                    env["ENHANCER_MODEL_ID"] = mc.id
                    env["ENHANCER_MODEL_REV"] = mc.rev
                else:
                    env["MODEL_PATH"] = mc.path
                    env["MODEL_ID"] = mc.id
                    env["MODEL_REV"] = mc.rev
                env["MODEL_DTYPE"] = mc.dtype

        args = [
            self.agent_cfg.worker_python,
            "-X",
            "faulthandler",
            "-m",
            self.agent_cfg.worker_module,
            "--stage",
            stage_name,
            "--socket",
            str(self.socket_path),
        ]

        logger.info("Executor %s spawning worker for stage '%s': %s", self.cfg.id, stage_name, " ".join(args))
        self.process = await asyncio.create_subprocess_exec(
            *args,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            preexec_fn=worker_preexec(self.agent_cfg.worker_oom_score_adj),
        )
        self.pid = self.process.pid or 0
        self.metrics.set_executor_pid(self.cfg.id, self.pid)
        self.on_worker_count_change()
        self._monitor_task = asyncio.create_task(self._monitor_process(self.process, cmd.stage))

    async def _monitor_process(self, proc: asyncio.subprocess.Process, stage_enum: int) -> None:
        stage_name = STAGE_ENUM_TO_NAME.get(stage_enum, "unknown")
        try:
            if proc.stdout:
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    logger.info("[%s:%s] %s", self.cfg.id, stage_name, line.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass

        rc = await proc.wait()
        if self.process is not proc:
            return

        self.process = None
        self.pid = 0
        self.metrics.set_executor_pid(self.cfg.id, None)
        self.on_worker_count_change()

        # Check if a task was in flight when the worker exited
        task = self.active_task
        if task and not task.completed:
            task.completed = True
            self.active_task = None
            sig = -rc if rc < 0 else (rc - 128 if rc > 128 else 0)
            klog = get_kernel_log_tail(self.cfg.device)
            if task.cancelled:
                fclass = agent_pb2.FAILURE_CANCELLED
            elif self.kill_detail:
                fclass = agent_pb2.FAILURE_OOM_KILL
            else:
                fclass = classify_worker_crash(rc, sig, self.cfg.device, klog)
            fclass_name = FAILURE_CLASS_TO_NAME.get(fclass, "python_error")
            self.metrics.task_outcomes_total.labels(
                executor=self.cfg.id,
                stage=stage_name,
                outcome="failed",
                **{"class": fclass_name},
            ).inc()
            self.metrics.worker_exits_total.labels(
                executor=self.cfg.id,
                stage=stage_name,
                **{"class": fclass_name},
            ).inc()
            shutil.rmtree(task.scratch_dir, ignore_errors=True)
            self.emit_msg(
                agent_pb2.AgentMessage(
                    failed=agent_pb2.Failed(
                        task=task.assign.task,
                        attempt=task.assign.attempt,
                        failure_class=fclass,
                        detail=self.kill_detail or f"Worker process exited with code {rc}",
                        exit_code=rc,
                        signal=sig,
                        kernel_log_tail=klog,
                    )
                )
            )
            self.stage = agent_pb2.STAGE_UNSPECIFIED
            self.model = None
            self.set_state(agent_pb2.EXECUTOR_STATE_FAILED, detail=f"worker_crashed:{fclass_name}")
            self.set_state(agent_pb2.EXECUTOR_STATE_IDLE)
            return

        if rc != 0 and self.state not in (agent_pb2.EXECUTOR_STATE_STOPPING, agent_pb2.EXECUTOR_STATE_IDLE):
            sig = -rc if rc < 0 else (rc - 128 if rc > 128 else 0)
            klog = get_kernel_log_tail(self.cfg.device)
            fclass = (
                agent_pb2.FAILURE_OOM_KILL
                if self.kill_detail
                else classify_worker_crash(rc, sig, self.cfg.device, klog)
            )
            fclass_name = FAILURE_CLASS_TO_NAME.get(fclass, "python_error")
            self.metrics.worker_exits_total.labels(
                executor=self.cfg.id,
                stage=stage_name,
                **{"class": fclass_name},
            ).inc()
            self.stage = agent_pb2.STAGE_UNSPECIFIED
            self.model = None
            self.set_state(agent_pb2.EXECUTOR_STATE_FAILED, detail=f"{fclass_name}: {self.kill_detail or f'exit {rc}'}")
            self.set_state(agent_pb2.EXECUTOR_STATE_IDLE)
        else:
            self.metrics.worker_exits_total.labels(
                executor=self.cfg.id,
                stage=stage_name,
                **{"class": "clean"},
            ).inc()
            self.stage = agent_pb2.STAGE_UNSPECIFIED
            self.model = None
            if self.state != agent_pb2.EXECUTOR_STATE_IDLE:
                self.set_state(agent_pb2.EXECUTOR_STATE_IDLE)

    async def _send_worker_json(self, payload: Dict[str, Any]) -> None:
        if not self.worker_writer:
            return
        data = (json.dumps(payload) + "\n").encode("utf-8")
        self.worker_writer.write(data)
        await self.worker_writer.drain()

    async def _handle_worker_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.worker_writer = writer
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                msg = json.loads(text)
                event = msg.get("event")

                vram = msg.get("vram")
                if isinstance(vram, dict):
                    used = int(vram.get("allocated") or vram.get("reserved") or 0)
                    self.metrics.set_executor_mem_used(self.cfg.id, used)

                if event == "hello":
                    ok = await self._on_worker_hello(msg)
                    if not ok:
                        break
                elif event == "claim":
                    await self._on_worker_claim()
                elif event == "progress":
                    await self._on_worker_progress(msg)
                elif event == "done":
                    await self._on_worker_done(msg)
                elif event == "error":
                    await self._on_worker_error(msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Worker socket error on %s: %s", self.cfg.id, e)
        finally:
            if self.worker_writer is writer:
                self.worker_writer = None
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _on_worker_hello(self, msg: Dict[str, Any]) -> bool:
        """Validate PCI device isolation (§10.3) and transition STARTING -> READY."""
        devices = msg.get("devices")
        pci = msg.get("pci", "")
        if not isinstance(devices, list):
            devices = [pci] if pci else []

        # §10.3: The worker's `hello` must report **exactly one device with the expected PCI address**;
        # anything else -> kill it, report `failed` with class `device_mismatch`, mark executor disabled.
        expected_pci = self.cfg.device
        if len(devices) != 1 or (expected_pci and devices[0] != expected_pci):
            detail = (
                f"device_mismatch: executor {self.cfg.id} expected single device '{expected_pci}', "
                f"worker reported {devices}"
            )
            logger.error(detail)
            stage_name = STAGE_ENUM_TO_NAME.get(self.stage, "unknown")
            self.metrics.worker_exits_total.labels(
                executor=self.cfg.id,
                stage=stage_name,
                **{"class": "device_mismatch"},
            ).inc()
            if self.active_task and not self.active_task.completed:
                task = self.active_task
                task.completed = True
                self.active_task = None
                shutil.rmtree(task.scratch_dir, ignore_errors=True)
                self.emit_msg(
                    agent_pb2.AgentMessage(
                        failed=agent_pb2.Failed(
                            task=task.assign.task,
                            attempt=task.assign.attempt,
                            failure_class=agent_pb2.FAILURE_DEVICE_MISMATCH,
                            detail=detail,
                        )
                    )
                )
            if self.process and self.process.returncode is None:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
            self.pid = 0
            self.metrics.set_executor_pid(self.cfg.id, None)
            self.on_worker_count_change()
            self.stage = agent_pb2.STAGE_UNSPECIFIED
            self.model = None
            self.set_state(agent_pb2.EXECUTOR_STATE_FAILED, detail=detail)
            self.set_state(agent_pb2.EXECUTOR_STATE_IDLE)
            return False

        self.pid = int(msg.get("pid") or self.pid or 0)
        self.load_s = float(msg.get("load_s") or 0.0)
        self.versions = msg.get("versions") or {}
        self.metrics.set_executor_pid(self.cfg.id, self.pid)

        stage_name = STAGE_ENUM_TO_NAME.get(self.stage, "unknown")
        model_id = self.model.id if self.model else ""
        self.metrics.worker_load_seconds.labels(
            executor=self.cfg.id,
            stage=stage_name,
            model=model_id,
        ).observe(self.load_s)

        if self.active_task is None:
            self.set_state(agent_pb2.EXECUTOR_STATE_READY)
        return True

    async def _on_worker_claim(self) -> None:
        if self.release_requested and self.active_task is None:
            self.set_state(agent_pb2.EXECUTOR_STATE_STOPPING)
            await self._send_worker_json({"action": "exit"})
            return

        if self.active_task and not self.active_task.dispatched_to_worker:
            await self._dispatch_active_task_to_worker()
            return

        # Answer at once: this runs on the socket reader, and an idle worker claims every
        # second, so waiting here would queue its later events (done, error) behind a
        # growing backlog of claims. An assignment that arrives later is pushed to the
        # worker directly; it accepts a job while it idles.
        await self._send_worker_json({"action": "none"})

    def _build_worker_job_payload(self, task: ActiveTask) -> Dict[str, Any]:
        assign = task.assign
        paths: Dict[str, str] = {}
        paths.update(task.input_paths)
        for name, out_path in task.output_paths.items():
            key = ARTIFACT_PATH_KEYS.get(name, name)
            paths[key] = str(out_path)

        params: Dict[str, Any] = {}
        if self.model:
            params["model_id"] = self.model.id
            params["model_rev"] = self.model.rev
            params["model_dtype"] = self.model.dtype

        which = assign.WhichOneof("params")
        if which == "enhance":
            params["prompt"] = assign.enhance.prompt
            if assign.enhance.max_new_tokens:
                params["max_new_tokens"] = assign.enhance.max_new_tokens
            if assign.enhance.references:
                params["references"] = [r.sha256 for r in assign.enhance.references]
        elif which == "encode":
            params["prompt"] = assign.encode.prompt
            params["prompt_source"] = assign.encode.prompt_source or "original"
            if assign.encode.negative_prompt:
                params["negative_prompt"] = assign.encode.negative_prompt
            params["true_cfg_scale"] = assign.encode.true_cfg_scale or 1.0
            if assign.encode.references:
                params["references"] = [r.sha256 for r in assign.encode.references]
                params["reference_res"] = assign.encode.reference_res
        elif which == "denoise":
            params["width"] = assign.denoise.width
            params["height"] = assign.denoise.height
            params["steps"] = assign.denoise.steps
            params["seed"] = assign.denoise.seed
            params["true_cfg_scale"] = assign.denoise.true_cfg_scale or 1.0
            params["references"] = assign.denoise.references
        elif which == "decode":
            params["decode_mode"] = DECODE_MODE_TO_NAME.get(assign.decode.mode, "untiled")

        return {
            "action": "job",
            "id": assign.task,
            "paths": paths,
            "params": params,
        }

    async def _dispatch_active_task_to_worker(self) -> None:
        task = self.active_task
        if not task or task.dispatched_to_worker:
            return
        task.dispatched_to_worker = True
        await self._send_worker_json(self._build_worker_job_payload(task))

    async def _on_worker_progress(self, msg: Dict[str, Any]) -> None:
        task = self.active_task
        if not task or msg.get("id") != task.assign.task:
            return
        step = int(msg.get("step") or 0)
        steps = int(msg.get("steps") or 0)
        sps = float(msg.get("s_per_step") or 0.0)
        stats: Dict[str, float] = {}
        if isinstance(msg.get("enhance_stats"), dict):
            # The enhancer reports tokens, not steps: pass its stats through as they are,
            # and keep step/steps meaningful for generic consumers (tokens vs. the limit).
            stats = {
                str(k): float(v)
                for k, v in msg["enhance_stats"].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if step == 0:
                step = int(stats.get("generated_tokens", 0))
                steps = int(stats.get("max_new_tokens", 0))
                tps = stats.get("generation_tps", 0.0)
                sps = round(1.0 / tps, 4) if tps > 0 else 0.0

        self.emit_msg(
            agent_pb2.AgentMessage(
                progress=agent_pb2.Progress(
                    task=task.assign.task,
                    attempt=task.assign.attempt,
                    step=step,
                    steps=steps,
                    s_per_step=sps,
                    sub_stage=str(msg.get("sub_stage") or ""),
                    stats=stats,
                )
            )
        )

    async def _on_worker_done(self, msg: Dict[str, Any]) -> None:
        task = self.active_task
        if not task or msg.get("id") != task.assign.task or task.completed:
            return
        task.completed = True
        self.active_task = None
        self.finishing = True
        stage_name = STAGE_ENUM_TO_NAME.get(task.assign.stage, "unknown")
        elapsed = max(0.0, time.monotonic() - task.started_at)
        self.metrics.task_seconds.labels(executor=self.cfg.id, stage=stage_name).observe(elapsed)

        try:
            # Upload outputs (§7.4: 3 tries with short backoff 1 s, 3 s)
            uploaded_refs: List[agent_pb2.ArtifactRef] = []
            for upload_spec in task.assign.uploads:
                out_file = task.output_paths.get(upload_spec.name)
                if not out_file or not out_file.is_file():
                    raise FileNotFoundError(f"Worker did not produce expected output {upload_spec.name}")
                sha256, size_bytes = compute_file_sha256(out_file)
                ok, err_detail = await self._upload_artifact_with_retry(upload_spec.url, out_file, sha256)
                if not ok:
                    self.metrics.task_outcomes_total.labels(
                        executor=self.cfg.id,
                        stage=stage_name,
                        outcome="failed",
                        **{"class": "upload_failed"},
                    ).inc()
                    self.emit_msg(
                        agent_pb2.AgentMessage(
                            failed=agent_pb2.Failed(
                                task=task.assign.task,
                                attempt=task.assign.attempt,
                                failure_class=agent_pb2.FAILURE_UPLOAD_FAILED,
                                detail=err_detail,
                            )
                        )
                    )
                    return

                uploaded_refs.append(
                    agent_pb2.ArtifactRef(
                        name=upload_spec.name,
                        url=upload_spec.url,
                        sha256=sha256,
                        bytes=size_bytes,
                    )
                )

            raw_timings = msg.get("timings") or {}
            numeric_timings: Dict[str, float] = {"load_s": self.load_s}
            for k, v in raw_timings.items():
                if isinstance(v, (int, float)):
                    numeric_timings[str(k)] = float(v)

            peak_mem = int(msg.get("peak_vram") or 0)
            model_ref = self.model or agent_pb2.ModelRef(
                id=self.agent_cfg.models[0].id if self.agent_cfg.models else "Qwen/Qwen-Image-2.1",
                rev=self.agent_cfg.models[0].rev if self.agent_cfg.models else "main",
                dtype=self.agent_cfg.models[0].dtype if self.agent_cfg.models else "bf16",
            )
            provenance = agent_pb2.Provenance(
                agent=self.agent_cfg.agent_id,
                executor=self.cfg.id,
                device=self.cfg.device,
                backend_kind=self.agent_cfg.backend_kind,
                model=model_ref,
                torch_version=str(self.versions.get("torch") or "unknown"),
                worker_version=str(self.versions.get("worker") or self.agent_cfg.agent_version),
            )

            self.metrics.task_outcomes_total.labels(
                executor=self.cfg.id,
                stage=stage_name,
                outcome="done",
                **{"class": "none"},
            ).inc()
            self.emit_msg(
                agent_pb2.AgentMessage(
                    done=agent_pb2.Done(
                        task=task.assign.task,
                        attempt=task.assign.attempt,
                        artifacts=uploaded_refs,
                        timings=numeric_timings,
                        peak_mem_bytes=peak_mem,
                        provenance=provenance,
                    )
                )
            )
        except Exception as e:
            logger.exception("Post-processing/upload error for task %s: %s", task.assign.task, e)
            self.metrics.task_outcomes_total.labels(
                executor=self.cfg.id,
                stage=stage_name,
                outcome="failed",
                **{"class": "upload_failed"},
            ).inc()
            self.emit_msg(
                agent_pb2.AgentMessage(
                    failed=agent_pb2.Failed(
                        task=task.assign.task,
                        attempt=task.assign.attempt,
                        failure_class=agent_pb2.FAILURE_UPLOAD_FAILED,
                        detail=str(e),
                    )
                )
            )
        finally:
            shutil.rmtree(task.scratch_dir, ignore_errors=True)
            self.finishing = False
            if self.release_requested:
                await self.release_worker()
            elif self.is_worker_alive:
                self.set_state(agent_pb2.EXECUTOR_STATE_READY)

    async def _on_worker_error(self, msg: Dict[str, Any]) -> None:
        task = self.active_task
        if not task or msg.get("id") != task.assign.task or task.completed:
            return
        task.completed = True
        self.active_task = None
        stage_name = STAGE_ENUM_TO_NAME.get(task.assign.stage, "unknown")
        detail = str(msg.get("message") or "Worker reported error")
        fclass = (
            agent_pb2.FAILURE_CANCELLED
            if task.cancelled or detail.lower() == "cancelled"
            else agent_pb2.FAILURE_PYTHON_ERROR
        )
        fclass_name = FAILURE_CLASS_TO_NAME.get(fclass, "python_error")
        self.metrics.task_outcomes_total.labels(
            executor=self.cfg.id,
            stage=stage_name,
            outcome="failed",
            **{"class": fclass_name},
        ).inc()
        shutil.rmtree(task.scratch_dir, ignore_errors=True)
        self.emit_msg(
            agent_pb2.AgentMessage(
                failed=agent_pb2.Failed(
                    task=task.assign.task,
                    attempt=task.assign.attempt,
                    failure_class=fclass,
                    detail=detail,
                )
            )
        )
        if self.release_requested:
            await self.release_worker()
        elif self.is_worker_alive:
            self.set_state(agent_pb2.EXECUTOR_STATE_READY)

    def _resolve_url(self, url: str) -> str:
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return f"{self.agent_cfg.backend_http_url}/{url.lstrip('/')}"

    async def _download_inputs(self, task: ActiveTask) -> None:
        if not task.assign.inputs:
            return
        inputs_dir = task.scratch_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        headers = {"Authorization": f"Bearer {self.agent_cfg.agent_token}"}
        async with self.http_client_factory() as client:
            for inp in task.assign.inputs:
                url = self._resolve_url(inp.url)
                res = await client.get(url, headers=headers, timeout=60.0)
                res.raise_for_status()
                dest = inputs_dir / inp.name
                dest.write_bytes(res.content)
                actual_sha, _ = compute_file_sha256(dest)
                if inp.sha256 and actual_sha.lower() != inp.sha256.lower():
                    raise ValueError(
                        f"Input artifact {inp.name} SHA-256 mismatch: expected {inp.sha256}, got {actual_sha}"
                    )
                key = ARTIFACT_PATH_KEYS.get(inp.name, inp.name)
                task.input_paths[key] = str(dest)

    async def _upload_artifact_with_retry(self, raw_url: str, path: Path, sha256: str) -> Tuple[bool, str]:
        """Upload an output with 3 tries and short backoff (1 s, 3 s), §7.4."""
        url = self._resolve_url(raw_url)
        data = path.read_bytes()
        headers = {
            "Authorization": f"Bearer {self.agent_cfg.agent_token}",
            "X-Content-SHA256": sha256,
            "Content-Type": "application/octet-stream",
        }
        delays = [0.0] + list(self.agent_cfg.upload_retry_delays)
        last_err = ""
        async with self.http_client_factory() as client:
            for attempt_idx, delay in enumerate(delays):
                if delay > 0:
                    await asyncio.sleep(delay)
                try:
                    res = await client.put(url, content=data, headers=headers, timeout=60.0)
                    if res.status_code in (200, 201, 204):
                        return True, ""
                    last_err = f"HTTP {res.status_code}: {res.text}"
                    # 409: this attempt was fenced out; 422: the backend rejected the payload.
                    # Neither changes on a retry.
                    if res.status_code in (409, 422):
                        return False, last_err
                except Exception as e:
                    last_err = f"Upload exception on try {attempt_idx + 1}: {e}"
        return False, f"Upload failed after {len(delays)} tries: {last_err}"

    async def run_assigned_task(self, assign: agent_pb2.Assign) -> None:
        scratch_dir = self.agent_cfg.scratch_dir / "tasks" / f"{assign.task}-{assign.attempt}"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
        (scratch_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (scratch_dir / "outputs").mkdir(parents=True, exist_ok=True)

        task = ActiveTask(assign=assign, scratch_dir=scratch_dir)
        for up in assign.uploads:
            task.output_paths[up.name] = scratch_dir / "outputs" / up.name
        # Default output path even if uploads is empty in unit tests
        stage_default_file = {
            agent_pb2.STAGE_ENHANCE: "enhanced.json",
            agent_pb2.STAGE_ENCODE: "embeds.safetensors",
            agent_pb2.STAGE_DENOISE: "latent.safetensors",
            agent_pb2.STAGE_DECODE: "image.png",
        }.get(assign.stage)
        if stage_default_file and stage_default_file not in task.output_paths:
            task.output_paths[stage_default_file] = scratch_dir / "outputs" / stage_default_file

        self.active_task = task
        self.set_state(agent_pb2.EXECUTOR_STATE_BUSY)

        try:
            await self._download_inputs(task)
            await self._dispatch_active_task_to_worker()
        except Exception as e:
            logger.error("Failed preparing inputs for task %s: %s", assign.task, e)
            task.completed = True
            self.active_task = None
            shutil.rmtree(scratch_dir, ignore_errors=True)
            self.emit_msg(
                agent_pb2.AgentMessage(
                    failed=agent_pb2.Failed(
                        task=assign.task,
                        attempt=assign.attempt,
                        failure_class=agent_pb2.FAILURE_PYTHON_ERROR,
                        detail=f"Input download failed: {e}",
                    )
                )
            )
            if self.is_worker_alive:
                self.set_state(agent_pb2.EXECUTOR_STATE_READY)

    async def cancel_task(self, task_id: str, attempt: int) -> None:
        if self.active_task and self.active_task.assign.task == task_id and self.active_task.assign.attempt == attempt:
            self.active_task.cancelled = True
            await self._send_worker_json({"action": "cancel", "id": task_id})

    async def release_worker(self) -> None:
        if (self.active_task and not self.active_task.completed) or self.finishing:
            self.release_requested = True
            return
        self.release_requested = False
        if not self.is_worker_alive:
            if self.state != agent_pb2.EXECUTOR_STATE_IDLE:
                self.set_state(agent_pb2.EXECUTOR_STATE_IDLE)
            return

        self.set_state(agent_pb2.EXECUTOR_STATE_STOPPING)
        await self._send_worker_json({"action": "exit"})

        try:
            if self.process:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            await self.kill_worker(reason="release_timeout")

    async def kill_worker(self, reason: str = "", emit_idle: bool = True) -> None:
        proc = self.process
        if proc and proc.returncode is None:
            logger.info("Killing worker PID %s on %s (%s)", proc.pid, self.cfg.id, reason)
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        self.process = None
        self.pid = 0
        self.metrics.set_executor_pid(self.cfg.id, None)
        self.on_worker_count_change()
        await self.stop_socket_server()
        if emit_idle:
            self.stage = agent_pb2.STAGE_UNSPECIFIED
            self.model = None
            self.set_state(agent_pb2.EXECUTOR_STATE_IDLE)


class AriesAgent:
    """Thin agent daemon managing executors, worker subprocesses, and gRPC control stream."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        http_client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.metrics = AgentMetrics()
        self.http_client_factory = http_client_factory or (lambda: httpx.AsyncClient())
        self.send_queue: asyncio.Queue[agent_pb2.AgentMessage] = asyncio.Queue()
        self.executors: Dict[str, ExecutorRuntime] = {}
        self.running = False
        self.session_id: Optional[str] = None
        self._inhibitor_proc: Optional[ subprocess.Popen ] = None
        self._ram_guard_task: Optional[asyncio.Task] = None
        self._ram_low_samples = 0

        for exec_cfg in self.config.executors:
            self.metrics.init_executor(exec_cfg.id, exec_cfg.device, exec_cfg.mem_total_bytes)
            self.executors[exec_cfg.id] = ExecutorRuntime(
                cfg=exec_cfg,
                agent_cfg=self.config,
                metrics=self.metrics,
                emit_msg=self.enqueue_message,
                on_worker_count_change=self._update_sleep_inhibitor,
                http_client_factory=self.http_client_factory,
            )

    def enqueue_message(self, msg: agent_pb2.AgentMessage) -> None:
        self.send_queue.put_nowait(msg)

    def _update_sleep_inhibitor(self) -> None:
        """Hold a systemd sleep inhibitor while any worker subprocess is alive (§10.4)."""
        if not self.config.enable_sleep_inhibitor:
            return
        any_alive = any(ex.is_worker_alive for ex in self.executors.values())
        if any_alive and self._inhibitor_proc is None:
            inhibit_bin = shutil.which("systemd-inhibit")
            if inhibit_bin:
                try:
                    self._inhibitor_proc = subprocess.Popen(
                        [
                            inhibit_bin,
                            "--what=sleep",
                            "--who=aries-agent",
                            "--why=GPU stage worker active",
                            "sleep",
                            "infinity",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:
                    logger.debug("Could not start systemd-inhibit: %s", e)
        elif not any_alive and self._inhibitor_proc is not None:
            try:
                self._inhibitor_proc.terminate()
                self._inhibitor_proc.wait(timeout=1.0)
            except Exception:
                pass
            self._inhibitor_proc = None

    def check_host_ram(self) -> Optional[str]:
        """One host-RAM guard sample; returns the killed executor's id, if any.

        The kernel's OOM killer does not save this host: when RAM runs out it keeps
        dropping and re-reading the memory-mapped weights and the machine freezes instead
        (HARDWARE.md F9). So when MemAvailable stays under the floor for
        `ram_guard_samples` samples in a row, kill the worker holding the most host RAM.
        Its task fails as `oom_kill` and the backend retries it as usual.
        """
        floor = self.config.ram_guard_floor_bytes
        available = get_host_ram_available_bytes()
        if floor <= 0 or available >= floor:
            self._ram_low_samples = 0
            return None
        self._ram_low_samples += 1
        if self._ram_low_samples < self.config.ram_guard_samples:
            return None
        self._ram_low_samples = 0
        alive = [rt for rt in self.executors.values() if rt.is_worker_alive and not rt.kill_detail]
        if not alive:
            return None
        victim = max(alive, key=lambda rt: rt.worker_ram_bytes())
        victim.kill_for_host_ram(
            f"host RAM guard: {available / 2**30:.1f} GiB available, "
            f"below the {floor / 2**30:.1f} GiB floor"
        )
        return victim.cfg.id

    async def _ram_guard_loop(self) -> None:
        while self.running:
            await asyncio.sleep(self.config.ram_guard_interval_s)
            try:
                self.check_host_ram()
            except Exception:
                logger.exception("Host RAM guard check failed")

    def build_register_message(
        self, pending_results: Optional[List[Tuple[str, int, str]]] = None
    ) -> agent_pb2.Register:
        """Build the `Register` message encoding capabilities and limits (§5.1, §7.2)."""
        models_pb = [
            agent_pb2.ModelRef(id=m.id, rev=m.rev, dtype=m.dtype)
            for m in self.config.models
        ]
        main_models = [m.id for m in self.config.models if "PE" not in m.id and not m.id.endswith("-PE-T2I")]
        pe_models = [m.id for m in self.config.models if "PE" in m.id or m.id.endswith("-PE-T2I")]
        if not main_models and self.config.models:
            main_models = [self.config.models[0].id]
        if not pe_models and self.config.models:
            pe_models = [self.config.models[-1].id]

        executors_pb: List[agent_pb2.Executor] = []
        for ex in self.config.executors:
            decode_modes = [
                DECODE_NAME_TO_MODE[m]
                for m in ex.decode_modes
                if m in DECODE_NAME_TO_MODE
            ] or [agent_pb2.DECODE_MODE_UNTILED]

            # With PE-I2I installed, enhance also takes jobs with references (on that model).
            i2i = self.config.enhancer_i2i_model_id
            stages = [
                agent_pb2.StageCapability(
                    stage=agent_pb2.STAGE_ENHANCE,
                    models=pe_models,
                    max_px=0,
                    max_references=self.config.max_reference_images if i2i else 0,
                    reference_models=[i2i] if i2i else [],
                ),
                agent_pb2.StageCapability(
                    stage=agent_pb2.STAGE_ENCODE,
                    models=main_models,
                    max_px=0,
                    max_references=self.config.max_reference_images,
                ),
                agent_pb2.StageCapability(
                    stage=agent_pb2.STAGE_DENOISE,
                    models=main_models,
                    max_px=ex.max_px,
                    max_references=self.config.max_reference_images,
                ),
                agent_pb2.StageCapability(
                    stage=agent_pb2.STAGE_DECODE,
                    models=main_models,
                    max_px=ex.max_px,
                    modes=decode_modes,
                ),
            ]
            executors_pb.append(
                agent_pb2.Executor(
                    id=ex.id,
                    device=ex.device,
                    mem_total_bytes=ex.mem_total_bytes,
                    stages=stages,
                )
            )

        # Full-size untiled decodes need the RAM floor (tiled ones stay on the card). A split
        # decode holds ~0.1 GB of host RAM, so only plain untiled ones, which spill
        # 13-17 GB, also take the memory-heavy slot.
        split = self.config.decode_split_above_px
        decode_requires = [agent_pb2.Requirement(host_ram_available_bytes=self.config.decode_host_ram_floor_bytes)]
        if split <= 0 or split > self.config.decode_above_px:
            decode_requires.append(
                agent_pb2.Requirement(max_concurrent_per_agent=self.config.decode_max_concurrent_per_agent)
            )
        limits_pb = [
            agent_pb2.Limit(
                stage=agent_pb2.STAGE_DECODE,
                above_px=self.config.decode_above_px,
                requires=decode_requires,
                decode_mode="untiled",
            )
        ]
        if self.config.max_reference_images > self.config.denoise_refs_limit_above:
            limits_pb.append(
                agent_pb2.Limit(
                    stage=agent_pb2.STAGE_DENOISE,
                    above_px=0,
                    above_references=self.config.denoise_refs_limit_above,
                    requires=[
                        agent_pb2.Requirement(host_ram_available_bytes=self.config.decode_host_ram_floor_bytes),
                        agent_pb2.Requirement(max_concurrent_per_agent=1),
                    ],
                )
            )

        running_pb: List[agent_pb2.RunningTask] = []
        for ex_id, rt in self.executors.items():
            if rt.active_task and not rt.active_task.completed:
                running_pb.append(
                    agent_pb2.RunningTask(
                        task=rt.active_task.assign.task,
                        attempt=rt.active_task.assign.attempt,
                        executor=ex_id,
                    )
                )
        # Finished tasks whose Done/Failed is still queued: the backend must not requeue them,
        # the result follows right after Register.
        for task_id, attempt, ex_id in pending_results or []:
            running_pb.append(agent_pb2.RunningTask(task=task_id, attempt=attempt, executor=ex_id))

        return agent_pb2.Register(
            protocol=1,
            agent=self.config.agent_id,
            agent_version=self.config.agent_version,
            backend_kind=self.config.backend_kind,
            metrics_url=self.config.metrics_url,
            host_ram_total_bytes=self.config.get_host_ram_total_bytes(),
            models=models_pb,
            executors=executors_pb,
            limits=limits_pb,
            running=running_pb,
        )

    def validate_assignment(self, assign: agent_pb2.Assign) -> Optional[str]:
        """Validate `Assign` against this agent's own declaration (§6.4).

        Returns a rejection reason string if the assignment contradicts the declaration,
        or None if valid.
        """
        rt = self.executors.get(assign.executor)
        if rt is None:
            return f"unknown executor '{assign.executor}'"

        if assign.stage not in STAGE_ENUM_TO_NAME:
            return f"undeclared stage '{assign.stage}' on executor '{assign.executor}'"

        if rt.active_task is not None and not rt.active_task.completed:
            return f"executor '{assign.executor}' is already busy with task '{rt.active_task.assign.task}'"

        if rt.state != agent_pb2.EXECUTOR_STATE_READY or rt.stage != assign.stage:
            return (
                f"executor '{assign.executor}' is not ready for stage "
                f"{STAGE_ENUM_TO_NAME.get(assign.stage)} (current state={rt.state}, stage={rt.stage})"
            )

        which = assign.WhichOneof("params")
        if assign.stage == agent_pb2.STAGE_DENOISE:
            if which != "denoise":
                return "denoise assignment missing DenoiseParams"
            px = int(assign.denoise.width) * int(assign.denoise.height)
            if rt.cfg.max_px > 0 and px > rt.cfg.max_px:
                return f"denoise size {assign.denoise.width}x{assign.denoise.height} ({px} px) exceeds executor max_px {rt.cfg.max_px}"
            if assign.denoise.references > self.config.max_reference_images:
                return f"{assign.denoise.references} references exceed max_references {self.config.max_reference_images}"
        elif assign.stage == agent_pb2.STAGE_DECODE:
            if which != "decode":
                return "decode assignment missing DecodeParams"
            mode_name = DECODE_MODE_TO_NAME.get(assign.decode.mode, "")
            if mode_name not in rt.cfg.decode_modes:
                return f"decode mode '{mode_name}' not supported by executor '{assign.executor}'"
        elif assign.stage == agent_pb2.STAGE_ENCODE:
            if which != "encode":
                return "encode assignment missing EncodeParams"
            if len(assign.encode.references) > self.config.max_reference_images:
                return f"{len(assign.encode.references)} references exceed max_references {self.config.max_reference_images}"
        elif assign.stage == agent_pb2.STAGE_ENHANCE:
            if which != "enhance":
                return "enhance assignment missing EnhanceParams"
            refs = len(assign.enhance.references)
            if refs > self.config.max_reference_images:
                return f"{refs} references exceed max_references {self.config.max_reference_images}"
            i2i = self.config.enhancer_i2i_model_id
            if refs and (not i2i or rt.model is None or rt.model.id != i2i):
                return "the loaded enhancer does not read reference images"

        return None

    def validate_start_worker(self, cmd: agent_pb2.StartWorker) -> Optional[str]:
        rt = self.executors.get(cmd.executor)
        if rt is None:
            return f"unknown executor '{cmd.executor}'"
        if cmd.stage not in STAGE_ENUM_TO_NAME:
            return f"undeclared stage '{cmd.stage}'"
        if cmd.model and cmd.model.id:
            declared = [m for m in self.config.models if m.id == cmd.model.id and (not cmd.model.rev or m.rev == cmd.model.rev)]
            if not declared:
                return f"model '{cmd.model.id}@{cmd.model.rev}' not present on agent '{self.config.agent_id}'"
        return None

    async def handle_backend_message(self, msg: agent_pb2.BackendMessage) -> bool:
        """Process one message from the backend. Returns False to close the stream."""
        which = msg.WhichOneof("msg")
        if which == "welcome":
            self.session_id = msg.welcome.session
            self.metrics.set_connected(True)
            logger.info("Connected to backend (session=%s, protocol=%d)", self.session_id, msg.welcome.protocol)
            return True

        if which == "start_worker":
            sw = msg.start_worker
            err = self.validate_start_worker(sw)
            if err:
                logger.error("Invalid StartWorker from backend: %s", err)
                return True
            rt = self.executors[sw.executor]
            await rt.start_worker(sw)
            return True

        if which == "assign":
            assign = msg.assign
            rejection = self.validate_assignment(assign)
            if rejection:
                logger.warning("Rejecting assignment (%s attempt %d): %s", assign.task, assign.attempt, rejection)
                self.enqueue_message(
                    agent_pb2.AgentMessage(
                        rejected=agent_pb2.Rejected(
                            task=assign.task,
                            attempt=assign.attempt,
                            reason=rejection,
                        )
                    )
                )
                return True
            rt = self.executors[assign.executor]
            asyncio.create_task(rt.run_assigned_task(assign))
            return True

        if which == "cancel":
            for rt in self.executors.values():
                await rt.cancel_task(msg.cancel.task, msg.cancel.attempt)
            return True

        if which == "release":
            rt = self.executors.get(msg.release.executor)
            if rt:
                asyncio.create_task(rt.release_worker())
            return True

        return True

    def _channel_options(self) -> List[Tuple[str, Any]]:
        """HTTP/2 keepalive options per Architecture V3 §7.1."""
        return [
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 20000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
        ]

    def _create_channel(self) -> grpc.aio.Channel:
        options = self._channel_options()
        if self.config.ca_cert_path and Path(self.config.ca_cert_path).is_file():
            root_certs = Path(self.config.ca_cert_path).read_bytes()
            creds = grpc.ssl_channel_credentials(root_certificates=root_certs)
            return grpc.aio.secure_channel(self.config.backend_grpc_url, creds, options=options)
        return grpc.aio.insecure_channel(self.config.backend_grpc_url, options=options)

    def prepare_stream_queue(self) -> None:
        """Give a new stream its own queue: Register first, then kept results and states.

        A dead stream's request iterator can stay suspended on its queue; on a shared queue it
        would take the next stream's Register (the backend then refuses that stream, again and
        again). So each stream gets a fresh queue, and the old one gets a stop marker.
        """
        # Keep task results produced while disconnected; drop stale state and progress, which
        # are re-announced below.
        old = self.send_queue
        kept: List[agent_pb2.AgentMessage] = []
        while not old.empty():
            try:
                item = old.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not None and item.WhichOneof("msg") in ("done", "failed", "rejected"):
                kept.append(item)
        old.put_nowait(None)  # ends a request iterator still waiting on the old stream
        self.send_queue = asyncio.Queue()

        pending: List[Tuple[str, int, str]] = []
        for item in kept:
            body = getattr(item, item.WhichOneof("msg"))
            if item.WhichOneof("msg") != "rejected":
                ex_id = body.provenance.executor if item.WhichOneof("msg") == "done" else ""
                pending.append((body.task, body.attempt, ex_id))

        self.enqueue_message(agent_pb2.AgentMessage(register=self.build_register_message(pending)))
        for item in kept:
            self.enqueue_message(item)
        for rt in self.executors.values():
            rt.announce_state()

    async def run_once(self) -> None:
        """Open a single gRPC Connect stream, register, and serve messages until disconnect."""
        self.prepare_stream_queue()

        queue = self.send_queue  # this stream's queue; a later stream gets its own

        async def request_iterator():
            while self.running:
                item = await queue.get()
                if item is None:
                    break
                yield item

        async with self._create_channel() as channel:
            stub = agent_pb2_grpc.AgentGatewayStub(channel)
            metadata = (("authorization", f"Bearer {self.config.agent_token}"),)
            call = stub.Connect(request_iterator(), metadata=metadata)
            try:
                async for backend_msg in call:
                    cont = await self.handle_backend_message(backend_msg)
                    if not cont:
                        break
            finally:
                self.metrics.set_connected(False)

    async def run(self) -> None:
        """Main lifecycle loop: start /metrics server and maintain gRPC stream with backoff."""
        self.running = True
        await self.metrics.start_http_server(self.config.metrics_host, self.config.metrics_port)
        if self.config.ram_guard_floor_bytes > 0:
            self._ram_guard_task = asyncio.create_task(self._ram_guard_loop())
        backoff = self.config.reconnect_min_s
        first_connect = True

        try:
            while self.running:
                if not first_connect:
                    self.metrics.agent_reconnects_total.inc()
                first_connect = False

                try:
                    t0 = time.monotonic()
                    await self.run_once()
                    if time.monotonic() - t0 > 10.0:
                        backoff = self.config.reconnect_min_s
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning("Agent stream disconnected (%s); reconnecting in %.1fs", e, backoff)

                if not self.running:
                    break
                await asyncio.sleep(backoff)
                backoff = min(self.config.reconnect_max_s, backoff * 2.0)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        self.running = False
        self.send_queue.put_nowait(None)  # type: ignore[arg-type]
        if self._ram_guard_task is not None:
            self._ram_guard_task.cancel()
            self._ram_guard_task = None
        for rt in self.executors.values():
            await rt.kill_worker(reason="agent_shutdown", emit_idle=False)
        self._update_sleep_inhibitor()
        await self.metrics.stop_http_server()
