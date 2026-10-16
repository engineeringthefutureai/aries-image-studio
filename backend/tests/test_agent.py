"""Tests for aries_agent (Architecture V3 §5-§10, §12)."""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import grpc
import httpx
import pytest

from aries.agent.v1 import agent_pb2, agent_pb2_grpc
from aries_agent.agent import AriesAgent
from aries_agent.config import AgentConfig, ExecutorConfig, ModelConfig
from aries_agent.hardware import classify_worker_crash


def test_agent_never_imports_torch():
    """§10.1 guard: The agent never imports torch (or any GPU runtime)."""
    code = (
        "import sys; "
        "import aries_agent, aries_agent.config, aries_agent.hardware, aries_agent.metrics, aries_agent.agent; "
        "agent = aries_agent.agent.AriesAgent(); "
        "assert 'torch' not in sys.modules, f'torch was imported: {sys.modules.get(\"torch\")}'; "
        "assert 'diffusers' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )
    env = {
        **os.environ,
        "ARIES_AGENT_TOKEN": "test-token",
        "ARIES_EXECUTORS_JSON": '[{"id": "xpu0", "device": "0000:04:00.0"}]',
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"


def _make_test_config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        agent_id="aries",
        agent_version="0.3.0",
        backend_kind="xpu",
        backend_grpc_url="127.0.0.1:50051",
        backend_http_url="http://127.0.0.1:8000",
        agent_token="test-secret-token",
        metrics_host="127.0.0.1",
        metrics_port=0,
        metrics_url="http://aries.lan:9464/metrics",
        scratch_dir=tmp_path / "scratch",
        worker_python=sys.executable,
        models=[
            ModelConfig(id="Qwen/Qwen-Image-2.1", rev="sha256main", dtype="bf16", path="/models/main"),
            ModelConfig(id="Qwen/Qwen-Image-2.1-PE-T2I", rev="sha256pe", dtype="bf16", path="/models/pe"),
        ],
        executors=[
            ExecutorConfig(
                id="xpu0",
                device="0000:04:00.0",
                selector="level_zero:0",
                mem_total_bytes=25668481024,
                max_px=4194304,
                decode_modes=["untiled"],
            ),
            ExecutorConfig(
                id="xpu1",
                device="0000:08:00.0",
                selector="level_zero:1",
                mem_total_bytes=25668481024,
                max_px=4194304,
                decode_modes=["untiled"],
            ),
        ],
        upload_retry_delays=[0.01, 0.02],
        enable_sleep_inhibitor=False,
    )


def test_register_declaration_shape(tmp_path: Path):
    """§5.1: Verify Register declaration matches the Aries two-card B60 capability model."""
    cfg = _make_test_config(tmp_path)
    agent = AriesAgent(config=cfg)
    reg = agent.build_register_message()

    assert reg.protocol == 1
    assert reg.agent == "aries"
    assert reg.backend_kind == "xpu"
    assert len(reg.models) == 2
    assert len(reg.executors) == 2
    assert reg.executors[0].id == "xpu0"
    assert reg.executors[0].device == "0000:04:00.0"
    assert reg.executors[0].mem_total_bytes == 25668481024

    stage_map = {sc.stage: sc for sc in reg.executors[0].stages}
    assert stage_map[agent_pb2.STAGE_ENHANCE].models == ["Qwen/Qwen-Image-2.1-PE-T2I"]
    assert stage_map[agent_pb2.STAGE_ENCODE].models == ["Qwen/Qwen-Image-2.1"]
    assert stage_map[agent_pb2.STAGE_DENOISE].max_px == 4194304
    assert stage_map[agent_pb2.STAGE_DECODE].max_px == 4194304
    assert list(stage_map[agent_pb2.STAGE_DECODE].modes) == [agent_pb2.DECODE_MODE_UNTILED]

    assert len(reg.limits) == 1
    lim = reg.limits[0]
    assert lim.stage == agent_pb2.STAGE_DECODE
    assert lim.above_px == 1048576
    kinds = {r.WhichOneof("kind"): getattr(r, r.WhichOneof("kind")) for r in lim.requires}
    assert kinds["host_ram_available_bytes"] == 6442450944
    assert kinds["max_concurrent_per_agent"] == 1


@pytest.mark.asyncio
async def test_agent_metrics_http_endpoint(tmp_path: Path):
    """§9: Verify agent /metrics endpoint exposes all required Prometheus series."""
    cfg = _make_test_config(tmp_path)
    agent = AriesAgent(config=cfg)
    port = await agent.metrics.start_http_server("127.0.0.1", 0)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/metrics")
            assert resp.status_code == 200
            text = resp.text
            assert "aries_agent_connected" in text
            assert "aries_agent_reconnects_total" in text
            assert 'aries_executor_state{executor="xpu0",state="idle"} 1.0' in text
            assert 'aries_executor_mem_total_bytes{device="0000:04:00.0",executor="xpu0"} 2.5668481024e+10' in text
            assert "aries_executor_mem_overflow_bytes" in text
            assert "aries_host_ram_available_bytes" in text
    finally:
        await agent.metrics.stop_http_server()


@pytest.mark.asyncio
async def test_assignment_validation_rejections(tmp_path: Path):
    """§6.4: Agent rejects assignments that contradict its declaration."""
    cfg = _make_test_config(tmp_path)
    agent = AriesAgent(config=cfg)

    # 1. Unknown executor
    await agent.handle_backend_message(
        agent_pb2.BackendMessage(
            assign=agent_pb2.Assign(
                task="t-1",
                attempt=1,
                executor="xpu99",
                stage=agent_pb2.STAGE_ENCODE,
                encode=agent_pb2.EncodeParams(prompt="hi", prompt_source="original"),
            )
        )
    )
    msg1 = await agent.send_queue.get()
    assert msg1.WhichOneof("msg") == "rejected"
    assert "unknown executor" in msg1.rejected.reason

    # 2. Executor is IDLE (not READY for stage)
    await agent.handle_backend_message(
        agent_pb2.BackendMessage(
            assign=agent_pb2.Assign(
                task="t-2",
                attempt=1,
                executor="xpu0",
                stage=agent_pb2.STAGE_ENCODE,
                encode=agent_pb2.EncodeParams(prompt="hi", prompt_source="original"),
            )
        )
    )
    msg2 = await agent.send_queue.get()
    assert msg2.WhichOneof("msg") == "rejected"
    assert "not ready" in msg2.rejected.reason

    # 3. Put xpu0 in READY(STAGE_DENOISE) and test exceeding max_px (4096x4096)
    rt0 = agent.executors["xpu0"]
    rt0.stage = agent_pb2.STAGE_DENOISE
    rt0.state = agent_pb2.EXECUTOR_STATE_READY
    await agent.handle_backend_message(
        agent_pb2.BackendMessage(
            assign=agent_pb2.Assign(
                task="t-3",
                attempt=1,
                executor="xpu0",
                stage=agent_pb2.STAGE_DENOISE,
                denoise=agent_pb2.DenoiseParams(width=4096, height=4096, steps=20, seed=1),
            )
        )
    )
    msg3 = await agent.send_queue.get()
    assert msg3.WhichOneof("msg") == "rejected"
    assert "exceeds executor max_px" in msg3.rejected.reason


@pytest.mark.asyncio
async def test_pci_device_mismatch_kills_worker(tmp_path: Path):
    """§10.3: Worker hello reporting multiple cards or wrong PCI address is killed with device_mismatch."""
    cfg = _make_test_config(tmp_path)
    agent = AriesAgent(config=cfg)
    rt = agent.executors["xpu0"]
    rt.stage = agent_pb2.STAGE_ENCODE
    await rt.start_socket_server()

    # Connect a fake worker that reports seeing BOTH cards (the 11 Oct bug)
    reader, writer = await asyncio.open_unix_connection(str(rt.socket_path))
    hello = {
        "event": "hello",
        "stage": "encode",
        "card": 0,
        "pid": 12345,
        "pci": "0000:04:00.0,0000:08:00.0",
        "devices": ["0000:04:00.0", "0000:08:00.0"],
        "device_count": 2,
        "load_s": 1.2,
    }
    writer.write((json.dumps(hello) + "\n").encode("utf-8"))
    await writer.drain()

    # Wait for agent to process hello and emit FAILED -> IDLE
    msg_failed = await asyncio.wait_for(agent.send_queue.get(), timeout=3.0)
    msg_idle = await asyncio.wait_for(agent.send_queue.get(), timeout=3.0)

    assert msg_failed.executor_state.state == agent_pb2.EXECUTOR_STATE_FAILED
    assert "device_mismatch" in msg_failed.executor_state.detail
    assert msg_idle.executor_state.state == agent_pb2.EXECUTOR_STATE_IDLE

    writer.close()
    await writer.wait_closed()
    await rt.stop_socket_server()


@pytest.mark.asyncio
async def test_end_to_end_task_and_upload_retry(tmp_path: Path):
    """§7.3, §10.3, §10.4: Worker hello -> Assign -> input download -> run -> upload -> Done + Provenance."""
    cfg = _make_test_config(tmp_path)
    input_bytes = b"safetensors-latent-bytes"
    input_sha = hashlib.sha256(input_bytes).hexdigest()
    uploaded: Dict[str, bytes] = {}
    upload_attempts = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal upload_attempts
        if request.method == "GET" and request.url.path == "/agent/v1/artifacts/in/latent.safetensors":
            return httpx.Response(200, content=input_bytes)
        if request.method == "PUT" and request.url.path == "/agent/v1/artifacts/out/image.png":
            upload_attempts += 1
            if upload_attempts == 1:
                # Fail first try to verify backoff retry (§7.3)
                return httpx.Response(503, text="temporary error")
            uploaded["image.png"] = request.content
            return httpx.Response(200, text="ok")
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    agent = AriesAgent(config=cfg, http_client_factory=lambda: httpx.AsyncClient(transport=transport))
    rt = agent.executors["xpu0"]
    rt.stage = agent_pb2.STAGE_DECODE
    rt.model = agent_pb2.ModelRef(id="Qwen/Qwen-Image-2.1", rev="sha256main", dtype="bf16")
    await rt.start_socket_server()

    # Fake worker process over Unix socket
    async def fake_worker():
        r, w = await asyncio.open_unix_connection(str(rt.socket_path))
        w.write(
            (
                json.dumps(
                    {
                        "event": "hello",
                        "stage": "decode",
                        "card": 0,
                        "pid": 4321,
                        "pci": "0000:04:00.0",
                        "devices": ["0000:04:00.0"],
                        "device_count": 1,
                        "versions": {"torch": "2.6.0+xpu", "worker": "0.3.0"},
                        "load_s": 0.42,
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        await w.drain()

        # Claim job: the agent answers "none" at once and pushes the job when it is
        # assigned, as the real worker expects while idling.
        w.write((json.dumps({"event": "claim"}) + "\n").encode("utf-8"))
        await w.drain()
        while True:
            job_msg = json.loads((await r.readline()).decode("utf-8"))
            if job_msg["action"] != "none":
                break
        assert job_msg["action"] == "job"
        assert job_msg["id"] == "task-dec-1"

        # Verify input file was downloaded into scratch dir
        in_file = Path(job_msg["paths"]["latent_safetensors"])
        assert in_file.read_bytes() == input_bytes

        # Write output image.png
        out_file = Path(job_msg["paths"]["image_png"])
        out_file.write_bytes(b"PNG-IMAGE-OUTPUT")

        w.write(
            (
                json.dumps(
                    {
                        "event": "done",
                        "id": "task-dec-1",
                        "timings": {"decode_s": 5.1, "save_s": 0.2},
                        "peak_vram": 25760000000,
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        await w.drain()
        w.close()
        await w.wait_closed()

    worker_task = asyncio.create_task(fake_worker())

    # 1. Wait for READY state from hello
    ready_msg = await asyncio.wait_for(agent.send_queue.get(), timeout=3.0)
    assert ready_msg.executor_state.state == agent_pb2.EXECUTOR_STATE_READY
    assert ready_msg.executor_state.pid == 4321
    assert abs(ready_msg.executor_state.load_s - 0.42) < 1e-6

    # 2. Send Assign
    assign_msg = agent_pb2.BackendMessage(
        assign=agent_pb2.Assign(
            task="task-dec-1",
            attempt=1,
            executor="xpu0",
            stage=agent_pb2.STAGE_DECODE,
            inputs=[
                agent_pb2.ArtifactRef(
                    name="latent.safetensors",
                    url="/agent/v1/artifacts/in/latent.safetensors",
                    sha256=input_sha,
                    bytes=len(input_bytes),
                )
            ],
            uploads=[
                agent_pb2.ArtifactRef(
                    name="image.png",
                    url="/agent/v1/artifacts/out/image.png",
                )
            ],
            decode=agent_pb2.DecodeParams(mode=agent_pb2.DECODE_MODE_UNTILED),
        )
    )
    await agent.handle_backend_message(assign_msg)

    busy_msg = await asyncio.wait_for(agent.send_queue.get(), timeout=3.0)
    assert busy_msg.executor_state.state == agent_pb2.EXECUTOR_STATE_BUSY

    done_msg = await asyncio.wait_for(agent.send_queue.get(), timeout=3.0)
    assert done_msg.WhichOneof("msg") == "done"
    assert done_msg.done.task == "task-dec-1"
    assert done_msg.done.attempt == 1
    assert done_msg.done.peak_mem_bytes == 25760000000
    assert done_msg.done.provenance.agent == "aries"
    assert done_msg.done.provenance.executor == "xpu0"
    assert done_msg.done.provenance.device == "0000:04:00.0"
    assert done_msg.done.provenance.backend_kind == "xpu"
    assert done_msg.done.provenance.model.id == "Qwen/Qwen-Image-2.1"
    assert done_msg.done.provenance.model.rev == "sha256main"
    assert done_msg.done.provenance.torch_version == "2.6.0+xpu"
    assert upload_attempts == 2
    assert uploaded["image.png"] == b"PNG-IMAGE-OUTPUT"

    # Scratch dir for the task must be deleted after Done (§10.4)
    scratch_task_dir = cfg.scratch_dir / "tasks" / "task-dec-1-1"
    assert not scratch_task_dir.exists()

    await worker_task
    await rt.stop_socket_server()


def test_crash_classification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """§4, §12: Verify crash classification from exit code, sysfs devcoredump, and kernel log."""
    assert classify_worker_crash(137, 9, "0000:04:00.0", "") == agent_pb2.FAILURE_OOM_KILL
    assert (
        classify_worker_crash(1, 0, "0000:04:00.0", "xe 0000:04:00.0: Fault response: Unsuccessful -EINVAL")
        == agent_pb2.FAILURE_GPU_FAULT
    )
    assert (
        classify_worker_crash(1, 0, "0000:04:00.0", "Engine reset ccs Timedout job")
        == agent_pb2.FAILURE_ENGINE_RESET
    )
    assert classify_worker_crash(1, 0, "0000:04:00.0", "", "cancelled") == agent_pb2.FAILURE_CANCELLED

    # Simulate active devcoredump in sysfs
    pci_dir = tmp_path / "0000:04:00.0" / "devcoredump"
    pci_dir.mkdir(parents=True)
    monkeypatch.setenv("ARIES_SYSFS_PCI_DEVICES_PATH", str(tmp_path))
    assert classify_worker_crash(1, 0, "0000:04:00.0", "") == agent_pb2.FAILURE_WEDGED


@pytest.mark.asyncio
async def test_start_worker_idempotency(tmp_path: Path):
    """Verify StartWorker is idempotent and does not kill an alive worker starting for the same stage."""
    cfg = _make_test_config(tmp_path)
    agent = AriesAgent(config=cfg)
    rt = agent.executors["xpu0"]

    sw_cmd = agent_pb2.StartWorker(
        executor="xpu0",
        stage=agent_pb2.STAGE_ENCODE,
        model=agent_pb2.ModelRef(id="Qwen/Qwen-Image-2.1"),
    )

    # Put rt in STARTING state with a mock process alive
    class MockProcess:
        def __init__(self):
            self.pid = 9999
            self.returncode = None
            self.killed = False

        def kill(self):
            self.killed = True

        async def wait(self):
            self.returncode = -9
            return -9

    proc = MockProcess()
    rt.process = proc  # type: ignore
    rt.pid = 9999
    rt.stage = agent_pb2.STAGE_ENCODE
    rt.model = agent_pb2.ModelRef(id="Qwen/Qwen-Image-2.1")
    rt.state = agent_pb2.EXECUTOR_STATE_STARTING

    # Call start_worker with the exact same stage & model
    await rt.start_worker(sw_cmd)

    # Process must NOT have been killed, state remains STARTING
    assert not proc.killed
    assert rt.process is proc
    assert rt.pid == 9999
    assert rt.state == agent_pb2.EXECUTOR_STATE_STARTING



def test_encode_payload_carries_true_cfg_scale(tmp_path: Path):
    """Encode must know the guidance scale to write negative embeddings when it is > 1."""
    agent = AriesAgent(config=_make_test_config(tmp_path))
    rt = agent.executors["xpu0"]
    from aries_agent.agent import ActiveTask

    assign = agent_pb2.Assign(
        task="t1", attempt=1, executor="xpu0", stage=agent_pb2.STAGE_ENCODE,
        encode=agent_pb2.EncodeParams(prompt="p", prompt_source="original", true_cfg_scale=4.0),
    )
    payload = rt._build_worker_job_payload(ActiveTask(assign=assign, scratch_dir=tmp_path))
    assert payload["params"]["true_cfg_scale"] == 4.0


@pytest.mark.asyncio
async def test_enhance_progress_forwards_sub_stage_and_stats(tmp_path: Path):
    agent = AriesAgent(config=_make_test_config(tmp_path))
    rt = agent.executors["xpu0"]
    from aries_agent.agent import ActiveTask

    assign = agent_pb2.Assign(task="t1", attempt=2, executor="xpu0", stage=agent_pb2.STAGE_ENHANCE)
    rt.active_task = ActiveTask(assign=assign, scratch_dir=tmp_path)
    await rt._on_worker_progress({
        "id": "t1",
        "sub_stage": "thinking",
        "enhance_stats": {"generated_tokens": 300, "max_new_tokens": 2048, "generation_tps": 20.0, "flag": True},
    })
    prog = (await agent.send_queue.get()).progress
    assert prog.sub_stage == "thinking"
    assert dict(prog.stats) == {"generated_tokens": 300.0, "max_new_tokens": 2048.0, "generation_tps": 20.0}
    assert (prog.step, prog.steps) == (300, 2048)


def test_reconnect_keeps_results_and_reports_them_running(tmp_path: Path):
    """Done/Failed queued while disconnected survive the reconnect; progress does not."""
    agent = AriesAgent(config=_make_test_config(tmp_path))
    done = agent_pb2.AgentMessage(done=agent_pb2.Done(
        task="t-done", attempt=1, provenance=agent_pb2.Provenance(executor="xpu1")))
    agent.enqueue_message(agent_pb2.AgentMessage(progress=agent_pb2.Progress(task="t-done", attempt=1)))
    agent.enqueue_message(done)
    agent.enqueue_message(agent_pb2.AgentMessage(failed=agent_pb2.Failed(task="t-fail", attempt=2)))

    agent.prepare_stream_queue()

    kinds = []
    while not agent.send_queue.empty():
        kinds.append(agent.send_queue.get_nowait())
    assert [m.WhichOneof("msg") for m in kinds] == ["register", "done", "failed"]
    running = {(r.task, r.attempt, r.executor) for r in kinds[0].register.running}
    assert running == {("t-done", 1, "xpu1"), ("t-fail", 2, "")}


@pytest.mark.asyncio
async def test_duplicate_start_worker_reannounces_state(tmp_path: Path):
    """The backend marks the executor `starting` when it sends StartWorker; an ignored
    duplicate must still tell it the worker's real state."""
    agent = AriesAgent(config=_make_test_config(tmp_path))
    rt = agent.executors["xpu0"]

    class Alive:
        pid = 1
        returncode = None

    rt.process = Alive()  # type: ignore[assignment]
    rt.stage = agent_pb2.STAGE_ENCODE
    rt.model = agent_pb2.ModelRef(id="Qwen/Qwen-Image-2.1")
    rt.state = agent_pb2.EXECUTOR_STATE_READY
    await rt.start_worker(agent_pb2.StartWorker(
        executor="xpu0", stage=agent_pb2.STAGE_ENCODE, model=agent_pb2.ModelRef(id="Qwen/Qwen-Image-2.1")))
    msg = (await agent.send_queue.get()).executor_state
    assert (msg.executor, msg.state) == ("xpu0", agent_pb2.EXECUTOR_STATE_READY)


@pytest.mark.asyncio
async def test_idle_claims_are_answered_at_once(tmp_path: Path):
    """An idle worker claims every second; blocking on a claim queued its later events
    (done, error) behind a growing backlog."""
    agent = AriesAgent(config=_make_test_config(tmp_path))
    rt = agent.executors["xpu0"]
    await rt.start_socket_server()
    try:
        r, w = await asyncio.open_unix_connection(str(rt.socket_path))
        for _ in range(5):
            w.write(b'{"event": "claim"}\n')
        await w.drain()
        replies = [json.loads(await asyncio.wait_for(r.readline(), timeout=0.5)) for _ in range(5)]
        assert [m["action"] for m in replies] == ["none"] * 5
        w.close()
    finally:
        await rt.stop_socket_server()


@pytest.mark.asyncio
async def test_dead_stream_iterator_cannot_take_the_next_register(tmp_path: Path):
    """A dead stream's request iterator stays suspended on its queue. It must not receive
    the next stream's Register (the backend refuses a stream that does not start with it)."""
    agent = AriesAgent(config=_make_test_config(tmp_path))
    old_queue = agent.send_queue
    zombie = asyncio.create_task(old_queue.get())  # the old stream's iterator, still waiting
    await asyncio.sleep(0)

    agent.prepare_stream_queue()

    assert await asyncio.wait_for(zombie, timeout=1.0) is None  # told to stop
    first = agent.send_queue.get_nowait()
    assert first.WhichOneof("msg") == "register"
