"""Tests for aries_agent (Architecture V3 §5-§10, §12)."""
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from aries.agent.v1 import agent_pb2
from aries_agent.agent import ActiveTask, AriesAgent
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

    assert stage_map[agent_pb2.STAGE_ENCODE].max_references == 4
    assert stage_map[agent_pb2.STAGE_DENOISE].max_references == 4
    assert stage_map[agent_pb2.STAGE_DECODE].max_references == 0

    # Full-size untiled decodes need the RAM floor (tiled ones stay on the card); split (the
    # default) they stay on the card too, so they do not take the memory-heavy slot.
    assert len(reg.limits) == 2
    lim = reg.limits[0]
    assert (lim.stage, lim.above_px, lim.HasField("above_references")) == (agent_pb2.STAGE_DECODE, 3500000, False)
    assert lim.decode_mode == "untiled"
    kinds = {r.WhichOneof("kind"): getattr(r, r.WhichOneof("kind")) for r in lim.requires}
    assert kinds == {"host_ram_available_bytes": 6442450944}
    # Denoise with more than 2 references spills into host RAM (docs/HARDWARE.md §1.1).
    refs = reg.limits[1]
    assert (refs.stage, refs.above_px, refs.above_references) == (agent_pb2.STAGE_DENOISE, 0, 2)
    kinds = {r.WhichOneof("kind"): getattr(r, r.WhichOneof("kind")) for r in refs.requires}
    assert kinds == {"host_ram_available_bytes": 6442450944, "max_concurrent_per_agent": 1}

    # Plain untiled decode (split off) spills: it also takes the shared slot.
    cfg.decode_split_above_px = 0
    reg = agent.build_register_message()
    kinds = {r.WhichOneof("kind"): getattr(r, r.WhichOneof("kind")) for r in reg.limits[0].requires}
    assert kinds == {"host_ram_available_bytes": 6442450944, "max_concurrent_per_agent": 1}

    # A cap at or below the threshold never spills: no denoise limit to declare.
    cfg.max_reference_images = 2
    assert len(agent.build_register_message().limits) == 1


def test_reference_params_reach_the_worker(tmp_path: Path):
    """Encode gets the references' hashes and resolution, denoise their count; the files
    themselves arrive under their artifact names."""
    agent = AriesAgent(config=_make_test_config(tmp_path))
    rt = agent.executors["xpu0"]
    refs = [agent_pb2.Reference(sha256="a" * 64), agent_pb2.Reference(sha256="b" * 64)]
    enc = agent_pb2.Assign(
        task="t-enc", attempt=1, executor="xpu0", stage=agent_pb2.STAGE_ENCODE,
        encode=agent_pb2.EncodeParams(prompt="p", references=refs, reference_res=1024),
    )
    task = ActiveTask(assign=enc, scratch_dir=tmp_path)
    task.input_paths["ref0.image"] = "/in/ref0.image"
    task.output_paths["ref1.safetensors"] = tmp_path / "ref1.safetensors"
    payload = rt._build_worker_job_payload(task)
    assert payload["params"]["references"] == ["a" * 64, "b" * 64]
    assert payload["params"]["reference_res"] == 1024
    assert payload["paths"]["ref0.image"] == "/in/ref0.image"
    assert payload["paths"]["ref1.safetensors"] == str(tmp_path / "ref1.safetensors")

    den = agent_pb2.Assign(
        task="t-den", attempt=1, executor="xpu0", stage=agent_pb2.STAGE_DENOISE,
        denoise=agent_pb2.DenoiseParams(width=1024, height=1024, steps=20, seed=1, references=2),
    )
    assert rt._build_worker_job_payload(ActiveTask(assign=den, scratch_dir=tmp_path))["params"]["references"] == 2

    enh = agent_pb2.Assign(
        task="t-enh", attempt=1, executor="xpu0", stage=agent_pb2.STAGE_ENHANCE,
        enhance=agent_pb2.EnhanceParams(prompt="p", max_new_tokens=4096),
    )
    assert rt._build_worker_job_payload(ActiveTask(assign=enh, scratch_dir=tmp_path))["params"]["max_new_tokens"] == 4096
    enh.enhance.max_new_tokens = 0  # unset: the worker's default
    assert "max_new_tokens" not in rt._build_worker_job_payload(ActiveTask(assign=enh, scratch_dir=tmp_path))["params"]

    # More references than declared: rejected like any other contradiction.
    rt.state, rt.stage = agent_pb2.EXECUTOR_STATE_READY, agent_pb2.STAGE_DENOISE
    den.denoise.references = 5
    assert "exceed max_references 4" in agent.validate_assignment(den)
    rt.stage = agent_pb2.STAGE_ENCODE
    enc.encode.references.extend(refs * 2)
    assert "exceed max_references 4" in agent.validate_assignment(enc)


I2I_ID = "Qwen/Qwen-Image-2.1-PE-I2I"


def _make_i2i_config(tmp_path: Path) -> AgentConfig:
    """The test agent with the image-aware enhancer installed (docs/ENHANCER_I2I.md)."""
    cfg = _make_test_config(tmp_path)
    cfg.models.append(ModelConfig(id=I2I_ID, rev="sha256i2i", dtype="int4", path="/models/pe-i2i"))
    cfg.enhancer_i2i_model_id = I2I_ID
    return cfg


def test_enhance_declares_the_image_enhancer(tmp_path: Path):
    enhance = {sc.stage: sc for sc in AriesAgent(config=_make_test_config(tmp_path)).build_register_message().executors[0].stages}[agent_pb2.STAGE_ENHANCE]
    assert (enhance.max_references, list(enhance.reference_models)) == (0, [])

    reg = AriesAgent(config=_make_i2i_config(tmp_path)).build_register_message()
    assert [m.id for m in reg.models][-1] == I2I_ID
    for ex in reg.executors:
        enhance = {sc.stage: sc for sc in ex.stages}[agent_pb2.STAGE_ENHANCE]
        assert list(enhance.models) == ["Qwen/Qwen-Image-2.1-PE-T2I", I2I_ID]
        assert list(enhance.reference_models) == [I2I_ID]
        assert enhance.max_references == 4


def test_config_reads_the_image_enhancer_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARIES_AGENT_TOKEN", "t")
    monkeypatch.setenv("ARIES_EXECUTORS_JSON", '[{"id": "xpu0", "device": "0000:04:00.0"}]')
    monkeypatch.delenv("ARIES_ENHANCER_I2I_MODEL_PATH", raising=False)
    assert AgentConfig.from_env().enhancer_i2i_model_id is None
    monkeypatch.setenv("ARIES_ENHANCER_I2I_MODEL_PATH", str(tmp_path / "not-downloaded"))
    assert AgentConfig.from_env().enhancer_i2i_model_id is None
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "torchao"}}))
    monkeypatch.setenv("ARIES_ENHANCER_I2I_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("ARIES_ENHANCER_I2I_MODEL_DTYPE", "int4")
    cfg = AgentConfig.from_env()
    assert cfg.enhancer_i2i_model_id == I2I_ID
    i2i = cfg.models[-1]
    assert (i2i.id, i2i.path, i2i.dtype) == (I2I_ID, str(tmp_path), "int4")


def test_config_takes_up_to_the_models_reference_limit(monkeypatch):
    monkeypatch.setenv("ARIES_AGENT_TOKEN", "t")
    monkeypatch.setenv("ARIES_EXECUTORS_JSON", '[{"id": "xpu0", "device": "0000:04:00.0"}]')
    monkeypatch.setenv("ARIES_MAX_REFERENCE_IMAGES", "10")
    assert AgentConfig.from_env().max_reference_images == 10
    monkeypatch.setenv("ARIES_MAX_REFERENCE_IMAGES", "11")
    with pytest.raises(ValueError, match="0 to 10 reference images"):
        AgentConfig.from_env()


def test_enhance_references_reach_the_image_enhancer(tmp_path: Path):
    agent = AriesAgent(config=_make_i2i_config(tmp_path))
    rt = agent.executors["xpu0"]
    refs = [agent_pb2.Reference(sha256="a" * 64), agent_pb2.Reference(sha256="b" * 64)]
    enh = agent_pb2.Assign(
        task="t-enh", attempt=1, executor="xpu0", stage=agent_pb2.STAGE_ENHANCE,
        enhance=agent_pb2.EnhanceParams(prompt="p", references=refs),
    )
    task = ActiveTask(assign=enh, scratch_dir=tmp_path)
    task.input_paths["ref0.image"] = "/in/ref0.image"
    payload = rt._build_worker_job_payload(task)
    assert payload["params"]["references"] == ["a" * 64, "b" * 64]
    assert payload["paths"]["ref0.image"] == "/in/ref0.image"

    rt.state, rt.stage = agent_pb2.EXECUTOR_STATE_READY, agent_pb2.STAGE_ENHANCE
    rt.model = agent_pb2.ModelRef(id=I2I_ID)
    assert agent.validate_assignment(enh) is None
    # A text enhancer must not get references.
    rt.model = agent_pb2.ModelRef(id="Qwen/Qwen-Image-2.1-PE-T2I")
    assert "does not read reference images" in agent.validate_assignment(enh)
    rt.model = agent_pb2.ModelRef(id=I2I_ID)
    enh.enhance.references.extend(refs * 2)
    assert "exceed max_references 4" in agent.validate_assignment(enh)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id,path,dtype", [
    (None, "/models/pe", "bf16"),
    ("Qwen/Qwen-Image-2.1-PE-T2I", "/models/pe", "bf16"),
    (I2I_ID, "/models/pe-i2i", "int4"),
])
async def test_enhance_worker_loads_the_requested_enhancer(tmp_path: Path, monkeypatch, model_id, path, dtype):
    """StartWorker.model picks the enhancer; without one the text enhancer is the default."""
    agent = AriesAgent(config=_make_i2i_config(tmp_path))
    rt = agent.executors["xpu0"]
    spawned = {}

    class Proc:
        pid, returncode, stdout = 4242, None, None

    async def fake_exec(*args, env, **kwargs):
        spawned["env"] = env
        spawned["preexec_fn"] = kwargs.get("preexec_fn")
        return Proc()

    async def no_socket():
        return None

    async def no_monitor(proc, stage):
        return None

    monkeypatch.setattr("aries_agent.agent.asyncio.create_subprocess_exec", fake_exec)
    monkeypatch.setattr(rt, "start_socket_server", no_socket)
    monkeypatch.setattr(rt, "_monitor_process", no_monitor)
    cmd = agent_pb2.StartWorker(executor="xpu0", stage=agent_pb2.STAGE_ENHANCE)
    if model_id:
        cmd.model.CopyFrom(agent_pb2.ModelRef(id=model_id))
    await rt.start_worker(cmd)
    env = spawned["env"]
    assert (env["ENHANCER_MODEL_PATH"], env["MODEL_DTYPE"]) == (path, dtype)
    assert spawned["preexec_fn"] is not None  # the worker's oom_score_adj (test below)
    assert env["MODEL_PATH"] == "/models/main"


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
async def test_agent_health_endpoints(tmp_path: Path):
    """/healthz answers while the agent runs; /readyz only once Studio has welcomed it."""
    cfg = _make_test_config(tmp_path)
    agent = AriesAgent(config=cfg)
    port = await agent.metrics.start_http_server("127.0.0.1", 0)
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get(f"http://127.0.0.1:{port}/healthz")
            assert health.status_code == 200
            body = health.json()
            assert body["status"] == "ok" and body["connected"] is False
            assert body["executors"]["xpu0"] == "idle"
            assert (await client.get(f"http://127.0.0.1:{port}/readyz")).status_code == 503

            welcome = agent_pb2.BackendMessage(welcome=agent_pb2.Welcome(session="s1", protocol=1))
            assert await agent.handle_backend_message(welcome)
            ready = await client.get(f"http://127.0.0.1:{port}/readyz?probe=1")
            assert ready.status_code == 200 and ready.json()["connected"] is True
            assert "aries_agent_connected 1.0" in (await client.get(f"http://127.0.0.1:{port}/metrics")).text

            agent.metrics.set_connected(False)
            assert (await client.get(f"http://127.0.0.1:{port}/readyz")).status_code == 503
            assert (await client.get(f"http://127.0.0.1:{port}/healthz")).status_code == 200
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


@pytest.mark.asyncio
async def test_cancel_during_input_download_never_reaches_the_worker(tmp_path: Path):
    """A Cancel that arrives while the inputs download is reported as cancelled; the job was
    otherwise sent to the worker afterwards and ran to the end."""
    cfg = _make_test_config(tmp_path)
    started, release = asyncio.Event(), asyncio.Event()

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(200, content=b"latent")

    transport = httpx.MockTransport(mock_handler)
    agent = AriesAgent(config=cfg, http_client_factory=lambda: httpx.AsyncClient(transport=transport))
    rt = agent.executors["xpu0"]
    sent: List[Dict[str, Any]] = []

    async def record(msg: Dict[str, Any]) -> None:
        sent.append(msg)

    rt._send_worker_json = record  # type: ignore[method-assign]
    assign = agent_pb2.Assign(
        task="task-dec-2",
        attempt=1,
        executor="xpu0",
        stage=agent_pb2.STAGE_DECODE,
        inputs=[agent_pb2.ArtifactRef(name="latent.safetensors", url="/agent/v1/artifacts/in/latent.safetensors")],
    )
    run = asyncio.create_task(rt.run_assigned_task(assign))
    await asyncio.wait_for(started.wait(), timeout=3.0)
    await rt.cancel_task("task-dec-2", 1)
    release.set()
    await asyncio.wait_for(run, timeout=3.0)

    msgs = []
    while not agent.send_queue.empty():
        msgs.append(agent.send_queue.get_nowait())
    failed = [m.failed for m in msgs if m.WhichOneof("msg") == "failed"]
    assert len(failed) == 1
    assert failed[0].failure_class == agent_pb2.FAILURE_CANCELLED
    assert not any(m.get("action") == "job" for m in sent), sent
    assert rt.active_task is None


@pytest.mark.asyncio
async def test_worker_claim_during_input_download_waits_for_the_inputs(tmp_path: Path):
    """Seen live with reference images: an idle worker's claim arrived while the embeds were
    still downloading, got the job without them, and failed with KeyError
    'embeds_safetensors'."""
    cfg = _make_test_config(tmp_path)
    started, release = asyncio.Event(), asyncio.Event()

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(200, content=b"embeds")

    transport = httpx.MockTransport(mock_handler)
    agent = AriesAgent(config=cfg, http_client_factory=lambda: httpx.AsyncClient(transport=transport))
    rt = agent.executors["xpu0"]
    sent: List[Dict[str, Any]] = []

    async def record(msg: Dict[str, Any]) -> None:
        sent.append(msg)

    rt._send_worker_json = record  # type: ignore[method-assign]
    assign = agent_pb2.Assign(
        task="task-den-1",
        attempt=1,
        executor="xpu0",
        stage=agent_pb2.STAGE_DENOISE,
        inputs=[agent_pb2.ArtifactRef(name="embeds.safetensors", url="/agent/v1/artifacts/e/1/embeds.safetensors")],
        denoise=agent_pb2.DenoiseParams(width=1024, height=1024, steps=4, seed=1),
    )
    run = asyncio.create_task(rt.run_assigned_task(assign))
    await asyncio.wait_for(started.wait(), timeout=3.0)
    await rt._on_worker_claim()
    assert [m["action"] for m in sent] == ["none"]

    release.set()
    await asyncio.wait_for(run, timeout=3.0)
    jobs = [m for m in sent if m["action"] == "job"]
    assert len(jobs) == 1
    assert Path(jobs[0]["paths"]["embeds_safetensors"]).read_bytes() == b"embeds"


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


def test_enhancer_dtype_from_quantized_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A torchao int4 enhancer is declared as int4 while the main model stays bf16."""
    main, pe = tmp_path / "main", tmp_path / "pe"
    main.mkdir()
    pe.mkdir()
    (main / "config.json").write_text(json.dumps({"model_type": "qwen"}))
    (pe / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
                "quantization_config": {
                    "quant_method": "torchao",
                    "quant_type": {"default": {"_type": "Int4WeightOnlyConfig", "_data": {"group_size": 128}}},
                },
            }
        )
    )
    monkeypatch.setenv("ARIES_AGENT_TOKEN", "t")
    monkeypatch.setenv("ARIES_EXECUTORS_JSON", json.dumps([{"id": "xpu0"}]))
    monkeypatch.setenv("ARIES_MODEL_PATH", str(main))
    monkeypatch.setenv("ARIES_ENHANCER_MODEL_PATH", str(pe))
    monkeypatch.delenv("ARIES_ENHANCER_MODEL_DTYPE", raising=False)
    monkeypatch.delenv("MODEL_DTYPE", raising=False)
    dtypes = {m.id: m.dtype for m in AgentConfig.from_env().models}
    assert dtypes == {"Qwen/Qwen-Image-2.1": "bf16", "Qwen/Qwen-Image-2.1-PE-T2I": "int4"}

    monkeypatch.setenv("ARIES_ENHANCER_MODEL_DTYPE", "int4-g128")
    assert AgentConfig.from_env().models[1].dtype == "int4-g128"


async def _spawn_sleeper(rt) -> None:
    rt.process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)", stdout=asyncio.subprocess.PIPE
    )
    rt.pid = rt.process.pid
    rt.stage = agent_pb2.STAGE_DENOISE
    rt.state = agent_pb2.EXECUTOR_STATE_BUSY
    rt._monitor_task = asyncio.create_task(rt._monitor_process(rt.process, agent_pb2.STAGE_DENOISE))


@pytest.mark.asyncio
async def test_ram_guard_kills_hungriest_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """HARDWARE.md F9: below the host-RAM floor the worker holding most RAM is killed as oom_kill."""
    cfg = _make_test_config(tmp_path)
    agent = AriesAgent(config=cfg)
    small, big = agent.executors["xpu0"], agent.executors["xpu1"]
    for rt in (small, big):
        await _spawn_sleeper(rt)
    monkeypatch.setattr(small, "worker_ram_bytes", lambda: 4 << 30)
    monkeypatch.setattr(big, "worker_ram_bytes", lambda: 20 << 30)
    scratch = tmp_path / "task"
    scratch.mkdir()
    big.active_task = ActiveTask(
        assign=agent_pb2.Assign(task="t1", attempt=1, stage=agent_pb2.STAGE_DENOISE), scratch_dir=scratch
    )

    monkeypatch.setenv("ARIES_HOST_RAM_AVAILABLE_BYTES", str(8 << 30))
    assert agent.check_host_ram() is None
    monkeypatch.setenv("ARIES_HOST_RAM_AVAILABLE_BYTES", str(2 << 30))
    assert agent.check_host_ram() is None  # one low sample is not enough
    assert agent.check_host_ram() == "xpu1"

    msg = await asyncio.wait_for(agent.send_queue.get(), timeout=5.0)
    assert msg.failed.task == "t1"
    assert msg.failed.failure_class == agent_pb2.FAILURE_OOM_KILL
    assert "host RAM guard: 2.0 GiB available, below the 4.0 GiB floor" in msg.failed.detail
    assert small.is_worker_alive and not big.is_worker_alive
    assert agent.metrics.ram_guard_kills_total.labels(executor="xpu1", stage="denoise")._value.get() == 1

    agent.config.ram_guard_floor_bytes = 0  # turned off
    assert agent.check_host_ram() is None
    assert agent.check_host_ram() is None
    await small.kill_worker(reason="test")


@pytest.mark.asyncio
async def test_worker_output_line_over_limit_keeps_draining(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """A line over the stream limit is dropped and the pipe stays drained: the worker, which
    writes far more than a pipe holds afterwards, still runs to its end."""
    agent = AriesAgent(config=_make_test_config(tmp_path))
    rt = agent.executors["xpu0"]
    script = "import sys; print('x' * 8192); [print('line', i) for i in range(20000)]; print('last')"
    rt.process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE, limit=1024
    )
    proc = rt.process
    with caplog.at_level(logging.INFO, logger="aries_agent"):
        await asyncio.wait_for(rt._monitor_process(proc, agent_pb2.STAGE_DENOISE), timeout=20.0)
    assert proc.returncode == 0
    assert "dropped" in caplog.text
    assert "] last" in caplog.text


def test_worker_preexec_raises_oom_score_adj(tmp_path: Path, monkeypatch):
    """Workers are the kernel's first OOM victims, never the desktop session."""
    from aries_agent.agent import worker_preexec

    written = {}
    real_open = open

    def fake_open(path, mode="r", *a, **k):
        if path == "/proc/self/oom_score_adj":
            target = tmp_path / "adj"
            written["path"] = target
            return real_open(target, mode, *a, **k)
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    worker_preexec(1000)()
    monkeypatch.undo()
    assert written["path"].read_text() == "1000"
    assert AgentConfig().worker_oom_score_adj == 1000

    # The real thing, in a child: the value sticks.
    proc = subprocess.run(
        [sys.executable, "-c", "print(open('/proc/self/oom_score_adj').read().strip())"],
        preexec_fn=worker_preexec(900), capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == "900"


@pytest.mark.asyncio
async def test_release_during_upload_waits_for_the_upload(tmp_path: Path):
    """A release (lend, idle grace) that lands while a finished task's outputs upload must
    not stop the worker under the upload, nor be followed by READY while the worker exits:
    the worker exits once the upload is done, and Studio hears STOPPING, never READY."""
    agent = AriesAgent(config=_make_test_config(tmp_path))
    rt = agent.executors["xpu0"]
    rt.stage = agent_pb2.STAGE_DECODE
    exited = asyncio.Event()

    class _Proc:
        pid, returncode = 4321, None

        async def wait(self):
            await exited.wait()
            self.returncode = 0
            return 0

    rt.process = _Proc()
    sent: List[Dict[str, Any]] = []

    async def send_worker_json(obj):
        sent.append(obj)
        if obj.get("action") == "exit":
            asyncio.get_running_loop().call_later(0.05, exited.set)  # exiting takes a moment

    rt._send_worker_json = send_worker_json
    out = tmp_path / "image.png"
    out.write_bytes(b"png")
    assign = agent_pb2.Assign(task="t1", attempt=1, stage=agent_pb2.STAGE_DECODE)
    assign.uploads.add(name="image.png", url="http://studio/out/image.png")
    rt.active_task = ActiveTask(assign=assign, scratch_dir=tmp_path / "scratch", output_paths={"image.png": out})
    exit_sent_mid_upload = []
    releases = []

    async def upload(url, path, sha):
        releases.append(asyncio.create_task(rt.release_worker()))  # arrives mid-upload
        for _ in range(5):
            await asyncio.sleep(0)
        exit_sent_mid_upload.append(bool(sent))
        return True, ""

    rt._upload_artifact_with_retry = upload
    await rt._on_worker_done({"id": "t1"})
    await asyncio.gather(*releases)

    assert exit_sent_mid_upload == [False]
    assert sent == [{"action": "exit"}]
    states = []
    while not agent.send_queue.empty():
        m = agent.send_queue.get_nowait()
        if m.HasField("executor_state"):
            states.append(m.executor_state.state)
    assert agent_pb2.EXECUTOR_STATE_READY not in states
    assert states[-1] == agent_pb2.EXECUTOR_STATE_STOPPING
