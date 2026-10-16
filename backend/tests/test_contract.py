"""The stage workers write the artifacts the backend's upload checks expect.

The contract lives in proto/aries/agent/v1/artifacts.json; studio/tests/contract_test.rs
checks the Rust side against the same file.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Stage modules import torch/diffusers/transformers; the same stand-ins as test_stages.py.
import _ml_stubs

_ml_stubs.install()

import pytest
from PIL import Image

from aries_worker.stages.decode import DecodeStageRunner
from aries_worker.stages.denoise import DenoiseStageRunner
from aries_worker.stages.encode import EncodeStageRunner
from aries_worker.stages.enhance import EnhanceStageRunner

CONTRACT = json.loads(
    (Path(__file__).resolve().parents[2] / "proto/aries/agent/v1/artifacts.json").read_text()
)
MODEL = {"model_id": "Qwen/Qwen-Image-2.1", "model_rev": "rev123", "model_dtype": "bf16"}


def _run_encode(tmp_path: Path, true_cfg_scale: float, with_mask: bool):
    runner = EncodeStageRunner()
    runner.pipe = MagicMock()
    # The real pipeline returns mask None when nothing is padded.
    runner.pipe.encode_prompt.return_value = (MagicMock(), MagicMock() if with_mask else None, None)
    with patch("aries_worker.stages.encode.atomic_safetensors_save", return_value=1) as save:
        runner.run_job(
            "task-enc",
            {"embeds_safetensors": str(tmp_path / "embeds.safetensors")},
            {"prompt": "a lighthouse", "prompt_source": "original", "true_cfg_scale": true_cfg_scale, **MODEL},
        )
    tensors, _path = save.call_args[0]
    return set(tensors), set(save.call_args[1]["metadata"])


@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize("true_cfg_scale", [1.0, 4.0])
def test_encode_writes_contract(tmp_path: Path, true_cfg_scale: float, with_mask: bool):
    spec = CONTRACT["embeds.safetensors"]
    tensors, metadata = _run_encode(tmp_path, true_cfg_scale, with_mask)
    cfg = true_cfg_scale > 1
    required = set(spec["tensors"]) | (set(spec["tensors_when_cfg"]) if cfg else set())
    allowed = required | set(spec["optional_tensors"]) | (set(spec["optional_tensors_when_cfg"]) if cfg else set())
    assert required <= tensors <= allowed
    assert set(spec["metadata"]) <= metadata


def test_denoise_writes_contract(tmp_path: Path):
    spec = CONTRACT["latent.safetensors"]
    runner = DenoiseStageRunner()
    runner.pipe = MagicMock()
    embeds = tmp_path / "embeds.safetensors"
    embeds.write_bytes(b"x")
    with (
        patch("aries_worker.stages.denoise.load_safetensors_with_metadata") as load,
        patch("aries_worker.stages.denoise.atomic_safetensors_save", return_value=1) as save,
    ):
        load.return_value = ({"prompt_embeds": MagicMock()}, {"model": "m", "rev": "r", "dtype": "bf16"})
        runner.run_job(
            "task-den",
            {"embeds_safetensors": str(embeds), "latent_safetensors": str(tmp_path / "latent.safetensors")},
            {"width": 2048, "height": 2048, "steps": 4, "seed": 7, "true_cfg_scale": 1.0},
        )
    tensors, _path = save.call_args[0]
    assert set(spec["tensors"]) <= set(tensors)
    assert set(spec["metadata"]) <= set(save.call_args[1]["metadata"])


def test_decode_reads_contract_tensor(tmp_path: Path):
    """Decode accepts a latent.safetensors holding exactly the contract's tensor names."""
    (latent_name,) = CONTRACT["latent.safetensors"]["tensors"]
    runner = DecodeStageRunner()
    runner.vae = MagicMock()
    runner.vae.config = MagicMock(spec=[])  # no latents_mean/std: skip denormalization
    decoded = MagicMock()
    decoded.ndim = 4
    decoded.__truediv__.return_value.__add__.return_value.clamp.return_value.cpu.return_value.permute.return_value.float.return_value.numpy.return_value = [
        MagicMock()
    ]
    runner.vae.decode.return_value = (decoded,)
    latent = MagicMock()
    latent.to.return_value = latent
    latent.ndim = 5
    path = tmp_path / "latent.safetensors"
    path.write_bytes(b"x")
    with (
        patch("aries_worker.stages.decode.load_safetensors_with_metadata") as load,
        patch("PIL.Image.fromarray", return_value=Image.new("RGB", (16, 16))),
    ):
        load.return_value = ({latent_name: latent}, {"width": "16", "height": "16"})
        runner.run_job("task-dec", {"latent_safetensors": str(path), "image_png": str(tmp_path / "image.png")}, {})
    assert (tmp_path / "image.png").exists()


def test_enhance_writes_contract(tmp_path: Path):
    spec = CONTRACT["enhanced.json"]
    runner = EnhanceStageRunner()
    tokenizer = MagicMock()
    input_ids = MagicMock()
    input_ids.shape = (1, 8)

    class Encoding(dict):
        def to(self, device):
            return self

    tokenizer.return_value = Encoding(input_ids=input_ids)
    tokenizer.batch_decode.return_value = ['plan</think>{"rewritten_prompt": "a photo", "wh_ratio": "1:1"}']
    runner.tokenizer = tokenizer
    runner.model = MagicMock()
    runner.model.generate.return_value = [MagicMock()]
    out = tmp_path / "enhanced.json"
    runner.run_job("task-enh", {"enhanced_json": str(out)}, {"prompt": "a photo"})
    data = json.loads(out.read_text())
    assert set(spec["fields"]) <= set(data)
    assert set(data) <= set(spec["fields"]) | set(spec["optional_fields"])


def test_agent_exports_the_metric_names_the_backend_scrapes():
    from prometheus_client import generate_latest

    from aries_agent.metrics import AgentMetrics

    names = json.loads(
        (Path(__file__).resolve().parents[2] / "proto/aries/agent/v1/agent_metrics.json").read_text()
    )
    metrics = AgentMetrics()
    metrics.init_executor("xpu0", "0000:04:00.0", 25 * 1024**3)
    text = generate_latest(metrics.registry).decode()
    label = names["executor_label"]
    assert any(line.startswith(names["host_ram_available_bytes"] + " ") for line in text.splitlines())
    for key in ("executor_mem_used_bytes", "executor_mem_total_bytes", "executor_mem_overflow_bytes"):
        assert f'{names[key]}{{' in text and f'{label}="xpu0"' in text, key
