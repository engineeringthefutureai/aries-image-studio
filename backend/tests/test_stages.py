import sys
from unittest.mock import MagicMock, patch

# In CPU test environments without Intel XPU / PyTorch wheels installed,
# inject test doubles for external ML packages so stage runner units can be tested.
mock_torch = MagicMock()
mock_torch.xpu.is_available.return_value = False
mock_torch.xpu.device_count.return_value = 0
sys.modules["torch"] = mock_torch
sys.modules["diffusers"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["transformers.generation"] = MagicMock()
sys.modules["transformers.generation.streamers"] = MagicMock()

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
import pytest
from PIL import Image

from aries_worker.client import WorkerSocketClient
from aries_worker.stages.enhance import EnhanceCancelled, EnhanceProgressStreamer, EnhanceStageRunner, _parse_enhanced_json
from aries_worker.stages.encode import EncodeStageRunner
from aries_worker.stages.denoise import DenoiseStageRunner
from aries_worker.stages.decode import DecodeStageRunner


@pytest.fixture(scope="module", autouse=True)
def clean_mock_modules():
    yield
    for m in ("torch", "diffusers", "transformers"):
        sys.modules.pop(m, None)


@pytest.fixture
def test_job_dir():
    d = Path(tempfile.mkdtemp(prefix="aries_stage_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_enhance_parser():
    """Verify thinking extraction and JSON extraction in enhancer output."""
    raw = "<think>Consider lighting and textures</think>```json\n{\"enhanced_prompt\": \"Cinematic neon city\", \"wh_ratio\": \"16:9\"}\n```"
    parsed = _parse_enhanced_json(raw, "original prompt")
    assert parsed["enhanced_prompt"] == "Cinematic neon city"
    assert parsed["wh_ratio"] == "16:9"
    assert "lighting" in parsed["thinking"]

    # Truncated JSON without closing brace (token limit hit mid-generation)
    truncated = '<think>Deep thoughts</think>{\n  "enhanced_prompt": "A single luminous crystal dominates the frame, surrounded by glowing bioluminescent spores in a dark cave",\n  "wh_ratio": "16:9"'
    parsed_trunc = _parse_enhanced_json(truncated, "original prompt")
    assert "A single luminous crystal dominates" in parsed_trunc["enhanced_prompt"]
    assert parsed_trunc["wh_ratio"] == "16:9"
    assert "Deep thoughts" in parsed_trunc["thinking"]

    # Model hallucinated key variations (enhanted_prompt, enhritten_prompt, rewritten_prompt)
    enhanted_case = '{\n  "enhanted_prompt": "A highly detailed, photorealistic 3D-style close-up shows Froggy G, a cute anthropomorphic frog",\n  "wh_ratio": "1:1"\n}'
    parsed_enhanted = _parse_enhanced_json(enhanted_case, "original")
    assert "Froggy G" in parsed_enhanted["enhanced_prompt"]

    enhritten_case = '{\n  "enhritten_prompt": "A single large amethyst-like quartz crystal stands upright",\n  "wh_ratio": "16:9"\n}'
    parsed_enhritten = _parse_enhanced_json(enhritten_case, "original")
    assert "amethyst-like quartz crystal" in parsed_enhritten["enhanced_prompt"]

    rewritten_case = '{\n  "rewritten_prompt": "A solitary granite lighthouse on a steep cliff",\n  "wh_ratio": "3:2"\n}'
    parsed_rewritten = _parse_enhanced_json(rewritten_case, "original")
    assert "solitary granite lighthouse" in parsed_rewritten["enhanced_prompt"]

    # Raw prompt fallback without JSON
    raw_fallback = "Just plain text without braces"
    parsed_fallback = _parse_enhanced_json(raw_fallback, "original prompt")
    assert parsed_fallback["enhanced_prompt"] == "Just plain text without braces"


class FakeBatchEncoding(dict):
    """Stands in for transformers' BatchEncoding: item access plus .to(device)."""

    def to(self, device):
        return self


def _fake_tokenizer(response: str, prompt_len: int = 42) -> MagicMock:
    input_ids = MagicMock()
    input_ids.shape = (1, prompt_len)
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "prompt text"
    tokenizer.return_value = FakeBatchEncoding(input_ids=input_ids)
    tokenizer.batch_decode.return_value = [response]
    return tokenizer


def test_enhance_runner_execution(test_job_dir):
    """EnhanceStageRunner writes enhanced.json from the model's JSON reply."""
    runner = EnhanceStageRunner()
    runner.tokenizer = _fake_tokenizer('{"enhanced_prompt": "Enhanced masterpiece", "wh_ratio": "4:3"}')
    runner.model = MagicMock()
    runner.model.generate.return_value = [MagicMock()]

    paths = {"enhanced_json": str(test_job_dir / "enhanced.json")}
    timings, peak_vram = runner.run_job("job-123", paths, {"prompt": "a sunset"})

    assert "enhance_s" in timings
    data = json.loads(Path(paths["enhanced_json"]).read_text())
    assert data["rewritten_prompt"] == "Enhanced masterpiece"
    assert data["wh_ratio"] == "4:3"


TOKEN_TEXT = {
    2: "thinking",
    3: " deeply {about braces}",
    4: "</think>",
    5: '{"rewritten_prompt":',
    6: '"photo"}',
}


def _decode(tokens, skip_special_tokens=False):
    return "".join(TOKEN_TEXT.get(t, " ") for t in tokens)


def test_enhance_progress_streamer():
    """Follows HF generate()'s call order: one put() with the whole prompt before prefill,
    then one put() per generated token."""
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = _decode
    emitted = []
    cancelled = False

    streamer = EnhanceProgressStreamer(
        tokenizer=tokenizer,
        prompt_tokens=50,
        progress_cb=lambda **kw: emitted.append(kw),
        cancel_check=lambda: cancelled,
        emit_interval=0.0,
    )
    assert streamer.status == "prefill"

    # The prompt echo arrives before prefill runs: still prefill, nothing counted.
    streamer.put([[100] * 50])
    assert streamer.status == "prefill"
    assert streamer.t_first_token is None
    assert streamer.generated_tokens == 0

    time.sleep(0.01)  # prefill
    streamer.put([2])
    assert streamer.status == "thinking"
    assert streamer.prefill_s >= 0.01
    assert 0 < streamer.prefill_tps <= 50 / 0.01

    # A brace inside the reasoning must not end the thinking phase.
    streamer.put([3])
    assert streamer.status == "thinking"

    # </think> ends reasoning; it counts as a thought token.
    streamer.put([4])
    assert streamer.status == "rewriting"
    assert streamer.thought_tokens == 3

    streamer.put([5])
    streamer.put([6])
    assert streamer.reply_tokens == 2
    assert streamer.generated_tokens == 5

    streamer.end()
    last = emitted[-1]
    assert last["sub_stage"] == "rewriting"
    assert "step" not in last
    stats = last["enhance_stats"]
    assert (stats["prompt_tokens"], stats["generated_tokens"]) == (50, 5)
    assert (stats["thought_tokens"], stats["reply_tokens"]) == (3, 2)
    assert stats["generation_tps"] > 0

    cancelled = True
    with pytest.raises(EnhanceCancelled):
        streamer.put([2])


def test_enhance_streamer_without_reasoning():
    """Output that starts straight with JSON is all reply."""
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = _decode
    streamer = EnhanceProgressStreamer(tokenizer=tokenizer, prompt_tokens=0, emit_interval=0.0)
    streamer.put([[100] * 7])
    assert streamer.prompt_tokens == 7
    streamer.put([5])
    assert streamer.status == "rewriting"
    streamer.put([6])
    assert (streamer.thought_tokens, streamer.reply_tokens) == (1, 1)


def test_enhance_runner_progress_cb(test_job_dir):
    """EnhanceStageRunner reports prefill first and returns token telemetry."""
    runner = EnhanceStageRunner()
    runner.tokenizer = _fake_tokenizer('{"rewritten_prompt": "Enhanced photo", "wh_ratio": "16:9"}')
    runner.tokenizer.decode.side_effect = _decode

    def fake_generate(**kwargs):
        streamer = kwargs["streamer"]
        streamer.put([[100] * 42])
        streamer.put([5])
        streamer.put([6])
        streamer.end()
        return [MagicMock()]

    runner.model = MagicMock()
    runner.model.generate.side_effect = fake_generate

    paths = {"enhanced_json": str(test_job_dir / "enhanced.json")}
    emitted = []
    timings, peak = runner.run_job("job-enh-1", paths, {"prompt": "a mountain"}, progress_cb=lambda **kw: emitted.append(kw))

    assert timings["prompt_tokens"] == 42
    assert timings["generated_tokens"] == 2
    assert timings["prefill_tps"] > 0
    assert emitted[0]["sub_stage"] == "prefill"
    assert emitted[0]["enhance_stats"]["prompt_tokens"] == 42
    assert emitted[-1]["sub_stage"] == "rewriting"


def test_encode_runner_execution(test_job_dir):
    """Test EncodeStageRunner saving embeds.pt."""
    runner = EncodeStageRunner()

    mock_pipe = MagicMock()
    mock_embeds = MagicMock()
    mock_embeds.shape = [1, 20, 4096]
    mock_embeds.cpu.return_value = "cpu_embeds"
    mock_mask = MagicMock()
    mock_mask.cpu.return_value = "cpu_mask"
    mock_pad = MagicMock()
    mock_pad.cpu.return_value = "cpu_pad"

    mock_pipe.encode_prompt.return_value = (mock_embeds, mock_mask, mock_pad)
    runner.pipe = mock_pipe
    runner.device = "xpu:0"

    paths = {
        "embeds_pt": str(test_job_dir / "embeds.pt"),
        "enhanced_json": str(test_job_dir / "enhanced.json"),  # file does not exist
    }

    with patch("torch.save") as mock_save:
        mock_save.side_effect = lambda obj, f: f.write(b"dummy")

        # 1. Without enhance flag (default): encodes original prompt, ignores non-existent enhanced.json
        timings, peak = runner.run_job("job-456", paths, {"prompt": "cyberpunk alley", "enhance": False})
        assert "encode_s" in timings
        assert "save_s" in timings
        mock_save.assert_called_once()
        saved_dict = mock_save.call_args[0][0]
        assert saved_dict["prompt_embeds"] == "cpu_embeds"
        assert saved_dict["prompt_source"] == "original"
        assert saved_dict["prompt"] == "cyberpunk alley"

        # 2. With enhance=True and missing file: must raise FileNotFoundError
        mock_save.reset_mock()
        with pytest.raises(FileNotFoundError):
            runner.run_job("job-456", paths, {"prompt": "cyberpunk alley", "enhance": True})

        # 3. With enhance=True and present file: encodes rewritten prompt
        enh_file = Path(paths["enhanced_json"])
        enh_file.write_text(json.dumps({"rewritten_prompt": "enhanced neon cyberpunk alley"}))
        timings_enh, peak_enh = runner.run_job("job-456", paths, {"prompt": "cyberpunk alley", "enhance": True})
        assert timings_enh["prompt_source"] == "enhanced"
        saved_dict_enh = mock_save.call_args[0][0]
        assert saved_dict_enh["prompt"] == "enhanced neon cyberpunk alley"
        assert saved_dict_enh["prompt_source"] == "enhanced"


def test_denoise_runner_execution(test_job_dir):
    """Test DenoiseStageRunner progress reporting, latent saving, and embeds.pt removal."""
    runner = DenoiseStageRunner()

    # Create dummy embeds.pt
    embeds_path = test_job_dir / "embeds.pt"
    embeds_path.write_bytes(b"dummy_embeds_content")

    mock_pipe = MagicMock()
    mock_latent = MagicMock()
    mock_latent.detach.return_value.cpu.return_value = "cpu_latent"
    mock_pipe.return_value.images = mock_latent
    runner.pipe = mock_pipe
    runner.is_loaded = True

    paths = {
        "embeds_pt": str(embeds_path),
        "latent_pt": str(test_job_dir / "latent.pt"),
    }

    progress_reports = []

    def on_progress(step, steps, sps):
        progress_reports.append((step, steps, sps))

    with patch("torch.load") as mock_load, patch("torch.save") as mock_save, patch("torch.Generator"):
        mock_save.side_effect = lambda obj, f: f.write(b"dummy")
        mock_load.return_value = {"prompt_embeds": "fake_emb"}
        timings, peak = runner.run_job(
            "job-789",
            paths,
            {"steps": 10, "seed": 42, "width": 2048, "height": 2048},
            progress_cb=on_progress,
        )

        assert "denoise_s" in timings
        assert timings["steps"] == 10
        mock_save.assert_called_once()
        # Check embeds.pt was unlinked
        assert not embeds_path.exists()


def test_decode_runner_execution(test_job_dir):
    """Test DecodeStageRunner loading latent.pt and writing image.png."""
    runner = DecodeStageRunner()

    latent_path = test_job_dir / "latent.pt"
    latent_path.write_bytes(b"dummy_latent_content")

    mock_vae = MagicMock()
    runner.vae = mock_vae
    runner.is_loaded = True

    paths = {
        "latent_pt": str(latent_path),
        "image_png": str(test_job_dir / "image.png"),
    }

    # Test with 3D sequence to 5D unpacking
    mock_vae.config.latents_mean = [0.0] * 64
    mock_vae.config.latents_std = [1.0] * 64
    mock_vae.config.z_dim = 64
    mock_vae.dtype = mock_torch.float32

    fake_decoded_tensor = MagicMock()
    fake_decoded_tensor.ndim = 5
    fake_decoded_tensor.__getitem__.return_value = fake_decoded_tensor
    fake_decoded_tensor.__truediv__.return_value.__add__.return_value.clamp.return_value.cpu.return_value.permute.return_value.float.return_value.numpy.return_value = [
        MagicMock()
    ]
    mock_vae.decode.return_value = (fake_decoded_tensor,)

    fake_3d_latents = MagicMock()
    fake_3d_latents.ndim = 3
    fake_3d_latents.shape = (1, 4, 64)
    fake_5d_latents = MagicMock()
    fake_5d_latents.ndim = 5
    fake_3d_latents.transpose.return_value.reshape.return_value = fake_5d_latents
    fake_3d_latents.to.return_value = fake_3d_latents

    with patch("torch.load") as mock_load, patch("PIL.Image.fromarray") as mock_fromarray:
        fake_img = Image.new("RGB", (64, 64), color="blue")
        mock_fromarray.return_value = fake_img

        mock_load.return_value = {
            "latent": fake_3d_latents,
            "width": 64,
            "height": 64,
        }

        timings, peak = runner.run_job(
            "job-decode",
            paths,
            {"decode_mode": "untiled", "width": 64, "height": 64},
        )

        assert "decode_s" in timings
        assert Path(paths["image_png"]).exists()
        fake_3d_latents.transpose.assert_called_once_with(1, 2)
        mock_vae.decode.assert_called_once()


@pytest.mark.asyncio
async def test_worker_socket_client_full_cycle(test_job_dir):
    """Test WorkerSocketClient connecting to a server and processing a job."""
    sock_path = test_job_dir / "test.sock"

    # Start simple mock manager server
    async def handle_worker(reader, writer):
        # 1. Read hello
        line = await reader.readline()
        hello = json.loads(line.decode("utf-8"))
        assert hello["event"] == "hello"

        # 2. Read claim
        line = await reader.readline()
        claim = json.loads(line.decode("utf-8"))
        assert claim["event"] == "claim"

        # 3. Send job
        writer.write(json.dumps({
            "action": "job",
            "id": "job-client-test",
            "paths": {"out": str(test_job_dir / "out.txt")},
            "params": {"test": 1},
        }).encode("utf-8") + b"\n")
        await writer.drain()

        # 4. Read done
        line = await reader.readline()
        done = json.loads(line.decode("utf-8"))
        assert done["event"] == "done"
        assert done["id"] == "job-client-test"

        # 5. Read claim
        line = await reader.readline()
        claim2 = json.loads(line.decode("utf-8"))
        assert claim2["event"] == "claim"

        # 6. Send exit
        writer.write(json.dumps({"action": "exit"}).encode("utf-8") + b"\n")
        await writer.drain()

        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(handle_worker, path=str(sock_path))

    # Create dummy runner
    mock_runner = MagicMock()
    mock_runner.load_s = 0.5
    mock_runner.run_job.return_value = ({"stage_s": 0.1}, 1024)

    client = WorkerSocketClient(
        stage="encode",
        card=0,
        socket_path=str(sock_path),
        runner=mock_runner,
        idle_grace=1.0,
    )

    await client.run()
    server.close()
    await server.wait_closed()
