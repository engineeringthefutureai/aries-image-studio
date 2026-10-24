import sys
from unittest.mock import MagicMock, patch

# In CPU test environments without Intel XPU / PyTorch wheels installed,
# inject test doubles for external ML packages so stage runner units can be tested.
import _ml_stubs

mock_torch = _ml_stubs.install()

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
import pytest
from PIL import Image

from aries_worker.client import WorkerSocketClient
from aries_worker.stages.enhance import MAX_NEW_TOKENS, EnhanceCancelled, EnhanceProgressStreamer, EnhanceStageRunner, _parse_enhanced_json
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

    # A reply without JSON after the reasoning is used as written.
    parsed_fallback = _parse_enhanced_json("plan</think>Just plain text without braces", "original prompt")
    assert parsed_fallback["enhanced_prompt"] == "Just plain text without braces"


def test_enhance_parser_never_takes_the_prompt_from_reasoning():
    """Seen live: PE-I2I with three images hit the limit mid-reasoning. The reasoning became
    the prompt, or "..." from where it drafted the output format."""
    unfinished = "Let me carefully analyze the input images and the user's instruction.\n\n<image1>: a photo of a man"
    parsed = _parse_enhanced_json(unfinished, "Turn me into an anime character")
    assert parsed["enhanced_prompt"] == "Turn me into an anime character"
    assert parsed["thinking"].startswith("Let me carefully analyze")
    drafting = 'The output format is {"rewritten_prompt": "...", "wh_ratio": "3:2"}. Let me construct'
    assert _parse_enhanced_json(drafting, "orig")["enhanced_prompt"] == "orig"
    # A bare JSON reply (no reasoning at all) is still read.
    assert _parse_enhanced_json('{"rewritten_prompt": "x", "wh_ratio": "3:2"}', "orig")["enhanced_prompt"] == "x"


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


def test_enhance_runner_uses_the_jobs_token_limit(test_job_dir):
    """A job's `max_new_tokens` (Advanced "Enhancer tokens") replaces the default (4096) and
    is what progress reports as the limit."""
    runner = EnhanceStageRunner()
    runner.tokenizer = _fake_tokenizer('{"rewritten_prompt": "A long infographic", "wh_ratio": "3:2"}')
    runner.model = MagicMock()
    runner.model.generate.return_value = [MagicMock()]
    paths = {"enhanced_json": str(test_job_dir / "enhanced.json")}
    emitted = []
    runner.run_job("job-tok", paths, {"prompt": "p", "max_new_tokens": 6144}, progress_cb=lambda **kw: emitted.append(kw))
    assert runner.model.generate.call_args.kwargs["max_new_tokens"] == 6144
    assert emitted[0]["enhance_stats"]["max_new_tokens"] == 6144

    runner.run_job("job-tok", paths, {"prompt": "p"})
    assert runner.model.generate.call_args.kwargs["max_new_tokens"] == MAX_NEW_TOKENS
    assert MAX_NEW_TOKENS == 4096


def _i2i_runner(response: str) -> EnhanceStageRunner:
    """A runner as `load()` leaves it for PE-I2I: a processor that builds the inputs."""
    runner = EnhanceStageRunner()
    runner.vision = True
    input_ids = MagicMock()
    input_ids.shape = (1, 5500)
    runner.processor = MagicMock()
    runner.processor.apply_chat_template.return_value = FakeBatchEncoding(input_ids=input_ids)
    runner.tokenizer = _fake_tokenizer(response)
    runner.model = MagicMock()
    runner.model.generate.return_value = [MagicMock()]
    return runner


def test_enhance_i2i_reads_the_reference_images(test_job_dir):
    """PE-I2I gets the job's references (upright, ~1 MP) before the instruction, thinks, and
    may answer with `ratio_follow` instead of a ratio."""
    runner = _i2i_runner('plan</think>{"rewritten_prompt": "x", "wh_ratio": "", "ratio_follow": "<image2>"}')
    paths = {"enhanced_json": str(test_job_dir / "enhanced.json")}
    for i, size in enumerate([(512, 256), (300, 300)]):
        Image.new("RGB", size, (i * 90, 20, 20)).save(test_job_dir / f"ref{i}.png")
        paths[f"ref{i}.image"] = str(test_job_dir / f"ref{i}.png")
    timings, _ = runner.run_job(
        "job-i2i", paths, {"prompt": "put image 1 into image 2", "max_new_tokens": 5000, "references": ["a" * 64, "b" * 64]}
    )

    call = runner.processor.apply_chat_template.call_args
    content = call.args[0][1]["content"]
    assert [part["type"] for part in content] == ["image", "image", "text"]
    assert content[0]["image"].size == (1440, 736)  # 2:1 at ~1024², sides multiples of 32
    assert content[1]["image"].size == (1024, 1024)
    assert content[2]["text"] == "put image 1 into image 2"
    assert call.kwargs["enable_thinking"] is True
    kwargs = runner.model.generate.call_args.kwargs
    assert (kwargs["temperature"], kwargs["top_p"], kwargs["top_k"]) == (1.0, 0.95, 20)
    assert kwargs["max_new_tokens"] == 5000
    data = json.loads(Path(paths["enhanced_json"]).read_text())
    assert data["rewritten_prompt"] == "x"
    assert data["ratio_follow"] == "<image2>"
    assert timings["references"] == 2 and timings["prompt_tokens"] == 5500


def test_enhance_ratio_follow_names_an_input_or_nothing():
    reply = 'p</think>{"rewritten_prompt": "x", "wh_ratio": "2:3"}'
    assert _parse_enhanced_json(reply, "o")["ratio_follow"] == ""
    bad = 'p</think>{"rewritten_prompt": "x", "ratio_follow": "the first one"}'
    assert _parse_enhanced_json(bad, "o")["ratio_follow"] == ""
    # A reply cut by the token limit still yields both.
    cut = 'p</think>{"rewritten_prompt": "a goblin in the cafe", "ratio_follow": "<image1>", "wh_ra'
    parsed = _parse_enhanced_json(cut, "o")
    assert parsed["enhanced_prompt"] == "a goblin in the cafe"
    assert parsed["ratio_follow"] == "<image1>"


@pytest.mark.parametrize("reasoning,expect_calls,expect_limits,cut", [
    # Done within the limit: one generate().
    (2000, 1, [4096], False),
    # Runs out at 3072 (4096 minus the reply reserve): goes on with 8192 and finishes.
    (5000, 2, [4096, 8192 - 3072], False),
    # Never stops: 4096 -> 8192 -> 16384 -> 24576, then closed at 23040 for a reply in the last 1536.
    (99999, 5, [4096, 8192 - 3072, 16384 - 6656, 24576 - 14848, 24576 - 23040], True),
])
def test_enhance_reasoning_that_runs_out_gets_a_doubled_limit(test_job_dir, reasoning, expect_calls, expect_limits, cut):
    """Reasoning that reaches the limit minus the reply reserve continues with the limit
    doubled (the same as a rerun with twice the limit, minus the redone reasoning), up to
    24576; there it is closed with </think> and the model replies in what is left."""
    runner = _i2i_runner("")
    paths = {"enhanced_json": str(test_job_dir / "enhanced.json")}
    Image.new("RGB", (64, 64)).save(test_job_dir / "ref0.png")
    paths["ref0.image"] = str(test_job_dir / "ref0.png")
    calls, thought = [], [0]

    def generate(**kwargs):
        calls.append(kwargs)
        streamer = kwargs["streamer"]
        streamer.put([0] * 10)  # the prompt echo
        stop = (kwargs.get("stopping_criteria") or [None])[0]
        if stop is None:  # the forced reply
            streamer.put([5, 6])
            return MagicMock()
        for _ in range(kwargs["max_new_tokens"]):  # reasoning, one token at a time
            if thought[0] >= reasoning:
                streamer.put([4])  # </think>
                streamer.put([5, 6])
                break
            streamer.put([2])
            thought[0] += 1
            stop(MagicMock(shape=(1, 1)), None)
            if stop.triggered:
                break
        return MagicMock()

    runner.model.generate.side_effect = generate
    runner.tokenizer.batch_decode.return_value = ['plan\n</think>\n\n{"rewritten_prompt": "a goblin", "wh_ratio": "1:1"}']
    runner.tokenizer.decode.side_effect = _decode
    timings, _ = runner.run_job("job", paths, {"prompt": "p", "max_new_tokens": 4096, "references": ["a" * 64]})

    assert [c["max_new_tokens"] for c in calls] == expect_limits
    assert len(calls) == expect_calls
    assert ("thinking_cut" in timings) is cut
    assert ("limit_raised_from" in timings) is (expect_calls > 1)
    assert timings["max_new_tokens"] == {1: 4096, 2: 8192, 5: 24576}[expect_calls]
    assert json.loads(Path(paths["enhanced_json"]).read_text())["rewritten_prompt"] == "a goblin"


def test_enhance_default_limit_grows_with_the_references(test_job_dir):
    """4096, plus 2048 per reference image (each adds ~1,000 prompt tokens to look at)."""
    runner = _i2i_runner('p</think>{"rewritten_prompt": "x", "wh_ratio": "1:1"}')
    paths = {"enhanced_json": str(test_job_dir / "e.json")}
    for i in range(3):
        Image.new("RGB", (64, 64)).save(test_job_dir / f"ref{i}.png")
        paths[f"ref{i}.image"] = str(test_job_dir / f"ref{i}.png")
    runner.run_job("job", paths, {"prompt": "p", "references": ["a" * 64] * 3})
    assert runner.model.generate.call_args.kwargs["max_new_tokens"] == 4096 + 3 * 2048
    runner.run_job("job", paths, {"prompt": "p"})
    assert runner.model.generate.call_args.kwargs["max_new_tokens"] == 4096
    from aries_worker.stages.enhance import default_max_new_tokens
    assert default_max_new_tokens(10) == 24576


def test_text_enhancer_refuses_references(test_job_dir):
    runner = EnhanceStageRunner()
    runner.tokenizer = _fake_tokenizer('{"rewritten_prompt": "x", "wh_ratio": "1:1"}')
    runner.model = MagicMock()
    with pytest.raises(ValueError, match="reference images"):
        runner.run_job("job", {"enhanced_json": str(test_job_dir / "e.json")}, {"prompt": "p", "references": ["a" * 64]})


@pytest.mark.parametrize("vision", [False, True])
def test_enhance_load_picks_the_variant_from_the_checkpoint(test_job_dir, monkeypatch, vision):
    config = {"architectures": ["X"], **({"vision_config": {"depth": 27}} if vision else {})}
    (test_job_dir / "config.json").write_text(json.dumps(config))
    monkeypatch.setenv("ENHANCER_MODEL_PATH", str(test_job_dir))
    import aries_worker.stages.enhance as enhance

    with (
        patch.object(enhance, "AutoProcessor") as processor,
        patch.object(enhance, "AutoModelForImageTextToText") as i2i,
        patch.object(enhance, "AutoTokenizer") as tokenizer,
        patch.object(enhance, "AutoModelForCausalLM") as t2i,
    ):
        runner = EnhanceStageRunner()
        runner.load()
    assert runner.vision is vision
    assert i2i.from_pretrained.called is vision and processor.from_pretrained.called is vision
    assert t2i.from_pretrained.called is not vision and tokenizer.from_pretrained.called is not vision
    if vision:
        assert runner.tokenizer is processor.from_pretrained.return_value.tokenizer


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
    assert stats["max_new_tokens"] == MAX_NEW_TOKENS
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
    """EncodeStageRunner encodes the prompt Studio resolved and writes embeds.safetensors."""
    runner = EncodeStageRunner()
    mock_pipe = MagicMock()
    mock_pipe.encode_prompt.return_value = (MagicMock(), MagicMock(), MagicMock())
    runner.pipe = mock_pipe
    paths = {"embeds_safetensors": str(test_job_dir / "embeds.safetensors")}

    with patch("aries_worker.stages.encode.atomic_safetensors_save", return_value=64) as save:
        timings, _ = runner.run_job(
            "job-456", paths, {"prompt": "enhanced neon alley", "prompt_source": "enhanced"}
        )
        assert "encode_s" in timings and "save_s" in timings
        assert timings["prompt_source"] == "enhanced"
        tensors, path = save.call_args[0]
        assert set(tensors) == {"prompt_embeds", "prompt_embeds_mask", "image_pad_mask"}
        assert save.call_args[1]["metadata"]["prompt"] == "enhanced neon alley"
        assert str(path) == paths["embeds_safetensors"]

    with pytest.raises(ValueError):
        runner.run_job("job-456", paths, {"prompt": ""})


def test_denoise_runner_execution(test_job_dir):
    """DenoiseStageRunner reports progress and writes latent.safetensors."""
    runner = DenoiseStageRunner()
    embeds_path = test_job_dir / "embeds.safetensors"
    embeds_path.write_bytes(b"x")
    mock_pipe = MagicMock()

    def fake_pipe(**kwargs):
        for step in range(kwargs["num_inference_steps"]):
            kwargs["callback_on_step_end"](mock_pipe, step, None, {})
        out = MagicMock()
        out.images.detach.return_value.cpu.return_value = MagicMock()
        return out

    mock_pipe.side_effect = fake_pipe
    runner.pipe = mock_pipe
    paths = {"embeds_safetensors": str(embeds_path), "latent_safetensors": str(test_job_dir / "latent.safetensors")}
    progress_reports = []

    with (
        patch("aries_worker.stages.denoise.load_safetensors_with_metadata") as load,
        patch("aries_worker.stages.denoise.atomic_safetensors_save", return_value=64) as save,
        patch("torch.Generator"),
    ):
        load.return_value = ({"prompt_embeds": MagicMock()}, {"prompt_source": "original"})
        timings, _ = runner.run_job(
            "job-789",
            paths,
            {"steps": 10, "seed": 42, "width": 2048, "height": 2048},
            progress_cb=lambda step, steps, sps: progress_reports.append((step, steps)),
        )

    assert "denoise_s" in timings and timings["steps"] == 10
    assert progress_reports[-1] == (10, 10)
    save.assert_called_once()
    assert save.call_args[1]["metadata"]["seed"] == "42"


def test_denoise_with_references_runs_its_own_loop(test_job_dir):
    """References: stored latents and the mask go through our loop, never the pipeline call."""
    runner = DenoiseStageRunner()
    runner.pipe = MagicMock()
    runner.pipe.config.sample_sigmas = [1.0, 0.5]
    runner.pipe.scheduler.step.side_effect = lambda pred, t, latents, return_dict: (latents,)
    paths = {
        "embeds_safetensors": str(test_job_dir / "embeds.safetensors"),
        "latent_safetensors": str(test_job_dir / "latent.safetensors"),
        "ref0.safetensors": str(test_job_dir / "ref0.safetensors"),
    }
    Path(paths["embeds_safetensors"]).write_bytes(b"x")
    loaded = {
        paths["embeds_safetensors"]: (
            {"prompt_embeds": MagicMock(), "image_pad_mask": MagicMock()},
            {"prompt_source": "original", "reference_sizes": "[[1024, 1024]]"},
        ),
        paths["ref0.safetensors"]: ({"latents": MagicMock()}, {"width": "1024", "height": "1024"}),
    }
    progress = []
    with (
        patch("aries_worker.stages.denoise.load_safetensors_with_metadata", side_effect=lambda p, device: loaded[str(p)]),
        patch("aries_worker.stages.denoise.atomic_safetensors_save", return_value=64) as save,
        patch("aries_worker.stages.denoise.retrieve_timesteps", return_value=([MagicMock(), MagicMock()], 2)),
        patch("aries_worker.stages.denoise.calculate_shift", return_value=1.0),
        patch("torch.Generator"),
    ):
        timings, _ = runner.run_job(
            "job-ref",
            paths,
            {"steps": 2, "seed": 7, "width": 1024, "height": 1024, "references": 1},
            progress_cb=lambda step, steps, sps: progress.append(step),
        )

    runner.pipe.assert_not_called()
    assert runner.pipe.transformer.call_count == 2
    img_shapes = runner.pipe.transformer.call_args.kwargs["img_shapes"]
    assert img_shapes == [[(1, 64, 64), (1, 64, 64)]]
    assert progress == [1, 2]
    assert timings["references"] == 1
    assert save.call_args.kwargs["metadata"]["references"] == "1"

    # Embeds encoded for other references than the latents: refuse rather than mix them up.
    loaded[paths["embeds_safetensors"]][1]["reference_sizes"] = "[[512, 512]]"
    with (
        patch("aries_worker.stages.denoise.load_safetensors_with_metadata", side_effect=lambda p, device: loaded[str(p)]),
        pytest.raises(ValueError, match="encoded for references"),
    ):
        runner.run_job("job-ref", paths, {"steps": 2, "seed": 7, "width": 1024, "height": 1024, "references": 1})


def test_decode_split_columns_cover_the_image_once():
    """Strips meet at their cuts, each reaching `overlap` past them, and are small enough."""
    from aries_worker.stages.decode import split_columns

    # 2752x1536: 172 decoder columns of 16 px -> two halves, 32 columns of overlap.
    strips = split_columns(172, 16, 1536, 3_200_000, 32)
    assert strips == [(0, 118, 0, 86), (54, 172, 86, 172)]
    for columns, height in [(172, 1536), (128, 2048), (150, 1792), (300, 3000)]:
        strips = split_columns(columns, 16, height, 3_200_000, 32)
        assert len(strips) >= 2
        assert strips[0][2] == 0 and strips[-1][3] == columns
        for (a, b, ka, kb), nxt in zip(strips, strips[1:] + [None], strict=True):
            assert a == max(0, ka - 32) and b == min(columns, kb + 32)
            if nxt:
                assert kb == nxt[2]  # contiguous, no gap, no double cover
            assert (b - a) * 16 * height <= 3_200_000 or len(strips) == columns
    # A bigger image gets more strips.
    assert len(split_columns(300, 16, 3000, 3_200_000, 32)) > 2
    # Portrait 1792x2400 is split across its 150 rows, like landscape across columns: 2 strips.
    assert len(split_columns(150, 16, 1792, 3_200_000, 32)) == 2


@pytest.mark.parametrize("mode,size,env,split", [
    ("untiled", (2752, 1536), None, True),     # full size: split
    ("untiled", (1376, 768), None, False),     # ½ size: plain untiled
    ("tiled", (2752, 1536), None, False),      # tiling asked for: tiled
    ("untiled", (2752, 1536), "0", False),     # split turned off
])
def test_decode_splits_large_untiled_decodes(test_job_dir, monkeypatch, mode, size, env, split):
    import numpy as np
    import aries_worker.stages.decode as decode

    if env is not None:
        monkeypatch.setenv("ARIES_DECODE_SPLIT_ABOVE_PX", env)
    else:
        monkeypatch.delenv("ARIES_DECODE_SPLIT_ABOVE_PX", raising=False)
    runner = DecodeStageRunner()
    runner.vae = MagicMock()
    runner._prepare_latents = MagicMock(return_value="z")
    w, h = size
    decoded = MagicMock(ndim=4)
    decoded.__truediv__.return_value.__add__.return_value.clamp.return_value.cpu.return_value.permute.return_value.float.return_value.numpy.return_value = [np.zeros((8, 8, 4), np.float32)]
    runner.vae.decode.return_value = (decoded,)
    latent_path = test_job_dir / "latent.safetensors"
    latent_path.write_bytes(b"x")
    with (
        patch.object(decode, "load_safetensors_with_metadata", return_value=({"latent": MagicMock()}, {"width": str(w), "height": str(h)})),
        patch.object(decode, "decode_split", return_value=(np.zeros((8, 8, 4), np.float32), 2)) as split_fn,
    ):
        timings, _ = runner.run_job("job", {"latent_safetensors": str(latent_path), "image_png": str(test_job_dir / "image.png")},
                                    {"decode_mode": mode})
    assert split_fn.called is split
    assert runner.vae.decode.called is not split
    assert timings["decode_parts"] == (2 if split else 1)
    assert timings["decode_mode"] == mode
    assert (test_job_dir / "image.png").exists()


@pytest.mark.parametrize("mode", ["untiled", "tiled"])
def test_decode_runner_execution(test_job_dir, mode):
    """DecodeStageRunner unpacks a packed latent from latent.safetensors and writes image.png,
    with VAE tiling only when the assignment asks for it."""
    runner = DecodeStageRunner()

    latent_path = test_job_dir / "latent.safetensors"
    latent_path.write_bytes(b"dummy_latent_content")

    mock_vae = MagicMock()
    runner.vae = mock_vae
    runner.is_loaded = True

    paths = {
        "latent_safetensors": str(latent_path),
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

    with (
        patch("aries_worker.stages.decode.load_safetensors_with_metadata") as mock_load,
        patch("PIL.Image.fromarray") as mock_fromarray,
    ):
        fake_img = Image.new("RGB", (64, 64), color="blue")
        mock_fromarray.return_value = fake_img

        mock_load.return_value = ({"latent": fake_3d_latents}, {"width": "64", "height": "64"})

        timings, peak = runner.run_job(
            "job-decode",
            paths,
            {"decode_mode": mode, "width": 64, "height": 64},
        )

        assert timings["decode_mode"] == mode
        if mode == "tiled":
            mock_vae.enable_tiling.assert_called_once()
            mock_vae.disable_tiling.assert_not_called()
        else:
            mock_vae.disable_tiling.assert_called_once()
            mock_vae.enable_tiling.assert_not_called()
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
        assert "pci" in hello
        assert isinstance(hello["devices"], list)

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
        socket_path=str(sock_path),
        runner=mock_runner,
        idle_grace=1.0,
    )

    await client.run()
    server.close()
    await server.wait_closed()


def test_v3_safetensors_encode_and_denoise(test_job_dir):
    """Test V3 safetensors path for encode (with resolved prompt_source and metadata) and denoise (with CPU generator)."""
    runner = EncodeStageRunner()
    mock_pipe = MagicMock()
    mock_embeds = MagicMock()
    mock_embeds.cpu.return_value.contiguous.return_value = "contig_embeds"
    mock_mask = MagicMock()
    mock_mask.cpu.return_value.contiguous.return_value = "contig_mask"
    mock_pipe.encode_prompt.return_value = (mock_embeds, mock_mask, None)
    runner.pipe = mock_pipe

    embeds_path = test_job_dir / "embeds.safetensors"
    with patch("aries_worker.stages.encode.atomic_safetensors_save") as mock_st_save:
        mock_st_save.return_value = 512
        timings, _ = runner.run_job(
            "task-enc-v3",
            {"embeds_safetensors": str(embeds_path)},
            {
                "prompt": "resolved enhanced prompt",
                "prompt_source": "enhanced",
                "model_id": "Qwen/Qwen-Image-2.1",
                "model_rev": "rev123",
                "model_dtype": "bf16",
            },
        )
        assert timings["prompt_source"] == "enhanced"
        mock_st_save.assert_called_once()
        tensors_arg, path_arg = mock_st_save.call_args[0]
        meta_arg = mock_st_save.call_args[1]["metadata"]
        assert tensors_arg["prompt_embeds"] == "contig_embeds"
        assert meta_arg["prompt"] == "resolved enhanced prompt"
        assert meta_arg["prompt_source"] == "enhanced"
        assert meta_arg["model"] == "Qwen/Qwen-Image-2.1"
        assert meta_arg["rev"] == "rev123"
        assert meta_arg["dtype"] == "bf16"

    # Denoise with safetensors and CPU generator check
    den_runner = DenoiseStageRunner()
    den_pipe = MagicMock()
    mock_lat = MagicMock()
    mock_lat.detach.return_value.cpu.return_value.contiguous.return_value = "contig_latent"
    den_pipe.return_value.images = mock_lat
    den_runner.pipe = den_pipe

    embeds_path.write_bytes(b"fake_safetensors")
    latent_path = test_job_dir / "latent.safetensors"
    with (
        patch("aries_worker.stages.denoise.load_safetensors_with_metadata") as mock_st_load,
        patch("aries_worker.stages.denoise.atomic_safetensors_save") as mock_st_save_den,
        patch("torch.Generator") as mock_gen_cls,
    ):
        mock_st_load.return_value = (
            {"prompt_embeds": "fake_emb"},
            {"prompt_source": "enhanced", "model": "Qwen/Qwen-Image-2.1", "rev": "rev123", "dtype": "bf16"},
        )
        mock_st_save_den.return_value = 2048
        den_timings, _ = den_runner.run_job(
            "task-den-v3",
            {"embeds_safetensors": str(embeds_path), "latent_safetensors": str(latent_path)},
            {"width": 2048, "height": 2048, "steps": 20, "seed": 777, "true_cfg_scale": 1.0},
        )
        # Verify noise generator was created on "cpu" per V3 §3.1
        mock_gen_cls.assert_called_once_with("cpu")
        assert den_timings["prompt_source"] == "enhanced"
        mock_st_save_den.assert_called_once()



def test_pci_from_level_zero_uuid():
    from aries_worker.stages.loader import pci_from_level_zero_uuid

    # As reported by torch.xpu on the two Arc Pro B60 cards of the reference host.
    assert pci_from_level_zero_uuid("868011e2-0000-0000-0400-000000000000") == "0000:04:00.0"
    assert pci_from_level_zero_uuid("868011e2-0000-0000-0800-000000000000") == "0000:08:00.0"
    assert pci_from_level_zero_uuid("de10f0ab-0000-0000-0100-000000000000") == ""  # not Intel
    assert pci_from_level_zero_uuid("garbage") == ""


def test_probe_xpu_devices_ignores_phantom_count(monkeypatch):
    """device_count() says 2 under a one-card selector; only index 0 opens."""
    from aries_worker.stages import loader

    def props(idx):
        if idx > 0:
            raise RuntimeError("The device index is out of range. It must be in [0, 1)")
        return MagicMock(uuid="868011e2-0000-0000-0800-000000000000")

    monkeypatch.setattr(loader.torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(loader.torch.xpu, "get_device_properties", props)
    assert loader.probe_xpu_devices() == ["0000:08:00.0"]
