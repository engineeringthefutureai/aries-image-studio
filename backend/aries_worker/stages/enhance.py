"""Enhance stage: the prompt enhancer rewrites the user prompt -> enhanced.json.

Two enhancers share the stage (docs/ENHANCER_I2I.md): Qwen-Image-2.1-PE-T2I reads text,
Qwen-Image-2.1-PE-I2I also reads the job's reference images. A worker loads one and picks
the code path from the checkpoint (I2I has a `vision_config`).
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from PIL import Image

from aries_worker.stages.files import atomic_json_save
from aries_worker.stages.loader import (
    flush_xpu_cache,
    get_device,
    get_peak_vram_bytes,
    reset_peak_vram,
    resolve_enhancer_model_path,
)
from aries_worker.stages.references import DEFAULT_REFERENCE_RES, open_reference, reference_size

logger = logging.getLogger("aries_worker.stages.enhance")

SYSTEM_PROMPT_FALLBACK = """# Image Prompt Rewriting Expert

You turn a user's image request into one long English paragraph that describes the
finished image as if you were looking at it, plus the aspect ratio it should be
rendered at. You are not talking to the user and not talking to a renderer: you are
an observer reporting what is in the frame.

## Output format

Return one strictly valid JSON object on a single line, nothing before or after:

{"rewritten_prompt": "<the description>", "wh_ratio": "<e.g. 3:2>"}"""


# Default limit (reasoning plus reply): 4096, plus 2048 per reference image; jobs override
# it. Also the finish line for progress. PE-T2I stops well before 4096; PE-I2I used
# 2,900-4,100 with 1-4 images in testing and ran past 4096 with three live.
MAX_NEW_TOKENS = 4096
TOKENS_PER_REFERENCE = 2048
MAX_DEFAULT_TOKENS = 24576
# Reasoning that runs out of room continues with the limit doubled, up to this (Studio's
# own cap, ~13 minutes at 31 tok/s; the PE-I2I model card generates with 24000); only there
# is it closed for the reply.
TOKENS_CEILING = 24576


def default_max_new_tokens(references: int) -> int:
    return min(MAX_DEFAULT_TOKENS, MAX_NEW_TOKENS + TOKENS_PER_REFERENCE * references)
# Tokens kept for the reply: reasoning that reaches the rest of the limit gets a doubled
# limit, or at the ceiling is closed (budget forcing) and the model writes its JSON in what
# is left. Replies run 400-800 tokens.
REPLY_RESERVE_MAX = 1536
THINK_CLOSE = "\n</think>\n\n"
THINK_END = "</think>"
VALID_RATIOS = {"1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"}
# PE-I2I: keep the aspect of input N instead of choosing a ratio.
RATIO_FOLLOW_RE = re.compile(r'"ratio_follow"\s*:\s*"(<image\d+>)"')


def load_system_prompt() -> str:
    """System prompt: the model directory's copy if present, else the packaged one."""
    for candidate in (
        Path(resolve_enhancer_model_path()) / "system_prompt.txt",
        Path(__file__).parent / "system_prompt.txt",
    ):
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return SYSTEM_PROMPT_FALLBACK


def _parse_enhanced_json(result_text: str, original_prompt: str) -> Dict[str, Any]:
    """Parse thinking and JSON output from model response."""
    parsed = _parse_reply(result_text, original_prompt)
    payload_text = result_text.split(THINK_END, 1)[-1]
    match = RATIO_FOLLOW_RE.search(payload_text)
    parsed["ratio_follow"] = match.group(1) if match else ""
    return parsed


def _parse_reply(result_text: str, original_prompt: str) -> Dict[str, Any]:
    thinking = ""
    payload_text = result_text

    if "</think>" in result_text:
        parts = result_text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        payload_text = parts[1].strip()
    else:
        # The chat template opens <think>, so text without </think> is reasoning that never
        # finished, unless it is a bare JSON reply. Never take a prompt out of reasoning (it
        # drafts the format, e.g. "rewritten_prompt": "..."): keep the original prompt.
        bare = result_text.replace("<think>", "", 1).strip()
        if not bare.startswith(("{", "```")):
            return {"enhanced_prompt": original_prompt, "wh_ratio": "1:1", "thinking": bare}
        payload_text = bare

    # 1. Try standard JSON extraction (markdown block or outer braces)
    json_str = ""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", payload_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_match = re.search(r"\{.*\}", payload_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)

    if json_str:
        try:
            data = json.loads(json_str)
            wh = data.get("wh_ratio", "1:1")
            if wh not in VALID_RATIOS:
                wh = "1:1"
            # Prioritized key check for standard and potential typo/hallucinated fields
            enh_prompt = (
                data.get("rewritten_prompt")
                or data.get("enhanced_prompt")
                or data.get("enhritten_prompt")
                or data.get("enhanted_prompt")
            )
            # If not matching the common keys, find any non-ratio string value
            if not enh_prompt:
                for k, v in data.items():
                    if k not in ("wh_ratio", "ratio_follow") and isinstance(v, str) and len(v.strip()) > 0:
                        enh_prompt = v
                        break

            if enh_prompt and isinstance(enh_prompt, str):
                return {
                    "enhanced_prompt": enh_prompt.strip(),
                    "wh_ratio": wh,
                    "thinking": thinking,
                }
        except json.JSONDecodeError:
            pass

    # 2. Resilient regex extraction if JSON was truncated or malformed
    ep_match = re.search(
        r'"(?:rewritten_prompt|enhanced_prompt|enhritten_prompt|enhanted_prompt|[^"]*prompt[^"]*)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)',
        payload_text,
        re.IGNORECASE,
    ) or re.search(
        r'\{\s*"[^"]+"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)',
        payload_text,
    )
    if ep_match:
        extracted = ep_match.group(1).replace(r'\"', '"').replace(r'\n', '\n').strip()
        wh_match = re.search(r'"wh_ratio"\s*:\s*"([^"]+)"', payload_text)
        wh = wh_match.group(1) if wh_match else "1:1"
        return {
            "enhanced_prompt": extracted if extracted else original_prompt,
            "wh_ratio": wh if wh in VALID_RATIOS else "1:1",
            "thinking": thinking,
        }

    # 3. Fallback clean up
    clean_text = re.sub(r"```json|```", "", payload_text).strip()
    clean_text = re.sub(r'^\s*\{\s*"[^"]+"\s*:\s*"?', "", clean_text)
    clean_text = re.sub(r'"?\s*,?\s*"(?:wh_ratio|ratio_follow)".*$', "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'"?\s*\}?\s*$', "", clean_text).strip()
    return {
        "enhanced_prompt": clean_text if clean_text else original_prompt,
        "wh_ratio": "1:1",
        "thinking": thinking,
    }



class EnhanceCancelled(Exception):
    """Raised from the streamer when the agent cancels the job."""


def _flatten_ids(value: Any) -> Iterator[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_ids(item)
    else:
        yield int(value)


def reply_reserve(max_new_tokens: int) -> int:
    return min(REPLY_RESERVE_MAX, max_new_tokens // 4)


class ThinkingBudget:
    """`generate()` stopping criterion: stop when reasoning uses up its share of the limit."""

    def __init__(self, streamer: "EnhanceProgressStreamer", budget: int):
        self.streamer = streamer
        self.budget = budget
        self.triggered = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        stop = not self.streamer.has_exited_think and self.streamer.thought_tokens >= self.budget
        self.triggered = self.triggered or stop
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)


def _continue_inputs(inputs: Dict[str, Any], seq: Any) -> Dict[str, Any]:
    """`generate()` inputs for `seq` (the prompt plus what follows it): per-token tensors
    (attention mask, token types) grow with it; images stay as they are."""
    n = inputs["input_ids"].shape[-1]
    shape = tuple(inputs["input_ids"].shape)
    out = dict(inputs)
    for key, value in inputs.items():
        if key != "input_ids" and tuple(getattr(value, "shape", ())) == shape:
            fill = torch.ones if key == "attention_mask" else torch.zeros
            extra = fill((value.shape[0], seq.shape[-1] - n), dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, extra], dim=-1)
    out["input_ids"] = seq
    return out


class EnhanceProgressStreamer:
    """Token streamer for `generate()` that tracks sub-stages and throughput.

    HF `generate()` calls `put()` once with the full prompt *before* the prefill
    forward pass, then once per generated token, then `end()`. So prefill ends at
    the first generated token, not at the prompt put.

    The PE model's chat template opens `<think>` in the prompt, so generation starts
    in the reasoning phase and switches to the JSON reply at `</think>`. Output that
    starts directly with `{` has no reasoning phase.
    """

    def __init__(
        self,
        tokenizer: Any,
        prompt_tokens: int,
        progress_cb: Optional[Callable[..., None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        emit_interval: float = 0.25,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.progress_cb = progress_cb
        self.cancel_check = cancel_check
        self.emit_interval = emit_interval
        self.max_new_tokens = max_new_tokens

        self.t_start = time.monotonic()
        self.t_first_token: Optional[float] = None
        self.t_last_emit = 0.0
        self.prompt_seen = False

        self.prefill_s = 0.0
        self.prefill_tps = 0.0
        self.generation_tps = 0.0

        self.status = "prefill"
        self.has_exited_think = False
        self.thought_tokens = 0
        self.reply_tokens = 0
        self.generated_tokens = 0
        self.generated_text = ""

    def put(self, value: Any) -> None:
        if self.cancel_check and self.cancel_check():
            raise EnhanceCancelled("Generation cancelled by user")

        token_ids: List[int] = list(_flatten_ids(value))
        if not token_ids:
            return

        if not self.prompt_seen:
            # The prompt echo: prefill has not run yet.
            self.prompt_seen = True
            if not self.prompt_tokens:
                self.prompt_tokens = len(token_ids)
            return

        now = time.monotonic()
        if self.t_first_token is None:
            self.t_first_token = now
            self.prefill_s = max(1e-4, now - self.t_start)
            self.prefill_tps = round(self.prompt_tokens / self.prefill_s, 1) if self.prompt_tokens else 0.0

        chunk = ""
        if self.tokenizer is not None:
            try:
                chunk = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            except Exception:
                pass
        self.generated_text += chunk
        self.generated_tokens += len(token_ids)

        # Tokens up to and including </think> count as reasoning.
        if self.has_exited_think:
            self.reply_tokens += len(token_ids)
        else:
            self.thought_tokens += len(token_ids)
            tail = self.generated_text[-(len(chunk) + len(THINK_END)):]
            if THINK_END in tail or self.generated_text.lstrip().startswith("{"):
                self.has_exited_think = True

        gen_elapsed = max(1e-4, now - self.t_first_token)
        self.generation_tps = round(self.generated_tokens / gen_elapsed, 1)

        prev_status = self.status
        self.status = "rewriting" if self.has_exited_think else "thinking"
        if self.status != prev_status or now - self.t_last_emit >= self.emit_interval:
            self.emit_progress()

    def end(self) -> None:
        self.emit_progress()

    def resume(self, max_new_tokens: int) -> None:
        """Generation goes on in another `generate()` (it starts by echoing its prompt, which
        now holds everything so far) under a raised limit."""
        self.prompt_seen = False
        self.max_new_tokens = max_new_tokens

    def close_thinking(self, text: str) -> None:
        """Reasoning was cut off and `text` closes it; the next `generate()` writes the reply
        and starts by echoing its prompt again."""
        self.prompt_seen = False
        self.has_exited_think = True
        self.status = "rewriting"
        self.generated_text += text

    def stats(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "max_new_tokens": self.max_new_tokens,
            "thought_tokens": self.thought_tokens,
            "reply_tokens": self.reply_tokens,
            "prefill_s": round(self.prefill_s, 3),
            "prefill_tps": self.prefill_tps,
            "generation_tps": self.generation_tps,
        }

    def emit_progress(self) -> None:
        self.t_last_emit = time.monotonic()
        if self.progress_cb:
            self.progress_cb(sub_stage=self.status, enhance_stats=self.stats())


def is_image_enhancer(model_path: str) -> bool:
    """PE-I2I's checkpoint has a vision tower (`vision_config`); PE-T2I's does not."""
    try:
        config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and "vision_config" in config


def load_reference_images(job_id: str, paths: Dict[str, str], count: int) -> List[Image.Image]:
    """The job's references upright, at the ~1 MP the text encoder reads them at."""
    images = []
    for i in range(count):
        image_path = paths.get(f"ref{i}.image")
        if not image_path:
            raise FileNotFoundError(f"{job_id}: reference {i} has no image input")
        image = open_reference(Path(image_path)).convert("RGB")
        size = reference_size(image.size[0], image.size[1], DEFAULT_REFERENCE_RES)
        images.append(image.resize(size, Image.LANCZOS))
    return images


class EnhanceStageRunner:
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.processor: Optional[Any] = None
        self.vision = False  # PE-I2I: reads reference images
        self.load_s = 0.0

    def load(self) -> float:
        t0 = time.monotonic()
        model_path = resolve_enhancer_model_path()
        self.vision = is_image_enhancer(model_path)
        logger.info(
            "Loading the %s enhancer from %s...", "image-aware (PE-I2I)" if self.vision else "text (PE-T2I)", model_path
        )

        # bf16, or a torchao-quantized checkpoint (its config carries the quantization);
        # device_map puts the weights straight on the card either way.
        if self.vision:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map=get_device(),
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map=get_device(),
                trust_remote_code=True,
            )

        self.load_s = round(time.monotonic() - t0, 3)
        logger.info("Enhance model loaded in %.2fs", self.load_s)
        return self.load_s

    def run_job(
        self,
        job_id: str,
        paths: Dict[str, str],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[..., None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("enhance stage not loaded")

        prompt = params.get("prompt", "")
        # Long, detailed prompts (multi-part infographics) need more than the default.
        ref_count = len(params.get("references") or [])
        max_new_tokens = int(params.get("max_new_tokens") or default_max_new_tokens(ref_count))
        if ref_count and not self.vision:
            raise ValueError(f"{job_id}: this enhancer does not read reference images")
        t0 = time.monotonic()
        reset_peak_vram()

        if self.vision:
            # PE-I2I: the images, then the instruction, in one user message; its recommended
            # sampling (model card).
            images = load_reference_images(job_id, paths, ref_count)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": load_system_prompt()}]},
                {"role": "user", "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": prompt},
                ]},
            ]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", enable_thinking=True,
            ).to(get_device())
            sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
        else:
            messages = [
                {"role": "system", "content": load_system_prompt()},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(get_device())
            sampling = {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05}
        input_ids = inputs["input_ids"]

        streamer = EnhanceProgressStreamer(
            tokenizer=self.tokenizer,
            prompt_tokens=int(input_ids.shape[-1]),
            progress_cb=progress_cb,
            cancel_check=cancel_check,
            max_new_tokens=max_new_tokens,
        )
        streamer.emit_progress()  # report the prompt size before prefill starts

        limit, raised, cut = max_new_tokens, False, False
        ceiling = max(max_new_tokens, TOKENS_CEILING)
        run_inputs = inputs
        try:
            with torch.inference_mode():
                while True:
                    budget = ThinkingBudget(streamer, limit - reply_reserve(limit))
                    outputs = self.model.generate(
                        **run_inputs,
                        max_new_tokens=max(1, limit - streamer.generated_tokens),
                        **sampling,
                        do_sample=True,
                        streamer=streamer,
                        stopping_criteria=[budget],
                    )
                    if not budget.triggered:
                        break
                    if limit < ceiling:
                        # Still reasoning: go on from here with the limit doubled, as a rerun
                        # with twice the limit would, without redoing the reasoning so far.
                        limit, raised = min(ceiling, limit * 2), True
                        logger.info(
                            "Enhancer reasoning for %s reached %d tokens; raising the limit to %d",
                            job_id, streamer.thought_tokens, limit,
                        )
                        streamer.resume(limit)
                        run_inputs = _continue_inputs(inputs, outputs)
                        continue
                    # At the ceiling: close the reasoning and let the model reply in what is left.
                    logger.info(
                        "Enhancer reasoning for %s reached %d tokens at the ceiling; closing it for the reply",
                        job_id, streamer.thought_tokens,
                    )
                    close = self.tokenizer(THINK_CLOSE, add_special_tokens=False, return_tensors="pt")["input_ids"]
                    seq = torch.cat([outputs, close.to(outputs.device)], dim=-1)
                    streamer.close_thinking(THINK_CLOSE)
                    outputs = self.model.generate(
                        **_continue_inputs(inputs, seq),
                        max_new_tokens=max(1, limit - streamer.generated_tokens),
                        **sampling,
                        do_sample=True,
                        streamer=streamer,
                    )
                    cut = True
                    break
            peak_vram = get_peak_vram_bytes()
        finally:
            flush_xpu_cache()

        if streamer.generated_tokens >= limit:
            logger.info("Enhancer output for %s hit max_new_tokens=%d; keeping it as is", job_id, limit)

        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, outputs)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        parsed = _parse_enhanced_json(response, prompt)
        enhanced = {
            "rewritten_prompt": parsed["enhanced_prompt"],
            "wh_ratio": parsed["wh_ratio"],
            "thinking": parsed["thinking"],
        }
        if parsed["ratio_follow"]:
            enhanced["ratio_follow"] = parsed["ratio_follow"]
        atomic_json_save(enhanced, Path(paths["enhanced_json"]))

        duration = round(time.monotonic() - t0, 3)
        logger.info(
            "Enhance completed for job %s in %.3fs (prefill %.1f tok/s, generation %.1f tok/s)",
            job_id, duration, streamer.prefill_tps, streamer.generation_tps,
        )
        timings = {"enhance_s": duration, **streamer.stats()}
        if raised:
            timings["limit_raised_from"] = max_new_tokens
        if cut:
            timings["thinking_cut"] = 1
        if ref_count:
            timings["references"] = ref_count
        return timings, peak_vram
