"""Enhance stage: Qwen-Image-2.1-PE-T2I rewrites the user prompt -> enhanced.json."""
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aries_worker.stages.files import atomic_json_save
from aries_worker.stages.loader import (
    flush_xpu_cache,
    get_device,
    get_peak_vram_bytes,
    reset_peak_vram,
    resolve_enhancer_model_path,
)

logger = logging.getLogger("aries_worker.stages.enhance")

SYSTEM_PROMPT_FALLBACK = """# Image Prompt Rewriting Expert

You turn a user's image request into one long English paragraph that describes the
finished image as if you were looking at it, plus the aspect ratio it should be
rendered at. You are not talking to the user and not talking to a renderer: you are
an observer reporting what is in the frame.

## Output format

Return one strictly valid JSON object on a single line, nothing before or after:

{"rewritten_prompt": "<the description>", "wh_ratio": "<e.g. 3:2>"}"""


# Reasoning plus reply rarely needs more; a run that hits the limit is kept as is
# (the salvaged prompt is already detailed). Also the finish line for progress.
MAX_NEW_TOKENS = 2048
THINK_END = "</think>"
VALID_RATIOS = {"1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"}


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
    thinking = ""
    payload_text = result_text

    if "</think>" in result_text:
        parts = result_text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        payload_text = parts[1].strip()
    elif "<think>" in result_text:
        parts = result_text.split("<think>", 1)
        thinking = parts[1].strip()

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
                    if k != "wh_ratio" and isinstance(v, str) and len(v.strip()) > 0:
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
    clean_text = re.sub(r'"?\s*,?\s*"wh_ratio".*$', "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'"?\s*\}?\s*$', "", clean_text).strip()
    return {
        "enhanced_prompt": clean_text if clean_text else original_prompt,
        "wh_ratio": "1:1",
        "thinking": thinking,
    }



class EnhanceCancelled(Exception):
    """Raised from the streamer when the manager cancels the job."""


def _flatten_ids(value: Any) -> Iterator[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_ids(item)
    else:
        yield int(value)


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


class EnhanceStageRunner:
    def __init__(self) -> None:
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.load_s = 0.0

    def load(self) -> float:
        t0 = time.monotonic()
        model_path = resolve_enhancer_model_path()
        logger.info("Loading Qwen-Image-2.1-PE-T2I from %s...", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(get_device())

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
        t0 = time.monotonic()
        reset_peak_vram()

        messages = [
            {"role": "system", "content": load_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(get_device())
        input_ids = inputs["input_ids"]

        streamer = EnhanceProgressStreamer(
            tokenizer=self.tokenizer,
            prompt_tokens=int(input_ids.shape[-1]),
            progress_cb=progress_cb,
            cancel_check=cancel_check,
        )
        streamer.emit_progress()  # report the prompt size before prefill starts

        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.05,
                    do_sample=True,
                    streamer=streamer,
                )
            peak_vram = get_peak_vram_bytes()
        finally:
            flush_xpu_cache()

        if streamer.generated_tokens >= MAX_NEW_TOKENS:
            logger.info("Enhancer output for %s hit max_new_tokens=%d; keeping it as is", job_id, MAX_NEW_TOKENS)

        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, outputs)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        parsed = _parse_enhanced_json(response, prompt)
        atomic_json_save({
            "rewritten_prompt": parsed["enhanced_prompt"],
            "wh_ratio": parsed["wh_ratio"],
            "thinking": parsed["thinking"],
        }, Path(paths["enhanced_json"]))

        duration = round(time.monotonic() - t0, 3)
        logger.info(
            "Enhance completed for job %s in %.3fs (prefill %.1f tok/s, generation %.1f tok/s)",
            job_id, duration, streamer.prefill_tps, streamer.generation_tps,
        )
        return {"enhance_s": duration, **streamer.stats()}, peak_vram
