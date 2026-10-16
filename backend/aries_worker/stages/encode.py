"""Encode stage: Qwen3-VL text encoder on one card -> embeds.safetensors."""
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers import QwenImage21Pipeline
from transformers import Qwen3VLForConditionalGeneration

from aries_worker.stages.files import atomic_safetensors_save
from aries_worker.stages.loader import (
    flush_xpu_cache,
    get_device,
    get_peak_vram_bytes,
    reset_peak_vram,
    resolve_model_dtype,
    resolve_model_id,
    resolve_model_path,
    resolve_model_rev,
)

logger = logging.getLogger("aries_worker.stages.encode")
EMBEDS_FORMAT = 1


def _to_cpu_contiguous(tensor: Any) -> Any:
    if tensor is None:
        return None
    cpu_t = tensor.cpu() if hasattr(tensor, "cpu") else tensor
    if hasattr(cpu_t, "contiguous"):
        return cpu_t.contiguous()
    return cpu_t


class EncodeStageRunner:
    def __init__(self) -> None:
        self.pipe: Optional[QwenImage21Pipeline] = None
        self.load_s = 0.0

    def load(self) -> float:
        t0 = time.monotonic()
        model_path = resolve_model_path()
        logger.info("Loading text encoder from %s onto %s", model_path, get_device())

        # Straight onto the card: never stage 16 GB in host RAM (HARDWARE.md F3).
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map=get_device(),
        )
        self.pipe = QwenImage21Pipeline.from_pretrained(
            model_path,
            text_encoder=text_encoder,
            transformer=None,
            vae=None,
            torch_dtype=torch.bfloat16,
        )
        if not callable(getattr(self.pipe, "encode_prompt", None)):
            raise RuntimeError("QwenImage21Pipeline.encode_prompt missing - diffusers version mismatch")

        self.load_s = round(time.monotonic() - t0, 3)
        logger.info("Text encoder loaded in %.2fs", self.load_s)
        return self.load_s

    def run_job(self, job_id: str, paths: Dict[str, str], params: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        if self.pipe is None:
            raise RuntimeError("encode stage not loaded")
        # Studio resolves the prompt to encode (original or enhanced, ARCHITECTURE_V3.md §7.2).
        prompt = params.get("prompt")
        if not prompt:
            raise ValueError(f"{job_id}: no prompt to encode")
        source = str(params.get("prompt_source") or "original")
        true_cfg_scale = float(params.get("true_cfg_scale", 1.0))
        negative_prompt = params.get("negative_prompt")
        need_negative = true_cfg_scale > 1.0 or (isinstance(negative_prompt, str) and len(negative_prompt) > 0)

        reset_peak_vram()
        try:
            t0 = time.monotonic()
            with torch.inference_mode():
                # Returns (prompt_embeds, prompt_embeds_mask, image_pad_mask)
                prompt_embeds, prompt_embeds_mask, image_pad_mask = self.pipe.encode_prompt(
                    prompt=prompt,
                    device=get_device(),
                    num_images_per_prompt=1,
                )
                neg_embeds = None
                neg_mask = None
                if need_negative:
                    neg_text = negative_prompt if isinstance(negative_prompt, str) else ""
                    neg_embeds, neg_mask, _ = self.pipe.encode_prompt(
                        prompt=neg_text,
                        device=get_device(),
                        num_images_per_prompt=1,
                    )
            encode_s = time.monotonic() - t0

            t1 = time.monotonic()
            model_id = params.get("model_id") or resolve_model_id("encode")
            model_rev = params.get("model_rev") or resolve_model_rev("encode")
            model_dtype = params.get("model_dtype") or resolve_model_dtype()

            tensors: Dict[str, Any] = {"prompt_embeds": _to_cpu_contiguous(prompt_embeds)}
            for name, tensor in (
                ("prompt_embeds_mask", prompt_embeds_mask),
                ("image_pad_mask", image_pad_mask),
                ("negative_prompt_embeds", neg_embeds),
                ("negative_prompt_embeds_mask", neg_mask),
            ):
                if tensor is not None:
                    tensors[name] = _to_cpu_contiguous(tensor)
            size = atomic_safetensors_save(
                tensors,
                Path(paths["embeds_safetensors"]),
                metadata={
                    "format": str(EMBEDS_FORMAT),
                    "prompt": prompt,
                    "prompt_source": source,
                    "model": str(model_id),
                    "rev": str(model_rev),
                    "dtype": str(model_dtype),
                },
            )
            save_s = time.monotonic() - t1
        finally:
            peak_vram = get_peak_vram_bytes()
            flush_xpu_cache()

        logger.info(
            "encode %s: %.3fs encode, %.3fs save, %d bytes, prompt=%s",
            job_id, encode_s, save_s, size, source,
        )
        return {
            "encode_s": round(encode_s, 3),
            "save_s": round(save_s, 3),
            "prompt_source": source,
        }, peak_vram

