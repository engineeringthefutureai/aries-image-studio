"""Encode stage: Qwen3-VL text encoder on one card -> embeds.pt."""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers import QwenImage21Pipeline
from transformers import Qwen3VLForConditionalGeneration

from aries_worker.stages.files import atomic_torch_save
from aries_worker.stages.loader import (
    flush_xpu_cache,
    get_device,
    get_peak_vram_bytes,
    reset_peak_vram,
    resolve_model_path,
)

logger = logging.getLogger("aries_worker.stages.encode")
EMBEDS_FORMAT = 1


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

    @staticmethod
    def _effective_prompt(job_id: str, paths: Dict[str, str], params: Dict[str, Any]) -> Tuple[str, str]:
        """The prompt to encode: the enhancer's rewrite if the job was enhanced, else the original."""
        if params.get("enhance"):
            enh_path = Path(paths["enhanced_json"])
            if not enh_path.is_file():
                raise FileNotFoundError(f"{job_id}: expected enhanced prompt file {enh_path} does not exist")
            try:
                data = json.loads(enh_path.read_text())
            except Exception as e:
                raise ValueError(f"{job_id}: failed to parse {enh_path}: {e}")
            prompt, source = data.get("rewritten_prompt"), "enhanced"
        else:
            prompt, source = params.get("prompt"), "original"

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{job_id}: empty {source} prompt")
        return prompt, source

    def run_job(self, job_id: str, paths: Dict[str, str], params: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        if self.pipe is None:
            raise RuntimeError("encode stage not loaded")
        prompt, source = self._effective_prompt(job_id, paths, params)

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
            encode_s = time.monotonic() - t0

            t1 = time.monotonic()
            size = atomic_torch_save({
                "format": EMBEDS_FORMAT,
                "prompt": prompt,
                "prompt_source": source,
                "prompt_embeds": prompt_embeds.cpu(),
                "prompt_embeds_mask": None if prompt_embeds_mask is None else prompt_embeds_mask.cpu(),
                "image_pad_mask": None if image_pad_mask is None else image_pad_mask.cpu(),
                "negative_prompt_embeds": None,  # denoise refuses true_cfg_scale > 1 without these
            }, Path(paths["embeds_pt"]))
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
