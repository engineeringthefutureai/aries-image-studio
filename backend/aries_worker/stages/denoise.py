"""Denoise stage: 7B DiT transformer on one card, embeds.pt -> latent.pt.

Emits progress every step. The latent is captured at the `vae.decode` input
(output_type="latent"), so the decode stage needs only the VAE.
"""
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

from aries_worker.stages.files import atomic_torch_save
from aries_worker.stages.loader import (
    flush_xpu_cache,
    get_device,
    get_peak_vram_bytes,
    reset_peak_vram,
    resolve_model_path,
)

logger = logging.getLogger("aries_worker.stages.denoise")

# Rolling window for the reported seconds-per-step.
SPS_WINDOW = 5


class DenoiseCancelled(Exception):
    """Raised from the step callback when the manager cancels the job."""


class DenoiseStageRunner:
    def __init__(self) -> None:
        self.pipe: Optional[QwenImage21Pipeline] = None
        self.load_s = 0.0

    def load(self) -> float:
        t0 = time.monotonic()
        model_path = resolve_model_path()
        logger.info("Loading QwenImage21 transformer from %s...", model_path)

        transformer = QwenImage21Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to(get_device())
        self.pipe = QwenImage21Pipeline.from_pretrained(
            model_path,
            transformer=transformer,
            text_encoder=None,
            vae=None,
            torch_dtype=torch.bfloat16,
        )

        self.load_s = round(time.monotonic() - t0, 3)
        logger.info("Transformer DiT loaded in %.2fs", self.load_s)
        return self.load_s

    def run_job(
        self,
        job_id: str,
        paths: Dict[str, str],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[..., None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        if self.pipe is None:
            raise RuntimeError("denoise stage not loaded")

        t0 = time.monotonic()
        reset_peak_vram()

        steps = int(params["steps"])
        seed = int(params["seed"])
        width = int(params["width"])
        height = int(params["height"])

        embeds_path = Path(paths["embeds_pt"])
        if not embeds_path.exists():
            raise FileNotFoundError(f"Required embeds.pt not found for job {job_id}: {embeds_path}")

        embeds = torch.load(embeds_path, map_location=get_device(), weights_only=False)
        if not isinstance(embeds, dict) or "prompt_embeds" not in embeds:
            raise ValueError(f"{job_id}: invalid embeds.pt payload structure")

        true_cfg = float(params.get("true_cfg_scale", 1.0))
        if true_cfg > 1.0 and embeds.get("negative_prompt_embeds") is None:
            raise ValueError(f"{job_id}: true_cfg_scale={true_cfg} requested, but embeds.pt has no negative_prompt_embeds")

        prompt_source = embeds.get("prompt_source", "unknown")
        generator = torch.Generator(get_device()).manual_seed(seed)
        step_times = []
        last_step_time = t0

        def step_callback(pipe, step_idx, timestep, kwargs):
            nonlocal last_step_time
            if cancel_check and cancel_check():
                raise DenoiseCancelled(f"Job {job_id} cancelled at step {step_idx}")

            now = time.monotonic()
            step_times.append(now - last_step_time)
            last_step_time = now

            recent = step_times[-SPS_WINDOW:]
            if progress_cb:
                progress_cb(step=step_idx, steps=steps, sps=round(sum(recent) / len(recent), 3))
            return kwargs

        try:
            with torch.inference_mode():
                output = self.pipe(
                    prompt=None,
                    prompt_embeds=embeds["prompt_embeds"],
                    prompt_embeds_mask=embeds.get("prompt_embeds_mask"),
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    generator=generator,
                    output_type="latent",
                    callback_on_step_end=step_callback,
                )
                latents = output.images if hasattr(output, "images") else output[0]
                cpu_latents = latents.detach().cpu()

            t1 = time.monotonic()
            atomic_torch_save({
                "latent": cpu_latents,
                "width": width,
                "height": height,
                "prompt_source": prompt_source,
            }, Path(paths["latent_pt"]))
            save_s = round(time.monotonic() - t1, 3)

            # ARCHITECTURE.md §6: embeds.pt is deleted once denoise succeeds.
            embeds_path.unlink(missing_ok=True)

            duration = round(time.monotonic() - t0, 3)
            avg_sps = round(duration / steps, 3) if steps > 0 else 0.0
            logger.info("Denoise completed for job %s in %.3fs (%.3f s/step)", job_id, duration, avg_sps)
            return {
                "denoise_s": duration,
                "save_s": save_s,
                "steps": steps,
                "s_per_step": avg_sps,
                "prompt_source": prompt_source,
            }, get_peak_vram_bytes()
        finally:
            flush_xpu_cache()
