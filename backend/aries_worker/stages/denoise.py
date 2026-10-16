"""Denoise stage: 7B DiT transformer on one card, embeds.safetensors -> latent.safetensors.

Emits progress every step. The latent is captured at the `vae.decode` input
(output_type="latent"), so the decode stage needs only the VAE.
"""
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

from aries_worker.stages.files import atomic_safetensors_save, load_safetensors_with_metadata
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

logger = logging.getLogger("aries_worker.stages.denoise")

# Rolling window for the reported seconds-per-step.
SPS_WINDOW = 5


class DenoiseCancelled(Exception):
    """Raised from the step callback when the agent cancels the job."""


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

        embeds_path = Path(paths["embeds_safetensors"])
        if not embeds_path.exists():
            raise FileNotFoundError(f"Required embeds file not found for job {job_id}: {embeds_path}")

        embeds, meta = load_safetensors_with_metadata(embeds_path, device=get_device())
        prompt_source = meta.get("prompt_source", "unknown")
        model_id = meta.get("model") or resolve_model_id("denoise")
        model_rev = meta.get("rev") or resolve_model_rev("denoise")
        model_dtype = meta.get("dtype") or resolve_model_dtype()

        if not isinstance(embeds, dict) or "prompt_embeds" not in embeds:
            raise ValueError(f"{job_id}: invalid embeds payload structure")

        true_cfg = float(params.get("true_cfg_scale", 1.0))
        if true_cfg > 1.0 and embeds.get("negative_prompt_embeds") is None:
            raise ValueError(f"{job_id}: true_cfg_scale={true_cfg} requested, but embeds has no negative_prompt_embeds")

        # ARCHITECTURE_V3.md §3.1: initial noise from torch.Generator("cpu").manual_seed(seed)
        generator = torch.Generator("cpu").manual_seed(seed)
        step_times = []
        last_step_time = t0

        def step_callback(pipe, step_idx, _timestep, kwargs):
            nonlocal last_step_time
            if cancel_check and cancel_check():
                raise DenoiseCancelled(f"Job {job_id} cancelled at step {step_idx}")

            now = time.monotonic()
            step_times.append(now - last_step_time)
            last_step_time = now

            recent = step_times[-SPS_WINDOW:]
            if progress_cb:
                # diffusers numbers the callback from 0, after that step ran: report steps done.
                progress_cb(step=step_idx + 1, steps=steps, sps=round(sum(recent) / len(recent), 3))
            return kwargs

        try:
            pipe_kwargs: Dict[str, Any] = {
                "prompt": None,
                "prompt_embeds": embeds["prompt_embeds"],
                "prompt_embeds_mask": embeds.get("prompt_embeds_mask"),
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "generator": generator,
                "output_type": "latent",
                "callback_on_step_end": step_callback,
            }
            if true_cfg > 1.0:
                pipe_kwargs["true_cfg_scale"] = true_cfg
                pipe_kwargs["negative_prompt_embeds"] = embeds.get("negative_prompt_embeds")
                if embeds.get("negative_prompt_embeds_mask") is not None:
                    pipe_kwargs["negative_prompt_embeds_mask"] = embeds.get("negative_prompt_embeds_mask")

            with torch.inference_mode():
                output = self.pipe(**pipe_kwargs)
                latents = output.images if hasattr(output, "images") else output[0]
                cpu_latents = latents.detach().cpu()

            t1 = time.monotonic()
            contig_latent = cpu_latents.contiguous() if hasattr(cpu_latents, "contiguous") else cpu_latents
            atomic_safetensors_save(
                {"latent": contig_latent},
                Path(paths["latent_safetensors"]),
                metadata={
                    "width": str(width),
                    "height": str(height),
                    "prompt_source": str(prompt_source),
                    "model": str(model_id),
                    "rev": str(model_rev),
                    "dtype": str(model_dtype),
                    "seed": str(seed),
                    "steps": str(steps),
                },
            )
            save_s = round(time.monotonic() - t1, 3)

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

