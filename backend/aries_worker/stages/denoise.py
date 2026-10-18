"""Denoise stage: 7B DiT transformer on one card, embeds.safetensors -> latent.safetensors.

Emits progress every step. The latent is captured at the `vae.decode` input
(output_type="latent"), so the decode stage needs only the VAE.

With reference images the pipeline's `__call__` cannot be used: it VAE-encodes the
references from pixels and takes neither stored latents nor `image_pad_mask`. Those jobs
run `_denoise_with_references`, the pipeline's loop over the stored tensors
(docs/REFERENCE_IMAGES.md §3.1).
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor

from aries_worker.stages.files import atomic_safetensors_save, load_safetensors_with_metadata
from aries_worker.stages.references import VAE_SCALE
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

    def _denoise_with_references(
        self,
        embeds: Dict[str, Any],
        ref_latents: List[Any],
        ref_sizes: List[Tuple[int, int]],
        width: int,
        height: int,
        steps: int,
        generator: Any,
        step_callback: Callable[..., Any],
    ) -> Any:
        """`QwenImage21Pipeline.__call__`'s denoising loop over precomputed tensors.

        Same noise, timesteps and conditioning KV cache as the pipeline; the transformer
        sees `[reference latents, target latents]` and only the target slice is stepped.
        """
        pipe, transformer, scheduler = self.pipe, self.pipe.transformer, self.pipe.scheduler
        device = get_device()
        prompt_embeds = embeds["prompt_embeds"]
        image_pad_mask = embeds["image_pad_mask"]
        image_latents = torch.cat([lat.to(prompt_embeds.dtype) for lat in ref_latents], dim=1)

        # Initial noise from the CPU generator, shaped and packed like `prepare_latents`.
        channels = transformer.config.in_channels
        h_lat, w_lat = 2 * (height // (VAE_SCALE * 2)), 2 * (width // (VAE_SCALE * 2))
        noise = randn_tensor(
            (1, 1, channels, h_lat, w_lat), generator=generator, device=torch.device(device), dtype=prompt_embeds.dtype
        )
        latents = pipe._pack_latents(noise, 1, channels, h_lat, w_lat)

        img_shapes = [[
            *[(1, rh // VAE_SCALE, rw // VAE_SCALE) for rw, rh in ref_sizes],
            (1, height // VAE_SCALE, width // VAE_SCALE),
        ]]
        sigmas = pipe.config.sample_sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(
            latents.shape[1],
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(scheduler, steps, device, sigmas=sigmas, mu=mu)
        # The mask spans the joint sequence: one slot per 2x2 group of target latents.
        img_mask = torch.cat([image_pad_mask, image_pad_mask.new_ones(image_pad_mask.shape[0], latents.shape[1] // 4)], dim=1)
        cache = QwenImage21KVCache(len(transformer.transformer_blocks)) if transformer.config.causal_condition else None
        scheduler.set_begin_index(0)

        for i, t in enumerate(timesteps):
            kv_mode = ("extract" if i == 0 else "cached") if cache is not None else None
            with transformer.cache_context("cond"):
                noise_pred = transformer(
                    hidden_states=torch.cat([image_latents, latents], dim=1),
                    timestep=t.expand(latents.shape[0]).to(latents.dtype) / 1000,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=embeds.get("prompt_embeds_mask"),
                    img_shapes=img_shapes,
                    img_mask=img_mask,
                    kv_cache=cache,
                    kv_cache_mode=kv_mode,
                    return_dict=False,
                )[0]
            latents = scheduler.step(noise_pred[:, -latents.size(1):], t, latents, return_dict=False)[0]
            step_callback(pipe, i, t, {})
        return latents

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

        # Stored reference latents (docs/REFERENCE_IMAGES.md), in prompt order.
        n_refs = int(params.get("references") or 0)
        ref_latents: List[Any] = []
        ref_sizes: List[Tuple[int, int]] = []
        for i in range(n_refs):
            ref_path = paths.get(f"ref{i}.safetensors")
            if not ref_path:
                raise FileNotFoundError(f"{job_id}: reference {i} has no latents input")
            tensors, ref_meta = load_safetensors_with_metadata(Path(ref_path), device=get_device())
            ref_latents.append(tensors["latents"])
            ref_sizes.append((int(ref_meta["width"]), int(ref_meta["height"])))
        if n_refs:
            if embeds.get("image_pad_mask") is None:
                raise ValueError(f"{job_id}: {n_refs} references, but embeds has no image_pad_mask")
            if true_cfg > 1.0:
                raise ValueError(f"{job_id}: reference images with true_cfg_scale > 1 are not supported")
            embed_sizes = [tuple(s) for s in json.loads(meta.get("reference_sizes") or "[]")]
            if embed_sizes != ref_sizes:
                raise ValueError(f"{job_id}: embeds were encoded for references {embed_sizes}, latents are {ref_sizes}")

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
                if n_refs:
                    latents = self._denoise_with_references(
                        embeds, ref_latents, ref_sizes, width, height, steps, generator, step_callback
                    )
                else:
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
                    "references": str(n_refs),
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
                **({"references": n_refs} if n_refs else {}),
            }, get_peak_vram_bytes()
        finally:
            flush_xpu_cache()

