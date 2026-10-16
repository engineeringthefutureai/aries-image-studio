"""Decode stage: VAE on one card, latent.safetensors -> image.png (untiled or tiled)."""
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers import AutoencoderKLQwenImage21
from PIL import Image

from aries_worker.stages.files import atomic_write, load_safetensors_with_metadata
from aries_worker.stages.loader import (
    flush_xpu_cache,
    get_device,
    get_peak_vram_bytes,
    reset_peak_vram,
    resolve_model_path,
)

logger = logging.getLogger("aries_worker.stages.decode")


class DecodeStageRunner:
    def __init__(self) -> None:
        self.vae: Optional[AutoencoderKLQwenImage21] = None
        self.load_s = 0.0

    def load(self) -> float:
        t0 = time.monotonic()
        model_path = resolve_model_path()
        logger.info("Loading VAE from %s...", model_path)

        self.vae = AutoencoderKLQwenImage21.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(get_device())

        self.load_s = round(time.monotonic() - t0, 3)
        logger.info("VAE loaded in %.2fs", self.load_s)
        return self.load_s

    def _prepare_latents(self, latent: Any, width: int, height: int) -> Any:
        """Unpack and denormalize a pipeline latent into the 5D tensor `vae.decode` expects."""
        latents = latent.to(device=get_device(), dtype=self.vae.dtype)

        # Packed sequence (batch, seq_len, channels) -> (batch, channels, 1, h, w)
        if latents.ndim == 3:
            scale = getattr(self.vae, "spatial_compression_ratio", 16)
            batch_size, _, channels = latents.shape
            h = 2 * (height // (scale * 2))
            w = 2 * (width // (scale * 2))
            latents = latents.transpose(1, 2).reshape(batch_size, channels, 1, h, w)

        config = self.vae.config
        if hasattr(config, "latents_mean") and hasattr(config, "latents_std"):
            shape = (1, config.z_dim, 1, 1, 1)
            mean = torch.tensor(config.latents_mean).view(shape).to(latents.device, latents.dtype)
            std = torch.tensor(config.latents_std).view(shape).to(latents.device, latents.dtype)
            latents = latents * std + mean
        elif getattr(config, "scaling_factor", None):
            latents = latents / config.scaling_factor
        return latents

    def run_job(self, job_id: str, paths: Dict[str, str], params: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        if self.vae is None:
            raise RuntimeError("decode stage not loaded")

        t0 = time.monotonic()
        reset_peak_vram()

        latent_path = Path(paths["latent_safetensors"])
        if not latent_path.exists():
            raise FileNotFoundError(f"Required latent file not found for job {job_id}: {latent_path}")

        decode_mode = params.get("decode_mode") or "untiled"
        tensors, meta = load_safetensors_with_metadata(latent_path, device=get_device())
        if "latent" not in tensors:
            raise ValueError(f"{job_id}: latent.safetensors missing 'latent' tensor")
        latent = tensors["latent"]
        width = int(meta.get("width") or params["width"])
        height = int(meta.get("height") or params["height"])

        if decode_mode == "tiled":
            self.vae.enable_tiling()
        else:
            self.vae.disable_tiling()

        try:
            with torch.inference_mode():
                decoded = self.vae.decode(self._prepare_latents(latent, width, height), return_dict=False)[0]
                if decoded.ndim == 5:
                    decoded = decoded[:, :, 0]
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                pixels = decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
                img = Image.fromarray((pixels * 255).round().astype("uint8"))

            t1 = time.monotonic()
            size = atomic_write(Path(paths["image_png"]), lambda f: img.save(f, format="PNG"))
            save_s = round(time.monotonic() - t1, 3)
            duration = round(time.monotonic() - t0, 3)

            logger.info("Decode completed for job %s in %.3fs (%s, %d bytes)", job_id, duration, decode_mode, size)
            return {
                "decode_s": duration,
                "save_s": save_s,
                "decode_mode": decode_mode,
            }, get_peak_vram_bytes()
        finally:
            flush_xpu_cache()

