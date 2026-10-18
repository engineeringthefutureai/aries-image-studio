"""Decode stage: VAE on one card, latent.safetensors -> image.png (untiled or tiled).

Large untiled decodes run as an exact split (docs/HARDWARE.md §1.3): the decoder's only
attention (conv_in + mid_block) runs once on the whole latent, at latent resolution; the
memory-heavy upsampling blocks, which are purely local, run on strips across the longer side
that overlap by `SPLIT_OVERLAP` latent lines, and the strips are joined in the middle of each
overlap.
There the strips agree bit for bit, and the image matches an untiled decode to within
1/255, without the ~17 GB of host RAM a full-size untiled decode spills through the driver.
"""
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

# Split untiled decodes above this many pixels (ARIES_DECODE_SPLIT_ABOVE_PX; 0: never).
# ½-size decodes peak under 8 GiB and never need it.
SPLIT_ABOVE_PX = 3_500_000
# Pixels per strip, overlap included: a 2752x1536 half with its overlap (2.9 MP) peaks at
# 11.6 GiB of a B60's 23.9 (with the per-frame caches dropped).
SPLIT_PART_PX = 3_200_000
# Latent lines each strip reaches past its cut. The upsampling blocks see ~12 lines around
# a pixel; 32 leaves the middle of every overlap identical in both strips.
SPLIT_OVERLAP = 32
CACHE_T = 2  # the decoder's causal-conv cache depth (diffusers' CACHE_T)


def split_columns(columns: int, px_per_column: int, height: int, part_px: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Strips for a decoder input `columns` lines long (across the side being split; `height`
    is the other side in pixels): (src_start, src_end, keep_start, keep_end) in latent lines.
    Each strip decodes [src_start, src_end) and contributes [keep_start, keep_end);
    neighbours meet at their shared cut."""
    def strips(parts: int) -> List[Tuple[int, int, int, int]]:
        cuts = [round(i * columns / parts) for i in range(parts + 1)]
        return [
            (max(0, cuts[i] - overlap), min(columns, cuts[i + 1] + overlap), cuts[i], cuts[i + 1])
            for i in range(parts)
        ]

    parts = 2
    while parts < columns and max(b - a for a, b, _, _ in strips(parts)) * px_per_column * height > part_px:
        parts += 1
    return strips(parts)


class _OneFrameCache(list):
    """The decoder's `feat_cache` for a single frame: keeps its markers ("Rep") but drops the
    activations it caches for later frames, which one image never reads (each conv's
    entry is written once, after that conv has run). Same pixels, less memory held."""

    def __setitem__(self, i: Any, value: Any) -> None:
        super().__setitem__(i, value if isinstance(value, str) else None)


def _cached_conv(conv: Any, x: Any, cache: List[Any], idx: List[int]) -> Any:
    """One causal conv as `Decoder3d.forward` runs it on a single-frame first chunk."""
    i = idx[0]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and cache[i] is not None:
        cache_x = torch.cat([cache[i][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
    x = conv(x, cache[i])
    cache[i] = cache_x
    idx[0] += 1
    return x


def _unpatchify(x: Any, patch_size: int) -> Any:
    """diffusers' `_unpatchify` for one frame: (b, c*p*p, f, h, w) -> (b, c, f, h*p, w*p)."""
    if patch_size == 1:
        return x
    b, cpp, f, h, w = x.shape
    c = cpp // (patch_size * patch_size)
    x = x.view(b, c, patch_size, patch_size, f, h, w)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return x.view(b, c, f, h * patch_size, w * patch_size)


def decode_split(vae: Any, z: Any, width: int, height: int, part_px: int = SPLIT_PART_PX,
                 overlap: int = SPLIT_OVERLAP) -> Tuple[Any, int]:
    """`vae.decode(z)` in strips (module docstring). Returns the image as HxWxC floats in
    [0, 1] (as the untiled path computes them) and the number of strips."""
    dec = vae.decoder
    vae.clear_cache()
    x = vae.post_quant_conv(z)
    # Strips across the longer side: rows for portrait images, columns otherwise.
    rows = height > width
    lines = x.shape[-2] if rows else x.shape[-1]
    side = height if rows else width
    px_per_line = side // lines
    if lines * px_per_line != side:
        raise ValueError(f"latent of {lines} lines does not decode to {side} px")
    slots = len(vae._feat_map)
    cache, idx = _OneFrameCache([None] * slots), [0]
    mid = dec.mid_block(_cached_conv(dec.conv_in, x, cache, idx), feat_cache=cache, feat_idx=idx)
    tail_idx = idx[0]
    del x, cache
    flush_xpu_cache()
    strips = split_columns(lines, px_per_line, width if rows else height, part_px, overlap)
    out = None
    for src_a, src_b, keep_a, keep_b in strips:
        part_cache, part_idx = _OneFrameCache([None] * slots), [tail_idx]
        y = mid[..., src_a:src_b, :] if rows else mid[..., src_a:src_b]
        for up_block in dec.up_blocks:
            y = up_block(y, feat_cache=part_cache, feat_idx=part_idx, first_chunk=True)
        y = _cached_conv(dec.conv_out, dec.nonlinearity(dec.norm_out(y)), part_cache, part_idx)
        if vae.config.patch_size is not None:
            y = _unpatchify(y, vae.config.patch_size)
        y = torch.clamp(y, min=-1.0, max=1.0)[:, :, 0]
        y = (y / 2 + 0.5).clamp(0, 1)
        lo, hi = (keep_a - src_a) * px_per_line, (keep_b - src_a) * px_per_line
        y = y[0, :, lo:hi, :] if rows else y[0, :, :, lo:hi]
        part = y.permute(1, 2, 0).float().cpu().numpy()
        del y, part_cache
        # Strips differ in size; cached blocks of one would fragment the card for the next
        # and push the driver into host RAM.
        flush_xpu_cache()
        if out is None:  # channels as the VAE decodes them (RGBA for Qwen-Image-2.1)
            out = np.zeros((height, width, part.shape[2]), dtype=np.float32)
        span = slice(keep_a * px_per_line, keep_b * px_per_line)
        if rows:
            out[span, :] = part
        else:
            out[:, span] = part
    vae.clear_cache()
    return out, len(strips)


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
        split_above = int(os.environ.get("ARIES_DECODE_SPLIT_ABOVE_PX") or SPLIT_ABOVE_PX)
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

        parts = 1
        try:
            with torch.inference_mode():
                z = self._prepare_latents(latent, width, height)
                if decode_mode == "untiled" and split_above > 0 and width * height > split_above:
                    pixels, parts = decode_split(self.vae, z, width, height)
                else:
                    decoded = self.vae.decode(z, return_dict=False)[0]
                    if decoded.ndim == 5:
                        decoded = decoded[:, :, 0]
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    pixels = decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
                img = Image.fromarray((pixels * 255).round().astype("uint8"))

            t1 = time.monotonic()
            size = atomic_write(Path(paths["image_png"]), lambda f: img.save(f, format="PNG"))
            save_s = round(time.monotonic() - t1, 3)
            duration = round(time.monotonic() - t0, 3)

            logger.info(
                "Decode completed for job %s in %.3fs (%s, %d part(s), %d bytes)", job_id, duration, decode_mode, parts, size
            )
            return {
                "decode_s": duration,
                "save_s": save_s,
                "decode_mode": decode_mode,
                "decode_parts": parts,
            }, get_peak_vram_bytes()
        finally:
            flush_xpu_cache()

