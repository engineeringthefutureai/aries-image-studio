"""Encode stage: Qwen3-VL text encoder on one card -> embeds.safetensors.

With reference images (docs/REFERENCE_IMAGES.md) it also reads their pixels with the prompt,
and VAE-encodes the ones Studio has no latents for into `ref<i>.safetensors` (step 1).
The VAE (0.6 GB) loads only when a task asks for that.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
from transformers import Qwen3VLForConditionalGeneration

from aries_worker.stages.files import atomic_safetensors_save
from aries_worker.stages.references import DEFAULT_REFERENCE_RES, open_reference, reference_size
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
        self.vae: Optional[AutoencoderKLQwenImage21] = None
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

    def _load_vae(self) -> float:
        """Load the VAE next to the text encoder the first time a reference needs latents."""
        if self.vae is not None:
            return 0.0
        t0 = time.monotonic()
        self.vae = AutoencoderKLQwenImage21.from_pretrained(
            resolve_model_path(), subfolder="vae", torch_dtype=torch.bfloat16
        ).to(get_device())
        self.pipe.register_modules(vae=self.vae)
        load_s = round(time.monotonic() - t0, 3)
        logger.info("VAE loaded for reference latents in %.2fs", load_s)
        return load_s

    def _encode_reference_latents(
        self, image: Any, size: Tuple[int, int], sha256: str, reference_res: int, out: Path, model: Dict[str, str]
    ) -> int:
        """Step 1: VAE latents of one reference, packed as the pipeline does, then saved."""
        width, height = size
        pixels = self.pipe.image_processor.preprocess(image, width=width, height=height).unsqueeze(2)
        encoded = self.pipe._encode_vae_image(pixels.to(get_device(), torch.bfloat16), generator=None)
        packed = self.pipe._pack_latents(encoded, 1, encoded.shape[1], encoded.shape[3], encoded.shape[4])
        return atomic_safetensors_save(
            {"latents": _to_cpu_contiguous(packed)},
            out,
            metadata={
                "width": str(width),
                "height": str(height),
                "reference_res": str(reference_res),
                "source_sha256": sha256,
                **model,
            },
        )

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
        ref_shas: List[str] = [str(s) for s in params.get("references") or []]
        reference_res = int(params.get("reference_res") or DEFAULT_REFERENCE_RES)
        if ref_shas and need_negative:
            raise ValueError(f"{job_id}: reference images with true_cfg_scale > 1 are not supported")
        model_meta = {
            "model": str(params.get("model_id") or resolve_model_id("encode")),
            "rev": str(params.get("model_rev") or resolve_model_rev("encode")),
            "dtype": str(params.get("model_dtype") or resolve_model_dtype()),
        }

        # References in prompt order, upright, with the size the pipeline resizes them to.
        ref_images: List[Any] = []
        ref_sizes: List[List[int]] = []
        for i in range(len(ref_shas)):
            image_path = paths.get(f"ref{i}.image")
            if not image_path:
                raise FileNotFoundError(f"{job_id}: reference {i} has no image input")
            image = open_reference(Path(image_path))
            ref_images.append(image)
            ref_sizes.append(list(reference_size(image.size[0], image.size[1], reference_res)))
        latents_todo = [i for i in range(len(ref_shas)) if f"ref{i}.safetensors" in paths]

        reset_peak_vram()
        vae_load_s = vae_s = 0.0
        try:
            with torch.inference_mode():
                if latents_todo:
                    vae_load_s = self._load_vae()
                    t_vae = time.monotonic()
                    for i in latents_todo:
                        self._encode_reference_latents(
                            ref_images[i], tuple(ref_sizes[i]), ref_shas[i], reference_res,
                            Path(paths[f"ref{i}.safetensors"]), model_meta,
                        )
                    vae_s = time.monotonic() - t_vae
                t_text = time.monotonic()
                # Returns (prompt_embeds, prompt_embeds_mask, image_pad_mask)
                prompt_embeds, prompt_embeds_mask, image_pad_mask = self.pipe.encode_prompt(
                    prompt=prompt,
                    image=[
                        self.pipe.image_processor.resize(img, width=w, height=h)
                        for img, (w, h) in zip(ref_images, ref_sizes, strict=True)
                    ] or None,
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
            encode_s = time.monotonic() - t_text

            t1 = time.monotonic()

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
                    **model_meta,
                    **({"reference_sizes": json.dumps(ref_sizes)} if ref_shas else {}),
                },
            )
            save_s = time.monotonic() - t1
        finally:
            peak_vram = get_peak_vram_bytes()
            flush_xpu_cache()

        logger.info(
            "encode %s: %.3fs encode, %.3fs save, %d bytes, prompt=%s, %d references (%d VAE-encoded in %.3fs)",
            job_id, encode_s, save_s, size, source, len(ref_shas), len(latents_todo), vae_s,
        )
        timings: Dict[str, Any] = {
            "encode_s": round(encode_s, 3),
            "save_s": round(save_s, 3),
            "prompt_source": source,
        }
        if ref_shas:
            timings.update(references=len(ref_shas), reference_latents_encoded=len(latents_todo))
        if latents_todo:
            timings.update(vae_load_s=vae_load_s, vae_encode_s=round(vae_s, 3))
        return timings, peak_vram

