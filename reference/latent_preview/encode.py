"""Encode sample images with the Qwen-Image-2.1 VAE on the CPU, for fit.py.

Writes one .npz per image: `z`, the latent normalised as the denoiser outputs it
(64, h/16, w/16), and `img`, the image resized to a multiple of 64 px (longer side 768).
Run where torch and diffusers are installed (the agent image), no GPU needed:

    python encode.py /srv/models/qwen-image-2.1 out/ image1.png image2.png ...
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21
from PIL import Image

model_dir, out_dir, paths = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
out_dir.mkdir(parents=True, exist_ok=True)
vae = AutoencoderKLQwenImage21.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32).eval()
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
for i, p in enumerate(paths):
    im = Image.open(p).convert("RGB")
    s = 768 / max(im.size)
    w, h = (max(64, int(im.size[0] * s) // 64 * 64), max(64, int(im.size[1] * s) // 64 * 64))
    a = np.asarray(im.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0
    # The VAE is RGBA: opaque alpha.
    x = torch.from_numpy(np.concatenate([a, np.ones_like(a[..., :1])], -1)).permute(2, 0, 1)[None, :, None] * 2 - 1
    t = time.time()
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    z = ((z - mean) / std)[0, :, 0].numpy()
    np.savez_compressed(out_dir / f"{i:03d}.npz", z=z.astype(np.float32), img=(a * 255).astype(np.uint8))
    print(i, p, (w, h), f"{time.time() - t:.1f}s", flush=True)
