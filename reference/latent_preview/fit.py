"""Fit Studio's latent preview (studio/src/preview.rs) from encode.py's output.

One linear map from a latent pixel's 3x3 neighbourhood (9 x 64 channels, plus a bias) to
the 4x4 RGB pixels it covers, by ridge regression. Every 5th image is held out and scored
against the half-size image. Writes the matrix as little-endian f32, rows = inputs:

    python fit.py out/ ../../studio/assets/latent_preview.bin
"""
import glob
import sys

import numpy as np
from PIL import Image

K = 4  # preview pixels per latent pixel, per side
files = sorted(glob.glob(sys.argv[1] + "/*.npz"))
held = set(range(0, len(files), 5))


def features(z):
    c, h, w = z.shape
    zp = np.pad(z, ((0, 0), (1, 1), (1, 1)), mode="edge")
    parts = [zp[:, dy:dy + h, dx:dx + w].reshape(c, -1).T for dy in range(3) for dx in range(3)]
    return np.concatenate(parts + [np.ones((h * w, 1), np.float32)], 1)


def target(img, h, w):
    b = 16 // K
    small = img.reshape(h * K, b, w * K, b, 3).mean((1, 3))
    return small.reshape(h, K, w, K, 3).transpose(0, 2, 1, 3, 4).reshape(h * w, K * K * 3)


xtx, xty, n = 0, 0, 0
for i, f in enumerate(files):
    if i in held:
        continue
    d = np.load(f)
    x, y = features(d["z"]), target(d["img"].astype(np.float32) / 255, *d["z"].shape[1:])
    xtx, xty, n = xtx + x.T @ x, xty + x.T @ y, n + len(x)
m = np.linalg.solve(xtx + 1e-3 * n * np.eye(xtx.shape[0]), xty).astype("<f4")

scores = []
for i in sorted(held):
    d = np.load(files[i])
    z, img = d["z"], d["img"]
    c, h, w = z.shape
    p = np.clip(features(z) @ m, 0, 1).reshape(h, w, K, K, 3).transpose(0, 2, 1, 3, 4).reshape(h * K, w * K, 3)
    size = (img.shape[1] // 2, img.shape[0] // 2)
    ref = np.asarray(Image.fromarray(img).resize(size, Image.BOX)).astype(np.float32) / 255
    up = np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize(size, Image.BILINEAR)).astype(np.float32) / 255
    scores.append(10 * np.log10(1 / np.mean((up - ref) ** 2)))
print(f"matrix {m.shape}, held-out PSNR vs half size: mean {np.mean(scores):.2f} dB, min {np.min(scores):.2f} (n={len(scores)})")
m.tofile(sys.argv[2])
