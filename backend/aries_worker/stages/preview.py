"""Per-step previews for the denoise stage (ARCHITECTURE_V3.md §7.5).

The same linear map as Studio's preview after denoise (studio/src/preview.rs): each latent
pixel and its 3x3 neighbourhood (64 channels each, plus a bias) to the 4x4 RGB pixels it
covers. Here it runs on the card as one 3x3 convolution (64 to 48 channels) and a pixel
reorder, on the clean-image estimate of the step, and comes out as a quarter-size JPEG
(an eighth for large images, when Studio asks for half the sides).
`latent_preview.bin` is a copy of Studio's `studio/assets/latent_preview.bin`; the contract
test keeps the two identical.
"""
import io
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

CHANNELS = 64
SCALE = 4
INPUTS = 9 * CHANNELS + 1
OUTPUTS = SCALE * SCALE * 3
JPEG_QUALITY = 75
MATRIX_PATH = Path(__file__).with_name("latent_preview.bin")


def load_matrix(path: Path = MATRIX_PATH) -> np.ndarray:
    """The fitted map: `INPUTS` rows (3x3 neighbourhood of channels, then the bias) of
    `OUTPUTS` columns (4x4 pixels, RGB)."""
    return np.fromfile(path, dtype="<f4").reshape(INPUTS, OUTPUTS)


def conv_weights(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The map as a convolution: weight `(48, 64, 3, 3)` and bias `(48,)`. Output channel
    `(sy * 4 + sx) * 3 + c` is colour `c` of pixel `(sy, sx)` in the 4x4 block."""
    w = matrix[: 9 * CHANNELS].reshape(3, 3, CHANNELS, OUTPUTS).transpose(3, 2, 0, 1)
    return np.ascontiguousarray(w), matrix[9 * CHANNELS].copy()


def clean_estimate(x_next: Any, x_prev: Any, sigma_prev: float, sigma_next: float) -> Any:
    """The model's guess of the finished latent after an Euler flow-matching step from
    `x_prev` (at `sigma_prev`) to `x_next`: the step was `x_next = x_prev + (sigma_next -
    sigma_prev) * v`, and the clean latent is `x - sigma * v`."""
    v = (x_next - x_prev) / (sigma_next - sigma_prev)
    return x_next - sigma_next * v


class StepPreview:
    """Renders packed latents `(1, h * w, 64)` to JPEG on their own device."""

    def __init__(self, matrix: Optional[np.ndarray] = None) -> None:
        self._matrix = load_matrix() if matrix is None else matrix
        self._weights: Optional[Tuple[Any, Any]] = None

    def render(self, latents: Any, width: int, height: int, downscale: int = 1) -> Tuple[bytes, float]:
        """`(jpeg, ms)`: the preview of `latents` for a `width` x `height` image, its sides
        divided by `downscale` (averaging blocks of pixels), and the time it took, on the
        card and encoding."""
        import torch
        import torch.nn.functional as F
        from PIL import Image

        t0 = time.monotonic()
        h, w = 2 * (height // 32), 2 * (width // 32)
        if self._weights is None or self._weights[0].device != latents.device:
            weight, bias = conv_weights(self._matrix)
            self._weights = (
                torch.from_numpy(weight).to(latents.device),
                torch.from_numpy(bias).to(latents.device),
            )
        weight, bias = self._weights
        z = latents.float().transpose(1, 2).reshape(1, CHANNELS, h, w)
        # Edge pixels repeat, as in the fit.
        out = F.conv2d(F.pad(z, (1, 1, 1, 1), mode="replicate"), weight, bias)
        # (1, sy, sx, c, h, w) -> (1, h, sy, w, sx, c): the 4x4 blocks into place.
        rgb = out.reshape(1, SCALE, SCALE, 3, h, w).permute(0, 4, 1, 5, 2, 3).reshape(h * SCALE, w * SCALE, 3)
        if downscale > 1:
            rgb = F.avg_pool2d(rgb.permute(2, 0, 1).unsqueeze(0), downscale)[0].permute(1, 2, 0)
        pixels = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue(), round((time.monotonic() - t0) * 1000, 2)
