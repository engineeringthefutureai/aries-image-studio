"""Reference images (docs/REFERENCE_IMAGES.md): the resize rule and how a reference is read.

Every reference is resized to about `reference_res`² pixels, sides multiples of 32, the
same way `QwenImage21Pipeline.__call__` does it. The text encoder reads those pixels; the
VAE encodes them once into `ref<i>.safetensors`, which later jobs reuse.
"""
import math
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps

DEFAULT_REFERENCE_RES = 1024
VAE_SCALE = 16  # one latent token per 16x16 pixel tile


def reference_size(width: int, height: int, reference_res: int) -> Tuple[int, int]:
    """`calculate_dimensions` of the pipeline: `reference_res`² area at the image's aspect."""
    ratio = width / height
    w = math.sqrt(reference_res * reference_res * ratio)
    h = w / ratio
    return int(round(w / 32) * 32), int(round(h / 32) * 32)


def open_reference(path: Path) -> Image.Image:
    """The image upright (phone photos carry EXIF orientation) in RGBA, as the pipeline reads it."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGBA")
