"""Device, memory and model-path helpers shared by the stage runners.

A spawned worker sees exactly one card (the manager sets ONEAPI_DEVICE_SELECTOR),
so inside the process that card is always `xpu:0`.
"""
import gc
import logging
import os
from typing import Dict

import torch
import diffusers

logger = logging.getLogger("aries_worker.loader")

DEVICE = "xpu:0"

DEFAULT_MODEL_PATH = "/models/qwen-image-2.1"
DEFAULT_ENHANCER_MODEL_PATH = "/models/qwen-image-2.1-pe-t2i"


def get_device() -> str:
    return DEVICE


def has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.device_count() > 0


def flush_xpu_cache() -> None:
    """Synchronize the device, collect garbage and release cached allocator blocks."""
    gc.collect()
    for op in (torch.xpu.synchronize, torch.xpu.empty_cache):
        try:
            op()
        except Exception as e:
            logger.debug("XPU %s failed: %s", getattr(op, "__name__", op), e)


def reset_peak_vram() -> None:
    try:
        torch.xpu.reset_peak_memory_stats(0)
    except Exception:
        pass


def get_peak_vram_bytes() -> int:
    try:
        return int(torch.xpu.max_memory_allocated(0))
    except Exception:
        return 0


def get_vram_stats() -> Dict[str, int]:
    """Allocated, reserved, peak and total memory on this worker's card, in bytes."""
    stats = {"allocated": 0, "reserved": 0, "peak": 0, "total": 0}
    if not has_xpu():
        return stats
    try:
        stats.update(
            allocated=int(torch.xpu.memory_allocated(0)),
            reserved=int(torch.xpu.memory_reserved(0)),
            peak=int(torch.xpu.max_memory_allocated(0)),
            total=int(torch.xpu.get_device_properties(0).total_memory),
        )
    except Exception as e:
        logger.debug("Reading VRAM stats failed: %s", e)
    return stats


def get_versions() -> Dict[str, str]:
    return {
        "torch": getattr(torch, "__version__", "unknown"),
        "diffusers": getattr(diffusers, "__version__", "unknown"),
    }


def resolve_model_path() -> str:
    """Local path to the Qwen-Image-2.1 weights (diffusers format)."""
    return os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)


def resolve_enhancer_model_path() -> str:
    """Local path to the Qwen-Image-2.1-PE-T2I prompt enhancer weights."""
    return os.environ.get("ENHANCER_MODEL_PATH", DEFAULT_ENHANCER_MODEL_PATH)
