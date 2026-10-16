"""Device, memory and model-path helpers shared by the stage runners.

A spawned worker sees exactly one card (the agent sets ONEAPI_DEVICE_SELECTOR),
so inside the process that card is always `xpu:0`.
"""
import gc
import logging
import os
import re
from pathlib import Path
from typing import Dict, List

import torch
import diffusers

from aries_worker import __version__ as worker_version

logger = logging.getLogger("aries_worker.loader")

DEVICE = "xpu:0"

DEFAULT_MODEL_PATH = "/models/qwen-image-2.1"
DEFAULT_ENHANCER_MODEL_PATH = "/models/qwen-image-2.1-pe-t2i"
DEFAULT_MODEL_ID = "Qwen/Qwen-Image-2.1"
DEFAULT_ENHANCER_MODEL_ID = "Qwen/Qwen-Image-2.1-PE-T2I"
PCI_PATTERN = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$")


def get_device() -> str:
    return os.environ.get("ARIES_DEVICE", DEVICE)


def has_xpu() -> bool:
    return hasattr(torch, "xpu") and bool(torch.xpu.is_available()) and int(torch.xpu.device_count()) > 0


def flush_xpu_cache() -> None:
    """Synchronize the device, collect garbage and release cached allocator blocks."""
    gc.collect()
    for op in (getattr(torch.xpu, "synchronize", None), getattr(torch.xpu, "empty_cache", None)):
        if callable(op):
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
        "worker": worker_version,
    }


def scan_xe_pci_addresses() -> List[str]:
    """Scan PCI addresses bound to the Linux xe driver."""
    xe_path = os.environ.get("ARIES_SYSFS_XE_PATH", "/sys/bus/pci/drivers/xe")
    if not os.path.exists(xe_path):
        return []
    return [entry for entry in sorted(os.listdir(xe_path)) if PCI_PATTERN.match(entry)]


def pci_from_level_zero_uuid(uuid: object) -> str:
    """PCI address encoded in an Intel GPU's Level Zero UUID, or "" if it is not one.

    Layout seen on Arc Pro B60: vendor id (LE), device id (LE), 4 zero bytes, then bus,
    device, function, e.g. 868011e2-0000-0000-0800-... is 0000:08:00.0.
    """
    try:
        raw = bytes.fromhex(str(uuid).replace("-", ""))
    except ValueError:
        return ""
    if len(raw) != 16 or raw[0:2] != b"\x86\x80":
        return ""
    domain = int.from_bytes(raw[4:6], "little")
    return f"{domain:04x}:{raw[8]:02x}:{raw[9]:02x}.{raw[10]:x}"


def probe_xpu_devices() -> List[str]:
    """Devices this process can actually open, as PCI addresses (or UUIDs).

    `torch.xpu.device_count()` is not trustworthy here: with ONEAPI_DEVICE_SELECTOR naming
    one card it still reports 2, and opening index 1 fails. So probe index by index.
    """
    devices: List[str] = []
    for idx in range(int(torch.xpu.device_count())):
        try:
            props = torch.xpu.get_device_properties(idx)
        except Exception:
            break
        uuid = getattr(props, "uuid", "")
        devices.append(pci_from_level_zero_uuid(uuid) or str(uuid) or f"xpu:{idx}")
    return devices


def get_visible_devices() -> List[str]:
    """Return the PCI addresses (or device ids) this worker process can use.

    The agent checks that the worker's `hello` reports exactly one device with the
    expected PCI address (ARCHITECTURE_V3.md §10.3). On XPU the answer comes from the
    runtime itself, not from the environment the agent set, so a selector that picks the
    wrong card is caught.
    """
    explicit = os.environ.get("ARIES_VISIBLE_DEVICES")
    if explicit is not None:
        return [d.strip() for d in explicit.split(",") if d.strip()]

    if has_xpu():
        return probe_xpu_devices()

    pci_list = scan_xe_pci_addresses()
    selector = os.environ.get("ONEAPI_DEVICE_SELECTOR", "")
    if selector and ":" in selector:
        _, indices_str = selector.split(":", 1)
        indices = [s.strip() for s in indices_str.split(",") if s.strip()]
        visible: List[str] = []
        for idx_str in indices:
            if idx_str.isdigit():
                idx = int(idx_str)
                if idx < len(pci_list):
                    visible.append(pci_list[idx])
                elif os.environ.get("ARIES_DEVICE_PCI"):
                    visible.append(os.environ["ARIES_DEVICE_PCI"])
                else:
                    visible.append(f"level_zero:{idx}")
        if visible:
            return visible

    if pci_list:
        return pci_list

    if os.environ.get("ARIES_DEVICE_PCI"):
        return [os.environ["ARIES_DEVICE_PCI"]]

    for env_var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        val = os.environ.get(env_var)
        if val:
            return [d.strip() for d in val.split(",") if d.strip()]

    return ["xpu:0"]


def get_pci_address() -> str:
    """Return the primary PCI address or device identifier for this worker."""
    devices = get_visible_devices()
    return devices[0] if len(devices) == 1 else ",".join(devices)


def resolve_model_path() -> str:
    """Local path to the Qwen-Image-2.1 weights (diffusers format)."""
    return os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)


def resolve_enhancer_model_path() -> str:
    """Local path to the Qwen-Image-2.1-PE-T2I prompt enhancer weights."""
    return os.environ.get("ENHANCER_MODEL_PATH", DEFAULT_ENHANCER_MODEL_PATH)


def resolve_model_id(stage: str = "encode") -> str:
    if stage == "enhance":
        return os.environ.get("ENHANCER_MODEL_ID", os.environ.get("MODEL_ID", DEFAULT_ENHANCER_MODEL_ID))
    return os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)


def resolve_model_rev(stage: str = "encode") -> str:
    if stage == "enhance" and os.environ.get("ENHANCER_MODEL_REV"):
        return os.environ["ENHANCER_MODEL_REV"]
    if os.environ.get("MODEL_REV"):
        return os.environ["MODEL_REV"]
    base = Path(resolve_enhancer_model_path() if stage == "enhance" else resolve_model_path())
    for rev_file in (base / "revision", base / ".git" / "HEAD"):
        try:
            if rev_file.is_file():
                content = rev_file.read_text(encoding="utf-8").strip()
                if content and not content.startswith("ref:"):
                    return content[:40]
        except OSError:
            pass
    return "main"


def resolve_model_dtype() -> str:
    return os.environ.get("MODEL_DTYPE", "bf16")

