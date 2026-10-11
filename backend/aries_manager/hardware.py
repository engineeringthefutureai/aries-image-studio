"""Host and card inspection via sysfs and procfs. Never calls xpu-smi (it wakes the
cards out of ASPM L1)."""
import glob
import logging
import os
import re
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger("aries_manager.hardware")

PCI_PATTERN = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$")


def scan_xe_pci_addresses() -> List[str]:
    """Scan PCI addresses bound to the Linux xe driver."""
    xe_path = "/sys/bus/pci/drivers/xe"
    if not os.path.exists(xe_path):
        return []
    return [entry for entry in sorted(os.listdir(xe_path)) if PCI_PATTERN.match(entry)]


def get_pci_address_map(num_cards: int = 2) -> Dict[int, str]:
    """Card index -> PCI address. With ZE_ENABLE_PCI_ID_DEVICE_ORDER=1, Level Zero numbers
    devices in PCI order, so the n-th xe device is card n. Unknown cards map to ""."""
    pci_list = scan_xe_pci_addresses()
    return {idx: pci_list[idx] if idx < len(pci_list) else "" for idx in range(num_cards)}


def get_host_available_ram_bytes() -> int:
    """Return host available system RAM in bytes."""
    return psutil.virtual_memory().available


def check_for_wedged_card(pci_address: str) -> Optional[str]:
    """A reason string if the card at `pci_address` has a devcoredump (it faulted), else None."""
    if not pci_address:
        return None
    coredump_path = f"/sys/bus/pci/devices/{pci_address}/devcoredump"
    if os.path.exists(coredump_path):
        return f"Device {pci_address} has active devcoredump at {coredump_path}"

    # Also check /sys/class/drm entries
    for drm_path in glob.glob("/sys/class/drm/card*/device"):
        try:
            real_pci = os.path.basename(os.path.realpath(drm_path))
            if real_pci == pci_address:
                cd_path = os.path.join(drm_path, "devcoredump")
                if os.path.exists(cd_path):
                    return f"Card at {pci_address} recorded devcoredump"
        except Exception:
            pass

    return None


def get_vram_overflow_bytes(pid: int) -> int:
    """Driver memory of `pid` that lives in system RAM, from the drm-*-system lines in
    /proc/<pid>/fdinfo (the largest value across its DRM file descriptors)."""
    fdinfo_dir = f"/proc/{pid}/fdinfo"
    if not os.path.exists(fdinfo_dir):
        return 0

    overflow_bytes = 0
    try:
        for entry in os.listdir(fdinfo_dir):
            file_path = os.path.join(fdinfo_dir, entry)
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        lower = line.lower()
                        if "drm-total-system" in lower or "drm-resident-system" in lower or "drm-system-memory" in lower:
                            parts = line.split(":")
                            if len(parts) >= 2:
                                val_parts = parts[1].strip().split()
                                if val_parts and val_parts[0].isdigit():
                                    num = int(val_parts[0])
                                    unit = val_parts[1].lower() if len(val_parts) > 1 else ""
                                    if "kib" in unit:
                                        num *= 1024
                                    elif "mib" in unit:
                                        num *= 1024 * 1024
                                    elif "gib" in unit:
                                        num *= 1024 * 1024 * 1024
                                    overflow_bytes = max(overflow_bytes, num)
            except (OSError, IOError):
                pass
    except Exception:
        pass

    return overflow_bytes
