"""Host and GPU inspection via sysfs, procfs and kernel logs.

Never imports torch and never calls xpu-smi (which wakes cards out of ASPM L1).
"""
import glob
import os
import re
import subprocess
from typing import List, Optional

from aries.agent.v1 import agent_pb2

PCI_PATTERN = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$")


def scan_xe_pci_addresses() -> List[str]:
    """Scan PCI addresses bound to the Linux xe driver."""
    xe_path = os.environ.get("ARIES_SYSFS_XE_PATH", "/sys/bus/pci/drivers/xe")
    if not os.path.exists(xe_path):
        return []
    return [entry for entry in sorted(os.listdir(xe_path)) if PCI_PATTERN.match(entry)]


def _read_meminfo_bytes(field_name: str, fallback: int) -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(field_name + ":"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1]) * 1024
    except OSError:
        pass
    return fallback


def get_host_ram_total_bytes() -> int:
    return _read_meminfo_bytes("MemTotal", 32212254720)


def get_host_ram_available_bytes() -> int:
    override = os.environ.get("ARIES_HOST_RAM_AVAILABLE_BYTES")
    if override is not None:
        return int(override)
    return _read_meminfo_bytes("MemAvailable", 16106127360)


def get_process_rss_bytes(pid: Optional[int]) -> int:
    """Resident host RAM of `pid` (VmRSS), 0 when it is gone."""
    if not pid or pid <= 0:
        return 0
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1]) * 1024
    except OSError:
        pass
    return 0


def get_device_mem_total_bytes(pci_address: str, default: int = 25668481024) -> int:
    """Read total VRAM bytes from sysfs if exposed by the driver, else return default."""
    if not pci_address:
        return default
    for candidate in (
        f"/sys/bus/pci/devices/{pci_address}/mem_info_vram_total",
        f"/sys/bus/pci/devices/{pci_address}/tile0/memory/vram0/physical_size",
    ):
        try:
            if os.path.isfile(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text.startswith("0x"):
                        return int(text, 16)
                    if text.isdigit():
                        return int(text)
        except OSError:
            pass
    return default


def check_for_wedged_card(pci_address: str) -> Optional[str]:
    """Return a reason string if the card at `pci_address` has an active devcoredump in sysfs."""
    if not pci_address:
        return None
    sysfs_root = os.environ.get("ARIES_SYSFS_PCI_DEVICES_PATH", "/sys/bus/pci/devices")
    coredump_path = os.path.join(sysfs_root, pci_address, "devcoredump")
    if os.path.exists(coredump_path):
        return f"Device {pci_address} has active devcoredump at {coredump_path}"

    for drm_path in glob.glob("/sys/class/drm/card*/device"):
        try:
            real_pci = os.path.basename(os.path.realpath(drm_path))
            if real_pci == pci_address:
                cd_path = os.path.join(drm_path, "devcoredump")
                if os.path.exists(cd_path):
                    return f"Card at {pci_address} recorded devcoredump"
        except OSError:
            pass

    return None


def get_vram_overflow_bytes(pid: Optional[int]) -> int:
    """Driver memory of `pid` that spilled into system RAM, read from /proc/<pid>/fdinfo."""
    if not pid or pid <= 0:
        return 0
    fdinfo_dir = f"/proc/{pid}/fdinfo"
    if not os.path.exists(fdinfo_dir):
        return 0

    overflow_bytes = 0
    try:
        for entry in os.listdir(fdinfo_dir):
            file_path = os.path.join(fdinfo_dir, entry)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        lower = line.lower()
                        if (
                            "drm-total-system" in lower
                            or "drm-resident-system" in lower
                            or "drm-system-memory" in lower
                        ):
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
            except OSError:
                pass
    except OSError:
        pass

    return overflow_bytes


def get_kernel_log_tail(pci_address: str, max_lines: int = 15) -> str:
    """Return the last kernel log lines matching `pci_address` or GPU fault keywords."""
    mock_log = os.environ.get("ARIES_MOCK_KERNEL_LOG")
    if mock_log is not None:
        return mock_log

    try:
        proc = subprocess.run(
            ["dmesg", "--nopager", "-T"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return ""
        lines = proc.stdout.splitlines()
        keywords = ("fault", "engine reset", "wedged", "timedout job", "oom-kill", "ttm")
        matched = [
            ln
            for ln in lines
            if (pci_address and pci_address in ln)
            or any(k in ln.lower() for k in keywords)
        ]
        return "\n".join(matched[-max_lines:])
    except Exception:
        return ""


def classify_worker_crash(
    exit_code: int,
    signal_num: int,
    pci_address: str,
    kernel_log_tail: str,
    error_detail: str = "",
) -> int:
    """Classify worker crash using HARDWARE.md §2 signatures."""
    if error_detail.strip().lower() == "cancelled":
        return agent_pb2.FAILURE_CANCELLED

    if exit_code == 137 or signal_num == 9 or "oom-kill" in kernel_log_tail.lower():
        return agent_pb2.FAILURE_OOM_KILL

    if check_for_wedged_card(pci_address) or "wedged" in kernel_log_tail.lower():
        return agent_pb2.FAILURE_WEDGED

    lower_log = kernel_log_tail.lower()
    if "fault" in lower_log and ("xe" in lower_log or "einval" in lower_log or "ccs" in lower_log):
        return agent_pb2.FAILURE_GPU_FAULT

    if "engine reset" in lower_log or "timedout job" in lower_log:
        return agent_pb2.FAILURE_ENGINE_RESET

    return agent_pb2.FAILURE_PYTHON_ERROR
