"""Crash-safe writes for stage outputs.

Every stage writes to a temp name, fsyncs, then renames, so a worker that dies
mid-write never leaves a half-written file that looks complete (ARCHITECTURE_V3.md §7.4).
"""
import json
import os
from pathlib import Path
from typing import Any, Callable, BinaryIO, Dict, Optional, Tuple


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(out: Path, write: Callable[[BinaryIO], None]) -> int:
    """Write `out` via `write(file)` to a temp file, then rename. Returns the size in bytes."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(out)
    _fsync_dir(out.parent)
    return out.stat().st_size


def atomic_safetensors_save(
    tensors: Dict[str, Any],
    out: Path,
    metadata: Optional[Dict[str, str]] = None,
) -> int:
    """Write tensors and string metadata in safetensors format atomically."""
    from safetensors.torch import save

    clean_meta = {str(k): str(v) for k, v in (metadata or {}).items() if v is not None}
    clean_tensors = {k: v for k, v in tensors.items() if v is not None}
    data = save(clean_tensors, metadata=clean_meta)
    return atomic_write(out, lambda f: f.write(data))


def load_safetensors_with_metadata(
    path: Path,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Load tensors and header metadata from a safetensors file."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    metadata: Dict[str, str] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata()
        if meta:
            metadata = dict(meta)
    tensors = load_file(str(path), device=device)
    return tensors, metadata


def atomic_json_save(obj: Any, out: Path) -> int:
    data = json.dumps(obj, indent=2).encode("utf-8")
    return atomic_write(out, lambda f: f.write(data))

