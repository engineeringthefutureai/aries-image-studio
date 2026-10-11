"""Crash-safe writes for stage outputs.

Every stage writes to a temp name, fsyncs, then renames, so a worker that dies
mid-write never leaves a half-written file that looks complete (ARCHITECTURE.md §5).
"""
import json
import os
from pathlib import Path
from typing import Any, Callable, BinaryIO


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


def atomic_torch_save(obj: Any, out: Path) -> int:
    import torch

    return atomic_write(out, lambda f: torch.save(obj, f))


def atomic_json_save(obj: Any, out: Path) -> int:
    data = json.dumps(obj, indent=2).encode("utf-8")
    return atomic_write(out, lambda f: f.write(data))
