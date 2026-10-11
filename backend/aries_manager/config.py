"""Manager configuration, read once from the environment."""
import os
from pathlib import Path

GiB = 1024 ** 3

# Storage
DATA_DIR = Path(os.environ.get("ARIES_DATA_DIR", "/var/lib/aries-studio"))
DB_PATH = Path(os.environ.get("ARIES_DB_PATH", str(DATA_DIR / "manager.sqlite")))
JOBS_DIR = Path(os.environ.get("ARIES_JOBS_DIR", str(DATA_DIR / "jobs")))

# Unix socket the stage workers connect to
SOCKET_PATH = Path(os.environ.get("ARIES_SOCKET_PATH", "/run/aries/manager.sock"))

# Scheduler (ARCHITECTURE.md §4)
MAX_WAIT_SECONDS = float(os.environ.get("ARIES_MAX_WAIT_SECONDS", "600"))
IDLE_GRACE_SECONDS = float(os.environ.get("ARIES_IDLE_GRACE_SECONDS", "15"))
# The enhancer stays warm longer: users tend to submit enhanced prompts in bursts.
ENHANCE_IDLE_GRACE_SECONDS = float(os.environ.get("ARIES_ENHANCE_IDLE_GRACE_SECONDS", "300"))
DEFAULT_DECODE_RAM_FLOOR_BYTES = int(os.environ.get("ARIES_DECODE_RAM_FLOOR_BYTES", str(6 * GiB)))
# A worker silent for this long is treated as dead.
HEARTBEAT_TIMEOUT_SECONDS = float(os.environ.get("ARIES_HEARTBEAT_TIMEOUT_SECONDS", "30"))

# Disk and garbage collection (ARCHITECTURE.md §6)
DISK_WATERMARK_RATIO = float(os.environ.get("ARIES_DISK_WATERMARK_RATIO", "0.80"))
TTL_DAYS = int(os.environ.get("ARIES_TTL_DAYS", "7"))
AUDIT_RETENTION_DAYS = int(os.environ.get("ARIES_AUDIT_RETENTION_DAYS", "30"))
GC_INTERVAL_SECONDS = float(os.environ.get("ARIES_GC_INTERVAL_SECONDS", "60"))

# Hardware
NUM_CARDS = int(os.environ.get("ARIES_NUM_CARDS", "2"))
GPU_NAME = os.environ.get("ARIES_GPU_NAME", "Intel Arc Pro B60")
# Reported until a worker on the card tells us the real total (B60: 23.9 GiB usable).
DEFAULT_CARD_VRAM_BYTES = int(os.environ.get("ARIES_CARD_VRAM_BYTES", "25669140480"))
