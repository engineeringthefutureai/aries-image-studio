"""Prometheus metrics for the manager (ARCHITECTURE.md §9).
Labels never carry prompt text, seeds or job ids."""
from prometheus_client import Counter, Gauge, Histogram

PREFIX = "aries_manager_"

QUEUE_DEPTH = Gauge(
    f"{PREFIX}queue_depth",
    "Current number of queued jobs waiting for stage execution",
    ["stage"],
)

STAGE_RUNS_TOTAL = Counter(
    f"{PREFIX}stage_runs_total",
    "Total number of stage worker subprocess runs",
    ["stage", "card", "exit"],
)

WORKER_CRASHES_TOTAL = Counter(
    f"{PREFIX}worker_crashes_total",
    "Total number of worker crashes with non-zero exit codes",
    ["stage", "card", "exit", "crash_class"],
)

WORKER_EVICTIONS_TOTAL = Counter(
    f"{PREFIX}worker_evictions_total",
    "Total number of worker evictions or terminations instructed by manager",
    ["stage", "card", "reason"],
)

EVICTION_QUEUE_DEPTH = Histogram(
    f"{PREFIX}eviction_queue_depth",
    "Histogram of queue depths across stages at the moment of worker eviction",
    ["evicted_stage", "target_stage", "reason"],
    buckets=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
)

EVICTION_JOBS_PROCESSED = Histogram(
    f"{PREFIX}eviction_jobs_processed",
    "Number of jobs processed by a worker before eviction",
    ["stage", "reason"],
    buckets=(0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
)

MODEL_LOAD_SECONDS = Histogram(
    f"{PREFIX}model_load_seconds",
    "Time taken to load model weights for a stage",
    ["stage"],
    buckets=(1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
)

CARD_STATE = Gauge(
    f"{PREFIX}card_state",
    "1 if the card is in the given state (idle, running, unavailable)",
    ["card", "state"],
)

VRAM_OVERFLOW_BYTES = Gauge(
    f"{PREFIX}vram_overflow_bytes",
    "Driver memory overflowed to host RAM from fdinfo",
    ["card"],
)

HOST_AVAILABLE_BYTES = Gauge(
    f"{PREFIX}host_available_bytes",
    "Host available system RAM in bytes",
)

JOBS_BY_STATE = Gauge(
    f"{PREFIX}jobs",
    "Total jobs grouped by current state",
    ["state"],
)

STORAGE_BYTES = Gauge(
    f"{PREFIX}storage_bytes",
    "Total bytes utilized in manager data directory",
)

ENHANCE_PREFILL_TPS = Histogram(
    f"{PREFIX}enhance_prefill_tokens_per_second",
    "Prompt enhancer prefill throughput in tokens per second",
    buckets=(50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
)

ENHANCE_GENERATION_TPS = Histogram(
    f"{PREFIX}enhance_generation_tokens_per_second",
    "Prompt enhancer generation throughput in tokens per second",
    buckets=(1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0),
)

ENHANCE_TOKENS_TOTAL = Counter(
    f"{PREFIX}enhance_tokens_total",
    "Total tokens processed by prompt enhancer",
    ["type"],
)
