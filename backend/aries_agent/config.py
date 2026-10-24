"""Configuration and capability declaration builder for aries_agent (Architecture V3 §5.1, §10.2)."""
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from aries_agent import __version__
from aries_agent.hardware import get_device_mem_total_bytes, get_host_ram_total_bytes, scan_xe_pci_addresses

logger = logging.getLogger("aries_agent.config")


# Largest UI preset: 4:3 at 2400 x 1792 (4.30 MP). Every preset ran untiled on v0.2.
DEFAULT_MAX_PX = 2400 * 1792
# Qwen-Image-2.1 takes at most 10 condition images (model card).
MODEL_MAX_REFERENCE_IMAGES = 10


@dataclass
class ExecutorConfig:
    id: str
    device: str
    selector: str
    mem_total_bytes: int = 25668481024  # 23.9 GiB B60 default
    max_px: int = DEFAULT_MAX_PX
    # Untiled is the default per job; tiled is a per-job option (faster, can leave seams).
    decode_modes: List[str] = field(default_factory=lambda: ["untiled", "tiled"])


@dataclass
class ModelConfig:
    id: str
    rev: str
    dtype: str
    path: str


def _detect_quant_dtype(model_path: str) -> Optional[str]:
    """Precision of a quantized checkpoint from its config.json (torchao writes the config
    class, e.g. Int4WeightOnlyConfig), or None for a plain one."""
    try:
        qc = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8")).get("quantization_config")
    except (OSError, ValueError, AttributeError):
        return None
    if not qc:
        return None
    text = json.dumps(qc)
    for bits in ("int4", "int8"):
        if bits in text.lower():
            return bits
    return str(qc.get("quant_method") or "quantized") if isinstance(qc, dict) else "quantized"


def _read_model_rev(model_path: str, env_var: str) -> str:
    """Commit of the weights on disk: env override, a `revision` file, a git checkout, or
    the metadata `hf download --local-dir` leaves (first line of each *.metadata is the
    commit sha). "unknown" rather than a made-up value, since it lands in provenance."""
    if os.environ.get(env_var):
        return os.environ[env_var]
    base = Path(model_path)
    for candidate in (base / "revision", base / ".git" / "HEAD"):
        try:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8").strip()
                if text and not text.startswith("ref:"):
                    return text[:40]
        except OSError:
            pass
    shas: Counter = Counter()
    for meta in (base / ".cache" / "huggingface" / "download").glob("**/*.metadata"):
        try:
            first = meta.read_text(encoding="utf-8").split("\n", 1)[0].strip()
        except OSError:
            continue
        if len(first) == 40 and all(c in "0123456789abcdef" for c in first):
            shas[first] += 1
    if shas:
        return shas.most_common(1)[0][0]
    return "unknown"


@dataclass
class AgentConfig:
    agent_id: str = "aries"
    agent_version: str = __version__
    backend_kind: str = "xpu"
    backend_grpc_url: str = "localhost:50051"
    backend_http_url: str = "http://localhost:8080"
    agent_token: str = ""
    ca_cert_path: Optional[str] = None
    metrics_host: str = "0.0.0.0"
    metrics_port: int = 9464
    metrics_url: str = "http://127.0.0.1:9464/metrics"
    scratch_dir: Path = field(default_factory=lambda: Path("/run/aries-agent"))
    worker_python: str = field(default_factory=lambda: sys.executable)
    worker_module: str = "aries_worker"
    models: List[ModelConfig] = field(default_factory=list)
    # The image-aware enhancer's model id (docs/ENHANCER_I2I.md); None: not installed.
    enhancer_i2i_model_id: Optional[str] = None
    executors: List[ExecutorConfig] = field(default_factory=list)
    decode_above_px: int = 3500000  # full size only; ½ size peaks under 8 GiB
    # Untiled decodes above this run as an exact split (HARDWARE.md §1.3): no spill, so
    # they keep the RAM floor but not the memory-heavy slot. 0: plain untiled, with both.
    decode_split_above_px: int = 3500000
    decode_host_ram_floor_bytes: int = 6442450944  # 6 GiB
    decode_max_concurrent_per_agent: int = 1
    # Reference images (docs/REFERENCE_IMAGES.md §5): each adds ~2.6 GiB to denoise; above
    # `denoise_refs_limit_above` the job spills into host RAM (the driver held ~7 GB for one
    # 4-reference denoise at full size), so it needs the decode floor and shares the
    # one-at-a-time slot with full-size decodes on this agent. Up to the model's limit,
    # MODEL_MAX_REFERENCE_IMAGES; 4 is what one 24 GB card takes at full size.
    max_reference_images: int = 4
    denoise_refs_limit_above: int = 2
    # Host-RAM guard: kill the hungriest worker when MemAvailable stays below the floor
    # (HARDWARE.md F9). 0 turns it off. Sampled fast: a model load and a spilling denoise
    # took ~10 GB within seconds.
    ram_guard_floor_bytes: int = 4294967296  # 4 GiB
    ram_guard_interval_s: float = 0.25
    ram_guard_samples: int = 2
    # Workers' oom_score_adj: if the kernel has to kill, it kills a worker (a retried task),
    # never the desktop session (whose processes run at 100-200).
    worker_oom_score_adj: int = 1000
    upload_retry_delays: List[float] = field(default_factory=lambda: [1.0, 3.0])
    reconnect_min_s: float = 1.0
    reconnect_max_s: float = 30.0
    enable_sleep_inhibitor: bool = True

    @classmethod
    def from_env(cls) -> "AgentConfig":
        agent_id = os.environ.get("ARIES_AGENT_ID", "aries")
        backend_kind = os.environ.get("ARIES_BACKEND_KIND", "xpu")
        grpc_url = os.environ.get("ARIES_BACKEND_GRPC_URL", "localhost:50051")
        for prefix in ("http://", "https://"):
            if grpc_url.startswith(prefix):
                grpc_url = grpc_url[len(prefix):]
        grpc_url = grpc_url.rstrip("/")
        http_url = os.environ.get("ARIES_BACKEND_HTTP_URL", "http://localhost:8080").rstrip("/")

        token = os.environ.get("ARIES_AGENT_TOKEN", "")
        token_file = os.environ.get("ARIES_AGENT_TOKEN_FILE", "")
        if not token and token_file and Path(token_file).is_file():
            token = Path(token_file).read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("ARIES_AGENT_TOKEN (or ARIES_AGENT_TOKEN_FILE) must be set")

        max_reference_images = int(os.environ.get("ARIES_MAX_REFERENCE_IMAGES", "4"))
        if not 0 <= max_reference_images <= MODEL_MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"ARIES_MAX_REFERENCE_IMAGES={max_reference_images}: Qwen-Image-2.1 takes 0 to "
                f"{MODEL_MAX_REFERENCE_IMAGES} reference images"
            )

        ca_cert = os.environ.get("ARIES_CA_CERT") or None
        metrics_host = os.environ.get("ARIES_METRICS_HOST", "0.0.0.0")
        metrics_port = int(os.environ.get("ARIES_METRICS_PORT", "9464"))
        listen = os.environ.get("ARIES_METRICS_LISTEN", "")
        if listen:  # "host:port", as the compose file and systemd unit set it
            host, _, port = listen.rpartition(":")
            metrics_host, metrics_port = host or metrics_host, int(port)
        metrics_url = os.environ.get("ARIES_METRICS_URL", f"http://127.0.0.1:{metrics_port}/metrics")
        scratch_dir = Path(os.environ.get("ARIES_SCRATCH_DIR", "/run/aries-agent"))
        worker_python = os.environ.get("ARIES_WORKER_PYTHON", sys.executable)
        worker_module = os.environ.get("ARIES_WORKER_MODULE", "aries_worker")

        model_path = os.environ.get("ARIES_MODEL_PATH") or os.environ.get("MODEL_PATH", "/models/qwen-image-2.1")
        enhancer_path = os.environ.get("ARIES_ENHANCER_MODEL_PATH") or os.environ.get(
            "ENHANCER_MODEL_PATH", "/models/qwen-image-2.1-pe-t2i"
        )
        dtype = os.environ.get("MODEL_DTYPE", "bf16")
        enhancer_dtype = os.environ.get("ARIES_ENHANCER_MODEL_DTYPE") or _detect_quant_dtype(enhancer_path) or dtype
        model_rev = _read_model_rev(model_path, "MODEL_REV")
        enhancer_rev = _read_model_rev(enhancer_path, "ENHANCER_MODEL_REV")

        models = [
            ModelConfig(
                id=os.environ.get("MODEL_ID", "Qwen/Qwen-Image-2.1"),
                rev=model_rev,
                dtype=dtype,
                path=model_path,
            ),
            ModelConfig(
                id=os.environ.get("ENHANCER_MODEL_ID", "Qwen/Qwen-Image-2.1-PE-T2I"),
                rev=enhancer_rev,
                dtype=enhancer_dtype,
                path=enhancer_path,
            ),
        ]
        enhancer_i2i_model_id: Optional[str] = None
        i2i_path = os.environ.get("ARIES_ENHANCER_I2I_MODEL_PATH")
        if i2i_path and not (Path(i2i_path) / "config.json").is_file():
            # The compose file mounts it whether or not it was downloaded.
            logger.warning("ARIES_ENHANCER_I2I_MODEL_PATH=%s has no config.json; Enhance stays off with references", i2i_path)
            i2i_path = ""
        if i2i_path:
            enhancer_i2i_model_id = os.environ.get("ENHANCER_I2I_MODEL_ID", "Qwen/Qwen-Image-2.1-PE-I2I")
            models.append(
                ModelConfig(
                    id=enhancer_i2i_model_id,
                    rev=_read_model_rev(i2i_path, "ENHANCER_I2I_MODEL_REV"),
                    dtype=os.environ.get("ARIES_ENHANCER_I2I_MODEL_DTYPE") or _detect_quant_dtype(i2i_path) or dtype,
                    path=i2i_path,
                )
            )

        executors: List[ExecutorConfig] = []
        executors_json = os.environ.get("ARIES_EXECUTORS_JSON")
        if executors_json:
            for item in json.loads(executors_json):
                executors.append(
                    ExecutorConfig(
                        id=item["id"],
                        device=item.get("device", ""),
                        selector=item.get("selector", "level_zero:0"),
                        mem_total_bytes=int(item.get("mem_total_bytes", 25668481024)),
                        max_px=int(item.get("max_px", DEFAULT_MAX_PX)),
                        decode_modes=list(item.get("decode_modes", ["untiled", "tiled"])),
                    )
                )
        else:
            pci_list = scan_xe_pci_addresses()
            num_cards = int(os.environ.get("ARIES_NUM_CARDS", "2"))
            if backend_kind == "xpu" and len(pci_list) < num_cards:
                raise ValueError(
                    f"ARIES_NUM_CARDS={num_cards} but only {len(pci_list)} xe device(s) found in sysfs; "
                    "set ARIES_EXECUTORS_JSON to declare executors explicitly"
                )
            for idx in range(num_cards):
                pci = pci_list[idx] if idx < len(pci_list) else ""
                mem = get_device_mem_total_bytes(pci, default=25668481024)
                executors.append(
                    ExecutorConfig(
                        id=f"{backend_kind}{idx}",
                        device=pci,
                        selector=f"level_zero:{idx}" if backend_kind == "xpu" else str(idx),
                        mem_total_bytes=mem,
                        max_px=int(os.environ.get("ARIES_MAX_PX", str(DEFAULT_MAX_PX))),
                        decode_modes=["untiled", "tiled"],
                    )
                )

        return cls(
            agent_id=agent_id,
            backend_kind=backend_kind,
            backend_grpc_url=grpc_url,
            backend_http_url=http_url,
            agent_token=token,
            ca_cert_path=ca_cert,
            metrics_host=metrics_host,
            metrics_port=metrics_port,
            metrics_url=metrics_url,
            scratch_dir=scratch_dir,
            worker_python=worker_python,
            worker_module=worker_module,
            models=models,
            enhancer_i2i_model_id=enhancer_i2i_model_id,
            executors=executors,
            decode_above_px=int(os.environ.get("ARIES_DECODE_ABOVE_PX", "3500000")),
            decode_split_above_px=int(os.environ.get("ARIES_DECODE_SPLIT_ABOVE_PX", "3500000")),
            decode_host_ram_floor_bytes=int(os.environ.get("ARIES_DECODE_RAM_FLOOR_BYTES", "6442450944")),
            decode_max_concurrent_per_agent=int(os.environ.get("ARIES_DECODE_MAX_CONCURRENT", "1")),
            max_reference_images=max_reference_images,
            denoise_refs_limit_above=int(os.environ.get("ARIES_DENOISE_REFS_LIMIT_ABOVE", "2")),
            ram_guard_floor_bytes=int(os.environ.get("ARIES_RAM_GUARD_FLOOR_BYTES", "4294967296")),
            ram_guard_interval_s=float(os.environ.get("ARIES_RAM_GUARD_INTERVAL_S", "0.25")),
            worker_oom_score_adj=int(os.environ.get("ARIES_WORKER_OOM_SCORE_ADJ", "1000")),
            enable_sleep_inhibitor=os.environ.get("ARIES_SLEEP_INHIBITOR", "1") != "0",
        )

    def get_host_ram_total_bytes(self) -> int:
        override = os.environ.get("ARIES_HOST_RAM_TOTAL_BYTES")
        if override:
            return int(override)
        return get_host_ram_total_bytes()
