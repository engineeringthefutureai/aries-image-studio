# aries-image-studio — engineering notes

Product, UX and the browser-facing API live in **docs/SPEC.md** and **docs/SPEC_UI.md**;
building, deploying and configuring in **docs/BUILD_AND_RUN.md**;
Architecture V3 (one Rust backend, thin GPU agents, protobuf contract, capability matching,
live resource scraping, provenance) lives in **docs/ARCHITECTURE_V3.md** (earlier v0.2
design in **docs/ARCHITECTURE.md**); measurements and failure modes in **docs/HARDWARE.md**.
Treat them as the source of truth and flag disagreements instead of silently
diverging. When behavior changes, update the doc in the same change.

## Layout

```
proto/aries/agent/v1/    Single source of truth for the AgentGateway gRPC protocol
                         (agent.proto, buf.yaml, buf.gen.yaml).
studio/                  Rust backend (axum, tonic, tokio, sqlx/SQLite WAL): SPA + public
                         /api/* + artifact HTTP + AgentGateway gRPC server + capability
                         matcher + scheduler + live /metrics scraper + blob store + WebP
                         thumbnailer + PNG tEXt provenance chunk writer.
backend/aries_agent/     Thin GPU agent (Python 3.12, asyncio, grpcio, httpx, prometheus_client,
                         NO torch): sysfs GPU probe, gRPC stream client, worker subprocess
                         lifecycle, PCI verification, HTTP artifact transfer, crash classifier.
backend/aries_worker/    Stage worker: one process = one stage (enhance | encode |
                         denoise | decode) on one GPU; JSON-lines client to the agent/manager;
                         writes .safetensors and PNG.
backend/tests/           pytest; stage & agent tests mock torch/diffusers/transformers.
web/                     Vite + React + TypeScript + Tailwind SPA.
deploy/studio/           Docker Compose for Studio.
deploy/aries/            Systemd unit for aries-agent + Docker Compose / Makefile.
docs/  mockups/  reference/
```

## Current state (v0.3.0 — Architecture V3)

- **Rust Backend (`studio/`)**: Serves the SPA and `/api/*` (`POST/GET /api/jobs`,
  `/api/images/{id}`, `/api/images/{id}/thumb`, `/api/system`, `/api/capabilities`,
  `/api/executors/{id}/{lend,return,disable,enable}`, `/metrics`), runs the gRPC
  `AgentGateway` server, scrapes each online agent's `/metrics` every 1 s, and owns
  SQLite state and the content-addressed blob store.
- **Thin Agent (`backend/aries_agent/`)**: Connects out to `AgentGateway`, declares
  capabilities and live resource limits (`host_ram_available_bytes`,
  `max_concurrent_per_agent`, `executor_mem_available_bytes`), validates assignments
  against its own declaration, checks each worker opened exactly its executor's card, and
  transfers `.safetensors` / `.json` / `.png` artifacts over HTTP with SHA-256 checks.

## Stack

- **studio (backend):** Rust (axum 0.7, tonic 0.12, prost 0.13, sqlx 0.8 SQLite WAL,
  image 0.25).
- **agent & workers:** Python 3.12, `diffusers` from git (needs `QwenImage21Pipeline`),
  `transformers>=5.17`, `safetensors`, torch/torchvision from the **XPU index** (workers
  only; `aries_agent` never imports `torch`). Base image `ubuntu:26.04` +
  `ppa:kobuk-team/intel-graphics` (`libze1 libze-intel-gpu1 intel-opencl-icd`),
  `/dev/dri` passed through, host `render`/`video` GIDs via `group_add`.
- **web:** React + TypeScript + Tailwind; no browser storage: jobs, images, favorites and
  preferences come from Studio. Polls `/api/jobs?changed_since=`,
  `/api/system`, and `/api/capabilities`.

## Hard-won facts about this hardware (do not rediscover)

Full detail in docs/HARDWARE.md.

- 2× Intel Arc Pro B60 24 GB (`xe` driver). An integrated GPU **may enumerate first**:
  pin devices with `ONEAPI_DEVICE_SELECTOR`, confirm with `sycl-ls`, and keep
  `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`. Each V3 worker's `hello` lists the devices torch can
  actually open, and the agent refuses anything but the executor's one PCI address.
- `torch.xpu.device_count()` over-reports: with `ONEAPI_DEVICE_SELECTOR=level_zero:1` it
  returns 2 and opening index 1 fails. Probe `get_device_properties(i)` index by index.
  The Level Zero UUID carries the PCI address (`868011e2-0000-0000-0800-…` is
  `0000:08:00.0`).
- **One big model per card, one process per stage.** Two large components on one card
  overflowed driver memory into host RAM and ended in an OOM kill; only process exit
  reliably frees VRAM, host RAM and driver overflow.
- **No VAE tiling by default** (tiling makes seams). Untiled 2048² decode peaks at
  25.76 GiB, so it needs its card to itself plus host RAM headroom; the agent declares a
  live limit (`host_ram_available_bytes >= 6 GiB`, `max_concurrent_per_agent = 1` above
  1 MP) enforced by the backend scheduler.
- No peer-to-peer between the cards: move whole components or files, never split layers.
- **`.safetensors`, never `torch.save` pickle files**, for cross-stage artifacts
  (`embeds.safetensors`, `latent.safetensors`), and initial noise is generated on CPU via
  `torch.Generator("cpu").manual_seed(seed)`.
- `pip install torch*` from PyPI silently replaces the XPU build. Use
  `--index-url https://download.pytorch.org/whl/xpu`; the image asserts `'+xpu' in
  torch.__version__`. `torchvision` is required even for T2I.
- HF `generate()` calls `streamer.put()` with the whole prompt **before** prefill, then
  once per token. The PE model's chat template opens `<think>` in the prompt.
- **Never call `xpu-smi`** (wakes the cards out of ASPM L1). Hardware metrics come from
  sysfs and `/proc/meminfo` via the agent's `/metrics` endpoint.
- Weights are local; run with `HF_HUB_OFFLINE=1`. Never download at runtime.

## Conventions

- No secrets or personal data in the repo; config via env.
- Keep stage workers and the thin agent free of scheduling/business logic: all placement,
  retries, stickiness, and prompt resolution belong in the Rust backend (`studio/`).
- Every job/task state transition is one SQLite transaction that bumps `version`; events
  or artifact uploads from stale attempts are rejected (`409 Conflict`).
- Stage outputs are written to a temp file and renamed (`aries_worker/stages/files.py`).
- Metric labels never carry prompt text, seeds or job ids.
- Stage artifact names (tensors, safetensors metadata, JSON fields) live in
  `proto/aries/agent/v1/artifacts.json`; change it together with the worker and the
  backend's validation, and the contract tests on both sides.
- No default secrets: `ARIES_AGENT_TOKEN` is required on both sides; `.env` is git-ignored.
- Checks before committing:
  - `cd studio && cargo test`
  - `cd backend && pytest`
  - `cd web && npm run build && npm run lint`
