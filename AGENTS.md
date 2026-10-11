# aries-studio — engineering notes

Product, UX and the browser-facing API live in **docs/SPEC.md** and **docs/SPEC_UI.md**;
backend execution (stages, worker protocol, queue and failure semantics, telemetry)
in **docs/ARCHITECTURE.md**; measurements and failure modes in **docs/HARDWARE.md**.
Treat them as the source of truth and flag disagreements instead of silently
diverging. When behavior changes, update the doc in the same change.

## Layout

```
backend/aries_manager/   Manager: FastAPI API, SQLite state machine, card scheduler,
                         Unix-socket server for workers, GC, Prometheus metrics.
backend/aries_worker/    Stage worker: one process = one stage (enhance | encode |
                         denoise | decode) on one card; JSON-lines client to the manager.
backend/tests/           pytest; stage tests mock torch/diffusers/transformers.
web/                     Vite + React + TypeScript + Tailwind SPA.
deploy/aries/            Docker Compose + Makefile for the backend on the GPU host.
docs/  mockups/  reference/
```

## Current state (v0.2.0)

- The browser talks to the manager directly (`/jobs`, `/status`, `/stage_runs`) and
  keeps history in localStorage/IndexedDB. The always-on **Studio** service from
  SPEC.md does not exist yet.
- **Next major version:** Studio as a Rust server (axum, sqlx/SQLite, image storage,
  serving the SPA) that owns history and talks to the manager; browser storage goes away.

## Stack

- **backend:** Python 3.12, FastAPI + uvicorn, `diffusers` from git (needs
  `QwenImage21Pipeline`), `transformers>=5.17`, torch/torchvision from the **XPU
  index**. Base image `ubuntu:26.04` + `ppa:kobuk-team/intel-graphics`
  (`libze1 libze-intel-gpu1 intel-opencl-icd`), `/dev/dri` passed through, host
  `render`/`video` GIDs via `group_add`.
- **web:** React + TypeScript + Tailwind; polls the manager.

## Hard-won facts about this hardware (do not rediscover)

Full detail in docs/HARDWARE.md.

- 2× Intel Arc Pro B60 24 GB (`xe` driver). An integrated GPU **may enumerate first**:
  pin devices with `ONEAPI_DEVICE_SELECTOR`, confirm with `sycl-ls`, and keep
  `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`.
- **One big model per card, one process per stage.** Two large components on one card
  overflowed driver memory into host RAM and ended in an OOM kill; only process exit
  reliably frees VRAM, host RAM and driver overflow.
- **No VAE tiling by default** (tiling makes seams). Untiled 2048² decode peaks at
  25.76 GiB, so it needs its card to itself plus host RAM headroom; the scheduler
  enforces a host RAM floor and one decode at a time. The ">4 GiB tensor" theory was
  disproven (largest decode tensor is 2.25 GiB).
- No peer-to-peer between the cards: move whole components or files, never split layers.
- `pip install torch*` from PyPI silently replaces the XPU build. Use
  `--index-url https://download.pytorch.org/whl/xpu`; the image asserts `'+xpu' in
  torch.__version__`. `torchvision` is required even for T2I.
- HF `generate()` calls `streamer.put()` with the whole prompt **before** prefill, then
  once per token. The PE model's chat template opens `<think>` in the prompt.
- **Never call `xpu-smi`** (wakes the cards out of ASPM L1). Hardware metrics come from
  host exporters.
- Weights are local; run with `HF_HUB_OFFLINE=1`. Never download at runtime.

## Conventions

- No secrets or personal data in the repo; config via env.
- Keep stage workers free of business logic: anything that isn't "run this stage"
  belongs in the manager (or, later, Studio).
- Every job state transition is one SQLite transaction that bumps the version; worker
  events for a job that is no longer in that stage (e.g. cancelled) are ignored.
- Stage outputs are written to a temp file and renamed (`aries_worker/stages/files.py`).
- Metric labels never carry prompt text, seeds or job ids.
- Checks before committing: `cd backend && pytest`; `cd web && npm run build && npm run lint`.
