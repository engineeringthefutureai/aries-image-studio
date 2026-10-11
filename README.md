<img src="web/public/favicon.svg" alt="" width="96" align="right">

# Aries Image Studio

Type a prompt, get pictures. A quiet, image-first web app for generating images with
[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) on a home server with
two Intel Arc Pro B60 GPUs. Single user, LAN only.

**Version 0.2.0** — first public version. It works end to end on the hardware it was
built for; see [Limitations](#limitations) before deploying it anywhere else.
Release notes: [docs/releases/v0.2.0.md](docs/releases/v0.2.0.md).

![Timeline with an enhanced prompt](docs/screenshots/timeline-enhanced.webp)

![Both GPUs denoising a batch in parallel](docs/screenshots/dual-gpu-denoise.webp)

## Features

- **Timeline**: one turn per submission, 1–4 images each, with live per-stage progress
  (enhancing, reading prompt, drawing, finishing), cancel and retry.
- **Composer**: 7 aspect ratios at ~2K, ½ and ¼ size presets, count, random or locked
  seed, Standard / Draft quality, optional **prompt enhancement** with the official
  Qwen-Image-2.1-PE-T2I rewriter (its reasoning is kept and viewable).
- **Gallery and lightbox**: favorites, search, zoom, vary, rerun exact, reuse settings,
  download.
- **Status panel**: each GPU's state, current stage and VRAM; queue depth per stage;
  host RAM; recent worker runs and why they exited.
- **Crash-tolerant backend**: each pipeline stage runs in its own process on one card,
  with intermediate results on disk. A GPU fault costs one stage, which is retried
  once; a card that wedges is taken out of rotation while the other keeps working.
- Prometheus metrics at `/metrics`.

## How it works

```
Browser (web/)  ──HTTP poll──▶  Manager (backend/aries_manager)        one process
  timeline, gallery,              • job queue + state machine (SQLite)
  history in browser storage      • card scheduler
                                  • per-job files: embeds → latent → PNG
                                  • spawns ▼ one stage worker per card
                                Stage worker (backend/aries_worker)    one process each
                                  enhance │ encode │ denoise │ decode — one model, one card
```

A job moves through `enhance` (optional) → `encode` → `denoise` → `decode`. Each stage
loads one model on one card in a fresh process; the process exits when idle, which is
the only reliable way to give all memory back on this stack. The scheduler gives a
free card to the stage whose oldest job has waited longest, so a backlog runs two
denoises in parallel while a trickle keeps one card denoising and the other cycling
the short stages. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Requirements

- Linux with the `xe` driver and Level Zero runtime; tested on Ubuntu 26.04 with
  2× Intel Arc Pro B60 (24 GB). An untiled 2048² decode needs a 24 GB card to itself.
- Docker with Compose.
- ~50 GB of model weights on local disk (the backend never downloads at runtime).
- A current Node.js LTS to build the web app.

Host setup, dependency order and every failure mode seen so far:
[docs/HARDWARE.md](docs/HARDWARE.md).

## Quick start

**Backend** (on the GPU host):

```bash
hf download Qwen/Qwen-Image-2.1        --local-dir /srv/models/qwen-image-2.1
hf download Qwen/Qwen-Image-2.1-PE-T2I --local-dir /srv/models/qwen-image-2.1-pe-t2i

cd deploy/aries
cp .env.example .env      # adjust device selector, model and data directories
make build up
make health               # {"status": "ok", "state": "ready", ...}
```

**Web** (anywhere that can reach the manager):

```bash
cd web
npm ci
VITE_BACKEND_URL=http://<gpu-host>:8000 npm run dev    # http://localhost:5173
```

For a static deployment, `npm run build` and serve `web/dist/` behind a reverse proxy
that forwards `/jobs`, `/status` and `/stage_runs` to the manager.

## Configuration

The manager reads its settings from the environment (`backend/aries_manager/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `/models/qwen-image-2.1` | Qwen-Image-2.1 weights (diffusers format) |
| `ENHANCER_MODEL_PATH` | `/models/qwen-image-2.1-pe-t2i` | Prompt enhancer weights |
| `ONEAPI_DEVICE_SELECTOR` | — | Level Zero devices to use, e.g. `level_zero:0,1` |
| `ARIES_NUM_CARDS` | `2` | Number of cards to schedule |
| `ARIES_DATA_DIR` | `/var/lib/aries-studio` | SQLite database and per-job files |
| `ARIES_SOCKET_PATH` | `/run/aries/manager.sock` | Unix socket for stage workers |
| `ARIES_IDLE_GRACE_SECONDS` | `15` | Idle time before a worker exits |
| `ARIES_ENHANCE_IDLE_GRACE_SECONDS` | `300` | Same, for the enhancer |
| `ARIES_MAX_WAIT_SECONDS` | `600` | Wait after which a busy card switches to a starving stage |
| `ARIES_DECODE_RAM_FLOOR_BYTES` | 6 GiB | Host RAM required before an untiled decode |
| `ARIES_DISK_WATERMARK_RATIO` | `0.80` | Refuse new jobs above this disk usage |
| `ARIES_TTL_DAYS` | `7` | Delete finished images never fetched after this |

## Development

```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                  # stage tests mock torch/diffusers; no GPU needed

cd ../web
npm ci && npm run build && npm run lint
```

## Repository layout

```
backend/        Python: aries_manager (API, queue, scheduler) and aries_worker (stages)
web/            React + TypeScript + Tailwind single-page app
deploy/aries/   Docker Compose + Makefile for the backend on the GPU host
docs/           Specs, architecture and hardware notes
mockups/        UI mockups (visual reference)
reference/      Batch scripts that validated the pipeline on the hardware
```

| Doc | Covers |
|---|---|
| [docs/SPEC.md](docs/SPEC.md) | Product, UX, data model, browser-facing API |
| [docs/SPEC_UI.md](docs/SPEC_UI.md) | UI behavior: states, interactions, keyboard, accessibility |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Staged execution: manager, workers, protocol, failures, telemetry |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Measurements, setup and failure modes on Intel Arc |
| [docs/releases/](docs/releases/) | Release notes |

## Limitations

- **No authentication.** The browser talks to the manager directly; keep it on a
  trusted network.
- **History lives in the browser** (localStorage and IndexedDB): it is per browser and
  lost if site data is cleared.
- The manager must be reachable to submit jobs; nothing queues while the GPU host is
  off.
- Built and tested only on Intel Arc (XPU). Other accelerators would need changes in
  the stage workers.

## Roadmap

- **Next major version:** a Rust Studio server — always on, owns history, gallery and
  images in SQLite, accepts jobs while the GPU host sleeps — replacing browser storage.
- Prometheus alert rules and a Grafana dashboard.
- Wake-on-submit and lending cards to other GPU workloads.
- Edit with reference images, fast mode, live preview (docs/SPEC.md §8).

## License

MIT — see [LICENSE](LICENSE). The Qwen-Image model weights are under their own
license; check it before any non-personal use.
