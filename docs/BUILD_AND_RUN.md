# Aries Image Studio — Build and run

How to build Aries Image Studio 0.5.0 (Architecture V3) and run it: with Docker Compose
(recommended), or from source for development. For the 0.2.1 manager, use the
`release/v0.2.1` branch and its README.

## 1. What runs where

| Component | Code | Runs on | Listens on |
|---|---|---|---|
| **Studio** — web app, `/api/*`, image store, scheduler, gRPC gateway for agents | `studio/` (Rust) + `web/` (React, built into the Studio image) | any always-on Linux host; no GPU | `8080` HTTP (browser, artifacts, `/metrics`), `50051` gRPC (agents) |
| **Agent** — connects out to Studio, starts one stage worker per GPU, moves artifacts | `backend/aries_agent` + `backend/aries_worker` (Python) | the GPU host | `9464` agent `/metrics` (scraped by Studio), `/healthz`, `/readyz` |

Studio and the agent can share one machine or sit on two; the steps are the same; only
the host names change (section 6). The agent always dials Studio: Studio never needs to
reach the GPU host except to scrape its `/metrics`.

Studio needs three addresses right:

- the agent reaches Studio's gRPC at `ARIES_BACKEND_GRPC_URL` (agent side);
- the agent downloads and uploads artifacts at `ARIES_ARTIFACT_BASE_URL` (Studio side;
  Studio puts full URLs in each assignment);
- Studio scrapes the agent at `ARIES_METRICS_URL` (agent side; the agent announces it).

All three must resolve and connect **from inside the containers**, so use LAN names or
IP addresses, never `localhost`.

## 2. Requirements

**GPU host**

- Linux with the `xe` driver and the Level Zero runtime. Tested on Ubuntu 26.04 with
  2× Intel Arc Pro B60 24 GB. An untiled 2048² decode needs a 24 GB card to itself and
  about 6 GiB of free host RAM. Host setup and known failure modes:
  [HARDWARE.md](HARDWARE.md).
- Docker Engine with the Compose plugin, and `make`.
- About 50 GB of disk for the model weights, and the `hf` CLI
  (`pip install -U "huggingface_hub[cli]"`) to download them once. Nothing downloads at
  runtime (`HF_HUB_OFFLINE=1`).

**Studio host**

- Docker Engine with the Compose plugin. Disk for the images you keep (a 2048² PNG is
  around 5–8 MB, plus about 300 KB of JPEG thumbnails).

**Building from source** (optional, section 9): Rust 1.85+ (`rustup`), Node.js 22+,
Python 3.12 with [`uv`](https://docs.astral.sh/uv/) or `venv`. `protoc` is not needed:
the Rust build uses a vendored copy.

## 3. Download the models (GPU host)

```bash
sudo install -d -o "$USER" /srv/models
hf download Qwen/Qwen-Image-2.1        --local-dir /srv/models/qwen-image-2.1
hf download Qwen/Qwen-Image-2.1-PE-T2I --local-dir /srv/models/qwen-image-2.1-pe-t2i
# Optional: the image-aware enhancer, for Enhance with reference images
hf download Qwen/Qwen-Image-2.1-PE-I2I --local-dir /srv/models/qwen-image-2.1-pe-i2i
```

**Optional: int4 enhancer.** A torchao int4 copy of the prompt enhancer needs ~8 GiB of card
memory instead of ~18 GB and decodes at ~36 tok/s. Quantize it once, on an Arc card (int4
packing is device-specific), with the agent image so torch and torchao match the ones that
will load it:

```bash
docker run -i --rm --device /dev/dri --group-add render -v /srv/models:/models \
  aries-image-studio-agent:latest python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Int4WeightOnlyConfig
src, out = "/models/qwen-image-2.1-pe-t2i", "/models/qwen-image-2.1-pe-t2i-int4"
q = TorchAoConfig(quant_type=Int4WeightOnlyConfig(group_size=128, int4_packing_format="plain_int32"))
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="xpu", quantization_config=q)
m.save_pretrained(out); AutoTokenizer.from_pretrained(src).save_pretrained(out)
PY
```

Then point `ARIES_ENHANCER_MODEL_PATH` (and the compose mount) at
`qwen-image-2.1-pe-t2i-int4`. The agent reads the precision from the checkpoint's config and
records `int4` in provenance. Re-quantize after changing the torch or torchao pins.

**Optional: int4 image enhancer.** PE-I2I (Enhance with reference images,
[ENHANCER_I2I.md](ENHANCER_I2I.md)) in int4 needs ~10 GiB instead of ~20 and is 1.6× faster
(31 tok/s). Its recipe differs from the one above: load it as an image-text model and keep
the vision tower and `lm_head` in bf16, and copy the processor and its own system prompt
(the T2I recipe saves a text-only model that cannot see the images):

```bash
docker run -i --rm --device /dev/dri --group-add render -v /srv/models:/models \
  aries-image-studio-agent:latest python - <<'PY'
import shutil, torch
from transformers import AutoModelForImageTextToText, AutoProcessor, TorchAoConfig
from torchao.quantization import Int4WeightOnlyConfig
src, out = "/models/qwen-image-2.1-pe-i2i", "/models/qwen-image-2.1-pe-i2i-int4"
q = TorchAoConfig(quant_type=Int4WeightOnlyConfig(group_size=128, int4_packing_format="plain_int32"),
                  modules_to_not_convert=["model.visual", "lm_head"])
m = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map="xpu", quantization_config=q)
m.save_pretrained(out); AutoProcessor.from_pretrained(src).save_pretrained(out)
for f in ("system_prompt.txt", "chat_template.jinja", "generation_config.json", "LICENSE", "README.md"):
    shutil.copy(f"{src}/{f}", f"{out}/{f}")
PY
```

Then point `ARIES_ENHANCER_I2I_MODEL_PATH` (and the compose mount) at
`qwen-image-2.1-pe-i2i-int4`.

Another location works: set `MODELS_DIR` in the agent's `.env` (section 5). The agent
reads each model's revision from the metadata `hf download` leaves, and records it in
every image's provenance.

## 4. Run Studio

```bash
git clone <this repository> aries-image-studio
cd aries-image-studio/deploy/studio
cp .env.example .env
```

Edit `deploy/studio/.env`:

```bash
# One shared secret for Studio and every agent. Generate it once:
#   openssl rand -hex 32
ARIES_AGENT_TOKEN=<64 hex characters>

# Studio's HTTP address as the GPU host sees it (used for artifact transfer).
ARIES_ARTIFACT_BASE_URL=http://studio.lan:8080

# Database, images and thumbnails (default /var/lib/aries-image-studio).
# STUDIO_DATA_DIR=/var/lib/aries-image-studio
```

Create the data directory. The container runs as uid 1000, so it must own it:

```bash
sudo install -d -o 1000 -g 1000 /var/lib/aries-image-studio
```

Build and start:

```bash
docker compose up -d --build      # builds the web app and the Rust binary (a few minutes the first time)
docker compose logs -f            # "HTTP server listening" and "AgentGateway gRPC server listening"
```

Check it:

```bash
curl -s http://localhost:8080/api/system      # {"agents":[], ...} until an agent connects
curl -s http://localhost:8080/metrics | head  # Prometheus text
```

Open `http://studio.lan:8080` in a browser. It shows the timeline. A submit is rejected
("no known executor registered in the last 7 days") until an agent has connected once.

## 5. Run the agent (GPU host)

```bash
cd aries-image-studio/deploy/aries
cp .env.example .env
```

Edit `deploy/aries/.env`:

```bash
ARIES_AGENT_TOKEN=<the same value as Studio's>
ARIES_BACKEND_GRPC_URL=studio.lan:50051          # Studio's gRPC port
ARIES_METRICS_URL=http://gpu-host.lan:9464/metrics  # this host's agent metrics, as Studio sees it
# ARIES_AGENT_ID=aries                           # unique per GPU host
# MODELS_DIR=/srv/models                         # holds qwen-image-2.1/, qwen-image-2.1-pe-t2i/ (and -pe-i2i/)
```

Build and start with `make`, not plain `docker compose`. The Makefile finds the host's
numeric `render` and `video` group IDs and passes them to the container, so the workers
can open `/dev/dri`:

```bash
make build      # torch XPU, transformers, diffusers from git: large, ~15 min the first time
make up
make logs       # "Connected to backend (session=…)" when connected
make metrics    # the agent's own /metrics
make health     # /healthz (JSON: connected, executor states) and /readyz's status
```

On port 9464 the agent also answers `GET /healthz`, `200` while it runs, with
`{status, connected, connection_age_s, uptime_s, executors}`, and `GET /readyz`, `200`
once Studio has accepted its connection and `503` while it is disconnected. The Compose
file's healthcheck uses `/healthz`, so the container is not restarted just because
Studio is down.

The image checks that torch is the XPU build (`'+xpu' in torch.__version__`) and fails
the build otherwise.

Confirm on Studio:

```bash
curl -s http://studio.lan:8080/api/system | python3 -m json.tool | head -40
```

The agent should show `"online": true` with one executor per card (`xpu0`, `xpu1`),
each with its PCI address. The status pill in the header turns to **Ready**.

**Cards and device order.** By default the agent finds the `xe` cards in sysfs and
declares `ARIES_NUM_CARDS` (default 2) executors, one Level Zero index per card in PCI
order. Every worker reports the PCI address torch actually opened, and the agent refuses
a worker on the wrong card (failure class `device_mismatch`). If an integrated GPU shifts the
indices, confirm them with `sycl-ls` and declare the executors explicitly:

```bash
ARIES_EXECUTORS_JSON=[{"id":"xpu0","device":"0000:04:00.0","selector":"level_zero:0"},{"id":"xpu1","device":"0000:08:00.0","selector":"level_zero:1"}]
```

Never run `xpu-smi` on the GPU host: it wakes the cards out of their low-power state.
Card memory and host RAM come from sysfs and `/proc/meminfo` through the agent's
`/metrics`.

## 6. Both on one machine

Run sections 4 and 5 on the same host with its LAN name everywhere, for example
`aries-ai.lan`:

| Setting | File | Value |
|---|---|---|
| `ARIES_ARTIFACT_BASE_URL` | `deploy/studio/.env` | `http://aries-ai.lan:8080` |
| `ARIES_BACKEND_GRPC_URL` | `deploy/aries/.env` | `aries-ai.lan:50051` |
| `ARIES_METRICS_URL` | `deploy/aries/.env` | `http://aries-ai.lan:9464/metrics` |

`localhost` would point at each container itself. If the LAN name does not resolve
inside containers, use the host's IP address.

## 7. Make the first image

In the browser, type a prompt and press Enter. The default is one 2048² image at 20
steps (Standard). For a quick check pick **¼** size and **Draft**. The first job loads each model
from disk, so it takes longer than the next ones.

From the command line:

```bash
curl -s -X POST http://studio.lan:8080/api/jobs \
  -H 'content-type: application/json' \
  -d '{"prompt":"a lighthouse at dusk","width":512,"height":512,"steps":8}'
curl -s http://studio.lan:8080/api/jobs | python3 -m json.tool | less
```

Width and height must be multiples of 32, at most the agent's limit (2400×1792 by
default, `ARIES_MAX_PX`). Add `"enhance": true` to have Qwen-Image-2.1-PE rewrite the
prompt first. The full API is in [ARCHITECTURE_V3.md §8](ARCHITECTURE_V3.md).

## 8. Day-to-day operation

**Update to a new version**

```bash
git pull
cd deploy/studio && docker compose up -d --build
cd ../aries && make build up
```

`up` recreates a container whose image changed. Update while nothing is running if you
can. A stage that is running when its agent stops counts as a failed attempt, and each
stage gets two.

**Stop and start**

```bash
cd deploy/studio && docker compose down      # or: docker compose up -d
cd deploy/aries  && make down                # or: make up
```

History, images and favorites live in `STUDIO_DATA_DIR` and survive `down`, rebuilds and
reboots. Studio accepts jobs while the GPU host is off. They wait until an agent
connects. Back up the data directory to keep your history (stop Studio first, or copy the
`studio.db*` files together).

**Logs and metrics**

- `docker compose logs -f` (Studio) and `make logs` (agent).
- Studio's `http://studio.lan:8080/metrics` includes every connected agent's metrics with
  an `agent` label, so Prometheus needs only that one target.
- The status panel (click the pill in the header) shows each card's state, stage and
  memory, host RAM, and the queue.

## 9. Build from source (development)

Studio and the agent need the same `<token>`; make one with `openssl rand -hex 32`.

**Studio** (repo root):

```bash
cd web && npm ci && npm run build && cd ..        # Studio serves web/dist
ARIES_AGENT_TOKEN=<token> \
ARIES_DATA_DIR=./data \
ARIES_STATIC_DIR=web/dist \
ARIES_ARTIFACT_BASE_URL=http://<this host>:8080 \
cargo run --release -p aries-image-studio
```

**Web app with hot reload**, against any running Studio:

```bash
cd web
npm ci
VITE_BACKEND_URL=http://studio.lan:8080 npm run dev   # http://localhost:5173; proxies /api and /metrics
```

`VITE_BACKEND_URL` can also go in `web/.env.local` (git-ignored).

**Agent without Docker** (GPU host). The install order matters, because a plain
`pip install torch` replaces the XPU build:

```bash
cd backend
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install "torch==2.14.1+xpu" torchvision --index-url https://download.pytorch.org/whl/xpu
uv pip install "transformers==5.20.1" "torchao==0.18.0" accelerate safetensors pillow
uv pip install "git+https://github.com/huggingface/diffusers.git@b800f828f3951c4d206e154411bcbfa693cd322e"
uv pip install -e .
python -c "import torch; assert '+xpu' in torch.__version__"

ARIES_AGENT_TOKEN=<token> \
ARIES_BACKEND_GRPC_URL=studio.lan:50051 \
ARIES_METRICS_URL=http://<this host>:9464/metrics \
ARIES_MODEL_PATH=/srv/models/qwen-image-2.1 \
ARIES_ENHANCER_MODEL_PATH=/srv/models/qwen-image-2.1-pe-t2i \
ARIES_SCRATCH_DIR=/tmp/aries-agent \
ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 HF_HUB_OFFLINE=1 \
python -m aries_agent
```

The user running it needs to be in the `render` and `video` groups.
`deploy/aries/aries-agent.service` is a systemd unit for this layout (venv in
`/opt/aries-image-studio/.venv`, secrets in `/etc/aries/agent.env`).

**Tests and checks** (run all three before committing):

```bash
cd studio && cargo fmt --check && cargo clippy --all-targets && cargo test   # API, scheduler, contract tests
cd backend && pip install -e ".[dev]" && ruff check && pytest              # mocks torch/diffusers; no GPU needed
cd web && npm run build && npm run lint
```

**After changing `proto/aries/agent/v1/agent.proto`**: the Rust code regenerates on the
next build. Regenerate the Python stubs from the repo root:

```bash
python -m grpc_tools.protoc -I proto --python_out=backend --grpc_python_out=backend \
  proto/aries/agent/v1/agent.proto
```

Artifact names (tensors, safetensors metadata, JSON fields) live in
`proto/aries/agent/v1/artifacts.json`. Change it together with the worker, Studio's
validation and the contract tests on both sides.

## 10. Configuration reference

**Studio** (environment; the compose file sets the token, `ARIES_ARTIFACT_BASE_URL` and `RUST_LOG`, and mounts the data directory)

| Variable | Default | Meaning |
|---|---|---|
| `ARIES_AGENT_TOKEN` | — (required) | Shared secret agents present on gRPC and artifact HTTP |
| `ARIES_ARTIFACT_BASE_URL` | `http://127.0.0.1:8080` | Base of the artifact URLs sent to agents |
| `ARIES_DATA_DIR` | `./data` (`/app/data` in the image) | SQLite database, blobs, thumbnails |
| `ARIES_HTTP_ADDR` | `0.0.0.0:8080` | Web app, `/api`, artifacts, `/metrics` |
| `ARIES_GRPC_ADDR` | `0.0.0.0:50051` | AgentGateway |
| `ARIES_STATIC_DIR` | `./web/dist` (`/app/web/dist` in the image) | Built web app |
| `ARIES_IDLE_GRACE_S` | `45` | Idle time before a worker with nothing queued exits |
| `ARIES_MAX_WAIT_S` | `600` | Wait after which a busy card switches to a starving stage |
| `ARIES_SCRAPE_INTERVAL_MS` | `1000` | Agent `/metrics` scrape interval |
| `ARIES_SCRAPE_STALE_S` | `5` | A scrape older than this blocks memory-limited work |
| `ARIES_RECONNECT_GRACE_S` | `30` | How long a disconnected agent keeps its running tasks |
| `ARIES_DISK_WATERMARK_PCT` | `80` | Refuse new jobs above this disk usage |
| `ARIES_REFERENCE_RES` | `1024` | Reference images are resized to about this side squared (~1 MP); multiples of 32 from 256. Changing it re-encodes references once (their stored latents are keyed by it) |
| `ARIES_CLOUD_ENHANCERS` | `off` | `on` offers cloud prompt enhancers for each key below (CLOUD_ENHANCERS.md). The prompt and any reference images then go to that provider, and its terms apply |
| `ANTHROPIC_API_KEY` · `ARIES_ANTHROPIC_MODEL` | unset · `claude-sonnet-5-5` | Claude as an enhancer |
| `GEMINI_API_KEY` | unset | Gemini API models as enhancers |
| `ARIES_GEMINI_MODELS` | `gemini-3.5-flash-lite:15:250000` | The Gemini models offered, each `model:rpm:tpm`: requests and tokens a minute Studio sends it (the free tiers' limits); requests past them wait. `ARIES_ANTHROPIC_RPM` · `_TPM` do the same for Claude, `0` (no limit) by default |
| `ARIES_GEMINI_THINKING` | `high` | The Gemini models' thinking level; `minimal` is faster, without reasoning |
| `ARIES_CLOUD_ENHANCE_CONCURRENCY` | `4` | Cloud requests in flight per enhancer (model) |
| `ARIES_CLOUD_ENHANCER_PROMPT_FILE` | unset | A rewriting prompt to use instead of the packaged one |
| `RUST_LOG` | `info` | Log filter, e.g. `info,aries_image_studio=debug` |

**Agent** (environment; `deploy/aries/compose.yaml` sets the paths)

| Variable | Default | Meaning |
|---|---|---|
| `ARIES_AGENT_TOKEN` / `ARIES_AGENT_TOKEN_FILE` | — (required) | Shared secret, inline or from a file |
| `ARIES_BACKEND_GRPC_URL` | `localhost:50051` | Studio's gRPC address |
| `ARIES_METRICS_URL` | `http://127.0.0.1:9464/metrics` | Where Studio scrapes this agent |
| `ARIES_METRICS_LISTEN` | `0.0.0.0:9464` | Agent `/metrics`, `/healthz` and `/readyz` listener |
| `ARIES_AGENT_ID` | `aries` | Unique name per GPU host |
| `ARIES_BACKEND_KIND` | `xpu` | Accelerator kind |
| `ARIES_MODEL_PATH` | `/models/qwen-image-2.1` | Qwen-Image-2.1 weights |
| `ARIES_ENHANCER_MODEL_PATH` | `/models/qwen-image-2.1-pe-t2i` | Prompt rewriter weights (bf16, or a torchao int4 copy, section 3) |
| `ARIES_ENHANCER_MODEL_DTYPE` | from the checkpoint (`int4`/`int8`), else `MODEL_DTYPE` | Enhancer precision recorded in provenance |
| `ARIES_ENHANCER_I2I_MODEL_PATH` | unset (compose: `/models/qwen-image-2.1-pe-i2i`) | Image-aware enhancer (PE-I2I, bf16 or int4): Enhance with reference images. Ignored when the directory has no `config.json` |
| `ARIES_ENHANCER_I2I_MODEL_DTYPE` | from the checkpoint, else `MODEL_DTYPE` | Its precision recorded in provenance |
| `ARIES_NUM_CARDS` | `2` | Executors to declare when found in sysfs |
| `ARIES_EXECUTORS_JSON` | — | Explicit executors (`id`, `device`, `selector`, optional `max_px`) |
| `ARIES_MAX_PX` | `4300800` (2400×1792) | Largest image per executor |
| `ARIES_DECODE_SPLIT_ABOVE_PX` | `3500000` | Untiled decodes above this run as an exact split (HARDWARE.md §1.3): same pixels within 1/255, ≤ 12.5 GiB on the card, no host-RAM spill, so they skip the memory-heavy slot. `0`: plain untiled |
| `ARIES_DECODE_ABOVE_PX` | `3500000` (3.5 MP: full size only) | Decodes above this need the RAM floor; plain untiled ones (not split) also the memory-heavy slot. ½-size decodes peak under 8 GiB and run freely |
| `ARIES_DECODE_RAM_FLOOR_BYTES` | `6442450944` (6 GiB) | Free host RAM required for those decodes |
| `ARIES_DECODE_MAX_CONCURRENT` | `1` | How many memory-heavy tasks (those decodes, and denoise above `ARIES_DENOISE_REFS_LIMIT_ABOVE` references) run at once on this agent (at least 1; one count shared by both); `2` allows one per card |
| `ARIES_MAX_REFERENCE_IMAGES` | `4` | Most reference images a job may carry on this agent, up to 10 (the model's limit); each adds ~2.6 GiB to denoise (HARDWARE.md §1.1), and 10 at the default `ARIES_REFERENCE_RES` hung the host (HARDWARE.md F9): try a smaller reference resolution before raising it past 4. `0` turns references off |
| `ARIES_DENOISE_REFS_LIMIT_ABOVE` | `2` | Denoise with more references than this spills into host RAM: it needs the decode RAM floor and takes the memory-heavy slot it shares with full-size decodes |
| `ARIES_RAM_GUARD_FLOOR_BYTES` | `4294967296` (4 GiB) | Host-RAM guard: when free host RAM stays below this for two samples, the agent kills the worker holding the most host RAM and its task fails as `oom_kill` (retried once). `0` turns it off |
| `ARIES_RAM_GUARD_INTERVAL_S` | `0.25` | How often the guard samples `MemAvailable` |
| `ARIES_WORKER_OOM_SCORE_ADJ` | `1000` | Workers' `oom_score_adj`: if the kernel has to OOM-kill, it takes a worker (its task is retried), never the desktop session |
| `ARIES_SCRATCH_DIR` | `/run/aries-agent` (`/tmp/aries-agent` in compose) | Per-task files, deleted after each task |
| `ARIES_SLEEP_INHIBITOR` | `1` | Hold a systemd sleep inhibitor while workers run (needs `systemd-inhibit`) |
| `ARIES_CA_CERT` | — | CA bundle for a TLS gRPC endpoint |
| `ARIES_BACKEND_HTTP_URL` | `http://localhost:8080` | Studio's HTTP base, used only when an artifact URL Studio sends is relative (Studio sends absolute ones built from `ARIES_ARTIFACT_BASE_URL`) |

## 11. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Agent log: `invalid bearer token` / `UNAUTHENTICATED` | The two `.env` files have different `ARIES_AGENT_TOKEN` values. |
| Agent keeps reconnecting, Studio shows no agent | `ARIES_BACKEND_GRPC_URL` does not resolve or connect from inside the container; try the IP address. Port 50051 must be open on the Studio host. |
| Submit rejected: `no known executor registered in the last 7 days` | No agent has connected yet. |
| Jobs stay queued: `waiting for metrics scrape from <agent>` | Studio cannot reach `ARIES_METRICS_URL`; check it with `curl` from the Studio host. |
| Tasks fail with `upload_failed` or download errors | `ARIES_ARTIFACT_BASE_URL` is wrong or unreachable from the GPU host. |
| Workers fail at start, `/dev/dri` permission denied | The container lacks the host's `render`/`video` GIDs: start the agent with `make up`, or set `RENDER_GID`/`VIDEO_GID` in `.env`. |
| Tasks fail with `device_mismatch` | Level Zero order differs from PCI order (often an integrated GPU): set `ARIES_EXECUTORS_JSON` (section 5). |
| Studio exits: permission denied on `/app/data` | The data directory is not owned by uid 1000 (section 4). |
| Agent build fails the `+xpu` assertion | A dependency pulled torch from PyPI. Keep the Dockerfile's install order. |
| Full-size decodes wait for a long time | Less than 6 GiB of free host RAM, or another large decode is running. The status panel shows host RAM. |

More failure modes, with measurements: [HARDWARE.md](HARDWARE.md).
