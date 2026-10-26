<img src="web/public/favicon.svg" alt="" width="96" align="right">

# Aries Image Studio

Type a prompt, get pictures. A quiet, image-first web app for generating images with
[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) on a home server with
two Intel Arc Pro B60 GPUs. For one person or a few, on a home network, with optional
sign-in with Google.

**Version 0.6.0.** A Rust Studio server owns jobs, history and images, and a thin
agent on the GPU host runs the pipeline. It works end to end on the hardware it was
built for; see [Limitations](#limitations) before deploying it anywhere else.
Release notes: [0.6.0](docs/releases/v0.6.0.md) · [0.5.0](docs/releases/v0.5.0.md) · [0.4.0](docs/releases/v0.4.0.md) · [0.3.1](docs/releases/v0.3.1.md) · [0.3.0](docs/releases/v0.3.0.md) · [0.2.1](docs/releases/v0.2.1.md) ·
[0.2.0](docs/releases/v0.2.0.md).

![Timeline with an enhanced prompt](docs/screenshots/timeline-enhanced.webp)

![Both GPUs denoising a batch in parallel](docs/screenshots/dual-gpu-denoise.webp)

## Features

- **Timeline**: one turn per send, its images in a carousel in send order, with live
  per-stage progress (enhancing, reading prompt, drawing, finishing), cancel and retry.
- **Composer**: folds to a one-line pill until opened. 7 aspect ratios at ~2K, ½ and ¼
  size presets, 1–4 images, Draft / Standard / High, optional **prompt enhancement**
  with the official Qwen-Image-2.1-PE-T2I rewriter (its reasoning is kept and viewable;
  bf16 or a torchao int4 copy), or optionally a cloud model (Claude Sonnet 5.5, Gemini
  3.5 Flash-Lite) when Studio has an API key: seconds instead of minutes, no card used.
  **Advanced**: custom steps, fixed seed, VAE tiling, and
  rows of different settings sent together for side-by-side comparison.
- **Reference images**: attach up to 4 images (pick, drop, paste, or "Use as reference" on
  any result) and refer to them as "image 1", "image 2" in the prompt to edit or combine
  them. Each reference is VAE-encoded once and stored, so later generations reuse it.
  Enhance works with references too when the image-aware Qwen-Image-2.1-PE-I2I enhancer is
  installed: it reads the images and writes a precise edit prompt.
- **Gallery and viewer**: favorites, search and filters over the whole history, zoom to
  actual pixels with pan and a navigator, Rerun / Vary into the composer, download. Every
  page has its own URL, and long lists load as they scroll.
- **Accounts (optional)**: sign in with Google, for the addresses you allow; no passwords.
  Each account sees only its own images, favorites and preferences.
- **Status page**: each GPU's state, current stage and VRAM; queue depth per stage;
  host RAM.
- **Crash-tolerant backend**: each pipeline stage runs in its own process on one card,
  with intermediate results on disk. A GPU fault costs one stage, which is retried
  once; a card that wedges is taken out of rotation while the other keeps working.
- Prometheus metrics at `/metrics`.

## How it works

```
Browser (web/) ──HTTP──▶ Studio (studio/, Rust)                       always on, no GPU
  timeline, gallery        • web app + /api, SQLite job state, image store
                           • scheduler: which stage runs on which card
                           • gRPC gateway ◀── agents dial in
                                   ▲
                                   │ gRPC + artifact HTTP
                           Agent (backend/aries_agent)                 on the GPU host
                             • declares cards, models, live memory (/metrics)
                             • spawns ▼ one stage worker per card
                           Stage worker (backend/aries_worker)         one process each
                             enhance │ encode │ denoise │ decode — one model, one card
```

A job moves through `enhance` (optional) → `encode` → `denoise` → `decode`. Each stage
loads one model on one card in a fresh process, and the process exits when idle: on
this stack that is the only reliable way to give all memory back. Studio gives a free
card to the stage whose work has waited longest, so a backlog runs two denoises in
parallel. Intermediate results (`.safetensors`) travel through Studio over HTTP, so a
failed stage is retried from its inputs. Details:
[docs/ARCHITECTURE_V3.md](docs/ARCHITECTURE_V3.md).

## Requirements

- **GPU host:** Linux with the `xe` driver and the Level Zero runtime; tested on Ubuntu
  26.04 with 2× Intel Arc Pro B60 (24 GB). A full-size image needs a 24 GB card and
  about 6 GiB of free host RAM.
- **Studio host:** any always-on Linux machine, no GPU. It can be the GPU host itself.
- Docker Engine with the Compose plugin on both, and `make` on the GPU host.
- Disk for the model weights: ~50 GB for image generation and the prompt enhancer, ~70 GB
  with the image-aware enhancer (less with int4 copies). They are downloaded once;
  nothing downloads at runtime.

Host setup, dependency order and every failure mode seen so far:
[docs/HARDWARE.md](docs/HARDWARE.md).

## Running it

The short version follows. Every step, with one-machine setups, building from source,
the full configuration and troubleshooting, is in
**[docs/BUILD_AND_RUN.md](docs/BUILD_AND_RUN.md)**.

### 1. Download the models (GPU host)

```bash
pip install -U "huggingface_hub[cli]"
sudo install -d -o "$USER" /srv/models
hf download Qwen/Qwen-Image-2.1        --local-dir /srv/models/qwen-image-2.1         # 31 GB, required
hf download Qwen/Qwen-Image-2.1-PE-T2I --local-dir /srv/models/qwen-image-2.1-pe-t2i  # 18 GB, required
hf download Qwen/Qwen-Image-2.1-PE-I2I --local-dir /srv/models/qwen-image-2.1-pe-i2i  # 18 GB, optional
```

| Model | What for | Without it |
|---|---|---|
| Qwen-Image-2.1 | Generating images, with or without reference images | Nothing works |
| Qwen-Image-2.1-PE-T2I | **Enhance**: rewrites a short prompt into a detailed one | The agent declares it regardless, so jobs with Enhance on fail |
| Qwen-Image-2.1-PE-I2I | **Enhance with reference images**: looks at the images and writes an edit prompt | Reference images still work; Enhance is off while they are attached |

The agent picks up PE-I2I by itself when the directory has a `config.json`; no setting
needed. Both enhancers also come as torchao int4 copies that need about half the card
memory and run faster. You make them once, on the GPU host, with the agent image:
[docs/BUILD_AND_RUN.md §3](docs/BUILD_AND_RUN.md#3-download-the-models-gpu-host).
Weights elsewhere: set `MODELS_DIR` in the agent's `.env`.

### 2. Start Studio

```bash
git clone <this repository> aries-image-studio
cd aries-image-studio/deploy/studio
cp .env.example .env
#   ARIES_AGENT_TOKEN=        openssl rand -hex 32; the same value goes to the agent
#   ARIES_ARTIFACT_BASE_URL=  http://<studio host>:8080, as the GPU host reaches it
sudo install -d -o 1000 -g 1000 /var/lib/aries-image-studio    # database and images
docker compose up -d --build
```

Open `http://<studio host>:8080`. It shows the timeline; sending waits for an agent.

### 3. Start the agent (GPU host)

```bash
cd aries-image-studio/deploy/aries
cp .env.example .env
#   ARIES_AGENT_TOKEN=        the same value as Studio's
#   ARIES_BACKEND_GRPC_URL=   <studio host>:50051
#   ARIES_METRICS_URL=        http://<gpu host>:9464/metrics, as Studio reaches it
make build up                  # ~15 min the first time; make passes the render/video groups
make logs                      # "Connected to backend" once connected
```

Use LAN names or IP addresses in both `.env` files, never `localhost`: each address is
used from inside a container. The status pill in Studio's header turns to **Ready**, and
the status page lists one executor per card. For a quick first image pick **¼** size and
**Draft**; the first job loads each model from disk and takes longer than the next.

| Port | Where | Who connects |
|---|---|---|
| 8080 | Studio | Browsers; agents (artifacts) |
| 50051 | Studio | Agents (gRPC) |
| 9464 | GPU host | Studio (the agent's `/metrics`) |

Keep 50051 and 9464 on the LAN. Agents authenticate with the shared token.

### 4. Optional: sign-in with Google

Without sign-in, anyone who can reach port 8080 uses Studio and sees everything. With it,
only the Google accounts you list get in, each with its own history; no passwords. Keep
the OAuth app in **Testing** and Studio for people you know: it is not meant to be open to
the public (docs/AUTH.md §3).

1. **Put Studio behind HTTPS.** Google sends the browser back only to an https address
   (or `localhost`). Any TLS proxy with a certificate browsers trust works, for example
   Caddy:

   ```
   studio.example.com {
       @internal path /metrics /agent/*
       respond @internal 404          # agents and Prometheus use the LAN address
       reverse_proxy studio.lan:8080
   }
   ```

   Google never connects to Studio, so a Studio reachable only on your LAN works, as long
   as browsers trust the certificate.
2. **Create an OAuth client** in the Google Cloud console (APIs & Services → Credentials):
   type **Web application**, authorized redirect URI
   `https://studio.example.com/auth/callback` (one per address, if Studio has several). On the consent screen add the privacy
   policy `https://studio.example.com/privacy` and terms `https://studio.example.com/terms`
   (Studio serves both, open without signing in), and add each allowed address as a test
   user (leave the app in Testing).
3. **Set Studio's `.env`** and restart it (`docker compose up -d`):

   ```bash
   ARIES_OIDC_CLIENT_ID=<client ID>
   ARIES_OIDC_CLIENT_SECRET=<client secret>
   ARIES_AUTH_ALLOWED_EMAILS=you@example.com,partner@example.com
   ARIES_PUBLIC_URL=https://studio.example.com   # comma-separate several ways in
   ARIES_LEGAL_OPERATOR=Your Name          # named on /privacy and /terms
   ARIES_LEGAL_CONTACT=you@example.com
   ```

The first address owns everything made before sign-in, and is the only one that may
lend or disable cards. Signed-in devices keep working when the internet is down; to sign
in a new device without Google, run
`docker compose exec studio aries-image-studio login-link you@example.com` and open the
single-use link it prints. Read the privacy policy and terms before you switch it on:
they are your commitments as the host. Details: [docs/AUTH.md](docs/AUTH.md).

### 5. Optional: cloud prompt enhancers

With `ARIES_CLOUD_ENHANCERS=on` and an `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` in Studio's
`.env`, the composer also offers Claude Sonnet 5.5 or Gemini 3.5 Flash-Lite as the
enhancer: seconds instead of minutes, and no card used. The prompt and any reference
images of a job that picks one go to that provider, under its terms.
Details: [docs/CLOUD_ENHANCERS.md](docs/CLOUD_ENHANCERS.md).

### Updating and backups

```bash
git pull
cd deploy/studio && docker compose up -d --build
cd ../aries && make build up
```

History, images and favorites live in `/var/lib/aries-image-studio` (`STUDIO_DATA_DIR`)
and survive rebuilds. To back up, stop Studio and copy that directory. Jobs sent while the
GPU host is off wait until its agent connects.

## Configuration

Both sides read environment variables from their `deploy/*/.env`, and each `.env.example`
explains its settings. Every variable is listed in
[docs/BUILD_AND_RUN.md §10](docs/BUILD_AND_RUN.md#10-configuration-reference).

## Development

```bash
cd studio && cargo test                              # Rust: API, scheduler, contract tests
cd backend && pip install -e ".[dev]" && pytest      # mocks torch/diffusers; no GPU needed
cd web && npm ci && npm run build && npm run lint
VITE_BACKEND_URL=http://<studio-host>:8080 npm run dev   # hot reload on :5173
```

Running Studio and the agent from source: [docs/BUILD_AND_RUN.md §9](docs/BUILD_AND_RUN.md#9-build-from-source-development).

## Repository layout

```
studio/          Rust Studio server: API, scheduler, SQLite, image store, gRPC gateway
backend/         Python: aries_agent (GPU agent) and aries_worker (stage workers)
proto/           gRPC protocol, artifact and metric contracts shared by both sides
web/             React + TypeScript + Tailwind single-page app
deploy/studio/   Docker Compose for Studio
deploy/aries/    Docker Compose + Makefile (and a systemd unit) for the agent
deploy/grafana/  Grafana dashboard
docs/            Specs, architecture, hardware notes, build and run guide
mockups/         UI mockups (visual reference)
reference/       Batch scripts that validated the pipeline on the hardware
```

| Doc | Covers |
|---|---|
| [docs/BUILD_AND_RUN.md](docs/BUILD_AND_RUN.md) | Building, deploying, configuring and troubleshooting |
| [docs/SPEC.md](docs/SPEC.md) | Product, UX, data model |
| [docs/SPEC_UI.md](docs/SPEC_UI.md) | UI behavior: states, interactions, keyboard, accessibility |
| [docs/ARCHITECTURE_V3.md](docs/ARCHITECTURE_V3.md) | Studio, agents, protocol, scheduling, failures, API |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The 0.2 manager design (superseded) |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Measurements, setup and failure modes on Intel Arc |
| [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) | Memory and performance work, with measurements |
| [docs/REFERENCE_IMAGES.md](docs/REFERENCE_IMAGES.md) | Reference images: design and measurements |
| [docs/ENHANCER_I2I.md](docs/ENHANCER_I2I.md) | The image-aware prompt enhancer (PE-I2I) |
| [docs/CLOUD_ENHANCERS.md](docs/CLOUD_ENHANCERS.md) | Claude and Gemini as prompt enhancers |
| [docs/AUTH.md](docs/AUTH.md) | Sign-in with Google, accounts, privacy policy and terms |
| [CHANGELOG.md](CHANGELOG.md) | Every change, by release |
| [docs/releases/](docs/releases/) | Release notes |

## Limitations

- **Sign-in is optional.** Without it (the default), anyone who can reach Studio's port
  can use it and sees everything; keep it on a trusted network, or switch on sign-in with
  Google ([docs/AUTH.md](docs/AUTH.md)). Agents authenticate with a shared token.
- **Cloud enhancers send data out.** Off by default; switched on, the prompt and any
  reference images of a job that picks one go to that provider, under its terms.
- Built and tested only on Intel Arc (XPU). Other accelerators would need changes in
  the stage workers.

## Roadmap

- Prometheus alert rules.
- Wake-on-submit and lending cards to other GPU workloads.
- Fast mode and live preview (docs/SPEC.md §8).

## License

MIT — see [LICENSE](LICENSE). The models are not part of this repository and come under
their own licenses, which apply to what you make with them: Qwen-Image-2.1 does not allow
commercial use. Read each model card before you download it.

Whoever hosts Studio is responsible for running it: with sign-in on, its privacy policy and
terms (`/privacy`, `/terms`) are the host's commitments to their users and to Google, not the
authors' ([docs/AUTH.md](docs/AUTH.md) §3).
