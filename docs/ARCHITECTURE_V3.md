# Aries Image Studio — Architecture V3: one backend, thin agents

Status: implemented in v0.3.0 (`studio/`, `backend/aries_agent/`). A new design, written
fresh; it does not edit the earlier docs. Where V3 and an earlier doc disagree about the runtime model, V3 wins. Measured
hardware facts and failure modes still come from `HARDWARE.md`; the stage workers already
built (enhance, encode, denoise, decode) are reused.

---

## 0. In one paragraph

A single Rust **backend** serves the UI, owns every job, file and decision, and schedules
work onto **executors** (one per GPU). Each GPU box runs one **agent** process. The agent
dials the backend, declares what its executors can do and where their limits are, then
does exactly what it is told: start a stage worker on executor N, feed it task T, upload
the result, stop the worker. The agent stores nothing and decides nothing. The backend
scrapes each agent's `/metrics` for live resource state. Adding a CUDA or ROCm box later
means writing an agent for it, not changing the backend.

---

## 1. Goals and non-goals

**Goals**

1. One place of truth: jobs, queue, files, history, placement and retries live in the
   backend only.
2. Hardware-agnostic contract: the backend reasons about stages, models, sizes and
   abstract limits, never about xe, CUDA or a card's driver quirks.
3. Process isolation stays: one stage worker per executor at a time, a fresh process per
   worker run, nothing else resident on that GPU.
4. Nothing waits forever: a job the system can never run is rejected at submit; a job that
   can't run *now* says why, live, in the UI.
5. The whole system is visible from the UI: agents, executors, what each can do, what each
   is doing.

**Non-goals (for now)**

- Tiled decode (declared in the schema, not supported).
- Splitting one stage across GPUs.
- Multiple backend replicas (single replica, SQLite, `Recreate` strategy).
- Agents on untrusted networks (LAN + shared token is enough).

---

## 2. Components

```
                   browser (SPA, light theme)
                          │  HTTPS: /api/*  (polling, §8)
                          ▼
┌──────────────────────── backend (Rust, k3s, 1 replica) ────────────────────────┐
│  axum: SPA · public API · artifact HTTP   ·   tonic: AgentGateway (gRPC)        │
│  scheduler · job state machine · capability matcher · metrics scraper          │
│  SQLite (WAL) · blob store on PVC · /metrics                                   │
└────────────▲───────────────────────────────▲───────────────────────────────────┘
             │ gRPC stream (agent dials out) │ HTTP scrape every 1 s (backend dials in)
             │ + HTTP GET/PUT artifacts      │
┌────────────┴───────────────────────────────┴──────────┐
│ agent (one process per box, systemd)                   │
│   executor xpu0 ── worker subprocess (one stage) ──┐   │
│   executor xpu1 ── worker subprocess (one stage) ──┤   │  Unix socket, JSON lines
│   /metrics · scratch dir (deleted per task)        │   │  (the existing worker protocol)
└────────────────────────────────────────────────────┴───┘
```

| | Backend | Agent | Stage worker |
|---|---|---|---|
| Language | Rust (axum, tokio, sqlx) | Python 3.12, asyncio, **no torch** — §10.1 | Python (diffusers), unchanged |
| Owns | jobs, tasks, artifacts, placement, retries, GC, UI | process lifecycle of its workers, transfer of one task's files | one model on one GPU |
| Persistent state | SQLite + blob store | **none** | none |
| Decides | everything | nothing (validates assignments, §6.4) | nothing |

Message types are defined once, in `proto/aries/agent/v1/agent.proto`, and generated for
both sides (§7.2), so the two can't drift.

---

## 3. Domain model

| Entity | What it is |
|---|---|
| **Job** | One user request: prompt, size, seed, steps, options (enhance on/off). The thing the UI shows. |
| **Task** | One stage of one job: `enhance`, `encode`, `denoise`, `decode`. A job is a chain of 3–4 tasks. |
| **Attempt** | One try at a task on one executor. Retries create new attempts; results from old attempts are rejected. |
| **Artifact** | A file a task produced: `enhanced.json`, `embeds.safetensors`, `latent.safetensors`, `image.png`. Stored by the backend, addressed by id + SHA-256. |
| **Agent** | A connected box. Identity, version, backend kind (`xpu`, `cuda`, `rocm`, `mps`), metrics URL. |
| **Executor** | One schedulable GPU on an agent. Has capabilities, limits, a state, and at most one resident worker. |
| **Worker run** | One worker process lifetime on one executor: stage, model, start/end, exit code, tasks served, load time. |

### 3.1 Stages and artifacts

| Stage | Input | Output | Notes |
|---|---|---|---|
| `enhance` (optional) | prompt, followed by `Aspect ratio: <preset nearest the job's size>` so the rewriter frames for the size that will render (its own `wh_ratio` is recorded, not used); with references their pixels (`ref<i>.image`), for the image-aware enhancer PE-I2I ([ENHANCER_I2I.md](ENHANCER_I2I.md)) | `enhanced.json` (`rewritten_prompt`, `wh_ratio`, `thinking`; PE-I2I may answer `ratio_follow: "<imageN>"` instead of a ratio, recorded, not used) | original prompt and the rewriter's reasoning kept on the job |
| `encode` | prompt (original or enhanced), `true_cfg_scale`, reference images (`ref<i>.image`) | `embeds.safetensors`: `prompt_embeds` (+ `negative_prompt_embeds` when `true_cfg_scale > 1`), and their masks when the pipeline returns them (it returns none when nothing is padded); header holds `prompt`, `prompt_source`, `model`, `rev`, `dtype`. With references also `image_pad_mask` and `reference_sizes`, and `ref<i>.safetensors` (VAE latents) for each reference Studio has none stored for | 0.25–2.5 MB; ~8 MB more per reference |
| `denoise` | embeds + seed + size + steps, stored reference latents (`ref<i>.safetensors`) | `latent.safetensors`: tensor `latent`, the input `vae.decode` receives; header holds `model`, `rev`, `dtype`, `width`, `height`, `seed`, `steps` | ~2 MB at 2048²; initial noise from `torch.Generator("cpu").manual_seed(seed)`; with references our own copy of the pipeline's loop (REFERENCE_IMAGES.md §3.1) |
| `decode` | latent | `image.png` | backend writes the PNG metadata chunk afterwards |

The exact tensor, metadata and field names are one file,
`proto/aries/agent/v1/artifacts.json`. `backend/tests/test_contract.py` checks the workers
write it and `studio/tests/contract_test.rs` checks the upload validation requires exactly
it, so the two sides cannot drift apart silently.

Rules that make stages portable across agents and backends:

- **safetensors, not `torch.save`.** Pickle files from the network are code execution;
  safetensors is data only, and the Rust side can read its header to check shape and
  dtype before accepting an upload.
- **Model revision must match across a job's stages.** `encode` output carries
  `model@rev`; the matcher only gives `denoise` to an executor declaring the same revision.
- **Noise is generated on the CPU from the seed.** Same seed → same starting latent on
  any backend. Remaining cross-backend differences are numeric rounding over the steps;
  provenance (§3.2) records them. v0.2 drew noise on the card, so a v0.2 seed does not
  reproduce its image under V3.
- **Guidance:** `true_cfg_scale` defaults to 1.0 (no classifier-free guidance, one
  transformer pass per step, HARDWARE.md §1). `encode` writes the negative embeddings only
  when a job sets it above 1; upload validation and `denoise` refuse a job above 1
  without them. Jobs with reference images cannot set it above 1 (`422`).
- **Reference images** (REFERENCE_IMAGES.md): a reference's VAE latents depend only on
  the image, the model revision and `ARIES_REFERENCE_RES`, so Studio stores them by those
  and any later job reuses them; only the first encode that needs them loads the VAE.
- **One encode per send:** encode output depends only on the effective prompt, guidance
  and references (not seed, steps, size or tiling). A queued encode whose inputs match a
  finished one takes its embeds (the attempt copies the source's provenance, timings and
  peak memory, timings marked `"shared": 1`); one whose inputs match
  an encode that is running or queued ahead of it waits for that one.

### 3.2 Provenance (audit trail, mandatory)

Every attempt that succeeds records `{agent, executor, device (PCI or name), backend kind,
model, rev, dtype, torch version, worker version, timings, peak memory}`. The backend
refuses a `Done` without a complete `Provenance`.

Every image is tagged with the provenance of all its stages, in three places: the
`task_attempts` rows, the PNG metadata chunk the backend writes, and the job detail in the
UI. Cross-backend numeric drift is not measured or managed in V3; the tags make it
explainable if it ever matters.

---

## 4. Job and task state machine

```
job:   submitted ─▶ running ─▶ done
                 └▶ rejected (no capable executor known, §6.2)
       running ─▶ failed (a task failed twice) │ cancelled

task:  waiting ─(previous task done)─▶ queued ─▶ assigned ─▶ running ─▶ uploading ─▶ done
       queued ─▶ blocked(reason) ─▶ queued          (capability or resource, live)
       assigned|running|uploading ─▶ (attempt lost or failed) ─▶ queued, attempts += 1
       attempts == 2 ─▶ failed(class)
```

- One SQLite transaction per transition; each bumps the job's `version` for
  `changed_since` polling.
- `blocked` is not a terminal state; it is `queued` with a reason the UI shows
  ("no executor online can decode 2048²", "waiting for host RAM", "executor xpu1 lent").
- `failed(class)` uses the crash classes from HARDWARE.md: `gpu_fault`, `engine_reset`,
  `wedged`, `oom_kill`, `python_error`, `lost` (agent vanished), `upload_failed`.
- Intermediate artifacts stay until the job finishes, so a failed decode retries from the
  latent, not from scratch.

---

## 5. Capability model

The agent turns its hardware into a declaration in task terms. The backend never sees a
driver name in a decision path.

### 5.1 Registration (first message on the stream)

Shown in proto3 JSON form for readability; the wire type is `Register` (§7.2).

```json
{
  "type": "register",
  "agent": "aries",
  "agent_version": "0.3.0",
  "backend_kind": "xpu",
  "metrics_url": "http://aries.lan:9464/metrics",
  "host": { "ram_total_bytes": 32212254720 },
  "models": [ { "id": "Qwen/Qwen-Image-2.1", "rev": "<sha>", "dtype": "bf16" },
              { "id": "Qwen/Qwen-Image-2.1-PE-T2I", "rev": "<sha>", "dtype": "bf16" },
              { "id": "Qwen/Qwen-Image-2.1-PE-I2I", "rev": "<sha>", "dtype": "int4" } ],
  "executors": [
    {
      "id": "xpu0",
      "device": "0000:04:00.0",
      "mem_total_bytes": 25668481024,
      "stages": {
        "enhance": { "models": ["Qwen/Qwen-Image-2.1-PE-T2I", "Qwen/Qwen-Image-2.1-PE-I2I"],
                     "reference_models": ["Qwen/Qwen-Image-2.1-PE-I2I"], "max_references": 4 },
        "encode":  { "models": ["Qwen/Qwen-Image-2.1"], "max_references": 4 },
        "denoise": { "models": ["Qwen/Qwen-Image-2.1"], "max_px": 4194304, "max_references": 4 },
        "decode":  { "models": ["Qwen/Qwen-Image-2.1"], "max_px": 4194304, "modes": ["untiled"] }
      }
    },
    { "id": "xpu1", "device": "0000:08:00.0", "...": "same shape" }
  ],
  "limits": [
    { "stage": "decode", "above_px": 3500000, "decode_mode": "untiled",
      "requires": { "host_ram_available_bytes": 6442450944 } },
    { "stage": "denoise", "above_px": 0, "above_references": 2,
      "requires": { "host_ram_available_bytes": 6442450944, "max_concurrent_per_agent": 1 } }
  ],
  "running": []
}
```

### 5.2 Live state

| Channel | Carries | Used for |
|---|---|---|
| **Stream events** (push, immediate) | executor state changes, task progress, done/failed | state machine, UI progress |
| **`/metrics` scrape** (backend pulls every `scrape_interval`, default 1 s) | memory used/free per executor, host RAM available, overflow into system RAM, worker pid, load-time and duration histograms, crash counters | `limits` checks, UI resource panel, Prometheus |

---

## 6. Scheduler (backend)

One scheduler tick runs on every relevant event (submit, agent message, operator action)
and once a second; a mutex keeps ticks from overlapping.

### 6.1 Executor states

`offline` · `idle` · `starting(stage)` · `ready(stage)` · `busy(stage, task)` · `stopping` ·
`lent` · `disabled(reason)`.

- The backend sets `starting` when it sends `StartWorker` and `busy` when it assigns; the
  agent reports the rest with `ExecutorState`.
- A `READY` that arrives while the backend holds the executor `busy` with a task is stale
  (the agent sends READY right after `Done`, possibly after the next assignment) and is
  ignored, so an executor never gets two tasks.
- `disabled` survives reconnects and restarts until an operator enables it. Crash classes
  `wedged`, `gpu_fault` and `device_mismatch`, and an assignment the agent rejects, disable
  the executor.

### 6.2 Feasibility at submit

Checked against known executors (seen in the last 7 days, online or not): every stage of
the job needs an executor that declares it at the job's pixel count (and, for encode and
denoise, at least the job's number of reference images; for enhance with references, a
model in `reference_models` taking that many), with one model revision common to encode,
denoise and decode. Otherwise `422 {rejected: reason}`. A limit with
`above_references` applies only to tasks with more references than that; a decode limit
with `decode_mode` only to decodes in that mode (tiled decodes stay on the card, so the
agent's full-size limit names `untiled`). Decode mode is matched per job: an executor that
declares only `tiled` serves jobs that ask for tiling.
`max_concurrent_per_agent` is one count shared by every limit that carries it: the tasks
that spill into host RAM (full-size decode, denoise with many references) take turns
whatever their stage, since side by side they ended in a global OOM (HARDWARE.md F10).
`/api/capabilities` answers the same question ahead of time for the composer's presets.

### 6.3 Placement

1. **Feed resident workers first:** a `ready(S)` worker gets the oldest queued `S` task
   that fits its model revision and live limits. For enhance the model must also be the
   one the task needs (`reference_models` for a task with references, preferably another
   for one without): a PE-T2I worker never gets a job with images, and a ready PE-T2I worker
   with only such jobs waiting restarts with PE-I2I (rule 3).
2. **Start workers on idle executors** for the oldest runnable task, stages with no worker
   first. Long stages (`enhance`, `denoise`) get another worker when the workers already
   starting or ready for that stage do not cover the queue, so a backlog runs on every
   card. Short stages (`encode`, `decode`) get one worker at a time: it works through its
   queue faster than a second one could load.
3. **Stay only while there is work:** a `ready(S)` worker keeps its model while an `S`
   task is queued. As soon as none is, and another stage has queued work this executor
   can run that no other worker covers (free workers for any stage, plus busy ones for the
   short stages), it switches to that stage. Work merely expected (a job still in the stage
   before) does not hold a card. A task that has waited `max_wait` (default 10 min) takes
   over an executor even while its own stage has work queued.
4. **Idle grace:** with nothing at all to switch to, a `ready` worker stays loaded for
   `idle_grace` (default 45 s, `ARIES_IDLE_GRACE_S`) for a quick follow-up job, then is
   released.
5. **Model revision affinity:** denoise and decode only go to executors declaring the
   revision encode used.

Queued tasks that cannot run get a live `blocked_reason` (offline, lent, disabled, waiting
for host RAM, stale metrics, ...), shown in the UI.

### 6.4 Validation on agent

The agent rejects an assignment only when it contradicts its own declaration or state
(unknown executor, wrong stage, already busy, size above `max_px`) with `Rejected`. The
backend fences rejections by attempt: a rejection for an attempt that is no longer current
is ignored. A current one requeues the task without using up its retry and disables the
executor (`protocol_violation`).

### 6.5 Lending

`POST /api/executors/{id}/lend` takes an executor out of rotation (after its current task),
`/return` gives it back; `/disable` and `/enable` do the same with a reason.

---

## 7. Agent ⇄ backend protocol

Messages are defined once in `proto/aries/agent/v1/agent.proto`; Rust code is generated at
build time, Python code is checked in under `backend/aries/agent/v1/` (regenerate with
`buf generate` or `python -m grpc_tools.protoc`, see `buf.gen.yaml`).

### 7.1 Transport and authentication

The agent dials `AgentGateway.Connect` (gRPC, default port 50051) and holds the stream
with HTTP/2 keepalives. Both gRPC and the artifact HTTP endpoints authenticate with
`Authorization: Bearer <ARIES_AGENT_TOKEN>`, compared in constant time. Neither side has a
default token: the backend refuses to start and the agent refuses to run without one.

### 7.2 Messages

- **Agent → backend:** `Register` (first message), `ExecutorState`, `Progress` (step,
  steps, s/step; the enhancer also sends `sub_stage` and its token stats, and its step is
  tokens against `max_new_tokens`), `Done` (artifacts, timings, peak memory, mandatory
  `Provenance`), `Failed` (crash class, exit code, signal, kernel log tail), `Rejected`.
- **Backend → agent:** `Welcome`, `StartWorker`, `Assign` (inputs to GET, uploads to PUT,
  stage parameters), `Cancel`, `Release`.
- A `Requirement` with no known kind (from a newer agent) fails the registration, so a
  limit is never silently ignored.

### 7.3 Disconnects and reconnects

- When the stream ends the agent is marked offline. Its in-flight tasks wait
  `reconnect_grace` (default 30 s); if it is still offline then, they are requeued (or
  failed with class `lost` on their second attempt).
- On every `Register` the backend reconciles: tasks it has on that agent that are not in
  `Register.running` are requeued the same way (the agent restarted, or the result was
  lost), tasks in it keep their executor `busy`, and tasks it lists that the backend no
  longer runs get a `Cancel`.
- Results produced while disconnected are not lost: the agent keeps queued `Done`,
  `Failed` and `Rejected` messages across the reconnect, lists those tasks in
  `Register.running`, sends them right after `Register`, and re-announces each executor's
  state.
- Attempt numbers only grow (a retry or a rejection starts a new attempt); `attempts` is
  the retry budget, reset by a retry and given back on a rejection.

### 7.4 Artifacts

- Inputs and outputs move over HTTP: `GET`/`PUT /agent/v1/artifacts/{task}/{attempt}/{name}`.
  Reference images and stored reference latents are fetched by hash,
  `GET /agent/v1/blobs/{sha256}` (agent token required).
- An upload carries `X-Content-SHA256` and is validated before it is stored (safetensors
  header and names per `artifacts.json`, PNG dimensions, `enhanced.json` fields); uploads
  for a stale attempt get `409`, invalid payloads `422`. The agent retries other failures
  twice (after 1 s and 3 s) and then reports `upload_failed`.
- Blobs are content-addressed (`blobs/<sha[0..2]>/<sha>`). On completion the backend
  writes the job's metadata into the PNG (`tEXt` chunk `aries-image-studio`) and, in a
  blocking task, JPEG thumbnails (quality 82) whose longer side is 512, 768 and 1024 px; a
  missing one is made on request. Intermediate artifacts are deleted when the job is done, after 24 h for
  failed or cancelled jobs, or at once above the disk watermark. Stored reference latents
  stay with their image; uploaded images no job uses are deleted a day after upload.

---

## 8. Public API (browser-facing)

| Endpoint | Purpose |
|---|---|
| `POST /api/jobs` | submit; `{id?, prompt, negative_prompt?, width, height, steps?, seed?, true_cfg_scale?, enhance?, meta?}`. `meta` is the UI's own metadata (a JSON object up to 4 KiB: turn id, aspect label, size and quality preset), stored and returned as is. `steps` is 1–100 (default 20); `tiling` (default false) decodes with VAE tiling and needs an executor that declares the `tiled` decode mode (agents declare both). `enhance_max_tokens` (256–24576, kept only with `enhance`) is the rewriter's token limit; unset, the worker's default: 4096 plus 2048 per reference image. Reasoning that runs out of room continues with the limit doubled, up to 24576, and only there is closed for the reply. `id` is the client's (1–64 chars of `[A-Za-z0-9_-]`); resubmitting it returns the existing job with `200`. `references` lists reference images in prompt order, each `{upload: sha256}` or `{image: job_id}` (an earlier result); not with `true_cfg_scale > 1`, nor with `enhance` unless an executor has an image-aware enhancer for that many, and at most what an executor declares. Returns `201`, `400` (bad id) or `422 {rejected: reason}` (§6.2) |
| `POST /api/uploads` · `GET /api/uploads/{sha256}` · `/thumb?size=` | upload a reference image (PNG, JPEG or WebP body, up to 32 MB, 16384 px a side and 64 MP; `422` otherwise) → `{id, sha256, width, height, mime}` (size upright by EXIF; the same image twice is one upload) · the image · a JPEG thumbnail like the job thumbnails (default 512) |
| `GET /api/jobs?changed_since=<version>` | `{version, jobs, deleted}`: jobs changed since a version (all without it), each with tasks (status, step, enhance `sub_stage` and `stats`), per-stage provenance, `meta` and `favorite`; `deleted` lists ids deleted since that version. Poll 1 s while anything is active, 10 s when idle |
| `GET /api/jobs/{id}` · `PATCH /api/jobs/{id}` · `DELETE /api/jobs/{id}` | detail · `{favorite}` · delete (cancels first; history is user-owned) |
| `GET /api/preferences` · `PUT /api/preferences` | the UI's preferences (composer settings, theme): one JSON object up to 16 KiB, `{}` until saved |
| `POST /api/jobs/{id}/cancel` · `/retry` | cancel · retry a failed or cancelled job from its last good artifact (`409` for any other state) |
| `GET /api/images/{id}` · `/thumb?size=512\|768\|1024` | PNG with metadata · JPEG thumbnail (longer side; default 1024, `400` for other sizes). Both are sent with `Cache-Control: public, max-age=31536000, immutable`, since a finished job's image never changes |
| `GET /api/system` | agents, executors, states, declared capabilities, latest scraped resources, scrape age, queue by stage, blocked reasons |
| `GET /api/capabilities` | for the composer: max pixels, most reference images (`max_references_now`/`_ever`), most Enhance reads (`enhance_max_references_now`/`_ever`, 0 without PE-I2I) and the full-size presets servable **now** vs **ever** |
| `POST /api/executors/{id}/lend` · `/return` · `/disable` · `/enable` | operator controls |
| `GET /metrics` | backend metrics (§9) |

The SPA keeps nothing in browser storage: it renders what this API returns, shows images
from `/api/images/{id}` (in the grids, the thumbnail size each tile needs through `srcset`;
the viewer shows the 1024 px one until the PNG has loaded) and writes favorites and preferences
back. There is no authentication on this API yet: keep the backend on a trusted network.

---

## 9. Metrics

- **Backend** (`/metrics`): `aries_jobs_total{state}`, `aries_tasks_queued{stage}`,
  `aries_tasks_blocked{stage,reason}`, `aries_task_wait_seconds`, `aries_job_e2e_seconds`,
  `aries_executor_switch_total`, `aries_agent_connected`, `aries_agent_scrape_age_seconds`,
  `aries_artifacts_bytes{kind}`.
- **Agent** (`/metrics`, scraped by the backend every second): host RAM available,
  per-executor memory used/total/overflow and state, worker load and task durations, task
  outcomes and worker exits by class, reconnects. The same listener serves `/healthz`
  (`200` while the agent runs: connected, connection age, uptime, executor states) and
  `/readyz` (`200` once the backend's `Welcome` arrived, `503` while disconnected).
- The backend's `/metrics` re-exports every agent's latest scrape with an `agent` label
  (one HELP/TYPE per family), so Prometheus needs a single target. A stale scrape is left
  out, so a sleeping GPU box drops out instead of going `down` and paging. The agent's own
  `aries_agent_connected` is not re-exported; the backend's per-agent one replaces it.
- The agent metric names the backend scrapes are pinned in
  `proto/aries/agent/v1/agent_metrics.json`, with tests on both sides; card memory
  available is total minus used.
- Labels never carry prompt text, seeds or job ids.

---

## 10. Agent

### 10.1 No torch

`aries_agent` never imports torch or any GPU runtime (a test enforces it). It reads GPU
facts from sysfs and `/proc/meminfo`, never from `xpu-smi`.

### 10.2 Configuration

Environment variables (`ARIES_*`): token (or `ARIES_AGENT_TOKEN_FILE`), backend gRPC and
HTTP URLs, agent id, metrics listen address and the URL the backend scrapes, model paths
(`ARIES_MODEL_PATH`, `ARIES_ENHANCER_MODEL_PATH`, optional `ARIES_ENHANCER_I2I_MODEL_PATH`), executors. Without
`ARIES_EXECUTORS_JSON` the agent declares one executor per `xe` device in sysfs
(`ARIES_NUM_CARDS`, default 2) with `max_px` 2400 × 1792 (the largest composer preset)
and refuses to start if sysfs shows fewer cards. Encode and denoise declare
`ARIES_MAX_REFERENCE_IMAGES` (default 4) references, with a denoise limit above
`ARIES_DENOISE_REFS_LIMIT_ABOVE` (default 2). Model revisions come from `MODEL_REV`, a
`revision` file, a git checkout or the metadata `hf download --local-dir` writes, else
`unknown`.

### 10.3 Worker processes and device isolation

- One worker process per executor, spawned with `ONEAPI_DEVICE_SELECTOR` naming its card
  (or the CUDA/ROCm equivalent) and `HF_HUB_OFFLINE=1`; it speaks the existing JSON-lines
  protocol over a Unix socket. Worker output goes to the agent log.
- The worker's `hello` lists the devices it can actually open, from the runtime: it
  probes `torch.xpu` index by index (`device_count()` over-reports under a one-card
  selector) and reads each card's PCI address from its Level Zero UUID. Anything but
  exactly the executor's PCI address kills the worker and disables the executor
  (`device_mismatch`), which catches a selector that picks the wrong card.
- The agent answers a worker's `claim` at once (`job` or `none`) and pushes a later
  assignment to the idle worker; it never blocks the socket reader on a claim, so `done`
  and `error` are read as soon as they arrive.
- A duplicate `StartWorker` for the stage a live worker already serves is ignored, and the
  agent re-announces the worker's state.

### 10.4 Sleep

While any worker runs, the agent holds a `systemd-inhibit` sleep lock, so the GPU host
does not suspend mid-job.

---

## 11. Deployment

- **Backend:** `studio/Dockerfile`; set `ARIES_AGENT_TOKEN`, `ARIES_DATA_DIR`,
  `ARIES_ARTIFACT_BASE_URL` (the URL agents use for artifacts) and the listen addresses.
- **Agent:** `deploy/aries/compose.yaml` (container) or `deploy/aries/aries-agent.service`
  (systemd, token in `/etc/aries/agent.env`). `deploy/aries/.env` holds the token and is
  git-ignored; `.env.example` lists the settings.
