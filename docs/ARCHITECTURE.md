# Aries Image Studio — Architecture: asynchronous, staged execution

> **Superseded.** Since 0.3.0 the manager is gone: Studio (Rust) schedules and thin agents
> run the stage workers, as described in [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md). This
> document is kept as the record of the 0.2 design; the stage workers and the hardware
> reasoning behind them still apply.

How the 0.2 backend ran the pipeline. It was the source of truth for the manager, the
stage workers, their protocol, queue and failure semantics, and telemetry. SPEC.md
covers the product and the browser-facing API; HARDWARE.md covers the measurements
behind the decisions here.

**Implementation status (v0.2.1):** the manager, stage workers and web UI are built.
The **Studio** service (§1) is not built yet: the browser talks to the manager
directly and keeps history in browser storage. See §10.

---

## 0. Why

Bringing the pipeline up on this hardware (HARDWARE.md) taught three things:

1. **Only process exit reliably frees memory.** VRAM, host RAM and driver-side
   overflow all came back cleanly only when a process ended.
2. **No card may hold more than one big component.** Text encoder + transformer on one
   card ran ~13 GiB of driver overflow into host RAM and caused an OOM kill.
3. **Generation is slow enough that loading models is cheap.** ~4 min per 2048² image
   vs seconds per model load.

So every stage runs as its own subprocess, on one card, with one model loaded, and a
manager schedules cards and keeps durable state.

---

## 1. Components

```
Browser ──poll──▶ Studio (k3s)                     Manager (Aries, one process)
                  • system of record: history,      • job queue + state machine (SQLite)
                    gallery, metadata (SQLite+PVC)  • card scheduler
                  • accepts jobs while Aries sleeps • per-job files (embeds, latents, images)
                  ──submit / poll / fetch / ack──▶  • garbage collection
                                                    • spawns ─▶ stage worker (card 0)
                                                    • spawns ─▶ stage worker (card 1)
```

| Component | Owns | Lifetime |
|---|---|---|
| **Studio** | user-facing history, images after fetch, metadata, UI | always on (k3s) |
| **Manager** | execution queue, job state, per-job files until acknowledged, card schedule | long-running on Aries; restartable without losing jobs |
| **Stage worker** | one model on one card; processes jobs of one stage | spawned per stage run; exits when idle |

The manager is not exposed to the network beyond Studio. Browsers only ever talk to Studio.
(Target design. In v0.2.1 the browser talks to the manager directly, and the manager
has no authentication: keep it on a trusted LAN.)

---

## 2. Stages

| Stage | Model | Card memory | Input → output | Typical time |
|---|---|---|---|---|
| `enhance` (optional) | Qwen-Image-2.1-PE-T2I, bf16 | ~18 GB | prompt → `enhanced.json` (`rewritten_prompt`, `wh_ratio`) | seconds |
| `encode` | Qwen3-VL 8B text encoder | ~16.3 GB | prompt → `embeds.pt` (`prompt_embeds` + mask, ~0.25–2.5 MB) | seconds |
| `denoise` | transformer (7B DiT) | ~13.3 GB + working memory (~16 GB peak observed) | embeds + seed → `latent.pt` (~2 MB, the tensor `vae.decode` receives) | 6.4 s/step at 2048² |
| `decode` | VAE | 0.6 GB weights; untiled peak ~25.8 GiB at 2048² | latent → `image.png` | ~5 s at 2048² |

- `denoise` passes `prompt_embeds` / `prompt_embeds_mask` and a `latents` tensor made
  from the seed (or a CPU generator), so no text encoder is loaded and runs reproduce
  across cards.
- The latent is captured at the `vae.decode` input (already unpacked and denormalized),
  so `decode` needs only the VAE.
- `decode` mode is per job (`untiled` or `tiled`). The web UI always asks for
  `untiled`; tiling stays available as a fallback mode.

---

## 3. Job state machine

```
submitted ─▶ [enhance_queued ─▶ enhancing ─▶] encode_queued ─▶ encoding
          ─▶ denoise_queued ─▶ denoising ─▶ decode_queued ─▶ decoding
          ─▶ stored ─▶ acknowledged ─▶ collected

any *ing  ─▶ (worker died or job error) ─▶ back to *_queued, attempts += 1
attempts == 2 on the same stage ─▶ failed(stage)        (no crash loops)
any queued/active ─▶ cancelled (on request)
stored ─▶ expired ─▶ collected                           (TTL, never fetched)
```

- Every transition is a single SQLite transaction and bumps a global `version`
  counter stored on the job (used for `changed_since` polling).
- A job carries everything needed to run any remaining stage from files on disk, so
  manager restarts and worker crashes lose at most the stage in progress.
- `failed` records the stage, the error text, and the crash class if known
  (`gpu_fault`, `engine_reset`, `wedged`, `oom_kill`, `python_error`).

---

## 4. Scheduler: cards are executors, stages are work types

Each card is either **idle**, **running a stage worker**, **lent** (to aries-modeld,
e.g. for llama-server) or **unavailable** (wedged / unhealthy).

When a card becomes free, the manager picks the stage to run on it:

1. Candidate stages = stages with queued jobs whose constraints this card can meet.
2. Pick the stage whose **oldest waiting job is oldest**.
3. **Stickiness:** a worker keeps claiming jobs of its stage while any remain, unless
   another stage's oldest job has waited longer than `max_wait` (default 10 min) — then
   it finishes its current job and exits so the card can switch.
4. A worker with no work waits `idle_grace` for new jobs before exiting: 15 s by
   default, 5 min for `enhance` (enhanced prompts tend to come in bursts and the model
   is the slowest to load). It exits at once if its card is needed for another
   stage's queued work that no other card can take.

Stage constraints:

| Stage | Constraint |
|---|---|
| `decode` | host available RAM ≥ `decode_ram_floor` (default 6 GiB) **and** no other decode running (a card running a worker is never shared, so the card is otherwise empty) |
| `encode` | at most one encode worker at a time |
| `enhance`, `denoise` | a free card, unless workers already starting on other cards cover every queued job of that stage |
| any | card not `unavailable` |

This one policy covers both useful layouts without hard-coding either:

- **Backlog:** both cards drain the same stage in parallel (two denoises at once ≈ 2×
  throughput).
- **Steady trickle:** one card stays on `denoise` (sticky) while the other cycles
  `enhance → encode → decode` for the jobs around it.

**Card health:** a worker that dies while its card has a devcoredump in sysfs marks
the card `unavailable` until the manager restarts (a wedged card needs a reboot).
The other card keeps working.

---

## 5. Worker protocol

Spawn:

```
python -m aries_worker --stage denoise --card 1 --socket /run/aries/manager.sock
       env: ONEAPI_DEVICE_SELECTOR=<that card only>, ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
```

Each worker sees exactly one card (`xpu:0` inside the worker). Messages are JSON lines
over a Unix socket:

| From | Message | Meaning |
|---|---|---|
| worker | `hello {stage, card, pid, versions}` | after the model is loaded; includes `load_s` |
| worker | `claim` | ask for the next job of my stage |
| manager | `job {id, paths, params}` or `none` | `none` → start the idle-grace timer |
| worker | `progress {id, step, steps, s_per_step}` | denoise, every step |
| worker | `progress {id, sub_stage, enhance_stats}` | enhance: `sub_stage` is `prefill` → `thinking` → `rewriting`; `enhance_stats` carries token counts, the token limit (`max_new_tokens`, 2048) and prefill/generation tokens per second; at most every 0.25 s plus on each sub-stage change |
| worker | `done {id, timings, peak_vram}` | outputs written to the job's directory |
| worker | `error {id, message}` | Python-level failure or cancellation; worker stays up |
| worker | `heartbeat` | every 5 s; 30 s with no message at all → manager drops the worker |
| manager | `cancel {id}` | interrupt at the next step boundary |
| manager | `exit` | finish the current job, then exit |

- A worker writes outputs to a temp name and renames on success, so a crash never
  leaves a half-written file looking complete.
- Every message may carry `vram {allocated, reserved, peak, total}` for the card.
- A worker's `done` or `error` for a job that was cancelled meanwhile is ignored.
- The manager records each worker run (`stage_runs`): card, stage, pid, start/end,
  exit code, jobs processed, `load_s`, why it exited (`idle_grace`, `max_wait`,
  `card_needed`, or the crash class) and the queue depths at that moment.
- On worker death the manager reads the exit code (137 → `oom_kill`) and the last kernel
  log lines for that card to classify the crash.

---

## 6. Storage and garbage collection

```
/var/lib/aries-studio/
  manager.sqlite                 jobs, stage_runs, cards, settings (WAL mode)
  jobs/<job id>/
    request.json                 prompt, aspect, seed, steps, decode mode, enhance flag
    enhanced.json                (if enhanced)
    embeds.pt
    latent.pt
    image.png
```

| File | Deleted when |
|---|---|
| `embeds.pt` | denoise succeeds |
| `latent.pt` | image acknowledged (kept until then so a failed or re-requested decode needs no re-denoise) |
| `image.png`, `enhanced.json` | Studio acknowledges, or TTL expires (default 7 days) |
| job row | 30 days after `collected` (keeps the audit trail short) |

GC runs every 60 s.

**Acknowledgement:** the client (Studio; for now the browser) fetches the image (`GET /jobs/{id}/image`, response carries
`X-Content-SHA256`), stores it, verifies the hash, then `DELETE /jobs/{id}/image`.
Only that DELETE moves the job to `acknowledged`. Fetch is idempotent until then.

**Disk watermark:** above 80% used, the manager stops accepting new jobs (`503` with a
reason) and runs GC immediately.

---

## 7. APIs

### Manager (internal: Studio only)

| Method | Path | Purpose |
|---|---|---|
| POST | `/jobs` | Submit one job or a list; the client supplies job ids (ULIDs), so resubmits are idempotent |
| GET | `/jobs` | Current jobs: in flight and stored awaiting pickup (`?all=true` for every job) |
| GET | `/jobs/{id}` | One job, full detail |
| POST | `/jobs/{id}/cancel` | Cancel queued or running |
| GET | `/jobs/{id}/image` | PNG + `X-Content-SHA256` |
| DELETE | `/jobs/{id}/image` | Acknowledge receipt → GC |
| DELETE | `/jobs/{id}` | Acknowledge a stored job, or delete a finished one (active jobs: 400, cancel first) |
| DELETE | `/jobs/failed` | Delete every failed job |
| GET | `/status` | Cards (state, stage, sub-stage, VRAM, overflow), queue depth per stage, host RAM, decode RAM floor. A card's `vram_peak` covers its current or last worker run |
| GET | `/health` | `/status` plus an overall `state`: `ready`, `busy` or `error` |
| GET | `/stage_runs?limit=N` | Recent worker runs (§5) |
| GET | `/metrics` | Prometheus |

`POST /jobs` returns `503` with a reason above the disk watermark.

Live state (per-step progress, card status) is served from memory; durable state from
SQLite. Both are behind the same endpoints.

### Studio (browser-facing; planned)

- `GET /api/jobs` returns current active and pending jobs with stage, progress and
  ETA. The browser polls every **1 s while any job is active**, every **10 s when idle**,
  and on tab focus.
- Studio polls the manager via `GET /jobs` while it has active jobs, fetches and
  acknowledges finished images, and serves them from its own storage.
- Submission, history, gallery and favorites endpoints as in SPEC §4.

Until Studio exists the web UI does Studio's job itself: it polls the manager's
`GET /jobs` on the same schedule, fetches and acknowledges finished images, and keeps
history and images in browser storage.

---

## 8. Failure handling

| Event | Handling |
|---|---|
| Worker crashes mid-job | Job back to its stage queue, `attempts += 1`; second failure → `failed(stage)`; crash class recorded |
| GPU fault / engine reset on a card | As above, plus card health check; `wedged` → card `unavailable` |
| Host OOM kill (exit 137) | As above; scheduler raises `decode_ram_floor` by 2 GiB for the session and logs it |
| Manager restart | On start: any job in an `*ing` state goes back to its queue (attempts unchanged — the manager, not the job, failed); cards reset to idle |
| Worker dies before `hello` | It never loaded its model, so no job is charged; that card waits before the next spawn (5 s, doubling up to 5 min) |
| Cancel while running | Job is `cancelled` at once; the worker stops at its next step (denoise) or token (enhance); its late `done`/`error` is ignored |
| Aries asleep | Studio keeps accepting jobs and submits them on wake (idempotent ids) |
| Studio unreachable | Manager keeps running jobs; images wait for acknowledgement until TTL |
| Disk above watermark | New submissions refused with a reason; GC runs |

---

## 9. Telemetry

The manager exports (prefix `aries_manager_`):

| Metric | Labels | Notes |
|---|---|---|
| `queue_depth` | `stage` | |
| `jobs` | `state` | jobs per status |
| `card_state` | `card`, `state` | 1 for the current state (`idle`, `running`, `unavailable`) |
| `stage_runs_total` | `stage`, `card`, `exit` | worker processes, by exit code |
| `worker_crashes_total` | `stage`, `card`, `exit`, `crash_class` | |
| `worker_evictions_total` | `stage`, `card`, `reason` | `max_wait` or `card_needed` |
| `eviction_queue_depth`, `eviction_jobs_processed` | | histograms at eviction time |
| `model_load_seconds` | `stage` | |
| `vram_overflow_bytes` | `card` | driver memory in system RAM, from fdinfo |
| `host_available_bytes` | | |
| `storage_bytes` | | per-job files on disk |
| `enhance_prefill_tokens_per_second`, `enhance_generation_tokens_per_second` | | histograms |
| `enhance_tokens_total` | `type` | `prompt`, `thought`, `reply` |

Labels never carry prompt text, seeds or job ids. Per-job timings (`encode_s`,
`denoise_s`, `s_per_step`, `decode_s`, `enhance_s`, token stats, `total_s`) are merged
into the job's `timings` as each stage finishes. The UI status panel shows each
card's current stage, VRAM and per-stage queue depth.

---

## 10. Not built yet

- **Studio** (§1, §7): the always-on service that owns history and images. The next
  major version adds it as a Rust server with SQLite storage; the browser's
  localStorage/IndexedDB history goes away then.
- **Card lending** to aries-modeld (the `lent` card state) and wake-on-submit.
- Per-stage `queue_wait_s` and `load_s` in job timings; `load_s` is recorded per
  worker run only.
- Prometheus scrape config, alert rules and a Grafana dashboard.

## 11. Open questions

- `idle_grace` and `max_wait` defaults: tune from real use.
- Whether two untiled decodes at sizes below 2048² may run concurrently (likely yes;
  measure host RAM).
- Peak VRAM of `denoise` alone on a card at 2048² (expected ~16 GB; measure).
- Enhancer on XPU: memory, latency, and whether upstream `pe_core.py`'s prompt is used verbatim.
- Card lending handshake with aries-modeld (who initiates, timeouts).
