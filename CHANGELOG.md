# Changelog

## 0.3.0 (Architecture V3)

Release notes: [docs/releases/v0.3.0.md](docs/releases/v0.3.0.md).

V3 replaces the manager with a Rust Studio backend and a thin GPU agent
([docs/ARCHITECTURE_V3.md](docs/ARCHITECTURE_V3.md)). Fixes on top of the first V3 commits:

- Jobs submitted from the UI keep the browser's id, so tiles update, finish and can be
  cancelled; resubmitting an id is idempotent.
- Denoise results are accepted (the backend expected a tensor named `latents`; workers
  write `latent`). Tensor and field names now live in one contract file, checked by
  tests on both sides.
- `true_cfg_scale` defaults to 1.0 (no guidance, as in 0.2), and encode receives the scale
  so jobs that set it get their negative embeddings.
- ½ and ¼ size presets are multiples of 32, which Qwen-Image-2.1 needs (it floored 1264
  to 1248 and Studio then rejected the PNG); Studio refuses other sizes at submit.
- The prompt rewriter is told the job's aspect ratio, so a 2:3 job no longer gets a
  "square-format" description.
- Every composer preset is accepted: the agent's default size limit was 2048², below the
  4:3, 3:2 and 16:9 presets.
- An executor no longer gets a second task (and then disabled) when the agent's READY
  after a result crosses the next assignment; stale rejections are ignored.
- An agent that restarts or loses its connection no longer leaves jobs stuck: tasks it no
  longer has are requeued on registration, and results produced while disconnected are
  delivered after reconnecting.
- A denoise backlog runs on both GPUs again.
- Workers report the card torch actually opened (from its Level Zero UUID); the agent
  refuses a worker on the wrong card.
- Enhance progress (sub-stage, tokens, time left) and the rewriter's reasoning work under
  V3; the inspector shows all stage timings again.
- No default agent token on either side; `.env` is git-ignored. Model revisions are read
  from the `hf download` metadata instead of made up. The agent honors
  `ARIES_MODEL_PATH`/`ARIES_ENHANCER_MODEL_PATH` and `ARIES_METRICS_LISTEN`.
- Worker output is logged at INFO, so tracebacks show up in the agent log.
- Job history loads in three queries instead of two per job.

Found by running the stack on the GPUs:

- Encode results are accepted when the pipeline returns no attention mask (it does when
  nothing is padded); the masks are optional in the contract.
- The backend reads the agent's real metric names, so host RAM and card memory are no
  longer seen as 0 (which blocked every full-size decode).
- Retrying a job, or a rejected assignment, no longer reuses an attempt number; that
  failed the assignment and stalled scheduling for every job.
- The agent answers an idle worker's claims at once instead of blocking its socket
  reader, which delayed results by minutes after an idle period.
- Uploads the backend rejected (422) are not retried.

After running it for real:

- A worker keeps its card only while its own stage has queued work. An idle worker
  switches at once to another stage's waiting work instead of waiting for work still
  being prepared elsewhere, or for its idle grace; idle grace now applies only when
  nothing else is waiting.
- Studio's `/metrics` re-exports every agent's metrics with an `agent` label, so
  Prometheus scrapes one target; a sleeping GPU box drops out instead of paging.
- The agent reconnects reliably: a dead stream's request iterator could take the next
  stream's Register, and the backend then refused every reconnect.
- The web app keeps nothing in browser storage: history, images, favorites, turn grouping
  and preferences (including the theme) load from Studio, so every browser shows the same
  thing. New: `PATCH /api/jobs/{id}` (favorite), `GET`/`PUT /api/preferences`, job `meta`,
  and `deleted` in `/api/jobs?changed_since=`. Grids use 1024 px thumbnails.
- Dead code removed: the v0.2 manager (`backend/aries_manager`, still on the
  `release/v0.2.1` branch) and its dependencies; the workers' `.pt` paths and `--card`;
  the unused `Drain` message; the `worker_runs` table; the web app's `/status` fallback
  and unused types, fields and phases; the duplicate root `Dockerfile` (a root
  `.dockerignore` keeps the Studio build context small).
- Denoise progress counts completed steps (the last report was 9/10).
- Studio has a Compose file (`deploy/studio/`), and
  [docs/BUILD_AND_RUN.md](docs/BUILD_AND_RUN.md) covers building, deploying, configuring
  and troubleshooting both sides; the README describes 0.3.0.
- Timeline: a batch's tiles sit in a grid instead of a strip whose other tiles were
  hidden behind the prompt; long prompts are clamped with "Show more"; the phone
  header no longer overflows. "Rerun with this" is gone ("To composer" stays). The
  header shows 0.3.0.
- Everything is named Aries Image Studio: the Rust crate and binary are
  `aries-image-studio`, the agent image and container `aries-image-studio-agent`, and
  each PNG's metadata `tEXt` chunk is `aries-image-studio` (was `aries-studio`).

Noise is now drawn on the CPU, so a seed from 0.2 no longer reproduces its image.

## 0.2.1

Release notes: [docs/releases/v0.2.1.md](docs/releases/v0.2.1.md).

- The enhancer's token limit rises from 1536 to 2048. Many runs used to stop at the
  limit; a run that hits it still counts as complete and keeps its prompt.
- Enhance progress is generated tokens against the expected finish line (85 % of that
  limit), with an estimated time left from the live speed, instead of fixed thirds
  per sub-stage. A run that goes past it shows an indeterminate bar and "almost
  done…".

Fixes for the known issues in 0.2.0:

- Reading prompt and Finishing show an indeterminate progress bar instead of a frozen
  0 % (encode and decode report no steps).
- Times shown on the timeline and gallery are generation time (the sum of the stage
  times, enhance included). The inspector shows the total time and a breakdown: each
  stage, generation time, queue & model loading, and total. Images from before 0.2.0
  now show their generation time too instead of N/A.
- The inspector reads the decode mode, per-stage times, peak VRAM and decoding GPU
  from the job instead of showing fixed hardware text; the timeline's "bf16 · Arc" chip
  is now "bf16".
- Status panel: a card's peak VRAM covers its current or last worker run instead of
  everything since the manager started, and the host RAM spill warning appears only
  above 1 GB (the driver keeps a small baseline there).

## 0.2.0

Release notes with screenshots: [docs/releases/v0.2.0.md](docs/releases/v0.2.0.md).

First public version, under the name **Aries Image Studio**, with its own mark (a ram
charging out of a pixel trail inside a picture frame) as logo, favicon and touch icon.

- Staged execution backend: a manager (queue, state machine, card scheduler, GC,
  HTTP API, Prometheus metrics) spawns one worker process per stage and card:
  `enhance` → `encode` → `denoise` → `decode`.
- Prompt enhancement with Qwen-Image-2.1-PE-T2I, with live sub-stages (reading prompt,
  thinking, rewriting) and token throughput.
- Web UI: timeline, composer, gallery, lightbox, status panel, keyboard shortcuts,
  light and dark themes. History is kept in browser storage.
- Crash handling: a failed stage is retried once, then the job fails with the stage and
  crash class; OOM kills raise the decode RAM floor; wedged cards leave rotation.

Fixed during release preparation:

- A worker that could not start (bad model path, broken runtime) was respawned every
  second forever; the card now backs off (5 s, doubling, up to 5 min).
- A cancelled job could be put back in the queue when its worker reported the
  cancellation, or advanced if the worker finished anyway.
- Enhancer prefill time was measured before prefill ran (HF `generate()` hands the
  prompt to the streamer first), so prefill throughput came out absurdly high.
- A `{` inside the enhancer's reasoning switched the status to "rewriting" early.
- The enhancer's prompt token count was always 0 (the tokenizer returns a
  `BatchEncoding`, not a `dict`).
- Seconds-per-step from one stage leaked into the next, and a retried enhance job
  showed its previous run's sub-stage.
- `total_s`, the jobs-by-state and storage gauges, and per-run model load time are now
  recorded.

Cleanup:

- Removed the v1 single-process worker, its tests and the API aliases (`/v1/*`,
  `/api/*`, no-op `/free`); the manager serves one set of paths.
- `worker/` is now `backend/`; docs moved to `docs/`; hardware notes merged into
  `docs/HARDWARE.md`.
