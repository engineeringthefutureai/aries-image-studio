# Changelog

## 0.3.1

- Page load shows a spinner until Studio's first job list arrives; the starter (and the
  gallery's empty state) appears only when there really is nothing, instead of flashing
  on every load.
- The viewer's previous/next (`←`/`→`) follows the timeline: left to right through a
  turn, then on to the older turn. It used to step through jobs newest first, right to
  left within a card.
- The agent image pins the stack it was measured with (`torch==2.14.1+xpu`,
  `transformers==5.20.1`) and adds `torchao==0.18.0`, so the prompt enhancer can run from a
  torchao int4 checkpoint (~8 GiB of card memory instead of ~18 GB, ~35 tok/s decode).
  The worker loads it with `device_map` and the agent declares the enhancer's own precision
  (read from the checkpoint, or `ARIES_ENHANCER_MODEL_DTYPE`) instead of the main model's;
  each stage's worker now gets its own model's `MODEL_DTYPE`. flash-linear-attention stays
  out (measured: +6% decode for ~67 s of Triton compile per fresh container).
- The composer folds to a one-line pill (full width at the bottom on a phone, bottom
  right on wider screens) showing what the next send makes; focusing it, `/` or Rerun /
  Vary opens it, and a send, `Esc` or a tap outside folds it again. The timeline's
  bottom padding shrinks to match.
- A turn shows its images in the order they were sent, left to right, which is the order
  Studio processes them; they used to flip to newest first once Studio reported them.
  Each job records its position in the send (`meta.index`). Sending scrolls to the top
  of the timeline, and a new turn's carousel starts at its first image and no longer
  jumps as images finish (it follows only the image you pick).
- The agent's decode limit (6 GiB free host RAM, one at a time) now starts above 3.5 MP
  (`ARIES_DECODE_ABOVE_PX`), so only full-size decodes are held back; ½-size decodes peak
  under 8 GiB and used to queue behind each other above 1 MP. `ARIES_DECODE_MAX_CONCURRENT`
  is now documented.
- The count picker (1x–4x) shows on a phone too; it was hidden below 640 px, so a
  phone kept whatever count was last set elsewhere.
- Composer, simplified: the main row has aspect, size, Draft 12 · Standard 20 · High 40,
  count and Enhance; every image gets its own random seed (they used to count up from
  one) and the seed is no longer shown.
  **Advanced** (closed by default) holds custom steps (1–100), a fixed seed, VAE tiling
  (off by default) and **rows**: each row a full set of settings (aspect, size, steps,
  Enhance, tiling) sent with the same prompt and seed, to compare e.g. steps or enhanced
  against plain. Count applies per row, up to 16 images per send. Advanced settings apply
  only while it is open, and a send closes it and resets them to the simple defaults. Each image of a turn shows what sets it apart. Rerun opens
  Advanced to keep the image's seed.
- Tiling per job: `POST /api/jobs` takes `tiling`; Studio sends a tiled decode and only to
  executors that declare it (agents now declare untiled and tiled), and the viewer shows
  the decode mode the job used.
- The composer's prompt field empties once a job is sent (settings stay).
- Rerun and Vary fill the composer instead of starting a job: prompt, aspect, size,
  steps and the image's Enhance setting, with the seed locked (Rerun) or Random (Vary).
  A bar above the prompt shows which image it came from and, if it had an enhanced
  prompt, an Original · Enhanced switch (Enhanced puts the rewritten prompt in with
  Enhance off). This replaces the viewer's Reuse button and both "To composer" buttons
  for enhanced prompts; Reuse also left Enhance as it was instead of taking the image's.
  The timeline's Rerun used to call retry, which Studio ignores for a finished job, so
  the image vanished from the timeline with nothing queued until a reload; Studio now
  answers `409` to a retry of a job that is not failed or cancelled. Cancelled tiles
  say Retry (they resume the job) instead of Rerun.
- Steps: Studio accepts 1 to 100. It refuses other counts with `422`
  (it accepted anything before) and defaults to 20 instead of 30. Rerun and Vary keep
  the image's step count (Reuse used to fall back to a preset).
- The agent has health endpoints on its metrics port: `/healthz` (`200` while it runs,
  with connection state, uptime and each executor's state) and `/readyz` (`200` only
  while connected to Studio). Its Compose file has a healthcheck on `/healthz`, and
  `make health` shows both. `aries_agent_connected` now turns 1 when Studio welcomes
  the agent rather than when the stream is opened.
- Viewer: clicking the image zooms in to 100 % at the pointer, and clicking again
  (without dragging) returns to Fit; the zoom-in cursor promised that, but only a
  double-click did it. Double-click no longer zooms.
- Timeline: a batch is a carousel again, at full tile height, instead of a grid of
  smaller tiles. It snaps tile by tile, fades and shows arrows at an edge with more
  beyond, has dots below, follows the selected image (from the pills or dots), and
  takes `←`/`→` when focused.
- Viewer: 100 % is the image's actual size (one image pixel per device pixel) instead
  of the fitted size, and a zoomed image can be explored: drag it, use the arrow keys,
  or click or drag the navigator thumbnail. Wheel and pinch zoom at the pointer,
  `1` jumps to 100 %, and zoom goes up to 800 %.
- Thumbnails are JPEGs at 512, 768 and 1024 px (`/api/images/{id}/thumb?size=`), about
  40, 80 and 140 KB, instead of a lossless 1024 px WebP that was nearly as large as
  the PNG (1.2 MB against 1.7 MB). Grid tiles pick the smallest size that is sharp on their screen, and
  the viewer shows the thumbnail until the full image has loaded. Images and
  thumbnails are cached by the browser instead of downloaded again on every view
  switch. Thumbnails of earlier images are made the first time they are asked for,
  and the old WebP files are removed when Studio starts.
- Phone header: the "Offline" status pill no longer pushes the page 13 px wider than
  the screen.

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
