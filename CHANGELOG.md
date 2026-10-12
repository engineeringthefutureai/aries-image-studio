# Changelog

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
