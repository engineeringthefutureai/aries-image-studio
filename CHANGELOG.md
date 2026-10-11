# Changelog

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
