# Optimizations (0.3.1 → 0.4.0)

What was changed to keep full-size jobs inside the hardware, and what it measured. All
numbers are from the aries host: 2× Intel Arc Pro B60 24 GB, 30 GB of host RAM. Detail and
failure modes live in [HARDWARE.md](HARDWARE.md); this page is the overview.

## 1. Full-size decode memory

**Problem.** A plain untiled decode of a full-size image peaked at 25.8 GiB on a 24 GB
card. The driver kept the overflow in host RAM, 13–17 GB of it, and next to a denoise on
the other card that ended in the global OOM that logged the desktop session out
(HARDWARE.md F10).

**Exact split decode** (`backend/aries_worker/stages/decode.py`, HARDWARE.md §1.3).

- The VAE decoder's only attention block (`mid_block`) runs once on the whole latent, so
  the image keeps its global context.
- The memory-heavy upsampling blocks are local convolutions. They run in two strips across
  the longer side, each reaching 32 latent lines past its cut; the join is a hard cut where
  both strips agree bit for bit.
- The allocator cache is emptied between strips, since strips of different sizes fragment
  it. Strips across the shorter side of a portrait image (6 narrow strips) still spilled
  13 GB, hence across the longer side.
- Applies to untiled decodes above 3.5 MP (`ARIES_DECODE_SPLIT_ABOVE_PX`; `0` turns it off).
- Rejected: running the whole decoder per half, the way tiling does. It shifts colours by
  up to 10/255 and peaked at 29.2 GiB.

**Per-frame caches dropped.** The decoder is a video decoder: each causal convolution keeps
a copy of its input for the next frame. A single image never reads them, so `_OneFrameCache`
discards the tensors and keeps only the markers the convolutions check.

| | Card peak | Host RAM held by the driver | Pixels vs untiled |
|---|---|---|---|
| Plain untiled | 25.8 GiB | 13–17 GB | — |
| Split | 18.1–19.6 GiB | 0.1–1.5 GB | max 1/255 |
| Split, per-frame caches dropped | **11.6–12.5 GiB** | **0.09 GB** | max 1/255 |

Pixels were checked against Studio's production PNGs of three latents captured from live
2752×1536 jobs: no pixel off by more than 1/255. Memory was checked at all seven full-size
presets (2026-10-18); every one decodes in about 5 s, in 2 strips.

## 2. Host RAM safety

- **Workers die first.** Workers start at `oom_score_adj` 1000
  (`ARIES_WORKER_OOM_SCORE_ADJ`), so the kernel's OOM killer takes a worker, whose task is
  retried, never the desktop session (its processes run at 100–200).
- **Faster RAM guard.** The agent samples host RAM every 0.25 s with a 4 GiB floor (was
  0.5 s and 3 GiB): the drop that caused F10 came faster than the old window.
- **One memory-heavy slot per agent.** `max_concurrent_per_agent` is one count shared by
  every limit that carries it, so a plain untiled full-size decode and a heavy denoise take
  turns instead of running side by side on the two cards.
- **Heavy denoise threshold lowered.** Denoise with more than 2 references (was 3) is
  memory-heavy: one 4-reference full-size denoise made the driver hold ~7 GB of host RAM,
  far more than torch reports.
- **Decodes need only what they use.** Split decodes keep the 6 GiB RAM floor but leave the
  memory-heavy slot. Tiled full-size decodes skip the floor too: limits can name a decode
  mode (`Limit.decode_mode`), and the agent's full-size limit covers untiled decodes only.
- **No build output in RAM.** The host's `/tmp` is a tmpfs, so Rust builds go to
  `~/.cache/aries-studio-target`.

## 3. Enhancer token budget

A quality fix rather than a speed one: enhanced prompts came out as reasoning text or "…"
because every run hit its token limit mid-reasoning.

- Default limit 4096 + 2048 per reference image, capped at 24,576 (the model card's
  example uses 24k). Jobs can set it under Advanced.
- A reply reserve of `min(1536, limit / 4)` tokens is kept for the answer: reasoning that
  reaches it is stopped.
- When reasoning still runs out, generation continues with the limit doubled, up to the
  cap; at the cap the reasoning is closed and the model answers.
- Unfinished reasoning is never used as the prompt.
- Measured: 3 references used 4,907 of the default 10,240 tokens; a forced
  2048 → 4096 → 8192 run ended in a proper answer.

## 4. Scheduler and Studio

- Each 1 s scheduler tick loads its jobs in one batch, four queries in all; it used to run
  four per queued or active task, holding the scheduler lock that the gRPC handlers wait on.
- A task's blocked reason is written only when it changes; it used to take a write
  transaction per queued task every tick.
- `compute_capabilities` computes the largest servable image with one helper instead of two
  copies of the same loop.
