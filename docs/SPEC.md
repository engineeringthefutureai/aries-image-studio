# Aries Image Studio — Product & API Spec

Type a prompt, get pictures. A quiet, image-first app that sits between a chat
timeline and a gallery. No node graphs, no LLM chat. Single user, LAN only.

Inspiration: the *feel* of lightweight Labs-style image tools — minimal chrome,
results as the content, every image one click from "do that again, differently."
Own design; don't copy any product's layout or branding.

**Non-goals:** workflow editing, multi-user accounts, LLM conversation, training.

This document covers the product, UX, data model and the browser-facing API.
How the backend executes jobs — stages, the worker protocol, queue and failure
semantics, backend telemetry — is in **ARCHITECTURE.md**. UI behavior in detail is in
**SPEC_UI.md**.

---

## 1. Architecture

```
Browser ──HTTPS──▶ Studio (always on)  ──HTTP──▶  Manager (GPU box, 2× B60)
                   ├─ static SPA                  ├─ queue + state machine (SQLite)
                   ├─ REST API (§4)               ├─ card scheduler
                   ├─ history (SQLite)            └─ spawns one stage worker per card
                   └─ images + thumbnails
```

**History lives in Studio, off the GPU box.** Reasons:
- The GPU box suspends most of the day (~4 W asleep vs ~75 W idle). History, gallery
  and metadata must stay browsable while it sleeps.
- The GPU stack *will* crash (xe GPU faults happen). A crash must cost at most the
  stage in flight, never history.
- Control plane in the cluster, GPU box as a replaceable agent.

**Status (v0.3.0):** Studio exists as the Rust backend in `studio/` (ARCHITECTURE_V3.md):
it owns jobs, images and provenance in SQLite, and GPU boxes run a thin agent instead of
the manager. The SPA talks to its `/api` (ARCHITECTURE_V3.md §8), which is job-oriented
rather than the generation/gallery API sketched in §4. The SPA keeps nothing in browser
storage: jobs, images, favorites, turn grouping and preferences all load from Studio, so
every browser shows the same history.

---

## 2. UX

### 2.1 Timeline (default view)
Chat-shaped, newest at the bottom, composer docked below it.

Each **turn** = one submission:
- Prompt text, and the enhanced prompt (with the enhancer's reasoning, collapsed) when Enhance was on
- Param chips: `16:9 · seed 48213 · 40 steps · 2m 31s`
- 1–4 image tiles (one per job in the batch), sized to the aspect ratio
- Pending tile: aspect-correct placeholder showing the job's phase (§7.1) — queued
  (and for which stage), enhancing, reading prompt, drawing with percent and ETA,
  finishing — and cancel
- Failed tile: short error, "Retry" (new job, same params)
- Lineage marker when created via Vary/Reuse: `↳ from #id`

### 2.2 Composer
- Multiline prompt. Enter = generate, Shift+Enter = newline
- Chips:
  - **Aspect**: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16 (official sizes, §5)
  - **Count**: 1–4
  - **Seed**: Random (default) or Locked with a value; batches use seed, seed+1, …
  - **Quality**: Draft (12 steps) · Standard (20) · High (40); under Advanced: custom
    steps (1–100), fixed seed, VAE tiling, and rows of different settings sent together
  - **Enhance** toggle: rewrite the prompt with the prompt enhancer before generating
- Last-used settings (and the theme) are saved in Studio (`/api/preferences`), the same
  in every browser

### 2.3 Gallery
- Masonry grid of all finished images, newest first, infinite scroll (thumbnails)
- Filters: Favorites, aspect, text search over prompts
- Click → lightbox

### 2.4 Lightbox (shared by timeline and gallery)
- Full-resolution image, fit / 1:1 zoom, ←/→ to move through the current list
- Metadata panel: prompt, enhanced prompt, seed, size, steps, model, precision,
  timings (queue wait, total, s/step), device, software versions
- Actions:
  - **Download** (PNG with embedded metadata)
  - **Copy prompt** / **Copy seed**
  - **Rerun** — fills the composer with this prompt + params and the seed locked;
    doubles as a reproducibility check. Original or enhanced prompt is chosen there
  - **Vary** — the same with a random seed. Neither submits by itself
  - **Favorite**, **Delete** (with confirm; removes files too)
  - *Later:* **Use as reference** → edit mode (see §8)

### 2.5 Generator status pill (top bar)
`Ready` · `Busy (n queued)` · `Error` · `Offline`. Click for the status panel: each
card's state and current stage, VRAM per card, queue depth per stage, host RAM, and
recent worker runs. Jobs submitted while the generator is offline stay queued once
Studio exists (§1); until then the browser must reach the manager to submit.

### 2.6 Keyboard
`/` focus composer · `t` timeline · `g` gallery · `←/→` lightbox nav · `f` favorite ·
`?` shortcuts · `Esc` close.

### 2.7 Visual direction
Images are the content; chrome recedes. Neutral palette with one accent, **light
theme by default** plus dark, rounded tiles, soft shadows, generous spacing. Works at phone width
(single column, composer docked). Smooth but brief transitions; no spinners where a
progress value exists.

---

## 3. Data model (Studio, SQLite; planned)

```
generation
  id            TEXT PK   (ULID)
  batch_id      TEXT      (ULID; one per submission/turn)
  parent_id     TEXT NULL (source generation for Vary/Reuse/Rerun)
  created_at    INTEGER   (unix ms)
  status        TEXT      queued | running | done | failed | cancelled
  prompt        TEXT
  enhanced_prompt TEXT NULL
  params        JSON      {width, height, steps, seed, aspect, quality}
  model         JSON NULL {id, precision}                       -- reported by the manager
  env           JSON NULL {devices, torch, diffusers}           -- reported by the manager
  timings       JSON NULL {enhance_s, encode_s, denoise_s, decode_s, total_s,
                           s_per_step, token stats, peak_vram}  -- §7.4
  error         TEXT NULL
  attempts      INTEGER   default 0
  favorite      INTEGER   default 0
  image_path    TEXT NULL   -- relative to /data/images
  thumb_path    TEXT NULL   -- 512px WebP
```

Seed is **always stored as a concrete number**, even when the user chose Random.

Every PNG carries a `tEXt` chunk `aries-image-studio` with the full JSON record, so files
are self-describing outside the app (written by Studio).

The SPA stores no records itself: each job carries its UI metadata (turn id, aspect
label, size and quality preset) and favorite flag in Studio.

---

## 4. Studio API (browser-facing, `/api`; planned)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/generations` | Submit `{prompt, aspect, count, seed?, steps, enhance}` → `{batch_id, ids[]}` |
| GET | `/api/generations?before=<id>&limit=50` | Timeline page (grouped client-side by `batch_id`) |
| GET | `/api/generations/{id}` | One record |
| PATCH | `/api/generations/{id}` | `{favorite}` |
| DELETE | `/api/generations/{id}` | Delete record + files |
| POST | `/api/generations/{id}/cancel` | Cancel queued or running |
| GET | `/api/gallery?cursor=&favorites=&aspect=&q=` | Done images only |
| GET | `/api/jobs` | Active and pending jobs with stage, progress and ETA; polled by the browser (ARCHITECTURE.md §7) |
| GET | `/api/status` | Generator status snapshot for the status panel |
| GET | `/metrics` | Prometheus (§7.2) |
| GET | `/images/{id}.png`, `/thumbs/{id}.webp` | Files (long cache headers) |

The API the SPA uses today is the job API in ARCHITECTURE_V3.md §8; the table above is the
fuller history API still planned on top of it.

---

## 5. Generation parameters

| Aspect | Size |
|---|---|
| 1:1 | 2048 × 2048 |
| 4:3 / 3:4 | 2400 × 1792 / 1792 × 2400 |
| 3:2 / 2:3 | 2528 × 1696 / 1696 × 2528 |
| 16:9 / 9:16 | 2752 × 1536 / 1536 × 2752 |

Default steps 40. The composer also offers ½ and ¼ size presets (each side scaled and
rounded to a multiple of 32, which Qwen-Image-2.1 requires; Studio rejects other sizes at
submit). No free-form sizes.

---

## 6. Queue & failure semantics

See ARCHITECTURE.md §3 (state machine) and §8 (failure handling). In short: a batch
of N is N jobs; each stage that fails is retried once, then the job fails with the
stage and crash class recorded, so a poison prompt cannot crash-loop the GPU box.
Cancel takes effect immediately for queued jobs and at the next step for running ones.

---

## 7. Telemetry

Two consumers, one source of truth:
- **The UI** needs live, per-job answers: what phase is this image in, how fast is
  it going, is the generator healthy.
- **Prometheus/Grafana** needs history and alerts: is step time drifting, how often
  do workers die, how long do model loads take, where does VRAM go.

The **manager measures** (ARCHITECTURE.md §9). **Hardware** (temperature, fan, card
power, wall watts) stays with host exporters; the manager only reports each card's
PCI address so dashboards can join the two.

### 7.1 Job phases

```
queued → [enhancing] → encoding → denoising (step n/N) → decoding → done | failed | cancelled
```

- Each phase is a stage of ARCHITECTURE.md §2; between stages a job is queued for the
  next one.
- `enhancing` (when the prompt enhancer is on) reports sub-stages: reading the prompt
  (prefill), thinking, rewriting.
- `decoding` is the VAE, ~5 s at 2K. The UI shows **"Finishing"** there instead of
  sitting at 40/40.
- The phase a failure happened in is the single most useful diagnostic: every GPU
  failure seen during bring-up was in decode (HARDWARE.md §2).

### 7.2 Studio `/metrics`

Studio exports its own metrics and re-exports every agent's, labelled by `agent`
(ARCHITECTURE_V3.md §9), so Prometheus scrapes one target. Planned additions:
`eta_error_seconds` (histogram of actual − predicted start, so an ETA model can be
judged) and an executor-state gauge (disabled, lent).

### 7.3 Alerts (planned)

A **sleeping GPU box never pages**: Studio leaves a stale agent's series out instead of
reporting it down, so rules over agent metrics simply stop matching.

- An executor disabled for > 5 min (needs the executor-state gauge above).
- `aries_worker_exits_total{class!="clean"}` increased — with stage and crash class.
- Denoise s/step p50 > 1.5× its 7-day median, sustained 15 min (throttling, fan
  problem, driver regression).
- `aries_tasks_queued` > 0 while `aries_agent_connected` is 0 for > 30 min — work is
  stranded.
- `aries_host_ram_available_bytes` < 4 GiB (swap is off; the OOM killer is next).

### 7.4 Per-generation record

Each generation keeps its own telemetry (§3 `timings`, `model`, `env`): per-stage
times, total, s/step, enhancer token stats, peak VRAM, decode mode. This is what the viewer's details panel shows, and what
quantization or decode-mode comparisons are read from — no separate benchmark
harness needed.

---

## 8. Later

- **Edit / remix.** "Use as reference" puts up to 10 images into the composer;
  submission goes to the same pipeline with `image=[...]`. Lineage via `parent_id`.
- **Fast mode.** Turbo DiT/LoRA as a Quality preset, if a diffusers-format variant
  is available.
- **Wake on submit.** If the GPU box is asleep, Studio asks aries-modeld to wake it.
- **Live preview.** Cheap latent→RGB approximation per step instead of a progress
  ring (full VAE decode per step is far too expensive at 2048²).
