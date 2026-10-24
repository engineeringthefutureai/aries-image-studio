# Aries Image Studio — Product & API Spec

Type a prompt, get pictures. A quiet, image-first app that sits between a chat
timeline and a gallery. No node graphs, no LLM chat. Single user, LAN only.

Inspiration: the *feel* of lightweight Labs-style image tools — minimal chrome,
results as the content, every image one click from "do that again, differently."
Own design; don't copy any product's layout or branding.

**Non-goals:** workflow editing, multi-user accounts, LLM conversation, training.

This document covers the product and UX. How Studio runs jobs — stages, agents, the
protocol, scheduling, failure handling, telemetry — and the browser-facing API are in
**ARCHITECTURE_V3.md**. UI behavior in detail is in **SPEC_UI.md**.

---

## 1. Architecture

```
Browser ──HTTP──▶ Studio (Rust, always on)  ◀──gRPC──  Agent (GPU box, 2× B60)
                  ├─ web app + /api (§4)               ├─ one stage worker per card
                  ├─ jobs, queue, scheduler            ├─ /metrics, scraped by Studio
                  └─ history (SQLite), images,         └─ stores nothing, decides nothing
                     thumbnails (blob store)
```

**History lives in Studio, off the GPU box.** Reasons:
- The GPU box suspends most of the day (~4 W asleep vs ~75 W idle). History, gallery
  and metadata must stay browsable while it sleeps.
- The GPU stack *will* crash (xe GPU faults happen). A crash must cost at most the
  stage in flight, never history.
- Control plane on an always-on host, GPU box as a replaceable agent.

**Status (0.5.0):** Studio is the Rust backend in `studio/` (ARCHITECTURE_V3.md): it owns
jobs, images and provenance in SQLite and a blob store, and each GPU box runs a thin agent.
The web app keeps nothing in browser storage: jobs, images, favorites, turn grouping and
preferences all load from Studio, so every browser shows the same history.

---

## 2. UX

### 2.1 Timeline (default view)
Chat-shaped, newest turn on top, older ones loading as the page scrolls; the composer is
docked at the bottom.

Each **turn** = one submission:
- Prompt text, and the enhanced prompt (with the enhancer's reasoning, collapsed) when Enhance was on
- The reference images, numbered as the prompt names them
- Param chips: `16:9 · seed 48213 · 20 steps · 1m 12s`
- 1–4 image tiles (one per job in the batch), sized to the aspect ratio
- Pending tile: aspect-correct placeholder showing the job's phase (§7.1) — queued
  (and for which stage), enhancing, reading prompt, drawing with percent and ETA,
  finishing — and cancel
- Failed or cancelled tile: short error, "Retry" (the same job again, from its last good
  stage)

### 2.2 Composer
- Multiline prompt. Enter = generate, Shift+Enter = newline
- Chips:
  - **Aspect**: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16 (official sizes, §5)
  - **Count**: 1–4
  - **Seed**: Random (default) or Locked with a value; batches use seed, seed+1, …
  - **Quality**: Draft (12 steps) · Standard (20) · High (40); under Advanced: custom
    steps (1–100), fixed seed, VAE tiling, and rows of different settings sent together
  - **Enhance** toggle: rewrite the prompt with the prompt enhancer before generating.
    With reference images attached it uses the image-aware enhancer (Qwen-Image-2.1-PE-I2I),
    which reads them; without one installed, Enhance is off while references are attached
    (ENHANCER_I2I.md). When Studio offers cloud enhancers (off by default), a picker next
    to it chooses Built-in (the default, the only one that works offline), Claude Sonnet
    5.5 or Gemini 3.5 Flash-Lite: seconds instead of minutes, no card used, and the prompt and
    references go to the provider (CLOUD_ENHANCERS.md)
- **Reference images**: up to 4 (the agent's limit) attached from files (picker, drop
  anywhere, paste) or from earlier results ("Use as reference"), in order: the prompt
  calls them "image 1", "image 2", … (REFERENCE_IMAGES.md)
- Last-used settings (and the theme) are saved in Studio (`/api/preferences`), the same
  in every browser

### 2.3 Gallery
- Masonry grid of all finished images, newest first, infinite scroll (thumbnails)
- Filters: Favorites, aspect, text search over prompts
- Click → image viewer

### 2.4 Image viewer (shared by timeline and gallery; a page at `/images/{id}`)
- Full-resolution image, fit / 1:1 zoom, ←/→ to move through the current list
- Metadata panel: prompt, enhanced prompt, seed, size, steps, per-stage timings with
  the card and model (id and revision) of each, queue & model loading, total, peak
  memory. Precision and software versions stay in the stored provenance and the PNG
- Actions:
  - **Download** (PNG with embedded metadata)
  - **Copy prompt** / **Copy seed**
  - **Rerun** — fills the composer with this prompt + params and the seed locked;
    doubles as a reproducibility check. Original or enhanced prompt is chosen there
  - **Vary** — the same with a random seed. Neither submits by itself
  - **Favorite**, **Delete** (with confirm; removes files too)
  - **Use as reference** — attaches the image to the composer as the next reference

### 2.5 Generator status pill (top bar)
`Ready` · `Busy (…)` · `Error` · `Offline`. Click for the status page (`/status`): each
card's state and current stage, memory per card, host RAM, the queue per stage with the
reason anything waits, and lend / return / disable / enable per card. Jobs submitted while
the GPU box is offline stay queued in Studio and run when its agent connects; a job no
known card can ever run is rejected at submit.

### 2.6 Keyboard
`/` focus composer · `t` timeline · `g` gallery · `←/→` viewer nav · `f` favorite ·
`?` shortcuts · `Esc` leave the viewer or the system view.

### 2.7 Visual direction
Images are the content; chrome recedes. Neutral palette with one accent, **light
theme by default** plus dark, rounded tiles, soft shadows, generous spacing. Works at phone width
(single column, composer docked). Smooth but brief transitions; no spinners where a
progress value exists.

---

## 3. Data model

Studio's SQLite database (ARCHITECTURE_V3.md §3): a **job** per image, with its prompt,
enhanced prompt and the enhancer's reasoning, size, steps, seed, options, reference
images, favorite flag and the UI's own metadata (turn id, aspect label, size and quality
preset); a **task** per stage; an **attempt** per try, with its provenance (agent, card,
model, revision, precision, software versions, timings, peak memory) and, for a failure,
its crash class. Files live in a content-addressed blob store; each finished image has
JPEG thumbnails at 512, 768 and 1024 px.

Seed is **always stored as a concrete number**, even when the user chose Random.

Every PNG carries a `tEXt` chunk `aries-image-studio` with the job's JSON record, so files
are self-describing outside the app (written by Studio). `tEXt` is Latin-1, so the JSON is
plain ASCII: other characters (a Chinese prompt, emoji) are `\uXXXX` escapes.

---

## 4. Studio API (browser-facing)

The job API in ARCHITECTURE_V3.md §8: submit and page jobs (`/api/jobs`, filters for the
gallery), poll changes (`?changed_since=`), images and thumbnails, reference uploads,
preferences, system state and capabilities, operator controls, and `/metrics`. A batch of
N images is N jobs sharing a turn id in their `meta`.

---

## 5. Generation parameters

| Aspect | Size |
|---|---|
| 1:1 | 2048 × 2048 |
| 4:3 / 3:4 | 2400 × 1792 / 1792 × 2400 |
| 3:2 / 2:3 | 2528 × 1696 / 1696 × 2528 |
| 16:9 / 9:16 | 2752 × 1536 / 1536 × 2752 |

Default steps 20 (Standard). The composer also offers ½ and ¼ size presets (each side scaled and
rounded to a multiple of 32, which Qwen-Image-2.1 requires; Studio rejects other sizes at
submit). No free-form sizes.

---

## 6. Queue & failure semantics

See ARCHITECTURE_V3.md §4 (state machine) and §7.3 (disconnects). In short: a batch
of N is N jobs; each stage that fails is retried once, then the job fails with the
stage and crash class recorded, so a poison prompt cannot crash-loop the GPU box.
Cancel takes effect immediately for queued jobs; a running stage's worker is stopped.

---

## 7. Telemetry

Two consumers, one source of truth:
- **The UI** needs live, per-job answers: what phase is this image in, how fast is
  it going, is the generator healthy.
- **Prometheus/Grafana** needs history and alerts: is step time drifting, how often
  do workers die, how long do model loads take, where does VRAM go.

The **agent measures** and Studio collects (ARCHITECTURE_V3.md §9). **Hardware**
(temperature, fan, card power, wall watts) stays with host exporters; the agent only
reports each card's PCI address so dashboards can join the two.

### 7.1 Job phases

```
queued → [enhancing] → encoding → denoising (step n/N) → decoding → done | failed | cancelled
```

- Each phase is a stage (ARCHITECTURE_V3.md §3.1); between stages a job is queued for
  the next one.
- `enhancing` (when the prompt enhancer is on) reports sub-stages: reading the prompt
  (prefill), thinking, rewriting.
- `decoding` is the VAE, ~5 s at 2K. The UI shows **"Finishing"** there instead of
  sitting at the last step.
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
- `aries_tasks_queued` > 0 while `aries_agent_connected` is 0 for > 30 min — work is
  stranded.
- `aries_host_ram_available_bytes` < 4 GiB (swap is off; the OOM killer is next).

### 7.4 Per-generation record

Each job keeps its own telemetry (its attempts' provenance, §3): per-stage times,
total, s/step, enhancer token stats, peak memory, decode mode. This is what the viewer's details panel shows, and what
quantization or decode-mode comparisons are read from — no separate benchmark
harness needed.

---

## 8. Later

- **Lineage.** Which image a Rerun or Vary came from: "↳ variation of …" on the turn,
  linking back to the source (SPEC_UI §4.1).
- **Fast mode.** Turbo DiT/LoRA as a Quality preset, if a diffusers-format variant
  is available.
- **Wake on submit.** If the GPU box is asleep, Studio wakes it (Wake-on-LAN or a small
  helper on the box).
- **Live preview.** Cheap latent→RGB approximation per step instead of a progress
  ring (full VAE decode per step is far too expensive at 2048²). It need not slow
  generation: the denoised estimate and a fixed 64→3 colour projection on the 1/16-size
  latent take milliseconds against seconds per step. It needs that projection fitted
  for this VAE and a way to send a small frame every few steps through the agent to the
  browser.
