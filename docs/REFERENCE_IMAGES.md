# Reference images (image attachments) — design

Status: **implemented** for 0.4 (unreleased): worker, agent, Studio and web, with tests on
each side; the hardware run of §9 step 5 is still to do. The prototype (four stage
scripts plus a runner) is kept locally, outside the repository. Measurements:
[HARDWARE.md §1.1](HARDWARE.md#11-reference-images-prototype-2026-10-16).
Builds on [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md); where this document and that one
disagree, this one describes the intended change.

## 1. Goal

Attach up to N images to a prompt (N = 4 by default, configurable) so Qwen-Image-2.1 edits
or composes from them: "make this a winter night", "put these characters in one scene",
"this product on a beach". A reference is VAE-encoded **once** and stored, so any number of
later generations from it skip the VAE until decode.

**Non-goals for the first version:** masks and drawn annotations for local edits, RGBA /
transparent output, more than 4 references at full resolution (§8). (The image-aware
prompt enhancer came later: [ENHANCER_I2I.md](ENHANCER_I2I.md).)

## 2. How the model uses a reference

Qwen-Image-2.1 accepts up to 10 condition images (model card). Each one is resized to
about 1 MP (1024² area, sides multiples of 32) and then used **twice**:

1. **VAE encode → image latents.** 64 channels at 1/16 of the size: ~4,100 tokens,
   ~0.5 MB. These are placed in front of the noise, and the transformer attends to them.
2. **Text encoder (Qwen3-VL) reads the original pixels together with the prompt** → prompt
   embeddings with ~1,000 image tokens per reference, plus `image_pad_mask` marking which
   embedding positions are image tokens.

Step 1 depends only on the image. Step 2 depends on the prompt and all references
together, in order.

## 3. Pipeline with references

| Step | Stage (process) | Reads | Writes | Per reference |
|---|---|---|---|---|
| 0 | `enhance` (optional) | prompt (+ the references' pixels for PE-I2I) | `enhanced.json` | — (with references only on PE-I2I, §7) |
| 1 | `encode`, VAE, **only for references without stored latents** | reference pixels | `reference.safetensors` per reference | 0.3–1.4 s, 2.7 GiB peak; ~0.5 MB |
| 2 | `encode`, text encoder | prompt + reference pixels | `embeds.safetensors` (+ `image_pad_mask`, reference sizes) | +0.85 GiB, ~8 MB, ~1,000 tokens |
| 3 | `denoise`, transformer | embeds + mask + stored reference latents (no pixels, no VAE) | `latent.safetensors` | +2.6 GiB (conditioning KV cache) |
| 4 | `decode`, VAE | latent | `image.png` | unchanged |

The encode stage loads the VAE (0.6 GB, next to the 16 GB text encoder) only when a job
brings a reference Studio has no latents for. The denoise and decode stages never see the
reference pixels.

### 3.1 Our denoise function

The pipeline's `__call__` cannot take precomputed condition latents or `image_pad_mask`:
with `prompt_embeds` plus `image` it raises, and it always VAE-encodes from pixels. The
denoise worker therefore runs **its own loop** (`DenoiseStageRunner._denoise_with_references`,
~60 lines adapted from `QwenImage21Pipeline.__call__`, as in the prototype):

- noise from `torch.Generator("cpu").manual_seed(seed)`, packed like the pipeline;
- `img_shapes` = each reference's latent size, then the target's;
- `img_mask` = `image_pad_mask` plus one slot per 2×2 group of target latents;
- timesteps from the pipeline's `sample_sigmas` and `calculate_shift`;
- the conditioning KV cache (`QwenImage21KVCache`, `extract` on step 0, `cached` after);
- each step: `[reference latents, latents]` → transformer → keep the target slice →
  scheduler step.

It uses diffusers helpers (scheduler, transformer, `_pack_latents`, `calculate_shift`,
`retrieve_timesteps`), so the agent image **pins diffusers** to the tested commit
(`b800f828f3951c4d206e154411bcbfa693cd322e`, 0.41.0.dev0) instead of installing git main.
Text-only jobs keep the pipeline's `__call__` at first; moving them onto the same function
is a later cleanup, after a parity check (same seed, same embeds → same latent).

## 4. Storage, sharing and caching

| Artifact | Key | Lifetime | Reused by |
|---|---|---|---|
| Reference image (PNG/JPEG/WebP as uploaded) | sha256 of the bytes | while any job or composer draft refers to it; unreferenced uploads after 24 h | every job that attaches it |
| Reference latents (`reference.safetensors`) | image sha256 + VAE model@rev + dtype + reference resolution | with the image | **any** later job with that image, any prompt |
| Embeds (`embeds.safetensors`) | effective prompt + its source + reference shas in order + reference resolution + negative prompt / `true_cfg_scale` (`encode_cache_key` in `scheduler.rs`); the model revision is checked on reuse | until its job is done (intermediate artifacts are deleted then) | every job in a send with the same prompt; not later reruns |

- **One encode per send, not per image.** Encode output does not depend on seed, steps,
  size or tiling. Each job keeps its own encode task, but the scheduler runs only the
  first of those with one key: the others wait while it is queued or running, then take
  its embeds without running (a new attempt with the source's provenance, timings and
  peak memory, timings marked `"shared": 1`, and an artifact row pointing at the same
  blob). The viewer shows the copied details with "(reused)". This also speeds up
  text-only sends. Rows with Enhance on get their own rewritten prompt, so their own
  encode.
- **Twenty generations from one reference** cost one VAE encode, one text encode and 20
  denoise + decode.
- Precomputing on upload is not worth it: the text encode needs the prompt, and step 1 is
  about a second, cheaper than waking a card for it. Step 1 runs inside the first encode
  that needs it.

## 5. Limits and configuration

Memory per job grows linearly with references (HARDWARE.md §1.1): 4 references spill
0.2–2 GiB into host RAM, 10 at full size hung the host (HARDWARE.md F9).

| Setting | Where | Default | Meaning |
|---|---|---|---|
| `ARIES_MAX_REFERENCE_IMAGES` | agent | `4` | Most references a job may carry on this agent; declared to Studio |
| `ARIES_DENOISE_REFS_LIMIT_ABOVE` | agent | `2` | Denoise with more references than this needs the host-RAM floor and shares the one-at-a-time slot with full-size decodes on the agent |
| `ARIES_REFERENCE_RES` | Studio | `1024` | Side of the area references are resized to (1024 → ~1 MP, ~4,100 tokens each). Part of the latents key, so Studio owns it and sends it with each encode |

- The agent declares `max_references` on its encode and denoise capabilities and a
  `Limit` for denoise keyed on reference count (proto change, §6). Studio refuses at
  submit (`422`) a job with more references than any executor declares, and the composer
  stops accepting more at `GET /api/capabilities`' `max_references`.
- With 4 references at ½ size a step takes ~4× longer than with 3 (small spill); setting
  `ARIES_MAX_REFERENCE_IMAGES=3` avoids spilling entirely.

## 6. Changes by component

### Protocol (`proto/aries/agent/v1/`)
- `StageCapability.max_references` (encode, denoise).
- `Limit`: an `above_references` condition next to `above_px` (and, for decode, a
  `decode_mode` condition: the full-size limit covers untiled decodes only).
- `EncodeParams.references` (each a `Reference {sha256}`, in order) and `reference_res`.
  The pixels arrive as inputs `ref<i>.image`; latents Studio lacks are uploads
  `ref<i>.safetensors`. The worker computes the resized sizes itself and records them.
- `DenoiseParams.references`: the count; the latents arrive as inputs `ref<i>.safetensors`,
  their sizes in their own metadata (checked against the embeds' `reference_sizes`).
- `artifacts.json`: new `ref<i>.safetensors` (tensor `latents`; metadata `width`,
  `height`, `reference_res`, `source_sha256`, `model`, `rev`, `dtype`); `embeds.safetensors`
  requires `image_pad_mask` and a `reference_sizes` metadata field when references are
  present. Contract tests on both sides change with it.

### Studio (`studio/`)
- `POST /api/uploads` (image body, up to 32 MB; PNG, JPEG or WebP) → `{id, sha256, width,
  height, mime}` (the id is the sha256; sizes upright by EXIF); `GET /api/uploads/{id}` and
  a thumbnail. Stored in the blob store by sha256. Agents fetch images and latents with
  `GET /agent/v1/blobs/{sha256}`.
- `POST /api/jobs` gains `references: [{upload: id} | {image: job_id}]` (order matters:
  "image 1", "image 2" in the prompt); an earlier result becomes an upload of its PNG, so
  every reference is stored the same way. `422` when there are too many, when `enhance`
  is set and no executor has an image-aware enhancer, when `true_cfg_scale > 1` is set, when an upload is unknown, or when no executor declares
  references.
- Encode sharing (§4): an encode keyed by its inputs, linked to every job that needs it.
- Reference latents cache: looked up by key at encode assignment; missing ones become
  uploads of that encode task.
- Scheduler: the reference-count limit (§5) alongside the decode limit.
- Provenance and the PNG `tEXt` chunk list each reference's sha256 and resized size.

### Agent and worker (`backend/`)
- Agent: declare `max_references` and the denoise limit from the env above; pass the new
  params through. No scheduling logic.
- `encode` worker: load the VAE only when the assignment asks for reference latents; write
  `reference.safetensors` per missing reference (step 1) and `embeds.safetensors` with the
  mask (step 2).
- `denoise` worker: the loop of §3.1 when references are present.
- Dockerfile: pin diffusers to the tested commit.

### Web (`web/`)
- Composer: attach button, drag and drop, paste; a strip of thumbnails that can be
  removed and reordered; the count shows in the folded pill. Enhance is disabled while
  references are attached, unless an image-aware enhancer is installed. Uploads start as soon as an image is added.
- "Use as reference" on any image (viewer and timeline).
- Timeline turn and viewer details show the references; Rerun and Vary carry them.
- Rows and seeds work as today: every image of a send uses the same references.

### Docs
SPEC.md §2.2/§2.4 and §8 (edit/remix moves from Later), SPEC_UI.md composer and turn,
ARCHITECTURE_V3.md §3.1, §5, §7.4 and §8, BUILD_AND_RUN.md settings.

## 7. Decisions taken

- **VAE encode of a reference happens once and is stored** (step 1 separate from denoise),
  so many generations reuse it.
- **Encode once per send** (and per distinct prompt).
- **Our own denoise loop** rather than wrapping the pipeline; diffusers pinned.
- **Default limit 4 references**, configurable per agent.
- **Enhance with references needs PE-I2I:** the text enhancer (PE-T2I) cannot see images
  and could rewrite away what an edit asks for, so without the image-aware enhancer
  ([ENHANCER_I2I.md](ENHANCER_I2I.md)) Enhance is off with references.
- **No guidance with references:** `true_cfg_scale > 1` is refused with references (the
  UI never sets it); supporting it means a negative encode with the same images and a
  second KV cache in our loop.
- **Attachments belong to one send,** like the prompt: sending clears them; Rerun and
  Vary bring an image's references back.

## 8. Open questions

- **Output size with references:** keep the aspect/size picker (current plan), or add an
  "Auto" aspect that follows the last reference, as the pipeline does by default.
- **More than 4 references:** a smaller reference resolution (512 → ~1,000 tokens each)
  would let 10 cost about what 3 do now; needs a quality check before offering it.
- **The ½-size spill slowdown** (4× per step for a 0.2 GiB spill): understand it, or keep
  the default at 3.
- **Text-only jobs on the same loop** after a parity check.

## 9. Plan

1. ✓ Worker: pin diffusers; encode with references and step-1 latents; the denoise loop;
   `artifacts.json` and contract tests.
2. ✓ Protocol and agent: capabilities, limit, params; agent tests.
3. ✓ Studio: uploads, `references` on jobs, latents cache, encode sharing, scheduler limit,
   provenance; Rust tests.
4. ✓ Web: attach, strip, "Use as reference", turn and viewer display (checked in a
   browser against Studio and a fake agent).
5. On the hardware, under the agent's host-RAM guard: 1, 3 and 4 references at ¼, ½ and
   full size, 20 steps, checking images and timings against HARDWARE.md §1.1.
6. ✓ Docs listed in §6.
