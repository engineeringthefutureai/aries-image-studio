# Image-aware prompt enhancer (PE-I2I) — design and build plan

Status: **shipped in 0.4.0** (§4 is the build plan it followed), measured on the
hardware (§2). Builds on
[REFERENCE_IMAGES.md](REFERENCE_IMAGES.md) (references, `ref<i>.image` inputs) and the
enhance stage of [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md) §3.1.

## 1. Goal

Before this, Enhance was off while reference images were attached: the enhancer
(Qwen-Image-2.1-PE-T2I) reads text only and could rewrite away what an edit asks for.
Qwen-Image-2.1-PE-I2I is the image-editing counterpart from the same release: it reads
the instruction **and the images** (`<image1>`, `<image2>`, …) and writes a precise edit
prompt. With it, Enhance works on jobs with references.

**One stage, two models.** PE-I2I is not a new stage: it sits where enhance sits, writes
the same `enhanced.json`, and shows the same progress. The agent declares both enhancer
models on its `enhance` capability; Studio picks I2I for a job with references and T2I
for one without; a worker loads one model, and picks the T2I or I2I code path at load
from the checkpoint (I2I has a `vision_config`).

## 2. Measurements (2026-10-17, one B60, agent image of 0.4-dev)

Five cases on images from Studio (refs resized to ~1 MP, like the text encoder sees them),
recommended sampling (temperature 1.0, top_p 0.95, top_k 20, thinking on), limit 16 000.
Scripts and raw results: local `prototypes/pe-i2i/` (`bench.py`, `run.sh`, `results-*.jsonl`).

| Case (images) | Prompt tokens | int4: tokens · tok/s · time · peak | bf16: tokens · tok/s · time · peak | Aspect pick |
|---|---|---|---|---|
| hat (1) | 5 534 | 3 480 · 31.2 · 117 s · 9.6 GiB | 3 514 · 19.1 · 189 s · 19.1 GiB | `ratio_follow <image1>` |
| chibi (1) | 5 564 | 3 060 · 31.1 · 101 s · 9.6 GiB | 2 240 · 19.3 · 118 s · 19.1 GiB | `<image1>` |
| text (1) | 5 569 | 2 922 · 31.2 · 96 s · 9.6 GiB | 1 981 · 19.4 · 105 s · 19.1 GiB | `<image1>` |
| two (2) | 6 547 | 3 182 · 31.0 · 106 s · 9.9 GiB | 2 245 · 19.1 · 120 s · 19.4 GiB | `<image2>` (the scene) |
| four (4) | 8 699 | 3 994 · 30.6 · 135 s · 10.6 GiB | 4 118 · 18.5 · 227 s · 20.1 GiB | `wh_ratio 2:3` |

- int4 weights 8.0 GiB on the card (bf16 17.5 GiB), load 8 s; prefill 2.4–5.2 s.
- The system prompt (`system_prompt.txt`, 18 KB) is ~4 500 of the prompt tokens; each
  ~1 MP image adds ~1 000.
- All 10 runs: valid JSON, reasoning closed with `</think>`, never near the limit. Both
  precisions describe the same content (current headline text and its typeface, the
  goblin's ears/nose/hat to keep, the same poster layout); single samples at
  temperature 1.0, so token counts vary run to run.
- **Use int4**: 1.6× faster at half the memory, no visible quality loss.
- Output JSON: `rewritten_prompt`, and exactly one of `wh_ratio` (new composition) or
  `ratio_follow` (`"<imageN>"`: keep that input's aspect).
- **Token limit**: runs used 2 900–4 100 tokens; range 256–24576. The first default (4096
  for both) was too low live, see §2.1.

### 2.1 First live use: reasoning ran out of room

Four images from one send (3 references: a photo, a chibi Batman, a battle scene; "turn me
into an anime character and make me fight against batman in the setting of the mages",
7,581 prompt tokens) all hit 4096 tokens still reasoning (~18 KB each, deliberating
over the output format). Three never wrote a reply and one got 28 tokens of it. The parser
then took the reasoning itself as the prompt, or a `"..."` placeholder from where the
reasoning drafted the JSON format, or the 28-token fragment. Fixes:

- **Default limit 4096 + 2048 per reference image** (worker; the composer shows the same
  default and sends a limit only when the user sets one).
- **Out of room → double and go on:** when reasoning reaches the limit minus a reply reserve
  (a quarter of the limit, at most 1,536), generation continues from where it stopped with
  the limit doubled (the same as rerunning with twice the limit, minus redoing the
  reasoning; the sequence is prefilled again, a few seconds), up to 24576 (the model card generates with 24000). At 24576 the
  reasoning is closed with `</think>` and the model replies in the reserve.
- **Never a prompt from reasoning:** text without `</think>` that is not a bare JSON reply
  leaves the original prompt in place.
- Timings record `limit_raised_from` and `thinking_cut`; the viewer shows them.

### The int4 checkpoint

`~/models/Qwen-Image-2.1-PE-I2I-int4-xpu` (8.6 GB): language model linears int4
(torchao `Int4WeightOnlyConfig(group_size=128, int4_packing_format="plain_int32")`, 248
layers), vision tower and `lm_head` bf16 (`modules_to_not_convert=["model.visual",
"lm_head"]`), with processor, tokenizer, chat template and `system_prompt.txt`. Made by
`prototypes/pe-i2i/quantize.py`; **the T2I recipe (`AutoModelForCausalLM`) is wrong for
I2I**: it saves `Qwen3_5ForCausalLM` without `model.visual` (no eyes) and drops the
processor and system prompt.

## 3. Design

| Piece | Change |
|---|---|
| Protocol | `EnhanceParams.references` (repeated `Reference`, field 3): pixels arrive as inputs `ref<i>.image`, like encode. `StageCapability.reference_models` (repeated string, field 6): which of `models` take reference images; for enhance, a job with references needs one of them, a job without prefers one that is not. `max_references` on the enhance capability. `artifacts.json`: `enhanced.json` gains optional `ratio_follow`. |
| Agent | `ARIES_ENHANCER_I2I_MODEL_PATH` (unset: no I2I), `ENHANCER_I2I_MODEL_ID` (default `Qwen/Qwen-Image-2.1-PE-I2I`); dtype detected like T2I; declared on `enhance` with `reference_models=[I2I id]`, `max_references=ARIES_MAX_REFERENCE_IMAGES`. Passes `references` to the worker. |
| Worker | `EnhanceStageRunner` picks the variant at load: I2I if the checkpoint's `config.json` has `vision_config`. I2I: `AutoProcessor` + `AutoModelForImageTextToText`, images (upright, ~1 MP via `references.reference_size`) in the user message before the text, `enable_thinking=True`, sampling 1.0 / 0.95 / top_k 20. Parse `ratio_follow`. Default limit 4096 + 2048 per reference, doubled while reasoning runs out (§2.1). |
| Studio | Enhance with references allowed when an executor's enhance capability takes that many (else 422 as today). Model choice by `reference_models`; resident-worker feeding (Rule 1) must match the model, not just the stage. Enhance assignment sends `ref<i>.image` inputs and `references`. `ratio_follow` stored on the job (new column), not used yet ("Auto" aspect stays a REFERENCE_IMAGES.md §8 question). `/api/capabilities` gains `enhance_max_references_now/_ever`. Token range 256–24576. |
| Web | Enhance stays enabled with references when `enhance_max_references_ever >= references.length`; otherwise disabled as now. Token field default 4096 + 2048 per reference, max 24576 (§2.1). |
| Deploy | compose: mount and env for the I2I model; BUILD_AND_RUN: download + the I2I quantize command. |

Not changed: the enhance request text (prompt plus `Aspect ratio: <preset>`, since the
job's size is fixed by the composer), encode, denoise, decode.

## 4. Build plan

Work on `main` in the tree as it is after `f1eb0e0`. After each step: the step's tests.
At the end: all checks (§5) and docs.

### Step 1 — protocol and contract
1. `proto/aries/agent/v1/agent.proto`:
   - `EnhanceParams { string prompt = 1; uint32 max_new_tokens = 2; repeated Reference references = 3; }`
     (comment: pixels as `ref<i>.image` inputs).
   - `StageCapability`: `repeated string reference_models = 6;` with a comment (empty: every
     model in `models` takes references up to `max_references`).
   - Update the `max_new_tokens` comment: 0 = worker default (4096).
2. `artifacts.json`: `enhanced.json.optional_fields` += `"ratio_follow"`.
3. Regenerate Python stubs (needs `grpcio-tools` in the venv):
   `python -m grpc_tools.protoc -I proto --python_out=backend --grpc_python_out=backend proto/aries/agent/v1/agent.proto`
   (gencode header must stay `Protobuf Python Version: 7.35.1`).
4. Rust test harness (`studio/tests/v3_backend_test.rs`): every `proto::StageCapability {…}`
   needs `reference_models: vec![]`; `proto::EnhanceParams` literals none (only built in
   scheduler.rs).

### Step 2 — worker (`backend/aries_worker/stages/enhance.py`)
1. `MAX_NEW_TOKENS = 4096` (update its comment: default; jobs override).
2. `EnhanceStageRunner.load()`: read `config.json` of `resolve_enhancer_model_path()`; if it
   has `vision_config` → `self.vision = True`, `self.processor = AutoProcessor.from_pretrained(path)`,
   `self.tokenizer = self.processor.tokenizer`, `self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map=get_device())`.
   Else the current T2I path. Log which.
3. `run_job()`:
   - `ref_shas = params.get("references") or []`; images from `paths[f"ref{i}.image"]` via
     `references.open_reference` (upright RGBA → convert RGB) resized to
     `reference_size(w, h, 1024)` (LANCZOS).
   - I2I: messages `[{"role":"system","content":[{"type":"text","text":system}]},{"role":"user","content":[*({"type":"image","image":im} for im in imgs), {"type":"text","text":prompt}]}]`;
     `inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=True).to(device)`;
     generate with `temperature=1.0, top_p=0.95, top_k=20, do_sample=True`.
     A job without references on an I2I worker (should not happen) still works: no images.
   - T2I: unchanged (temperature 0.7, top_p 0.95, repetition_penalty 1.05).
   - System prompt: `load_system_prompt()` already prefers the model directory's
     `system_prompt.txt` (the I2I checkpoint has its own).
   - Streamer: unchanged (`EnhanceProgressStreamer`, `tokenizer.decode`).
   - `_parse_enhanced_json`: also return `ratio_follow` (string, `""` if absent; accept
     `"<imageN>"` only, else `""`); write it to `enhanced.json` when non-empty. Keep
     `wh_ratio` validation against `VALID_RATIOS` as is.
   - Timings: add `references: n` when n > 0.
4. `tests/_ml_stubs.py`: `transformers` is already a MagicMock; `AutoProcessor` and
   `AutoModelForImageTextToText` come from it.
5. Tests (`backend/tests/test_stages.py`, `test_contract.py`):
   - I2I run: runner with `vision=True`, mock processor; two ref PNGs in tmp; assert
     `apply_chat_template` got 2 image parts then the text and `enable_thinking=True`;
     generate kwargs `temperature == 1.0`, `top_k == 20`, `max_new_tokens` from params;
     `enhanced.json` has `ratio_follow` from a reply like
     `'plan</think>{"rewritten_prompt": "x", "wh_ratio": "", "ratio_follow": "<image2>"}'`.
   - Load picks the variant: tmp model dir with/without `vision_config` in `config.json`
     (patch the transformers classes).
   - Default limit is 4096 (update `test_enhance_progress_streamer` expectations that use
     `MAX_NEW_TOKENS` — they import the constant, so only fixed numbers need care).
   - Contract: enhanced.json written by I2I has only contract fields.

### Step 3 — agent (`backend/aries_agent/`)
1. `config.py` `from_env()`: if `ARIES_ENHANCER_I2I_MODEL_PATH` is set, append
   `ModelConfig(id=os.environ.get("ENHANCER_I2I_MODEL_ID", "Qwen/Qwen-Image-2.1-PE-I2I"), rev=_read_model_rev(path, "ENHANCER_I2I_MODEL_REV"), dtype=os.environ.get("ARIES_ENHANCER_I2I_MODEL_DTYPE") or _detect_quant_dtype(path) or dtype, path=path)`.
   Add `enhancer_i2i_model_id: Optional[str]` to `AgentConfig` (None when unset) so the
   agent knows which model takes references.
2. `agent.py` `build_register_message()`: the enhance `StageCapability` gets
   `reference_models=[i2i id]` and `max_references=self.config.max_reference_images` when
   I2I is configured (else neither). `pe_models` already includes every id with "PE".
3. `start_worker()`: the env loop sets `ENHANCER_MODEL_PATH` from each "PE" model (last
   wins) but `cmd.model` then overrides with the right one — keep, and add a test that a
   StartWorker for the I2I id sets `ENHANCER_MODEL_PATH` to the I2I path and
   `MODEL_DTYPE` to its dtype.
4. `_build_worker_job_payload()`: enhance → `params["references"] = [r.sha256 for r in assign.enhance.references]` when any.
5. `validate_assignment()`: enhance with more references than `max_reference_images` →
   reject (as encode/denoise).
6. Tests (`tests/test_agent.py`): register shape with and without I2I env (use
   `_make_test_config` + a third ModelConfig and `enhancer_i2i_model_id`); payload; env
   for StartWorker; validation.

### Step 4 — Studio (`studio/src/`)
1. `db.rs`: `StageCapDecl.reference_models: Vec<String>` (`#[serde(default)]`); jobs column
   `ratio_follow TEXT` (`ensure_column`), `JobRecord.ratio_follow: Option<String>`;
   `complete_task_attempt`'s enhanced tuple gains ratio_follow (or a separate UPDATE).
2. `grpc.rs` registration: copy `sc.reference_models`. `handle_done_message` enhance:
   `EnhancedJson` (storage.rs) gets `#[serde(default)] ratio_follow: String`; store it.
3. `scheduler.rs`:
   - `ENHANCE_TOKENS_RANGE = 256..=16384` (update the doc comment: 16384 ≈ 9 min at 31 tok/s).
   - `executor_supports_stage_static`: for a stage whose capability has non-empty
     `reference_models`: refs > 0 → only models in it (and `refs <= max_references`);
     refs == 0 → prefer models not in it, fall back to any. The existing encode/denoise
     `max_references` check stays. Enhance with refs and no `reference_models` → None.
   - `check_submit_feasibility`: drop the blanket "enhance is not available with reference
     images"; instead, with refs and enhance, require an executor whose enhance capability
     takes `refs` (error: "no known executor can enhance with N reference images").
   - Rule 1 (feed `ready` workers): for enhance, the model `executor_supports_stage_static`
     returns must equal `ex.current_model_id` (a T2I worker must not get an I2I task and
     vice versa); same in the switch logic (Rules 2/3) — a ready T2I enhance worker with an
     I2I task waiting must switch (StartWorker with the I2I model).
   - `dispatch_task_to_executor` enhance: `inputs` += `ref<i>.image` (blob URLs, as encode),
     `EnhanceParams.references`.
   - `compute_capabilities`: `enhance_max_references_now/_ever` (max over executors of the
     enhance capability's `max_references` when `reference_models` is non-empty).
4. `api.rs`: nothing new beyond the job field (serialized automatically).
5. Tests (`studio/tests/v3_backend_test.rs`): register helper with an I2I model on xpu0
   enhance (`models: [T2I, I2I]`, `reference_models: [I2I]`, `max_references: 4`);
   - enhance + refs submits (201) with it, 422 without it;
   - a ref job's enhance StartWorker names the I2I model and its Assign has `ref0.image` and
     `references`; a text job's names T2I;
   - a ready T2I enhance worker is not fed an I2I task;
   - `enhanced.json` with `ratio_follow` stored on the job;
   - tokens 16384 accepted, 16385 refused.
   Contract test (`contract_test.rs`): enhanced.json optional fields include `ratio_follow`.

### Step 5 — web (`web/src/`)
1. `types.ts`: `ENHANCE_TOKENS_DEFAULT = 4096`, `MAX_ENHANCE_TOKENS = 16384`;
   `CapabilitiesInfo.enhance_max_references_now/_ever?`; `V3JobRecord.ratio_follow?`.
2. `services/preferences.ts`: `enhanceMaxTokens: 4096`.
3. `Composer.tsx`: `enhanceRefsOk = references.length <= (capabilities?.enhance_max_references_ever ?? 0)`;
   with references, Enhance (main and rows) is disabled only when `!enhanceRefsOk`
   (title explains), and `handleGenerate` forces `enhance:false` only then; `anyEnhance`
   shows the token field with references when allowed. Update `ENHANCE_OFF_TITLE`.
   The NumberField width may need `w-12` for 5 digits.
4. Lightbox/Timeline: nothing required (enhanced prompt shows as today).
5. Browser check with the fake agent (`prototypes/e2e/fake_agent.py`): declare
   `PE-I2I` on enhance with `reference_models` and `max_references=4`; its enhance handler
   should assert `ref<i>.image` inputs arrive and print `references`. Check: attach 1 image
   + Enhance → POST has `enhance:true` and references; the job's enhanced prompt shows;
   without I2I declared, Enhance stays disabled with references.

### Step 6 — deploy and docs
1. `deploy/aries/compose.yaml`: env `ARIES_ENHANCER_I2I_MODEL_PATH=/models/qwen-image-2.1-pe-i2i`
   and volume `${MODELS_DIR:-/srv/models}/qwen-image-2.1-pe-i2i:/models/qwen-image-2.1-pe-i2i:ro`.
   The user's live deploy has its own compose/.env: tell them which lines to add (their
   int4 copy is `~/models/Qwen-Image-2.1-PE-I2I-int4-xpu`); never edit `deploy/aries/.env`.
2. `docs/BUILD_AND_RUN.md` §3: download `Qwen/Qwen-Image-2.1-PE-I2I`, the I2I quantize
   command (from `prototypes/pe-i2i/quantize.py`, inline like the T2I one, `docker run -i`),
   env table rows `ARIES_ENHANCER_I2I_MODEL_PATH`, `ARIES_ENHANCER_I2I_MODEL_DTYPE`.
3. `docs/HARDWARE.md`: §1.2 PE-I2I measurements (the table of §2, short).
4. `docs/SPEC.md` §2.2 (Enhance with references uses PE-I2I when installed),
   `docs/SPEC_UI.md` (Enhance with references; token field 256–24576, default 4096),
   `docs/ARCHITECTURE_V3.md` §3.1 (enhance input: references), §5.1 (example), §8
   (`enhance_max_tokens` range, capabilities fields), `docs/REFERENCE_IMAGES.md` §7
   ("Enhance is off with references" → only without an I2I model; §1 non-goal removed),
   this doc's status → implemented, `AGENTS.md` stack line if needed, `README.md`
   features, `CHANGELOG.md` Unreleased.

### Step 7 — real-model check (GPU; agent cards idle first)
1. `curl -s localhost:9464/metrics | grep aries_executor_state` → both idle.
2. In `prototypes/pe-i2i/`, a small script that builds `EnhanceStageRunner` with
   `ENHANCER_MODEL_PATH=/models/Qwen-Image-2.1-PE-I2I-int4-xpu`, writes ref images as
   `ref0.image`…, calls `run_job` for the "two" case with `max_new_tokens 4096`, prints the
   timings and `enhanced.json`; run through `run.sh` (host-RAM watchdog, card 1). Also a
   T2I run (`/models/Qwen-Image-2.1-PE-T2I-int4-xpu`) to prove the old path still loads.
3. Report to the user; then "push dirty" only when asked.

## 5. Checks

- `cd studio && cargo test`, `cd backend && pytest`, `cd web && npm run build && npm run lint`
  (AGENTS.md).
- Browser check: Studio with a fake agent that declares PE-I2I on enhance
  (`reference_models`, `max_references: 4`): one attached image plus Enhance sends
  `enhance: true` with the references, the enhance assignment carries `ref0.image`, and the
  job stores `ratio_follow`; without PE-I2I declared Enhance stays disabled with references.
- Real models, one idle card: `EnhanceStageRunner` from the repo on the int4 checkpoints
  (2026-10-17): PE-I2I with two references (goblin into the café scene) 108 s, 6,555 prompt
  tokens, 3,378 generated at 32.9 tok/s, 9.9 GiB peak, a precise edit prompt with the
  scene's details kept; PE-T2I on the same code path 47 s, 1,461 tokens, 7.9 GiB.
