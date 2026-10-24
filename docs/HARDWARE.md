# Hardware, setup and failure modes

Measurements and lessons from bringing Qwen-Image-2.1 up on 2× Intel Arc Pro B60
(24 GB each, Linux `xe` driver). They explain why the backend runs one model per card
in its own process (ARCHITECTURE.md §0). `reference/` holds the batch scripts that
produced these numbers.

---

## 1. Measured facts

| Item | Value |
|---|---|
| Usable VRAM per B60 | 23.9 GiB (24,480 MiB) |
| Text encoder (Qwen3-VL 8B, bf16) | ~16.3 GB |
| Transformer (7B DiT, bf16) | ~13.3 GB; ~16 GB peak while denoising |
| VAE weights | ~0.6 GB |
| Prompt enhancer (Qwen-Image-2.1-PE-T2I, bf16) | ~18 GB |
| Prompt enhancer, torchao int4 (group 128) | 7.9 GiB peak vs 17.4 GiB bf16 (enhance worker, 2.4k-token system prompt); generation 33 tok/s vs 20 tok/s bf16, prefill ~800–1070 tok/s for both |
| Image enhancer (Qwen-Image-2.1-PE-I2I), int4 / bf16 | weights 8.0 / 17.5 GiB; 9.6–10.6 / 19.1–20.1 GiB peak with 1–4 references; 31 / 19 tok/s (section 1.2) |
| **Untiled decode peak at 2048²** | **25.76 GiB**: more than one card; torch counts ~2 GiB over, but the driver then holds **13–17 GB of host RAM** (`GPUActive`) |
| Split decode (§1.3), full-size presets | 11.6–12.5 GiB on the card, 0.09 GB of host RAM, ~5 s; pixels within 1/255 of untiled |
| Untiled decode peak at ½ size (~1 MP) | under 8 GiB, VAE weights included: fits a card with room to spare |
| Largest single tensor in the decode | 2.25 GiB (288 ch × 2048² × bf16) |
| Denoise speed, 2048² | 6.4–6.5 s/step |
| Decode time, 2048² | 5.2 s untiled (VAE alone on its card) · 2.9 s tiled |
| Prompt length | ~31 tokens for a one-sentence prompt (image tokens at 2048²: 16,384) |
| Prompt embeddings on disk | ~0.25 MB (one sentence); ~2.5 MB for a ~300-token enhanced prompt |
| Latent on disk | ~2 MB |
| `true_cfg_scale` default | 1.0: no classifier-free guidance, one transformer pass per step |

**The governing rule:** failures scale with how far a card's VRAM demand exceeds its
24 GB. The xe driver overflows a few GB of cold data into system RAM transparently;
beyond that it faults, hangs, or exhausts host RAM.

**No VAE tiling by default.** Tiling leaves visible seams. Full-size decodes run as an
exact split instead (§1.3); tiled decode stays available as a per-job fallback.

### 1.1 Reference images (prototype, 2026-10-16)

Measured with a local prototype of four stage scripts, kept outside the repository (design: [REFERENCE_IMAGES.md](REFERENCE_IMAGES.md)):
the four stages as separate processes, each in its own container on one B60, handing
`.safetensors` files on; agent image 0.3.1 (torch 2.14.1+xpu, diffusers
`b800f828`, bf16). References are resized to ~1 MP (1024² area, sides multiples of 32),
the pipeline's default. "Peak" is `torch.xpu.max_memory_allocated()` of that one process;
a B60 holds 23.9 GiB, anything above spills into host RAM through the driver.

| Per reference image (~1 MP) | Value |
|---|---|
| Step 1, VAE encode | 0.3–1.4 s, 2.7 GiB peak; latents 64 ch × (h/16 × w/16) ≈ **4,100 tokens, ~0.5 MB** |
| Step 2, text-encoder tokens | ≈ **1,000 image tokens** (Qwen3-VL: 16 px patches, 2×2 merge); embeds ~8 MB per reference |
| Denoise memory | ≈ **+2.6 GiB per reference**, almost all of it the conditioning KV cache |
| Encode memory | ≈ +0.85 GiB per reference over 17.2 GiB |

**Conditioning KV cache.** `causal_condition` lets the pipeline cache the keys and values of
the text and reference tokens after step 1 (32 layers × K and V × 4,096 × bf16 ≈ **0.5 MB per
conditioning token**). It explains the denoise peaks to within 1 GiB: weights ~13.3 GB +
cache + one step's activations. The cache is also why the first step is 2–4× slower than
the rest.

Denoise peak / time per step (after the first) / first step, 16:9 output, 6 steps:

| References | Conditioning tokens | Encode | ½ size 1376×768 | Full 2752×1536 |
|---|---|---|---|---|
| 1 | 5.2k | 17.2 GiB, 2.9 s | 16.5 GiB · 1.4 s · 5.0 s | 18.2 GiB · 7.2 s · 11.1 s |
| 3 | 15.4k | 18.9 GiB, 3.8 s | 21.6 GiB · 1.8 s · 8.5 s | 23.3 GiB · 13.6 s · 24.0 s |
| 4 | 20.4k | 19.7 GiB, 4.9 s | 24.1 GiB · **7.7 s** · 19.7 s | 25.8 GiB · 15.2 s · 28.7 s |
| 10 | 51.4k | 24.8 GiB, 16 s | **39.8 GiB** · 23.9 s · 51.2 s | **host hang (F9)** |

- Up to 3 references fit without spilling at every output size; 4 spill 0.2 GiB (½) to
  ~2 GiB (full), with host RAM never below 19.5 GiB free on a 30 GiB host. These are
  torch's allocator figures. **The driver holds much more:** live (2026-10-17), one
  4-reference full-size denoise raised `GPUActive` in `/proc/meminfo` from 0.2 to 7.2 GB,
  and free host RAM fell by ~9.4 GB with the worker's own memory. Two such denoises at once
  would leave ~3.5 GB of 30, so they run one at a time (F10).
- **A small spill can cost a lot:** 4 references at ½ size spilled ~0.2 GiB and each step
  took 4× longer than with 3; at full size a ~2 GiB spill cost only ~12%. Not understood
  yet (which tensors spill may matter: the cache is read every step).
- The first matrix's ¼-size runs used 688×384, which is not a multiple of 32, and failed with
  shape errors; Studio's ¼ 16:9 size is 704×384. Not a model limit.
- Decode is unchanged by references (½: 1.6 s, 7.0 GiB; full: 18.7 s, 26.0 GiB).
- Multi-reference composition works (characters from three pictures in one scene), and a
  one-reference edit at 20 steps kept character, props and layout while changing the scene.
  The 6-step matrix images are too rough to judge quality.

### 1.3 Split decode (2026-10-17)

The decoder's only attention is in its `mid_block`, at latent resolution (3.2 GiB for a
full-size latent); the upsampling blocks after it, which hold the memory, are 3×3
convolutions that see ~12 latent lines around a pixel. So the decode stage runs
`conv_in` + `mid_block` once on the whole latent and the rest on two strips across the
longer side, each reaching 32 latent lines (512 px) past the cut, and joins them at the
cut. Plain tiling instead runs the whole decoder per tile: its attention sees only the
tile, which shifts the colours everywhere (up to 10/255) and leaves seams.

Measured on one B60 with the agent image: three latents captured from live 2752×1536 jobs
against Studio's production untiled PNGs of the same latents, and random latents at the
other presets.

| | Untiled (production) | Split | Tiling-style (whole decoder per half) |
|---|---|---|---|
| Card peak, 2752×1536 | 25.76 GiB | 11.6 GiB (18.1 with per-frame caches) | 29.2 GiB |
| Host RAM held by the driver | 13–17 GB (live) | 0.09 GB (0.2–0.4 with caches) | 16 GB |
| Time | ~5 s + spill | ~5 s | 29 s |
| Difference from untiled | — | max 1/255 (no pixel more), PSNR 74 dB | max 10/255, PSNR 60 dB |

- In the middle of the overlap the two strips agree bit for bit; they differ only in the
  outermost 1–2 latent lines of each strip, which the join discards.
- Other presets: 2400×1792 and 1792×2400 12.2 GiB, 2048² 12.5 GiB, 2528×1696 and 1696×2528
  12.0 GiB, 1536×2752 11.6 GiB, all 2 strips, 0.09 GB of host RAM (2026-10-18). Before the
  decoder's per-frame caches were dropped (a single image never reads them) the same
  presets peaked at 18.8–19.6 GiB with 0.1–1.5 GB of host RAM. Strips across the shorter side of a portrait image (6
  narrow strips) still spilled 13 GB, hence across the longer side; and the allocator cache
  is emptied between strips, since strips of different sizes fragment it.

### 1.2 Image enhancer, PE-I2I (2026-10-17)

One B60, agent image of 0.4-dev, references resized to ~1 MP, the model card's sampling
(temperature 1.0, top_p 0.95, top_k 20, thinking on). Scripts kept outside the repository;
design: [ENHANCER_I2I.md](ENHANCER_I2I.md).

| Case (references) | Prompt tokens | int4: tokens · tok/s · time · peak | bf16: tokens · tok/s · time · peak |
|---|---|---|---|
| edit, add a hat (1) | 5,534 | 3,480 · 31.2 · 117 s · 9.6 GiB | 3,514 · 19.1 · 189 s · 19.1 GiB |
| restyle (1) | 5,564 | 3,060 · 31.1 · 101 s · 9.6 GiB | 2,240 · 19.3 · 118 s · 19.1 GiB |
| change text (1) | 5,569 | 2,922 · 31.2 · 96 s · 9.6 GiB | 1,981 · 19.4 · 105 s · 19.1 GiB |
| compose two (2) | 6,547 | 3,182 · 31.0 · 106 s · 9.9 GiB | 2,245 · 19.1 · 120 s · 19.4 GiB |
| poster of four (4) | 8,699 | 3,994 · 30.6 · 135 s · 10.6 GiB | 4,118 · 18.5 · 227 s · 20.1 GiB |

- The system prompt is ~4,500 tokens; each ~1 MP reference adds ~1,000. Prefill 2.4–5.2 s.
- Every run gave valid JSON with sensible aspect answers; int4 shows no visible loss, so int4
  is the recommended copy. Live, three references ran past 4096 still reasoning; the default is
  now 4096 plus 2048 per reference, and reasoning that runs out continues with the limit
  doubled (ENHANCER_I2I.md §2.1).
- int4 must keep the vision tower in bf16 (`modules_to_not_convert=["model.visual",
  "lm_head"]`); quantizing it as a causal LM drops the vision tower altogether.

---

## 2. Failure modes seen, and their real causes

| # | Symptom | Signature | Cause | Mitigation |
|---|---|---|---|---|
| F1 | Crash right after the last step | `xe … Fault response: Unsuccessful -EINVAL`, `EngineClass: 5 ccs` | Untiled decode on a card **already holding other components** | Decode runs alone on its card |
| F2 | Hang after decode, card unusable | `Engine reset … ccs`, `Timedout job … in no process`, `Xe has declared device … as wedged` | Large-tile decode with the VAE sharing a card with the transformer | Same as F1. A wedged card needs a reboot; save `/sys/class/drm/cardN/device/devcoredump/data` first |
| F3 | Process killed, host OOM | `oom-kill …`, `[TTM] Buffer eviction failed` | Host RAM exhausted: full-frame CPU decode (>14 GiB), or GPU overflow on top of an overcommitted card | No CPU decode; a host RAM floor before untiled decode |
| F4 | `device index is out of range … [0, 1)` | Python error at load | Only one card visible (selector, or a single `/dev/dri` node) | `ONEAPI_DEVICE_SELECTOR` must list both cards |
| F5 | `Torch not compiled with XPU enabled` | Python error | The PyPI (CPU) torch got installed | Python 3.12; torch/torchvision from the XPU index only |
| F6 | `Qwen3VLVideoProcessor requires the Torchvision library` | Python error | torchvision missing | Install it from the XPU index |
| F7 | Visible straight seams | Image | VAE tiling | Untiled decode |
| F8 | `device_map must be a string` | Python error | This diffusers version only accepts strategy names | Place whole components manually |
| F9 | Host hangs hard, no OOM line in the journal; on the next boot one card fails `Failed to enable GuC CT (-ENXIO)` and needs another reboot | Journal simply stops | Denoise with 10 ~1 MP references at full size: ~25 GiB of conditioning KV cache plus weights and a 16k-token target, far past the card, spilling into host RAM (the same run at ½ size peaked at 39.8 GiB) | Limit references (`ARIES_MAX_REFERENCE_IMAGES`, default 4) and give conditioned denoise the host-RAM floor and one-at-a-time limit; the agent's host-RAM guard kills the worker before RAM runs out (`ARIES_RAM_GUARD_FLOOR_BYTES`); MGLRU `min_ttl_ms` (§3.1) makes the kernel kill instead of freezing |

| F10 | Desktop session logged out (no host hang); the workers die after it | `kswapd0 invoked oom-killer … Killed process (systemd) uid=1000`, then `(python)` from the agent's container; `GPUActive` 14 GB in the report | A full-size untiled decode on one card and a full-size denoise with 3 references on the other, at once, with 4 GB in the `/tmp` tmpfs: driver spill (14 GB), workers (~10 GB) and tmpfs filled 30 GB faster than the RAM guard's 1 s window. The session went first because its processes run at `oom_score_adj` 100–200 | Full-size decode runs split and no longer spills (§1.3); memory-heavy tasks share one slot per agent (denoise above 2 references, and plain untiled decode above 3.5 MP when the split is off; split decodes keep the 6 GiB floor); workers run at `oom_score_adj` 1000; RAM guard every 0.25 s at a 4 GiB floor; no build output in the tmpfs `/tmp` |

### Hypotheses that turned out wrong

- **"Intel kernels can't address tensors >4 GiB."** No tensor in the decode exceeds
  2.25 GiB, and an untiled decode succeeds with the VAE alone. The recurring fault
  address `0x00008000fffe0000` is a fixed address in the driver's map, not an
  overflowed offset. Allocation limits, IGC flags and rebuilding PyTorch are
  irrelevant.
- **"Slow steps come from RAM overflow."** Step time is the same on one overcommitted
  card and on a clean split. The driver evicts the *idle* text encoder; the
  transformer stays resident.
- **"Two big components can share a card."** Text encoder + transformer on one card
  ran ~13 GiB of driver overflow into host RAM and ended in an OOM kill. Hence one
  model per card.

---

## 3. Host setup

Ubuntu 26.04 with the `xe` driver and the Intel graphics PPA:

```bash
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt install -y libze1 libze-intel-gpu1 intel-opencl-icd
sycl-ls     # note which level_zero:N are the B60s; an integrated GPU may be :0
```

- Expose only the discrete cards with `ONEAPI_DEVICE_SELECTOR` and set
  `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1` so device indices follow PCI order. The agent
  then gives each stage worker exactly one card and checks it opened only that one.
- The cards sit on different root ports: there is no peer-to-peer. Moving whole
  components or files between cards is fine; splitting layers across cards is not.
  `cannot be used for peer-to-peer DMA` at device init is expected and harmless.
- **Never call `xpu-smi`** from the app: it wakes the cards out of ASPM L1 and costs
  idle power. Hardware metrics (temperature, fan, power) belong to host exporters.
- Full-power generation pulls ~200 W per card. Keep any host-side fan or thermal
  guard enabled.

### 3.1 Running out of host RAM

Without swap, the kernel's OOM killer rarely fires on this host: when RAM runs out it
keeps dropping and re-reading the memory-mapped model weights, and the machine freezes
instead (F9). Two layers keep that from happening:

1. **The agent's host-RAM guard** (`ARIES_RAM_GUARD_FLOOR_BYTES`, default 4 GiB) samples
   `MemAvailable` every 0.25 s and kills the worker holding the most host RAM (resident
   memory plus driver spill) once it stays under the floor for two samples. It sees the
   whole host, including driver memory no container limit counts. Kills show up as
   `aries_ram_guard_kills_total`.
2. **Kernel thrash protection** for anything else: with MGLRU, `min_ttl_ms` makes the
   kernel OOM-kill when it cannot keep the working set in memory for that long, instead
   of thrashing.

   ```bash
   cat /sys/kernel/mm/lru_gen/enabled        # 0x0007 = MGLRU on (Ubuntu default)
   echo 1000 | sudo tee /sys/kernel/mm/lru_gen/min_ttl_ms    # now, until reboot
   echo 'w /sys/kernel/mm/lru_gen/min_ttl_ms - - - - 1000' |
     sudo tee /etc/tmpfiles.d/mglru-min-ttl.conf              # at every boot
   ```

When the kernel does OOM-kill, it should take a worker, not the desktop: desktop-session
processes run at `oom_score_adj` 100–200, which made a 3 MB `systemd --user` outrank a
2.8 GB worker (F10). The agent starts every worker at `oom_score_adj` 1000
(`ARIES_WORKER_OOM_SCORE_ADJ`), so a worker is always the first choice and its task is
retried as `oom_kill`.

Neither helps if the GPU driver itself hangs the kernel (F9's GuC failure on the next
boot hints the driver was involved). A hardware watchdog reboots a hung headless host:
load `sp5100_tco` (AMD) or `iTCO_wdt` (Intel) and set `RuntimeWatchdogSec=60` in
`/etc/systemd/system.conf`. systemd-oomd and Docker memory limits are not used: oomd
kills the whole agent container, and a memory limit misses driver memory and can thrash
inside the limit.

## 4. Model weights

Official diffusers-format weights from Qwen (~30 GB), plus the prompt enhancer:

```bash
hf download Qwen/Qwen-Image-2.1        --local-dir /srv/models/qwen-image-2.1
hf download Qwen/Qwen-Image-2.1-PE-T2I --local-dir /srv/models/qwen-image-2.1-pe-t2i
hf download Qwen/Qwen-Image-2.1-PE-I2I --local-dir /srv/models/qwen-image-2.1-pe-i2i  # optional
```

Mount them read-only and run with `HF_HUB_OFFLINE=1`: the backend never downloads at
runtime. Check the model licenses before any non-personal use.

## 5. Python dependencies

Order matters: **torch first, from the XPU index.**

```bash
uv venv --python 3.12 /opt/venv && . /opt/venv/bin/activate
uv pip install "torch==2.14.1+xpu" torchvision --index-url https://download.pytorch.org/whl/xpu
uv pip install "transformers==5.20.1" "torchao==0.18.0" accelerate safetensors pillow psutil
uv pip install "git+https://github.com/huggingface/diffusers.git"   # QwenImage21Pipeline is not released yet
uv pip install fastapi "uvicorn[standard]" prometheus-client
python -c "import torch; assert '+xpu' in torch.__version__, torch.__version__"
```

- `torchvision` is required even for text-to-image: the Qwen3-VL processor imports it.
- Installing any `torch*` package from PyPI silently replaces the XPU build. Always use
  the XPU index; the Docker build asserts `+xpu`.
- torch and torchao are pinned together: a torchao int4 checkpoint is serialized by one
  torchao version and packed for the device it was made on.
- **No flash-linear-attention (`fla-core`).** The enhancer (`Qwen3_5ForCausalLM`, 3 of 4
  layers linear attention) runs the reference PyTorch gated-delta-rule without it. Measured
  with the int4 enhancer and the real system prompt (2442 tokens), Python 3.12 container:
  prefill ~1070 tok/s either way, decode 34.7 → 36.9 tok/s with fla, but the first run in a
  fresh container spent ~67 s compiling Triton kernels. fla also needs `g++` (Triton XPU
  will not start its driver without a C++ compiler, and fla then falls back to CUDA and
  fails) and `libze-dev`. `ocloc` is not needed. Never install `fla-core[xpu]` from PyPI:
  the extra pulls a PyPI torch.

`backend/Dockerfile` does exactly this.

## 6. Diagnostics

```bash
sudo dmesg -T | grep -E 'xe 0000|Fault|GuC|engine reset' | tail -30   # GPU faults
docker compose logs -f                              # agent + worker output (deploy/aries)
curl -s localhost:9464/healthz | jq                  # agent: connected, executor states
curl -s http://studio.lan:8080/api/system | jq       # Studio: executors, live memory, queue, blocked reasons
sqlite3 data/studio.db "SELECT stage, executor, failure_class, exit_code, signal, failure_detail
  FROM task_attempts WHERE status = 'failed' ORDER BY finished_at DESC LIMIT 5"   # recent failures and why
```

- Workers run under `python -X faulthandler`, so a GPU fault that kills one leaves a
  Python stack in the agent log. The agent classifies the exit (`oom_kill`, `wedged`,
  `gpu_fault`, `python_error`, …) with the kernel log tail, and Studio stores it on the
  attempt and retries the task once.
- A crash right after the last denoise step means decode: check that nothing else held
  VRAM on that card, or that the job used tiled decode.
