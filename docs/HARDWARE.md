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
| **Untiled decode peak at 2048²** | **25.76 GiB**: more than one card; ~2 GiB overflows to host RAM |
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

**No VAE tiling by default.** Tiling leaves visible seams. Untiled decode needs a card
to itself and host RAM headroom; tiled decode stays available as a per-job fallback.

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
  `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1` so device indices follow PCI order. The manager
  then gives each stage worker exactly one card.
- The cards sit on different root ports: there is no peer-to-peer. Moving whole
  components or files between cards is fine; splitting layers across cards is not.
  `cannot be used for peer-to-peer DMA` at device init is expected and harmless.
- **Never call `xpu-smi`** from the app: it wakes the cards out of ASPM L1 and costs
  idle power. Hardware metrics (temperature, fan, power) belong to host exporters.
- Full-power generation pulls ~200 W per card. Keep any host-side fan or thermal
  guard enabled.

## 4. Model weights

Official diffusers-format weights from Qwen (~30 GB), plus the prompt enhancer:

```bash
hf download Qwen/Qwen-Image-2.1        --local-dir /srv/models/qwen-image-2.1
hf download Qwen/Qwen-Image-2.1-PE-T2I --local-dir /srv/models/qwen-image-2.1-pe-t2i
```

Mount them read-only and run with `HF_HUB_OFFLINE=1`: the backend never downloads at
runtime. Check the model licenses before any non-personal use.

## 5. Python dependencies

Order matters: **torch first, from the XPU index.**

```bash
uv venv --python 3.12 /opt/venv && . /opt/venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
uv pip install "transformers>=5.17" accelerate safetensors pillow psutil
uv pip install "git+https://github.com/huggingface/diffusers.git"   # QwenImage21Pipeline is not released yet
uv pip install fastapi "uvicorn[standard]" prometheus-client
python -c "import torch; assert '+xpu' in torch.__version__, torch.__version__"
```

- `torchvision` is required even for text-to-image: the Qwen3-VL processor imports it.
- Installing any `torch*` package from PyPI silently replaces the XPU build. Always use
  the XPU index; the Docker build asserts `+xpu`.

`backend/Dockerfile` does exactly this.

## 6. Diagnostics

```bash
sudo dmesg -T | grep -E 'xe 0000|Fault|GuC|engine reset' | tail -30   # GPU faults
docker compose logs -f                                                # manager + worker output
curl -s localhost:8000/stage_runs | jq '.[0:5]'                       # recent worker exits and why
```

- Workers run under `python -X faulthandler`, so a GPU fault that kills one leaves a
  Python stack in the manager log. The manager classifies the exit (`oom_kill`,
  `wedged`, `python_error`) and retries the stage once.
- A crash right after the last denoise step means decode: check that nothing else held
  VRAM on that card, or that the job used tiled decode.
