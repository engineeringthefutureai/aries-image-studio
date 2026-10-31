# Reference: the batch scripts that validated the pipeline

The scripts used to bring Qwen-Image-2.1 up on 2× Arc Pro B60 and take the
measurements in `docs/HARDWARE.md`. Not part of the app — kept as working examples
that run directly on the hardware.

- `gen.py` — resumable batch generator: placement modes (`--split`, `--offload`,
  single card), VAE tiling, fixed seeds, per-image timing + peak VRAM JSON, and
  `.running` / `.failed` markers so a GPU crash can't crash-loop on one prompt.
- `run.sh` — one worker per card (data-parallel throughput), auto-restart.
- `run-split.sh` — one pipeline across both cards, auto-restart.
- `prompts.jsonl` — the fixed test prompt set (text rendering, counting, hands,
  portrait, RGBA, wide, tall). Reusable as a smoke/regression set.
- `latent_preview/` — how the matrix behind Studio's preview before decode
  (`studio/assets/latent_preview.bin`) was made: `encode.py` encodes sample images with the
  VAE on the CPU, `fit.py` fits the linear map and scores it on held-out images. The
  current one came from 18 Qwen-Image-2.1 images at 768 px (held-out PSNR about 23 dB
  against the half-size image).
