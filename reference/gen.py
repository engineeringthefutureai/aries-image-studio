#!/usr/bin/env python3
"""Qwen-Image-2.1 batch worker. One process per GPU, resumable.

Usage:
  ONEAPI_DEVICE_SELECTOR=level_zero:0 python -X faulthandler gen.py --shard 0 --of 2
Prompts: prompts.jsonl, one object per line:
  {"id": "neon-sign", "prompt": "...", "w": 2048, "h": 2048, "steps": 40, "seed": 42}
"""
import argparse, json, os, sys, time
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("QWEN_IMAGE_PATH", "Qwen/Qwen-Image-2.1"))
ap.add_argument("--prompts", default="prompts.jsonl")
ap.add_argument("--out", default="out")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--of", type=int, default=1)
ap.add_argument("--offload", action="store_true", help="CPU offload if the pipeline won't fit in 24 GB")
ap.add_argument("--split", action="store_true",
                help="one pipeline across both cards (text encoder / DiT / VAE on separate devices)")
args = ap.parse_args()

out = Path(args.out); out.mkdir(exist_ok=True)
tag = f"[w{args.shard}]"
def log(*a): print(tag, time.strftime("%H:%M:%S"), *a, flush=True)

jobs = [json.loads(l) for l in open(args.prompts) if l.strip()]
jobs = [j for i, j in enumerate(jobs) if i % args.of == args.shard]

if args.split:
    # Whole components placed per device; only embeddings and the latent cross cards.
    pipe = QwenImage21Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="balanced", max_memory={0: "21GiB", 1: "21GiB"},
    )
    log("placement:", getattr(pipe, "hf_device_map", "?"))
else:
    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if args.offload:
        pipe.enable_model_cpu_offload(device="xpu")
    else:
        pipe.to("xpu")
pipe.vae.enable_tiling()   # avoids the >4 GiB tensor fault in the 2048² decode
pipe.vae.enable_slicing()
log("loaded", torch.xpu.get_device_name(0))

for j in jobs:
    png, marker, fail = out / f"{j['id']}.png", out / f"{j['id']}.running", out / f"{j['id']}.failed"
    if png.exists() or fail.exists():
        continue
    if marker.exists():                  # previous attempt died mid-image -> don't retry forever
        marker.rename(fail); log("FAILED (crashed previous run):", j["id"]); continue
    marker.touch()

    for d in range(torch.xpu.device_count()):
        torch.xpu.reset_peak_memory_stats(d)
    t0 = time.time()
    img = pipe(
        prompt=j["prompt"],
        width=j.get("w", 2048), height=j.get("h", 2048),
        num_inference_steps=j.get("steps", 40),
        generator=torch.Generator("xpu").manual_seed(j.get("seed", 42)),
    ).images[0]
    dt = time.time() - t0

    tmp = png.with_suffix(".tmp.png"); img.save(tmp); tmp.rename(png)
    meta = {**j, "seconds": round(dt, 1),
            "peak_vram_gib": [round(torch.xpu.max_memory_allocated(d) / 2**30, 2)
                              for d in range(torch.xpu.device_count())],
            "device": os.environ.get("ONEAPI_DEVICE_SELECTOR", "")}
    (out / f"{j['id']}.json").write_text(json.dumps(meta, indent=1))
    marker.unlink()
    log(f"done {j['id']}  {dt:.0f}s  peak {meta['peak_vram_gib']} GiB")

log("shard complete")
