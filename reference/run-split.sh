#!/usr/bin/env bash
# One worker, one pipeline spread across both B60s. Restarts after a crash;
# gen.py skips finished images and marks the crashing one .failed.
set -u
export HF_HUB_OFFLINE=1
# export QWEN_IMAGE_PATH=/models/qwen-image-2.1
# Expose ONLY the two B60s (check `sycl-ls`; adjust if the iGPU is level_zero:0)
export ONEAPI_DEVICE_SELECTOR="level_zero:0,1"
mkdir -p logs

for attempt in 1 2 3 4 5 6 7 8 9 10; do
  python -X faulthandler gen.py --split "$@" >> logs/split.log 2>&1 && break
  echo "[split] exited non-zero (attempt $attempt), restarting in 15s" >> logs/split.log
  sleep 15
done
echo "done: $(ls out/*.png 2>/dev/null | wc -l) images, $(ls out/*.failed 2>/dev/null | wc -l) failed"
