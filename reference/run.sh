#!/usr/bin/env bash
# Two workers, one per B60. Each restarts after a crash; gen.py skips finished
# images and marks the one that crashed as .failed, so the run always moves forward.
set -u
export HF_HUB_OFFLINE=1
# export QWEN_IMAGE_PATH=/models/qwen-image-2.1   # if using a plain local folder
mkdir -p logs

worker() {
  local card=$1
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    ONEAPI_DEVICE_SELECTOR=level_zero:$card \
      python -X faulthandler gen.py --shard "$card" --of 2 "${@:2}" \
      >> "logs/w$card.log" 2>&1 && return 0
    echo "[w$card] exited non-zero (attempt $attempt), restarting in 15s" >> "logs/w$card.log"
    sleep 15
  done
}

worker 0 "$@" &
worker 1 "$@" &
wait
echo "all done: $(ls out/*.png 2>/dev/null | wc -l) images, $(ls out/*.failed 2>/dev/null | wc -l) failed"
