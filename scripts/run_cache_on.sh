#!/usr/bin/env bash
set -e

python experiments/measure_vllm_ttft.py \
  --cache-label cache_on \
  --prefix-repeat 50 100 200 400 \
  --rounds 3 \
  --warmup 1 \
  --max-tokens 16
