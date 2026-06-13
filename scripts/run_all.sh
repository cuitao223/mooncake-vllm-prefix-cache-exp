#!/usr/bin/env bash
set -e

cat <<'EOF'
This script does not start or stop the vLLM server automatically.
Run the experiment with two terminals:

Step 1: start cache off server in terminal A
  bash scripts/start_vllm_cache_off.sh

Step 2: run cache off measurements in terminal B
  bash scripts/run_cache_off.sh

Step 3: stop server in terminal A with Ctrl-C

Step 4: start cache on server in terminal A
  bash scripts/start_vllm_cache_on.sh

Step 5: run cache on measurements in terminal B
  bash scripts/run_cache_on.sh

Step 6: check outputs/tables and outputs/figures
EOF
