# vLLM Prefix Cache Experiment for Mooncake Reproduction

This project is a real LLM serving auxiliary experiment for reproducing one
core premise behind `Mooncake: A KVCache-centric Disaggregated Architecture for
LLM Serving`:

When multiple requests share the exact same long prefix, prefix KVCache reuse
can reduce repeated prefill computation and lower TTFT for later requests.

This is not a full Mooncake reproduction. It does not implement multi-node RDMA,
prefill/decode disaggregation, distributed KVCache pools, Mooncake's
KVCache-centric scheduler, or overload-oriented early rejection. It is a
single-machine vLLM prefix caching validation experiment intended to support a
Mooncake reproduction report with real LLM measurements.

## Hardware And Software

Target machine:

- GPU: RTX 4090D 24GB
- Default model: `Qwen/Qwen2.5-0.5B-Instruct`
- Framework: vLLM
- API: vLLM OpenAI-compatible server

Metrics:

- TTFT
- total latency
- output chars
- later-request mean TTFT
- cache on/off speedup

## Setup On The 4090D Server

Run these commands on the rented GPU server, not on your local Windows machine:

```bash
cd mooncake-vllm-prefix-cache-exp
conda create -n mooncake-vllm python=3.10 -y
conda activate mooncake-vllm
pip install -r requirements.txt
chmod +x scripts/*.sh
```

If model download is slow, configure your Hugging Face mirror or token before
starting vLLM.

## Important vLLM Flag Check

vLLM CLI flags can change across versions. Before running the experiment, check:

```bash
vllm serve --help | grep prefix
```

Confirm whether these flags exist:

```text
--enable-prefix-caching
--no-enable-prefix-caching
```

If your installed vLLM version uses different names, update:

- `scripts/start_vllm_cache_off.sh`
- `scripts/start_vllm_cache_on.sh`

## Run The Experiment

Use two terminals on the GPU server.

### Cache Off

Terminal A:

```bash
bash scripts/start_vllm_cache_off.sh
```

Wait until vLLM says the server is ready.

Terminal B:

```bash
bash scripts/run_cache_off.sh
```

Then stop the server in terminal A with `Ctrl-C`.

### Cache On

Terminal A:

```bash
bash scripts/start_vllm_cache_on.sh
```

Terminal B:

```bash
bash scripts/run_cache_on.sh
```

You can print the step-by-step instructions with:

```bash
bash scripts/run_all.sh
```

## Custom Runs

Measure one cache setting manually:

```bash
python experiments/measure_vllm_ttft.py \
  --cache-label cache_on \
  --prefix-repeat 50 100 200 400 \
  --rounds 3 \
  --warmup 1 \
  --max-tokens 16
```

Useful variants:

```bash
python experiments/measure_vllm_ttft.py --cache-label cache_on --prefix-repeat 800 --rounds 5
python experiments/measure_vllm_ttft.py --cache-label cache_off --prefix-repeat 800 --rounds 5
```

If the server uses another port:

```bash
python experiments/measure_vllm_ttft.py \
  --base-url http://localhost:8001/v1 \
  --cache-label cache_on
```

## Outputs

CSV:

```text
outputs/tables/vllm_prefix_cache.csv
```

Figures:

```text
outputs/figures/vllm_ttft_by_request.png
outputs/figures/vllm_prefix_length_sweep.png
outputs/figures/vllm_speedup.png
```

The CSV is append-only. Running cache-off first and cache-on second will produce
one combined result table.

## How The Requests Are Built

The script constructs four requests:

```text
request 0 = common_prefix + question 0
request 1 = common_prefix + question 1
request 2 = common_prefix + question 2
request 3 = common_prefix + question 3
```

The `common_prefix` is exactly identical for all four requests. Only the final
question changes.

The experiment uses:

```text
temperature = 0
stream = True
max_tokens = 16 by default
```

Streaming is required because TTFT is measured as the time from API request
start to the first non-empty streamed `delta.content`.

## Why Request 0 May Not Be Faster

Request 0 is the first request that introduces the long common prefix. vLLM must
build the KVCache for that prefix. The later requests, request 1/2/3, are the
ones expected to benefit from prefix cache reuse.

Therefore, the key comparison is:

```text
cache_on vs cache_off for request_id > 0
```

Do not expect request 0 to always improve.

## Why max_tokens Is Small

This experiment focuses on prefill and TTFT. If `max_tokens` is large, decode
time can dominate total latency and hide the prefix-cache effect. The default is
therefore:

```text
max_tokens = 16
```

## How To Interpret Results

Open:

```text
outputs/tables/vllm_prefix_cache.csv
```

Focus on:

```text
cache_label
prefix_repeat
request_id
ttft
```

Expected trend:

- For `request_id = 0`, cache-on is not necessarily faster.
- For `request_id > 0`, cache-on should often have lower TTFT than cache-off.
- Larger `prefix_repeat` should make the cache-on benefit more visible.

Speedup is computed as:

```text
speedup = cache_off later-request mean TTFT / cache_on later-request mean TTFT
```

where later requests means:

```text
request_id > 0
```

## Common Problems

### OOM On 4090D 24GB

Lower the max model length:

```bash
MAX_MODEL_LEN=2048 bash scripts/start_vllm_cache_on.sh
```

Or reduce GPU memory utilization:

```bash
GPU_MEMORY_UTILIZATION=0.75 bash scripts/start_vllm_cache_on.sh
```

### Model Download Is Slow

Wait for Hugging Face download to finish, or set your mirror/token before
running vLLM.

### vLLM Prefix Flags Are Not Compatible

Run:

```bash
vllm serve --help | grep prefix
```

Then update the two start scripts if needed.

### cache_on/cache_off Difference Is Small

Try:

```bash
python experiments/measure_vllm_ttft.py --cache-label cache_on --prefix-repeat 800 --rounds 5
python experiments/measure_vllm_ttft.py --cache-label cache_off --prefix-repeat 800 --rounds 5
```

You can also increase:

```bash
MAX_MODEL_LEN=8192 bash scripts/start_vllm_cache_on.sh
```

For 4090D, start with `Qwen/Qwen2.5-0.5B-Instruct` to confirm the pipeline first.
After that, try `Qwen/Qwen2.5-1.5B-Instruct` if memory allows.

### TTFT Is Noisy

Try:

```bash
python experiments/measure_vllm_ttft.py --cache-label cache_on --rounds 5 --warmup 2
```

Also make sure no other GPU-heavy process is running.

### Server Is Not Running

If the measurement script fails with a connection error, start vLLM in another
terminal first:

```bash
bash scripts/start_vllm_cache_on.sh
```

Then rerun the measurement script.

## Project Structure

```text
mooncake-vllm-prefix-cache-exp/
  README.md
  requirements.txt
  scripts/
    start_vllm_cache_off.sh
    start_vllm_cache_on.sh
    run_cache_off.sh
    run_cache_on.sh
    run_all.sh
  experiments/
    measure_vllm_ttft.py
  outputs/
    tables/
    figures/
```

