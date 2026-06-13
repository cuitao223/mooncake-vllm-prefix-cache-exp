from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from openai import APIConnectionError, APIError, OpenAI
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


PREFIX_PARAGRAPH = (
    "这是一段用于构造长上下文的实验文本。它讨论大语言模型推理服务中的 "
    "KVCache、prefill、decoding、TTFT、TBT、prefix caching、Mooncake、"
    "request scheduling 和系统吞吐。当多个请求共享同一个长前缀时，系统可以复用"
    "已经计算过的 KVCache，从而减少重复 prefill 计算并降低首 token 延迟。"
)


QUESTIONS = [
    "请用一句话总结上面内容的主题。",
    "请列出上面内容中的三个关键词。",
    "请判断上面内容主要讨论系统、模型还是数据。",
    "请给这段内容写一个标题。",
]


def ensure_output_dirs() -> tuple[Path, Path]:
    """Create output directories and return tables and figures paths."""
    tables_dir = ROOT / "outputs" / "tables"
    figures_dir = ROOT / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def build_prompts(prefix_repeat: int) -> list[dict[str, str]]:
    """Build four prompts with identical common prefix and different questions."""
    common_prefix = "\n".join([PREFIX_PARAGRAPH] * prefix_repeat)
    prompts = []
    for request_id, question in enumerate(QUESTIONS):
        prompt = f"{common_prefix}\n\nQuestion: {question}\nAnswer:"
        prompts.append({"request_id": request_id, "question": question, "prompt": prompt})
    return prompts


def extract_delta_content(chunk: Any) -> str:
    """Extract text from an OpenAI chat completion streaming chunk."""
    if not chunk.choices:
        return ""
    delta = chunk.choices[0].delta
    content = getattr(delta, "content", None)
    return content or ""


def measure_one_request(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
) -> tuple[float, float, int, str]:
    """Measure TTFT and total latency for one streaming chat completion."""
    start_time = time.perf_counter()
    first_token_time: float | None = None
    output_chunks: list[str] = []

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        content = extract_delta_content(chunk)
        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            output_chunks.append(content)

    end_time = time.perf_counter()
    if first_token_time is None:
        first_token_time = end_time

    output_text = "".join(output_chunks)
    return first_token_time - start_time, end_time - start_time, len(output_text), output_text


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append measurement rows to CSV, creating the header if needed."""
    df = pd.DataFrame(rows)
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def load_results(path: Path) -> pd.DataFrame:
    """Load all accumulated results, or return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_ttft_by_request(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot mean TTFT by request id for each cache label."""
    if df.empty:
        return
    grouped = df.groupby(["cache_label", "request_id"], as_index=False)["ttft"].mean()

    plt.figure(figsize=(7, 4))
    for cache_label, part in grouped.groupby("cache_label"):
        part = part.sort_values("request_id")
        plt.plot(part["request_id"], part["ttft"], marker="o", label=cache_label)
    plt.xlabel("request_id")
    plt.ylabel("mean TTFT (s)")
    plt.title("vLLM Prefix Cache TTFT by Request")
    plt.xticks(sorted(df["request_id"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "vllm_ttft_by_request.png")
    plt.close()


def plot_prefix_length_sweep(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot request_id > 0 mean TTFT over prefix lengths."""
    if df.empty:
        return
    tail = df[df["request_id"] > 0]
    if tail.empty:
        return
    grouped = tail.groupby(["cache_label", "prefix_repeat"], as_index=False)["ttft"].mean()

    plt.figure(figsize=(7, 4))
    for cache_label, part in grouped.groupby("cache_label"):
        part = part.sort_values("prefix_repeat")
        plt.plot(part["prefix_repeat"], part["ttft"], marker="o", label=cache_label)
    plt.xlabel("prefix_repeat")
    plt.ylabel("mean TTFT for request_id > 0 (s)")
    plt.title("Prefix Length Sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "vllm_prefix_length_sweep.png")
    plt.close()


def plot_speedup(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot cache_off/cache_on speedup for request_id > 0 when both labels exist."""
    if df.empty:
        return
    tail = df[df["request_id"] > 0]
    grouped = tail.groupby(["cache_label", "prefix_repeat"], as_index=False)["ttft"].mean()
    pivot = grouped.pivot(index="prefix_repeat", columns="cache_label", values="ttft")
    if "cache_on" not in pivot.columns or "cache_off" not in pivot.columns:
        return
    pivot = pivot.dropna(subset=["cache_on", "cache_off"])
    if pivot.empty:
        return
    speedup = pivot["cache_off"] / pivot["cache_on"]

    plt.figure(figsize=(7, 4))
    plt.plot(speedup.index, speedup.values, marker="o")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("prefix_repeat")
    plt.ylabel("speedup")
    plt.title("Prefix Cache Speedup")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "vllm_speedup.png")
    plt.close()


def update_figures(csv_path: Path, figures_dir: Path) -> None:
    """Regenerate all figures from accumulated CSV results."""
    df = load_results(csv_path)
    if df.empty:
        return
    plot_ttft_by_request(df, figures_dir)
    plot_prefix_length_sweep(df, figures_dir)
    plot_speedup(df, figures_dir)


def print_summary(csv_path: Path) -> None:
    """Print summary tables for the accumulated CSV."""
    df = load_results(csv_path)
    if df.empty:
        print("No rows saved.")
        return
    summary = (
        df.groupby(["cache_label", "prefix_repeat", "request_id"], as_index=False)["ttft"]
        .mean()
        .rename(columns={"ttft": "mean_ttft"})
        .sort_values(["cache_label", "prefix_repeat", "request_id"])
    )
    print("\nSummary: cache_label, prefix_repeat, request_id, mean_ttft")
    print(summary.to_string(index=False))

    tail = df[df["request_id"] > 0]
    if not tail.empty:
        tail_summary = (
            tail.groupby(["cache_label", "prefix_repeat"], as_index=False)["ttft"]
            .mean()
            .rename(columns={"ttft": "mean_ttft_request_id_gt_0"})
            .sort_values(["cache_label", "prefix_repeat"])
        )
        print("\nSummary for request_id > 0")
        print(tail_summary.to_string(index=False))


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.cache_label not in {"cache_on", "cache_off"}:
        raise ValueError("--cache-label must be cache_on or cache_off")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if any(value <= 0 for value in args.prefix_repeat):
        raise ValueError("--prefix-repeat values must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure vLLM prefix cache TTFT with streaming API.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-label", required=True, choices=["cache_on", "cache_off"])
    parser.add_argument("--prefix-repeat", nargs="+", type=int, default=[50, 100, 200, 400])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output-csv", default=str(ROOT / "outputs" / "tables" / "vllm_prefix_cache.csv"))
    args = parser.parse_args()
    validate_args(args)

    tables_dir, figures_dir = ensure_output_dirs()
    csv_path = Path(args.output_csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    rows: list[dict[str, Any]] = []
    total_iterations = len(args.prefix_repeat) * (args.warmup + args.rounds) * len(QUESTIONS)

    try:
        with tqdm(total=total_iterations, desc=f"measure {args.cache_label}") as progress:
            for prefix_repeat in args.prefix_repeat:
                prompts = build_prompts(prefix_repeat)
                for logical_round in range(args.warmup + args.rounds):
                    is_warmup = logical_round < args.warmup
                    round_id = logical_round - args.warmup
                    for prompt_info in prompts:
                        ttft, total_latency, output_chars, output_text = measure_one_request(
                            client=client,
                            model=args.model,
                            prompt=prompt_info["prompt"],
                            max_tokens=args.max_tokens,
                        )
                        if not is_warmup:
                            rows.append(
                                {
                                    "cache_label": args.cache_label,
                                    "model": args.model,
                                    "prefix_repeat": prefix_repeat,
                                    "round_id": round_id,
                                    "request_id": prompt_info["request_id"],
                                    "question": prompt_info["question"],
                                    "ttft": ttft,
                                    "total_latency": total_latency,
                                    "output_chars": output_chars,
                                    "output_text": output_text,
                                }
                            )
                        progress.update(1)
    except (APIConnectionError, APIError, OSError) as exc:
        print(
            "\nFailed to call vLLM OpenAI-compatible server.\n"
            f"base_url: {args.base_url}\n"
            "Check that the server is running in another terminal and that the port is reachable.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted; saving completed non-warmup rows.", file=sys.stderr)

    append_csv(csv_path, rows)
    update_figures(csv_path, figures_dir)
    print(f"\nSaved {len(rows)} new rows to {csv_path}")
    print(f"Figures directory: {figures_dir}")
    print_summary(csv_path)


if __name__ == "__main__":
    main()
