#!/usr/bin/env python3
"""
Benchmark comparison script for AHA-OLMO2 vs baseline OLMo2.

Usage:
    python benchmark_aha_comparison.py [OPTIONS]

Options:
    --num-prompts N      Number of prompts to benchmark (default: 200)
    --input-len N        Input sequence length (default: 512)
    --output-len N       Output sequence length (default: 128)
    --save-results       Save results to JSON file
    --aha-only           Only benchmark AHA model
    --baseline-only      Only benchmark baseline model
    --gpu-memory FLOAT   GPU memory utilization (default: 0.8)
"""

import argparse
import json
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path


AHA_MODEL = "xuan-luo/AHA-OLMO2"
BASELINE_MODEL = "allenai/OLMo-2-0425-1B"


def run_benchmark(model: str, args: argparse.Namespace, trust_remote_code: bool = False) -> dict:
    """Run vllm benchmark and parse results."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "throughput",
        "--model", model,
        "--dtype", "half",
        "--num-prompts", str(args.num_prompts),
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--gpu-memory-utilization", str(args.gpu_memory),
    ]

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        output = result.stdout + result.stderr
        print(output)

        if result.returncode != 0:
            print(f"ERROR: Benchmark failed with return code {result.returncode}")
            return {"error": output, "model": model}

        # Parse throughput metrics from output
        metrics = parse_benchmark_output(output)
        metrics["model"] = model
        return metrics

    except subprocess.TimeoutExpired:
        print(f"ERROR: Benchmark timed out after 30 minutes")
        return {"error": "timeout", "model": model}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e), "model": model}


def parse_benchmark_output(output: str) -> dict:
    """Parse benchmark output to extract metrics."""
    metrics = {}

    # Main throughput line format:
    # "Throughput: XX.XX requests/s, XX.XX total tokens/s, XX.XX output tokens/s"
    throughput_pattern = r"Throughput:\s*([\d.]+)\s*requests/s,\s*([\d.]+)\s*total tokens/s,\s*([\d.]+)\s*output tokens/s"
    match = re.search(throughput_pattern, output)
    if match:
        metrics["throughput_requests_per_sec"] = float(match.group(1))
        metrics["throughput_tokens_per_sec"] = float(match.group(2))
        metrics["output_tokens_per_sec"] = float(match.group(3))

    # Additional metrics
    prompt_match = re.search(r"Total num prompt tokens:\s*([\d]+)", output)
    if prompt_match:
        metrics["total_prompt_tokens"] = int(prompt_match.group(1))

    output_match = re.search(r"Total num output tokens:\s*([\d]+)", output)
    if output_match:
        metrics["total_output_tokens"] = int(output_match.group(1))

    return metrics


def print_comparison(aha_results: dict, baseline_results: dict):
    """Print side-by-side comparison of results."""
    print("\n" + "="*78)
    print("              AHA vs OLMo2 Benchmark Results")
    print("="*78)

    # Check for errors
    if "error" in aha_results:
        print(f"\nAHA Model Error: {aha_results.get('error', 'Unknown error')}")
    if "error" in baseline_results:
        print(f"\nBaseline Model Error: {baseline_results.get('error', 'Unknown error')}")

    if "error" in aha_results or "error" in baseline_results:
        return

    # Print header
    print(f"\n{'Model':<35} | {'Total tok/s':>12} | {'Output tok/s':>12} | {'Req/s':>8}")
    print("-"*35 + "-+-" + "-"*12 + "-+-" + "-"*12 + "-+-" + "-"*8)

    # Get metrics
    aha_total = aha_results.get("throughput_tokens_per_sec", "N/A")
    aha_output = aha_results.get("output_tokens_per_sec", "N/A")
    aha_requests = aha_results.get("throughput_requests_per_sec", "N/A")

    baseline_total = baseline_results.get("throughput_tokens_per_sec", "N/A")
    baseline_output = baseline_results.get("output_tokens_per_sec", "N/A")
    baseline_requests = baseline_results.get("throughput_requests_per_sec", "N/A")

    # Format values
    def fmt(v):
        if isinstance(v, (int, float)):
            return f"{v:,.1f}"
        return str(v)

    print(f"{'xuan-luo/AHA-OLMO2':<35} | {fmt(aha_total):>12} | {fmt(aha_output):>12} | {fmt(aha_requests):>8}")
    print(f"{'allenai/OLMo-2-0425-1B':<35} | {fmt(baseline_total):>12} | {fmt(baseline_output):>12} | {fmt(baseline_requests):>8}")

    # Calculate differences
    print("-"*35 + "-+-" + "-"*12 + "-+-" + "-"*12 + "-+-" + "-"*8)

    def calc_diff(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b != 0:
            return f"{((a - b) / b) * 100:+.1f}%"
        return "N/A"

    diff_total = calc_diff(aha_total, baseline_total)
    diff_output = calc_diff(aha_output, baseline_output)
    diff_req = calc_diff(aha_requests, baseline_requests)

    print(f"{'Difference (AHA vs Baseline)':<35} | {diff_total:>12} | {diff_output:>12} | {diff_req:>8}")
    print("="*78 + "\n")


def save_results(aha_results: dict, baseline_results: dict, args: argparse.Namespace):
    """Save benchmark results to JSON file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_prompts": args.num_prompts,
            "input_len": args.input_len,
            "output_len": args.output_len,
        },
        "aha": aha_results,
        "baseline": baseline_results,
    }

    # Append to results file
    results_file = Path("benchmark_results.json")

    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(results)

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AHA-OLMO2 vs baseline OLMo2"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=500,
        help="Number of prompts to benchmark (default: 200)"
    )
    parser.add_argument(
        "--input-len", type=int, default=512,
        help="Input sequence length (default: 512)"
    )
    parser.add_argument(
        "--output-len", type=int, default=128,
        help="Output sequence length (default: 128)"
    )
    parser.add_argument(
        "--gpu-memory", type=float, default=0.8,
        help="GPU memory utilization (default: 0.8)"
    )
    parser.add_argument(
        "--save-results", action="store_true",
        help="Save results to benchmark_results.json"
    )
    parser.add_argument(
        "--aha-only", action="store_true",
        help="Only benchmark AHA model"
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only benchmark baseline model"
    )

    args = parser.parse_args()

    print("\n" + "="*78)
    print("           AHA-OLMO2 vs OLMo2-1B Performance Comparison")
    print("="*78)
    print(f"Config: {args.num_prompts} prompts, {args.input_len} input len, {args.output_len} output len")

    aha_results = {}
    baseline_results = {}

    # Run benchmarks
    if not args.baseline_only:
        aha_results = run_benchmark(AHA_MODEL, args, trust_remote_code=True)

    if not args.aha_only:
        baseline_results = run_benchmark(BASELINE_MODEL, args, trust_remote_code=False)

    # Print comparison
    if aha_results and baseline_results:
        print_comparison(aha_results, baseline_results)
    elif aha_results:
        print(f"\nAHA Results: {aha_results}")
    elif baseline_results:
        print(f"\nBaseline Results: {baseline_results}")

    # Save results
    if args.save_results and (aha_results or baseline_results):
        save_results(aha_results, baseline_results, args)


if __name__ == "__main__":
    main()
