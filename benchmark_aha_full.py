#!/usr/bin/env python3
"""
Benchmark comparison script for OLMo2, AHA (baseline), AHA (optimized), and Local-only.

Compares four models:
1. OLMo2 (baseline)    - Standard OLMo2 without AHA (lower bound performance)
2. AHA (baseline)      - Frozen AHA implementation
3. AHA (optimized)     - Your optimized AHA implementation
4. Local-only          - All sliding window attention (upper bound performance)

Usage:
    python benchmark_aha_full.py [OPTIONS]

Options:
    --num-prompts N      Number of prompts to benchmark (default: 500)
    --input-len N        Input sequence length (default: 512)
    --output-len N       Output sequence length (default: 128)
    --save-results       Save results to JSON file
    --gpu-memory FLOAT   GPU memory utilization (default: 0.8)
    --skip-local-only    Skip local-only (upper bound) benchmark
"""

import argparse
import json
import os
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path


# Standard OLMo2 (no AHA)
OLMO2_MODEL = "allenai/OLMo-2-0425-1B"

# AHA model (same HuggingFace repo for both baseline and optimized)
# The difference is controlled by VLLM_AHA_BASELINE env var
AHA_MODEL = "xuan-luo/AHA-OLMO2"

# Local-only model (same as OLMo2, but uses local-only implementation)
LOCAL_ONLY_MODEL = "allenai/OLMo-2-0425-1B"


def run_benchmark(model: str, args: argparse.Namespace, trust_remote_code: bool = False,
                  use_baseline: bool = False, use_local_only: bool = False) -> dict:
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

    # Display name for output
    if use_local_only:
        display_name = "Local-only (upper)"
    elif "OLMo-2" in model:
        display_name = "OLMo2 (baseline)"
    elif use_baseline:
        display_name = "AHA (baseline)"
    else:
        display_name = "AHA (optimized)"

    print(f"\n{'='*70}")
    print(f"Benchmarking: {display_name}")
    print(f"Model path: {model}")
    if use_local_only:
        print(f"Using: olmo2_local_only.py (all sliding window)")
    elif use_baseline:
        print(f"Using: olmo2_aha_baseline.py (frozen)")
    elif "AHA" in model.upper():
        print(f"Using: olmo2_aha.py (optimized)")
    print(f"{'='*70}\n")

    # Set environment for baseline vs optimized vs local-only
    env = os.environ.copy()
    if use_local_only:
        env["VLLM_LOCAL_ONLY"] = "1"
        # Make sure AHA baseline is not set
        if "VLLM_AHA_BASELINE" in env:
            del env["VLLM_AHA_BASELINE"]
    elif use_baseline:
        env["VLLM_AHA_BASELINE"] = "1"
        if "VLLM_LOCAL_ONLY" in env:
            del env["VLLM_LOCAL_ONLY"]
    else:
        if "VLLM_AHA_BASELINE" in env:
            del env["VLLM_AHA_BASELINE"]
        if "VLLM_LOCAL_ONLY" in env:
            del env["VLLM_LOCAL_ONLY"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            env=env,
        )

        output = result.stdout + result.stderr
        print(output)

        if result.returncode != 0:
            print(f"ERROR: Benchmark failed with return code {result.returncode}")
            return {"error": output[:500], "model": model, "display_name": display_name}

        # Parse throughput metrics from output
        metrics = parse_benchmark_output(output)
        metrics["model"] = model
        metrics["display_name"] = display_name
        return metrics

    except subprocess.TimeoutExpired:
        print(f"ERROR: Benchmark timed out after 30 minutes")
        return {"error": "timeout", "model": model, "display_name": display_name}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e), "model": model, "display_name": display_name}


def run_local_only_benchmark(args: argparse.Namespace) -> dict:
    """Run local-only benchmark (upper bound performance)."""
    return run_benchmark(LOCAL_ONLY_MODEL, args, trust_remote_code=False, use_local_only=True)


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


def print_comparison(olmo2_results: dict, aha_baseline_results: dict,
                     aha_optimized_results: dict, local_only_results: dict):
    """Print side-by-side comparison of results."""
    print("\n" + "="*90)
    print("                    OLMo2 vs AHA vs Local-only Benchmark Results")
    print("="*90)

    # Check for errors
    all_results = [
        ("OLMo2 (baseline)", olmo2_results),
        ("AHA (baseline)", aha_baseline_results),
        ("AHA (optimized)", aha_optimized_results),
        ("Local-only (upper)", local_only_results),
    ]

    has_error = False
    for name, results in all_results:
        if results and "error" in results:
            print(f"\n{name} Error: {results.get('error', 'Unknown')[:200]}")
            has_error = True

    if has_error:
        print("\nSome benchmarks failed. Showing available results:\n")

    # Format values
    def fmt(v):
        if isinstance(v, (int, float)):
            return f"{v:,.1f}"
        return "N/A"

    def get_metric(results, key):
        if results and "error" not in results:
            return results.get(key, "N/A")
        return "N/A"

    # Print header
    print(f"\n{'Model':<20} | {'Total tok/s':>12} | {'Output tok/s':>12} | {'Req/s':>8}")
    print("-"*20 + "-+-" + "-"*12 + "-+-" + "-"*12 + "-+-" + "-"*8)

    # Get metrics
    olmo2_total = get_metric(olmo2_results, "throughput_tokens_per_sec")
    olmo2_output = get_metric(olmo2_results, "output_tokens_per_sec")
    olmo2_requests = get_metric(olmo2_results, "throughput_requests_per_sec")

    baseline_total = get_metric(aha_baseline_results, "throughput_tokens_per_sec")
    baseline_output = get_metric(aha_baseline_results, "output_tokens_per_sec")
    baseline_requests = get_metric(aha_baseline_results, "throughput_requests_per_sec")

    optimized_total = get_metric(aha_optimized_results, "throughput_tokens_per_sec")
    optimized_output = get_metric(aha_optimized_results, "output_tokens_per_sec")
    optimized_requests = get_metric(aha_optimized_results, "throughput_requests_per_sec")

    local_total = get_metric(local_only_results, "throughput_tokens_per_sec")
    local_output = get_metric(local_only_results, "output_tokens_per_sec")
    local_requests = get_metric(local_only_results, "throughput_requests_per_sec")

    print(f"{'OLMo2 (baseline)':<20} | {fmt(olmo2_total):>12} | {fmt(olmo2_output):>12} | {fmt(olmo2_requests):>8}")
    print(f"{'AHA (baseline)':<20} | {fmt(baseline_total):>12} | {fmt(baseline_output):>12} | {fmt(baseline_requests):>8}")
    print(f"{'AHA (optimized)':<20} | {fmt(optimized_total):>12} | {fmt(optimized_output):>12} | {fmt(optimized_requests):>8}")
    print(f"{'Local-only (upper)':<20} | {fmt(local_total):>12} | {fmt(local_output):>12} | {fmt(local_requests):>8}")

    # Calculate differences
    print("-"*20 + "-+-" + "-"*12 + "-+-" + "-"*12 + "-+-" + "-"*8)

    def calc_diff(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b != 0:
            return f"{((a - b) / b) * 100:+.1f}%"
        return "N/A"

    # AHA baseline vs OLMo2 (shows AHA overhead)
    diff1_total = calc_diff(baseline_total, olmo2_total)
    diff1_output = calc_diff(baseline_output, olmo2_output)
    diff1_req = calc_diff(baseline_requests, olmo2_requests)
    print(f"{'AHA overhead':<20} | {diff1_total:>12} | {diff1_output:>12} | {diff1_req:>8}")

    # AHA optimized vs AHA baseline (shows optimization gain)
    diff2_total = calc_diff(optimized_total, baseline_total)
    diff2_output = calc_diff(optimized_output, baseline_output)
    diff2_req = calc_diff(optimized_requests, baseline_requests)
    print(f"{'Optimization gain':<20} | {diff2_total:>12} | {diff2_output:>12} | {diff2_req:>8}")

    # Local-only vs OLMo2 (shows max possible gain)
    diff3_total = calc_diff(local_total, olmo2_total)
    diff3_output = calc_diff(local_output, olmo2_output)
    diff3_req = calc_diff(local_requests, olmo2_requests)
    print(f"{'Local-only speedup':<20} | {diff3_total:>12} | {diff3_output:>12} | {diff3_req:>8}")

    # AHA efficiency - how close AHA gets to local-only upper bound
    # Calculated as: (AHA_optimized - OLMo2) / (Local_only - OLMo2) * 100
    def calc_efficiency(optimized, baseline, upper):
        if (isinstance(optimized, (int, float)) and isinstance(baseline, (int, float))
            and isinstance(upper, (int, float)) and upper != baseline):
            efficiency = ((optimized - baseline) / (upper - baseline)) * 100
            return f"{efficiency:.1f}%"
        return "N/A"

    eff_total = calc_efficiency(optimized_total, olmo2_total, local_total)
    eff_output = calc_efficiency(optimized_output, olmo2_output, local_output)
    eff_req = calc_efficiency(optimized_requests, olmo2_requests, local_requests)
    print(f"{'AHA efficiency':<20} | {eff_total:>12} | {eff_output:>12} | {eff_req:>8}")

    print("="*60 + "\n")

    print("Legend:")
    print("  AHA overhead       = AHA (baseline) vs OLMo2 (negative = slower)")
    print("  Optimization gain  = AHA (optimized) vs AHA (baseline) (positive = faster)")
    print("  Local-only speedup = Local-only vs OLMo2 (shows max possible gain)")
    print("  AHA efficiency     = How close AHA gets to the local-only upper bound")
    print("                       (100% = matches local-only, 0% = same as OLMo2)")
    print()


def save_results(olmo2_results: dict, aha_baseline_results: dict,
                 aha_optimized_results: dict, local_only_results: dict,
                 args: argparse.Namespace):
    """Save benchmark results to JSON file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_prompts": args.num_prompts,
            "input_len": args.input_len,
            "output_len": args.output_len,
        },
        "olmo2_baseline": olmo2_results,
        "aha_baseline": aha_baseline_results,
        "aha_optimized": aha_optimized_results,
        "local_only": local_only_results,
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
        description="Benchmark OLMo2 vs AHA (baseline) vs AHA (optimized) vs Local-only"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=500,
        help="Number of prompts to benchmark (default: 500)"
    )
    parser.add_argument(
        "--input-len", type=int, default=512,
        help="Input sequence length (default: 512)"
    )
    parser.add_argument(
        "--output-len", type=int, default=512,
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
        "--skip-olmo2", action="store_true",
        help="Skip OLMo2 baseline benchmark"
    )
    parser.add_argument(
        "--skip-aha-baseline", action="store_true",
        help="Skip AHA baseline benchmark"
    )
    parser.add_argument(
        "--skip-aha-optimized", action="store_true",
        help="Skip AHA optimized benchmark"
    )
    parser.add_argument(
        "--skip-local-only", action="store_true",
        help="Skip local-only (upper bound) benchmark"
    )

    args = parser.parse_args()

    print("\n" + "="*90)
    print("              OLMo2 vs AHA vs Local-only Performance Comparison")
    print("="*90)
    print(f"Config: {args.num_prompts} prompts, {args.input_len} input len, {args.output_len} output len")
    print(f"\nModels:")
    print(f"  1. OLMo2 (baseline)    - Standard OLMo2 without AHA (lower bound)")
    print(f"  2. AHA (baseline)      - Frozen AHA implementation (olmo2_aha_baseline.py)")
    print(f"  3. AHA (optimized)     - Your optimized AHA (olmo2_aha.py)")
    print(f"  4. Local-only (upper)  - All sliding window attention (upper bound)")

    olmo2_results = {}
    aha_baseline_results = {}
    aha_optimized_results = {}
    local_only_results = {}

    # Run benchmarks
    if not args.skip_olmo2:
        olmo2_results = run_benchmark(OLMO2_MODEL, args, trust_remote_code=False)

    if not args.skip_aha_baseline:
        aha_baseline_results = run_benchmark(AHA_MODEL, args, trust_remote_code=True, use_baseline=True)

    if not args.skip_aha_optimized:
        aha_optimized_results = run_benchmark(AHA_MODEL, args, trust_remote_code=True, use_baseline=False)

    if not args.skip_local_only:
        local_only_results = run_local_only_benchmark(args)

    # Print comparison
    print_comparison(olmo2_results, aha_baseline_results, aha_optimized_results, local_only_results)

    # Save results
    if args.save_results:
        save_results(olmo2_results, aha_baseline_results, aha_optimized_results, local_only_results, args)


if __name__ == "__main__":
    main()
