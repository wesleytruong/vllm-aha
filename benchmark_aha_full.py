#!/usr/bin/env python3
"""
Benchmark comparison script for OLMo2, AHA variants, and Local-only.

Compares seven models:
1. OLMo2 (baseline)    - Standard OLMo2 without AHA (lower bound performance)
2. AHA (baseline)      - Frozen AHA implementation
3. AHA (optimized)     - Your optimized AHA implementation
4. AHA (routed)        - Per-head routed AHA (skip unneeded attention types)
5. AHA (greedy)        - Greedy local-first AHA (1-2 kernel calls total)
6. AHA (flashinfer)    - FlashInfer per-head router kernel (fused 1-kernel decode)
7. Local-only          - All sliding window attention (upper bound performance)

Usage:
    python benchmark_aha_full.py [OPTIONS]

Options:
    --num-prompts N       Number of prompts to benchmark (default: 500)
    --input-len N         Input sequence length (default: 512)
    --output-len N        Output sequence length (default: 128)
    --max-num-seqs N      Maximum number of sequences in a batch (default: vLLM default)
    --save-results        Save results to JSON file
    --gpu-memory FLOAT    GPU memory utilization (default: 0.8)
    --skip-aha-main       Skip both AHA baseline and optimized benchmarks
    --skip-local-only     Skip local-only (upper bound) benchmark
    --skip-aha-routed     Skip AHA routed benchmark
    --skip-aha-greedy     Skip AHA greedy benchmark
    --skip-aha-flashinfer Skip AHA flashinfer benchmark
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import re
import tempfile
from datetime import datetime
from pathlib import Path


# Standard OLMo2 (no AHA)
OLMO2_MODEL = "allenai/OLMo-2-0425-1B"

# AHA model — all four variants use the same checkpoint, selected via
# --hf-overrides '{"attention_implementation": "..."}' at runtime.
AHA_MODEL = "xuan-luo/AHA-OLMO2"

# Local-only model (same as OLMo2, but uses local-only implementation)
LOCAL_ONLY_MODEL = "allenai/OLMo-2-0425-1B"

# Trained context length for both OLMo2 and AHA-OLMO2.
TRAINED_MAX_POS = 4096

# Tokenizer used to slice the real-text dataset to exact token lengths.
# OLMo2 and AHA-OLMO2 share the same tokenizer (AHA is fine-tuned from OLMo2),
# so a single JSONL works across all benchmarked variants.
PROMPT_TOKENIZER = OLMO2_MODEL

# Cache directory for pre-built prompt JSONLs.
DATASET_CACHE_DIR = Path(".benchmark_datasets")


def prepare_real_text_dataset(
    hf_path: str,
    hf_split: str,
    hf_config: str | None,
    text_column: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    tokenizer_name: str = PROMPT_TOKENIZER,
) -> Path:
    """Build a JSONL of `num_prompts` prompts, each exactly `input_len` tokens,
    sliced from a long-context HF dataset. Cached on disk by content hash so
    repeat runs don't re-tokenize.

    Returns the path to the JSONL file consumable by vLLM's `custom` dataset.
    """
    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        f"{hf_path}|{hf_config}|{hf_split}|{text_column}|{tokenizer_name}|"
        f"{num_prompts}|{input_len}|{output_len}".encode()
    ).hexdigest()[:16]
    out_path = DATASET_CACHE_DIR / f"{Path(hf_path).name}_{key}.jsonl"
    if out_path.exists():
        print(f"[dataset] reusing cached prompts: {out_path}")
        return out_path

    print(
        f"[dataset] building {num_prompts} prompts of {input_len} tokens from "
        f"{hf_path} (split={hf_split}, config={hf_config}, column={text_column})"
    )
    # Lazy imports so users without `datasets` installed can still use --dataset random.
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset(hf_path, hf_config, split=hf_split, streaming=True)

    needed_tokens = num_prompts * input_len
    token_buffer: list[int] = []
    for example in ds:
        text = example.get(text_column)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_buffer.extend(ids)
        if len(token_buffer) >= needed_tokens:
            break

    if len(token_buffer) < needed_tokens:
        raise RuntimeError(
            f"Dataset {hf_path}:{hf_split} only yielded {len(token_buffer)} "
            f"tokens; need {needed_tokens} for {num_prompts}x{input_len}. "
            "Try a different split or smaller --num-prompts/--input-len."
        )

    with out_path.open("w") as f:
        for i in range(num_prompts):
            chunk = token_buffer[i * input_len : (i + 1) * input_len]
            prompt = tokenizer.decode(chunk, skip_special_tokens=True)
            f.write(
                json.dumps({"prompt": prompt, "output_tokens": output_len}) + "\n"
            )

    print(f"[dataset] wrote {out_path}")
    return out_path


def build_hf_overrides(args: argparse.Namespace, impl: str | None) -> str | None:
    """Build the --hf-overrides JSON string for a single run.

    Returns None when no overrides apply (lets vLLM derive everything from
    the stock config).
    """
    overrides: dict = {}
    if impl is not None:
        overrides["attention_implementation"] = impl
    if args.max_model_len is not None:
        overrides["max_position_embeddings"] = args.max_model_len
    if args.rope_scaling != "none":
        overrides["rope_parameters"] = {
            "rope_type": args.rope_scaling,
            "factor": args.rope_scaling_factor,
            "original_max_position_embeddings": TRAINED_MAX_POS,
            "rope_theta": 500000.0,
        }
    return json.dumps(overrides) if overrides else None


def _build_throughput_cmd(
    model: str,
    args: argparse.Namespace,
    output_len: int,
    json_path: Path,
    trust_remote_code: bool,
) -> list[str]:
    """Construct the `vllm bench throughput` argv for one pass.

    `output_len` and `json_path` are parameterized so the same builder serves
    both the full run and the calibration pass (output_len=1).
    """
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "throughput",
        "--model",
        model,
        "--dtype",
        "half",
        "--num-prompts",
        str(args.num_prompts),
        "--gpu-memory-utilization",
        str(args.gpu_memory),
        "--output-json",
        str(json_path),
    ]

    dataset_path = getattr(args, "_dataset_path", None)
    if dataset_path is not None:
        # Real-text JSONL prepared up-front; same file used across all variants.
        cmd += [
            "--dataset-name",
            "custom",
            "--dataset-path",
            str(dataset_path),
            "--custom-output-len",
            str(output_len),
            "--skip-chat-template",
        ]
    else:
        cmd += [
            "--dataset-name",
            "random",
            "--random-input-len",
            str(args.input_len),
            "--random-output-len",
            str(output_len),
        ]

    if args.max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])

    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    return cmd


def _read_elapsed_time(json_path: Path) -> float | None:
    """Extract elapsed_time (seconds) from vLLM bench throughput's --output-json file."""
    try:
        data = json.loads(json_path.read_text())
        v = data.get("elapsed_time")
        return float(v) if v is not None else None
    except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
        return None


def run_benchmark(
    model: str,
    args: argparse.Namespace,
    trust_remote_code: bool = False,
    use_baseline: bool = False,
    use_local_only: bool = False,
    use_routed: bool = False,
    use_greedy: bool = False,
    use_flashinfer: bool = False,
) -> dict:
    """Run vllm benchmark and parse results.

    When `args.measure_phases` is set, a calibration pass with output_len=1
    runs first to capture T_prefill, then the full run captures T_total. The
    returned dict gains prefill/decode latency and tok/s fields.
    """
    # Display name for output
    if use_local_only:
        display_name = "Local-only (upper)"
    elif "OLMo-2" in model:
        display_name = "OLMo2 (baseline)"
    elif use_baseline:
        display_name = "AHA (baseline)"
    elif use_greedy:
        display_name = "AHA (greedy)"
    elif use_routed:
        display_name = "AHA (routed)"
    elif use_flashinfer:
        display_name = "AHA (flashinfer)"
    else:
        display_name = "AHA (optimized)"

    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {display_name}")
    print(f"Model path: {model}")
    if use_local_only:
        print(f"Using: olmo2_local_only.py (all sliding window)")
    elif use_baseline:
        print(f"Using: olmo2_aha.py impl=baseline (frozen reference)")
    elif use_greedy:
        print(f"Using: olmo2_aha.py impl=greedy (greedy local-first)")
    elif use_routed:
        print(f"Using: olmo2_aha.py impl=routed (per-head routed)")
    elif use_flashinfer:
        print(f"Using: olmo2_aha.py impl=flashinfer (FlashInfer router kernel)")
    elif "AHA" in model.upper():
        print(f"Using: olmo2_aha.py impl=dual (optimized)")
    print(f"{'=' * 70}\n")

    # Set environment for local-only (still uses env var); AHA variants use
    # --hf-overrides to set attention_implementation in the config.
    env = os.environ.copy()
    env.pop("VLLM_LOCAL_ONLY", None)

    impl: str | None = None
    if use_local_only:
        env["VLLM_LOCAL_ONLY"] = "1"
    elif use_baseline:
        impl = "baseline"
    elif use_greedy:
        impl = "greedy"
    elif use_routed:
        impl = "routed"
    elif use_flashinfer:
        impl = "flashinfer"
    elif "AHA" in model.upper():
        impl = "dual"

    overrides_json = build_hf_overrides(args, impl)

    def _do_pass(output_len: int, label: str) -> dict:
        """Run one subprocess pass and return parsed metrics (or error dict)."""
        with tempfile.NamedTemporaryFile(
            prefix=f"aha_{label}_", suffix=".json", delete=False
        ) as tmp:
            json_path = Path(tmp.name)
        try:
            cmd = _build_throughput_cmd(
                model, args, output_len, json_path, trust_remote_code
            )
            if overrides_json is not None:
                cmd += ["--hf-overrides", overrides_json]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minute timeout
                    env=env,
                )
            except subprocess.TimeoutExpired:
                print(f"ERROR: {label} pass timed out after 30 minutes")
                return {"error": "timeout"}
            except Exception as e:
                print(f"ERROR: {label} pass: {e}")
                return {"error": str(e)}

            output = result.stdout + result.stderr
            print(output)
            if result.returncode != 0:
                print(
                    f"ERROR: {label} pass failed with return code {result.returncode}"
                )
                return {"error": output[:500]}

            metrics = parse_benchmark_output(output)
            elapsed = _read_elapsed_time(json_path)
            if elapsed is not None:
                metrics["elapsed_time"] = elapsed
            return metrics
        finally:
            json_path.unlink(missing_ok=True)

    # Optional calibration pass (output_len=1) for prefill timing.
    # Approximation: T_calib ≈ T_prefill + 1 decode step (decode step is
    # negligible at long context, ~µs vs prefill seconds).
    calib_metrics: dict | None = None
    if getattr(args, "measure_phases", False):
        print(f"[phases] calibration pass: output_len=1 to isolate prefill")
        calib_metrics = _do_pass(output_len=1, label="calib")
        if "error" in calib_metrics:
            print(
                f"WARNING: calibration pass failed; phase metrics unavailable for "
                f"{display_name}"
            )

    # Full pass.
    metrics = _do_pass(output_len=args.output_len, label="full")
    metrics["model"] = model
    metrics["display_name"] = display_name
    if "error" in metrics:
        return metrics

    # Compute phase-decomposed metrics if calibration succeeded.
    if (
        calib_metrics is not None
        and "error" not in calib_metrics
        and "elapsed_time" in calib_metrics
        and "elapsed_time" in metrics
    ):
        T_calib = calib_metrics["elapsed_time"]
        T_total = metrics["elapsed_time"]
        T_decode = max(T_total - T_calib, 1e-9)
        total_prompt = metrics.get("total_prompt_tokens")
        total_output = metrics.get("total_output_tokens")
        if total_prompt and total_output:
            metrics["prefill_ms_per_token"] = T_calib * 1000.0 / total_prompt
            metrics["decode_ms_per_token"] = T_decode * 1000.0 / total_output
            metrics["prefill_tok_per_sec"] = total_prompt / T_calib
            metrics["decode_tok_per_sec"] = total_output / T_decode

    return metrics


def run_local_only_benchmark(args: argparse.Namespace) -> dict:
    """Run local-only benchmark (upper bound performance)."""
    return run_benchmark(
        LOCAL_ONLY_MODEL, args, trust_remote_code=False, use_local_only=True
    )


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


def print_comparison(
    olmo2_results: dict,
    aha_baseline_results: dict,
    aha_optimized_results: dict,
    aha_routed_results: dict,
    aha_greedy_results: dict,
    aha_flashinfer_results: dict,
    local_only_results: dict,
):
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 90)
    print("                    OLMo2 vs AHA vs Local-only Benchmark Results")
    print("=" * 90)

    # Check for errors
    all_results = [
        ("OLMo2 (baseline)", olmo2_results),
        ("AHA (baseline)", aha_baseline_results),
        ("AHA (optimized)", aha_optimized_results),
        ("AHA (routed)", aha_routed_results),
        ("AHA (greedy)", aha_greedy_results),
        ("AHA (flashinfer)", aha_flashinfer_results),
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

    # Phase-decomposed columns appear only when at least one variant has them.
    has_phases = any(
        results
        and "error" not in results
        and results.get("prefill_ms_per_token") is not None
        for _, results in all_results
    )

    def fmt_phase(v):
        if isinstance(v, (int, float)):
            return f"{v:.4f}"
        return "—"

    phase_header = (
        f" | {'Prefill ms/tok':>14} | {'Decode ms/tok':>14}" if has_phases else ""
    )
    phase_sep = "-+-" + "-" * 14 + "-+-" + "-" * 14 if has_phases else ""
    phase_blank = f" | {'':>14} | {'':>14}" if has_phases else ""

    def variant_phase_cells(results) -> str:
        if not has_phases:
            return ""
        p = get_metric(results, "prefill_ms_per_token")
        d = get_metric(results, "decode_ms_per_token")
        return f" | {fmt_phase(p):>14} | {fmt_phase(d):>14}"

    # Print header
    print(
        f"\n{'Model':<20} | {'Total tok/s':>12} | {'Output tok/s':>12} | {'Req/s':>8}{phase_header}"
    )
    print(
        "-" * 20 + "-+-" + "-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 8 + phase_sep
    )

    # Get metrics
    olmo2_total = get_metric(olmo2_results, "throughput_tokens_per_sec")
    olmo2_output = get_metric(olmo2_results, "output_tokens_per_sec")
    olmo2_requests = get_metric(olmo2_results, "throughput_requests_per_sec")

    baseline_total = get_metric(aha_baseline_results, "throughput_tokens_per_sec")
    baseline_output = get_metric(aha_baseline_results, "output_tokens_per_sec")
    baseline_requests = get_metric(aha_baseline_results, "throughput_requests_per_sec")

    optimized_total = get_metric(aha_optimized_results, "throughput_tokens_per_sec")
    optimized_output = get_metric(aha_optimized_results, "output_tokens_per_sec")
    optimized_requests = get_metric(
        aha_optimized_results, "throughput_requests_per_sec"
    )

    local_total = get_metric(local_only_results, "throughput_tokens_per_sec")
    local_output = get_metric(local_only_results, "output_tokens_per_sec")
    local_requests = get_metric(local_only_results, "throughput_requests_per_sec")

    print(
        f"{'OLMo2 (baseline)':<20} | {fmt(olmo2_total):>12} | {fmt(olmo2_output):>12} | {fmt(olmo2_requests):>8}{variant_phase_cells(olmo2_results)}"
    )
    print(
        f"{'AHA (baseline)':<20} | {fmt(baseline_total):>12} | {fmt(baseline_output):>12} | {fmt(baseline_requests):>8}{variant_phase_cells(aha_baseline_results)}"
    )
    routed_total = get_metric(aha_routed_results, "throughput_tokens_per_sec")
    routed_output = get_metric(aha_routed_results, "output_tokens_per_sec")
    routed_requests = get_metric(aha_routed_results, "throughput_requests_per_sec")

    greedy_total = get_metric(aha_greedy_results, "throughput_tokens_per_sec")
    greedy_output = get_metric(aha_greedy_results, "output_tokens_per_sec")
    greedy_requests = get_metric(aha_greedy_results, "throughput_requests_per_sec")

    fi_total = get_metric(aha_flashinfer_results, "throughput_tokens_per_sec")
    fi_output = get_metric(aha_flashinfer_results, "output_tokens_per_sec")
    fi_requests = get_metric(aha_flashinfer_results, "throughput_requests_per_sec")

    print(
        f"{'AHA (optimized)':<20} | {fmt(optimized_total):>12} | {fmt(optimized_output):>12} | {fmt(optimized_requests):>8}{variant_phase_cells(aha_optimized_results)}"
    )
    print(
        f"{'AHA (routed)':<20} | {fmt(routed_total):>12} | {fmt(routed_output):>12} | {fmt(routed_requests):>8}{variant_phase_cells(aha_routed_results)}"
    )
    print(
        f"{'AHA (greedy)':<20} | {fmt(greedy_total):>12} | {fmt(greedy_output):>12} | {fmt(greedy_requests):>8}{variant_phase_cells(aha_greedy_results)}"
    )
    print(
        f"{'AHA (flashinfer)':<20} | {fmt(fi_total):>12} | {fmt(fi_output):>12} | {fmt(fi_requests):>8}{variant_phase_cells(aha_flashinfer_results)}"
    )
    print(
        f"{'Local-only (upper)':<20} | {fmt(local_total):>12} | {fmt(local_output):>12} | {fmt(local_requests):>8}{variant_phase_cells(local_only_results)}"
    )

    # Calculate differences
    print(
        "-" * 20 + "-+-" + "-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 8 + phase_sep
    )

    def calc_diff(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b != 0:
            return f"{((a - b) / b) * 100:+.1f}%"
        return "N/A"

    # AHA baseline vs OLMo2 (shows AHA overhead)
    diff1_total = calc_diff(baseline_total, olmo2_total)
    diff1_output = calc_diff(baseline_output, olmo2_output)
    diff1_req = calc_diff(baseline_requests, olmo2_requests)
    print(
        f"{'AHA overhead':<20} | {diff1_total:>12} | {diff1_output:>12} | {diff1_req:>8}{phase_blank}"
    )

    # AHA optimized vs AHA baseline (shows optimization gain)
    diff2_total = calc_diff(optimized_total, baseline_total)
    diff2_output = calc_diff(optimized_output, baseline_output)
    diff2_req = calc_diff(optimized_requests, baseline_requests)
    print(
        f"{'Optimization gain':<20} | {diff2_total:>12} | {diff2_output:>12} | {diff2_req:>8}{phase_blank}"
    )

    # AHA routed vs AHA baseline (shows routed gain)
    diff_routed_total = calc_diff(routed_total, baseline_total)
    diff_routed_output = calc_diff(routed_output, baseline_output)
    diff_routed_req = calc_diff(routed_requests, baseline_requests)
    print(
        f"{'Routed gain':<20} | {diff_routed_total:>12} | {diff_routed_output:>12} | {diff_routed_req:>8}{phase_blank}"
    )

    # AHA greedy vs AHA baseline (shows greedy gain)
    diff_greedy_total = calc_diff(greedy_total, baseline_total)
    diff_greedy_output = calc_diff(greedy_output, baseline_output)
    diff_greedy_req = calc_diff(greedy_requests, baseline_requests)
    print(
        f"{'Greedy gain':<20} | {diff_greedy_total:>12} | {diff_greedy_output:>12} | {diff_greedy_req:>8}{phase_blank}"
    )

    # AHA flashinfer vs AHA baseline (shows flashinfer gain)
    diff_fi_total = calc_diff(fi_total, baseline_total)
    diff_fi_output = calc_diff(fi_output, baseline_output)
    diff_fi_req = calc_diff(fi_requests, baseline_requests)
    print(
        f"{'FlashInfer gain':<20} | {diff_fi_total:>12} | {diff_fi_output:>12} | {diff_fi_req:>8}{phase_blank}"
    )

    # AHA flashinfer vs OLMo2 (end-to-end speedup over stock OLMo2)
    diff_fi_olmo_total = calc_diff(fi_total, olmo2_total)
    diff_fi_olmo_output = calc_diff(fi_output, olmo2_output)
    diff_fi_olmo_req = calc_diff(fi_requests, olmo2_requests)
    print(
        f"{'FlashInfer vs OLMo2':<20} | {diff_fi_olmo_total:>12} | {diff_fi_olmo_output:>12} | {diff_fi_olmo_req:>8}{phase_blank}"
    )

    # Local-only vs OLMo2 (shows max possible gain)
    diff3_total = calc_diff(local_total, olmo2_total)
    diff3_output = calc_diff(local_output, olmo2_output)
    diff3_req = calc_diff(local_requests, olmo2_requests)
    print(
        f"{'Local-only speedup':<20} | {diff3_total:>12} | {diff3_output:>12} | {diff3_req:>8}{phase_blank}"
    )

    # AHA efficiency - how close AHA gets to local-only upper bound
    # Calculated as: (AHA_optimized - OLMo2) / (Local_only - OLMo2) * 100
    def calc_efficiency(optimized, baseline, upper):
        if (
            isinstance(optimized, (int, float))
            and isinstance(baseline, (int, float))
            and isinstance(upper, (int, float))
            and upper != baseline
        ):
            efficiency = ((optimized - baseline) / (upper - baseline)) * 100
            return f"{efficiency:.1f}%"
        return "N/A"

    eff_total = calc_efficiency(optimized_total, olmo2_total, local_total)
    eff_output = calc_efficiency(optimized_output, olmo2_output, local_output)
    eff_req = calc_efficiency(optimized_requests, olmo2_requests, local_requests)
    print(
        f"{'AHA opt efficiency':<20} | {eff_total:>12} | {eff_output:>12} | {eff_req:>8}{phase_blank}"
    )

    eff_r_total = calc_efficiency(routed_total, olmo2_total, local_total)
    eff_r_output = calc_efficiency(routed_output, olmo2_output, local_output)
    eff_r_req = calc_efficiency(routed_requests, olmo2_requests, local_requests)
    print(
        f"{'AHA routed eff':<20} | {eff_r_total:>12} | {eff_r_output:>12} | {eff_r_req:>8}{phase_blank}"
    )

    eff_g_total = calc_efficiency(greedy_total, olmo2_total, local_total)
    eff_g_output = calc_efficiency(greedy_output, olmo2_output, local_output)
    eff_g_req = calc_efficiency(greedy_requests, olmo2_requests, local_requests)
    print(
        f"{'AHA greedy eff':<20} | {eff_g_total:>12} | {eff_g_output:>12} | {eff_g_req:>8}{phase_blank}"
    )

    eff_f_total = calc_efficiency(fi_total, olmo2_total, local_total)
    eff_f_output = calc_efficiency(fi_output, olmo2_output, local_output)
    eff_f_req = calc_efficiency(fi_requests, olmo2_requests, local_requests)
    print(
        f"{'AHA flashinfer eff':<20} | {eff_f_total:>12} | {eff_f_output:>12} | {eff_f_req:>8}{phase_blank}"
    )

    print("=" * (60 + (34 if has_phases else 0)) + "\n")

    print("Legend:")
    print("  AHA overhead       = AHA (baseline) vs OLMo2 (negative = slower)")
    print(
        "  Optimization gain  = AHA (optimized) vs AHA (baseline) (positive = faster)"
    )
    print("  Routed gain        = AHA (routed) vs AHA (baseline) (positive = faster)")
    print("  Greedy gain        = AHA (greedy) vs AHA (baseline) (positive = faster)")
    print(
        "  FlashInfer gain    = AHA (flashinfer) vs AHA (baseline) (positive = faster)"
    )
    print(
        "  FlashInfer vs OLMo2= AHA (flashinfer) vs OLMo2 (end-to-end speedup over stock OLMo2)"
    )
    print("  Local-only speedup = Local-only vs OLMo2 (shows max possible gain)")
    print(
        "  AHA opt efficiency = How close AHA optimized gets to local-only upper bound"
    )
    print("  AHA routed eff     = How close AHA routed gets to local-only upper bound")
    print("  AHA greedy eff     = How close AHA greedy gets to local-only upper bound")
    print(
        "  AHA flashinfer eff = How close AHA flashinfer gets to local-only upper bound"
    )
    print("                       (100% = matches local-only, 0% = same as OLMo2)")
    print()


def save_results(
    olmo2_results: dict,
    aha_baseline_results: dict,
    aha_optimized_results: dict,
    aha_routed_results: dict,
    aha_greedy_results: dict,
    aha_flashinfer_results: dict,
    local_only_results: dict,
    args: argparse.Namespace,
):
    """Save benchmark results to JSON file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_prompts": args.num_prompts,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_num_seqs": args.max_num_seqs,
            "measure_phases": getattr(args, "measure_phases", False),
        },
        "olmo2_baseline": olmo2_results,
        "aha_baseline": aha_baseline_results,
        "aha_optimized": aha_optimized_results,
        "aha_routed": aha_routed_results,
        "aha_greedy": aha_greedy_results,
        "aha_flashinfer": aha_flashinfer_results,
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
        "--num-prompts",
        type=int,
        default=500,
        help="Number of prompts to benchmark (default: 500)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=128,
        help="Input sequence length (default: 512)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1024,
        help="Output sequence length (default: 128)",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences in a batch (default: vLLM default)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Override max context length. When set, also overrides "
            "max_position_embeddings in the HF config. Unset keeps the "
            f"trained default ({TRAINED_MAX_POS})."
        ),
    )
    parser.add_argument(
        "--rope-scaling",
        choices=["none", "linear", "yarn"],
        default="none",
        help=(
            "RoPE scaling type to inject via hf_overrides. Required for "
            "coherent generation beyond the trained length of "
            f"{TRAINED_MAX_POS}. Does not affect speed measurements."
        ),
    )
    parser.add_argument(
        "--rope-scaling-factor",
        type=float,
        default=None,
        help=(
            "RoPE scaling factor. Defaults to ceil(max_model_len / "
            f"{TRAINED_MAX_POS} * 2) / 2 when --rope-scaling is set."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["pg19", "random", "hf"],
        default="pg19",
        help=(
            "Prompt source. 'pg19' (default) slices real text from "
            "emozilla/pg19-test to exact token lengths. 'random' uses vLLM's "
            "random-token generator (legacy). 'hf' lets you point at any HF "
            "dataset via --hf-dataset-path."
        ),
    )
    parser.add_argument(
        "--hf-dataset-path",
        default=None,
        help=(
            "HF dataset id when --dataset=hf (e.g. 'pg19', 'tau/scrolls'). "
            "Ignored otherwise."
        ),
    )
    parser.add_argument(
        "--hf-dataset-config",
        default=None,
        help="Optional HF dataset config/subset name.",
    )
    parser.add_argument(
        "--hf-dataset-split",
        default="test",
        help="HF dataset split (default: test).",
    )
    parser.add_argument(
        "--hf-text-column",
        default="text",
        help="Column to read text from (default: 'text').",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to benchmark_results.json",
    )
    parser.add_argument(
        "--measure-phases",
        action="store_true",
        help=(
            "Run a calibration pass with output_len=1 per variant to isolate "
            "prefill time. Adds 'Prefill ms/tok' and 'Decode ms/tok' columns "
            "to the summary. Roughly doubles per-variant runtime when input "
            "length is short; ~1.05-1.2x overhead at long-context shapes."
        ),
    )
    parser.add_argument(
        "--skip-olmo2", action="store_true", help="Skip OLMo2 baseline benchmark"
    )
    parser.add_argument(
        "--skip-aha-main",
        action="store_true",
        help="Skip both AHA baseline and AHA optimized benchmarks",
    )
    parser.add_argument(
        "--skip-aha-baseline", action="store_true", help="Skip AHA baseline benchmark", default=True
    )
    parser.add_argument(
        "--skip-aha-optimized", action="store_true", help="Skip AHA optimized benchmark", default=True
    )
    parser.add_argument(
        "--skip-aha-routed", action="store_true", help="Skip AHA routed benchmark", default=True
    )
    parser.add_argument(
        "--skip-aha-greedy", action="store_true", help="Skip AHA greedy benchmark", default=True
    )
    parser.add_argument(
        "--skip-aha-flashinfer",
        action="store_true",
        help="Skip AHA flashinfer benchmark",
    )
    parser.add_argument(
        "--skip-local-only",
        action="store_true",
        help="Skip local-only (upper bound) benchmark",
    )

    args = parser.parse_args()

    if args.rope_scaling != "none" and args.rope_scaling_factor is None:
        effective_len = args.max_model_len or TRAINED_MAX_POS
        ratio = max(1.0, effective_len / TRAINED_MAX_POS)
        args.rope_scaling_factor = math.ceil(ratio * 2) / 2

    if args.rope_scaling != "none" and args.max_model_len is None:
        print(
            "WARNING: --rope-scaling set without --max-model-len; scaling "
            "will apply but max context remains at the trained length."
        )

    # Build the real-text prompt JSONL once and reuse across all variants.
    args._dataset_path = None
    if args.dataset == "pg19":
        args._dataset_path = prepare_real_text_dataset(
            hf_path="emozilla/pg19-test",
            hf_split="test",
            hf_config=None,
            text_column="text",
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
        )
    elif args.dataset == "hf":
        if args.hf_dataset_path is None:
            parser.error("--dataset=hf requires --hf-dataset-path")
        args._dataset_path = prepare_real_text_dataset(
            hf_path=args.hf_dataset_path,
            hf_split=args.hf_dataset_split,
            hf_config=args.hf_dataset_config,
            text_column=args.hf_text_column,
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
        )

    print("\n" + "=" * 90)
    print("              OLMo2 vs AHA vs Local-only Performance Comparison")
    print("=" * 90)
    print(
        f"Config: {args.num_prompts} prompts, {args.input_len} input len, {args.output_len} output len"
    )
    print(
        f"Max num seqs: {args.max_num_seqs if args.max_num_seqs is not None else 'vLLM default'}"
    )
    print(f"\nModels:")
    print(f"  1. OLMo2 (baseline)    - Standard OLMo2 without AHA (lower bound)")
    print(
        f"  2. AHA (baseline)      - Frozen AHA reference (olmo2_aha.py impl=baseline)"
    )
    print(f"  3. AHA (optimized)     - Dual-kernel AHA (olmo2_aha.py impl=dual)")
    print(f"  4. AHA (routed)        - Per-head routed AHA (olmo2_aha.py impl=routed)")
    print(
        f"  5. AHA (greedy)        - Greedy local-first AHA (olmo2_aha.py impl=greedy)"
    )
    print(
        f"  6. AHA (flashinfer)    - FlashInfer router kernel (olmo2_aha.py impl=flashinfer)"
    )
    print(f"  7. Local-only (upper)  - All sliding window attention (upper bound)")

    olmo2_results = {}
    aha_baseline_results = {}
    aha_optimized_results = {}
    aha_routed_results = {}
    aha_greedy_results = {}
    aha_flashinfer_results = {}
    local_only_results = {}

    skip_aha_baseline = args.skip_aha_main or args.skip_aha_baseline
    skip_aha_optimized = args.skip_aha_main or args.skip_aha_optimized

    # Run benchmarks
    if not args.skip_olmo2:
        olmo2_results = run_benchmark(OLMO2_MODEL, args, trust_remote_code=False)

    if not skip_aha_baseline:
        aha_baseline_results = run_benchmark(
            AHA_MODEL, args, trust_remote_code=True, use_baseline=True
        )

    if not skip_aha_optimized:
        aha_optimized_results = run_benchmark(
            AHA_MODEL, args, trust_remote_code=True, use_baseline=False
        )

    if not args.skip_aha_routed:
        aha_routed_results = run_benchmark(
            AHA_MODEL, args, trust_remote_code=True, use_routed=True
        )

    if not args.skip_aha_greedy:
        aha_greedy_results = run_benchmark(
            AHA_MODEL, args, trust_remote_code=True, use_greedy=True
        )

    if not args.skip_aha_flashinfer:
        aha_flashinfer_results = run_benchmark(
            AHA_MODEL, args, trust_remote_code=True, use_flashinfer=True
        )

    if not args.skip_local_only:
        local_only_results = run_local_only_benchmark(args)

    # Print comparison
    print_comparison(
        olmo2_results,
        aha_baseline_results,
        aha_optimized_results,
        aha_routed_results,
        aha_greedy_results,
        aha_flashinfer_results,
        local_only_results,
    )

    # Save results
    if args.save_results:
        save_results(
            olmo2_results,
            aha_baseline_results,
            aha_optimized_results,
            aha_routed_results,
            aha_greedy_results,
            aha_flashinfer_results,
            local_only_results,
            args,
        )


if __name__ == "__main__":
    main()
