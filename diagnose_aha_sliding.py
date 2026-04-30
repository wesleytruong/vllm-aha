#!/usr/bin/env python3
"""Diagnose whether AHA's shared KV cache is evicted by the local sliding window.

Hypothesis: _AHADualAttention gives local_attn per_layer_sliding_window=W and
shares KV cache with global_attn. vLLM's KV manager may evict entries past W
in the SHARED cache, which silently breaks the global path but makes AHA look
dramatically faster than OLMo2 at long context.

Sweep local_window_size and watch three signals:
  - "Maximum concurrency for X tokens per request" (vLLM startup log)
  - Peak "GPU KV cache usage" during decode (per-step log)
  - Final throughput

If the hypothesis holds:
  - window=128   → high concurrency, low peak KV%, high throughput
  - window=MAX   → OLMo2-like concurrency, high peak KV%, OLMo2-like throughput
If refuted: all windows give ~same numbers.
"""

import json
import os
import re
import subprocess
import sys
import time

AHA_MODEL = "xuan-luo/AHA-OLMO2"
OLMO2_MODEL = "allenai/OLMo-2-0425-1B"

INPUT_LEN = 8192
OUTPUT_LEN = 16000
MAX_MODEL_LEN = 32000
NUM_PROMPTS = 8
GPU_MEM = 0.8

RUNS = [
    ("OLMo2 (ref)",    OLMO2_MODEL, False, {}),
    ("AHA win=128",    AHA_MODEL,   True,  {"attention_implementation": "baseline", "local_window_size": 128}),
    ("AHA win=2048",   AHA_MODEL,   True,  {"attention_implementation": "baseline", "local_window_size": 2048}),
    ("AHA win=8192",   AHA_MODEL,   True,  {"attention_implementation": "baseline", "local_window_size": 8192}),
    ("AHA win=32000",  AHA_MODEL,   True,  {"attention_implementation": "baseline", "local_window_size": MAX_MODEL_LEN}),
]


def run_one(label, model, trust_remote, overrides):
    hf = dict(overrides)
    hf["max_position_embeddings"] = MAX_MODEL_LEN

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
        "--model", model, "--dtype", "half",
        "--num-prompts", str(NUM_PROMPTS),
        "--dataset-name", "random",
        "--random-input-len", str(INPUT_LEN),
        "--random-output-len", str(OUTPUT_LEN),
        "--gpu-memory-utilization", str(GPU_MEM),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--hf-overrides", json.dumps(hf),
    ]
    if trust_remote:
        cmd.append("--trust-remote-code")

    banner = "=" * 70
    print(f"\n{banner}\n[{label}]  overrides={hf}\n{banner}", flush=True)

    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - t0
    out = p.stdout + p.stderr

    m = {"label": label, "elapsed_s": round(elapsed, 1)}
    if p.returncode != 0:
        m["error"] = f"rc={p.returncode}"
        print(out[-2000:])
        return m

    if r := re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", out):
        m["max_concurrency"] = float(r.group(2))
        m["accounted_tokens_per_req"] = int(r.group(1).replace(",", ""))
    if r := re.search(r"GPU KV cache size: ([\d,]+) tokens", out):
        m["kv_cache_tokens"] = int(r.group(1).replace(",", ""))
    if r := re.search(r"Available KV cache memory: ([\d.]+) GiB", out):
        m["kv_gib"] = float(r.group(1))
    if r := re.search(
        r"Throughput:\s*([\d.]+)\s*requests/s,\s*([\d.]+)\s*total tokens/s,\s*([\d.]+)\s*output tokens/s",
        out,
    ):
        m["req_per_s"] = float(r.group(1))
        m["total_tok_s"] = float(r.group(2))
        m["output_tok_s"] = float(r.group(3))

    usages = [float(x) for x in re.findall(r"GPU KV cache usage: ([\d.]+)%", out)]
    if usages:
        m["peak_kv_usage_pct"] = max(usages)
        m["mean_kv_usage_pct"] = round(sum(usages) / len(usages), 1)

    print(
        f"  → concurrency={m.get('max_concurrency', '?')}x "
        f"peak_KV={m.get('peak_kv_usage_pct', '?')}% "
        f"out_tok/s={m.get('output_tok_s', '?')} "
        f"elapsed={m['elapsed_s']}s",
        flush=True,
    )
    return m


def main():
    results = []
    for label, model, trust, ov in RUNS:
        results.append(run_one(label, model, trust, ov))

    print("\n" + "=" * 110)
    print("SLIDING-WINDOW CACHE EVICTION DIAGNOSTIC")
    print(f"input_len={INPUT_LEN} output_len={OUTPUT_LEN} max_model_len={MAX_MODEL_LEN} num_prompts={NUM_PROMPTS}")
    print("=" * 110)
    hdr = (
        f"{'Config':<18}{'Conc':>8}{'Acct toks/req':>16}"
        f"{'KV tokens':>12}{'Peak KV%':>10}{'Mean KV%':>10}"
        f"{'Tot tok/s':>12}{'Out tok/s':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['label']:<18} FAILED ({r['error']})")
            continue
        print(
            f"{r['label']:<18}"
            f"{r.get('max_concurrency', 0):>7.2f}x"
            f"{r.get('accounted_tokens_per_req', 0):>16,}"
            f"{r.get('kv_cache_tokens', 0):>12,}"
            f"{r.get('peak_kv_usage_pct', 0):>9.1f}%"
            f"{r.get('mean_kv_usage_pct', 0):>9.1f}%"
            f"{r.get('total_tok_s', 0):>12.1f}"
            f"{r.get('output_tok_s', 0):>12.1f}"
        )

    print(
        "\nInterpretation:\n"
        "  Hypothesis confirmed if concurrency DROPS and peak_KV% RISES monotonically with window.\n"
        "  If win=32000 matches OLMo2 on all three metrics → shared cache is definitely evicted by local window.\n"
        "  If all AHA runs look identical → hypothesis refuted; speedup comes from something else."
    )

    out_path = "/tmp/aha_sliding_diag.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
