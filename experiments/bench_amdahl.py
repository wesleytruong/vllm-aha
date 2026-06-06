#!/usr/bin/env python3
"""Amdahl curve: e2e decode speedup vs context length.

Sweeps context lengths (from config.amdahl_contexts, GPU-aware), runs the e2e
decode benchmark at each (reloading the model per context since KV size differs),
and assembles the curve showing the real-gate e2e speedup rising toward the
kernel-level ceiling as attention's share of the decode step grows.

Each point also records the Amdahl decomposition derived purely from e2e timing:
  p_attn   = attention's fraction of the full-attention step
  s_kernel = real-gate attention time / full attention time (kernel speedup)
  e2e      = full-attention step / real-gate step  (what tokens/s actually sees)
Amdahl predicts e2e = 1 / ((1-p) + p*s); we report measured vs predicted.

Usage (run from the repo root so it finds bench_e2e_decode.py + .benchmark_datasets):
    python experiments/bench_amdahl.py [--batch 1] [--contexts 8192,32768,131072]
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from config import get_config  # noqa: E402


def main():
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--contexts", type=str, default="",
                    help="comma list; default = config.amdahl_contexts")
    ap.add_argument("--out-dir", default="results/amdahl")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    contexts = ([int(x) for x in args.contexts.split(",")] if args.contexts
                else cfg["amdahl_contexts"])
    os.makedirs(args.out_dir, exist_ok=True)
    # model_max must exceed the largest context + decode tokens
    model_max = max(cfg["model_max"], max(contexts) + 256)

    print(f"# Amdahl sweep: profile={cfg['profile']} contexts={contexts} "
          f"batch={args.batch} model_max={model_max}", flush=True)
    points = []
    for ctx in contexts:
        out_json = os.path.join(args.out_dir, f"e2e_ctx{ctx}_b{args.batch}.json")
        env = os.environ.copy()
        env.update(INPUT_LEN=str(ctx), AHA_BATCH=str(args.batch),
                   AHA_MODEL_MAX=str(model_max), AHA_GPU_MEM=str(cfg["gpu_mem"]),
                   AHA_E2E_OUT=out_json)
        print(f"\n=== context={ctx} ===", flush=True)
        r = subprocess.run([args.python, os.path.join(ROOT, "bench_e2e_decode.py")],
                           env=env, cwd=ROOT)
        if r.returncode != 0 or not os.path.exists(out_json):
            print(f"  FAILED ctx={ctx} (rc={r.returncode}); skipping", flush=True)
            continue
        with open(out_json) as f:
            points.append(json.load(f))

    # Assemble + print the curve
    summary = os.path.join(args.out_dir, f"amdahl_curve_b{args.batch}.json")
    with open(summary, "w") as f:
        json.dump(points, f, indent=2)
    print(f"\n{'ctx':>8} {'p_attn':>7} {'s_kern':>7} {'e2e_meas':>9} "
          f"{'e2e_pred':>9} {'real_tok/s':>11} {'full_tok/s':>11}", flush=True)
    for pt in points:
        p, s = pt["p_attn"], pt["s_kernel"]
        pred = 1.0 / ((1 - p) + p * s) if (1 - p + p * s) else float("nan")
        rt = pt["configs"]["real-gate"]["tok_s"]
        ft = pt["configs"]["all-global"]["tok_s"]
        print(f"{pt['context']:>8} {p:>7.3f} {s:>7.3f} {pt['e2e_speedup']:>8.2f}x "
              f"{pred:>8.2f}x {rt:>11.0f} {ft:>11.0f}", flush=True)
    print(f"\nwrote {summary}", flush=True)


if __name__ == "__main__":
    main()
