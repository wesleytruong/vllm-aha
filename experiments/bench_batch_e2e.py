#!/usr/bin/env python3
"""Batch x e2e decode throughput at a fixed (long) context.

The throughput regime: how does the real-gate e2e tokens/s advantage behave as
batch grows? Two competing effects — the fixed per-step CPU/scheduling overhead
amortizes over more tokens (helps the ratio), while the decode-attention kernel
speedup itself erodes with batch (hurts it). This measures the net.

Reloads the model per batch (KV size differs). Reports total throughput
(batch x tokens/s) and the real-vs-full speedup at each batch.

Usage (from repo root):
    python experiments/bench_batch_e2e.py [--context 32768] [--batches 1,4,8,16]
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
    ap.add_argument("--context", type=int, default=cfg["batch_context"])
    ap.add_argument("--batches", type=str, default="",
                    help="comma list; default = config.batches")
    ap.add_argument("--out-dir", default="results/batch_e2e")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse existing per-cell JSONs instead of recomputing "
                         "them — lets a cancelled sweep resume on resubmit")
    args = ap.parse_args()

    batches = ([int(x) for x in args.batches.split(",")] if args.batches
               else cfg["batches"])
    os.makedirs(args.out_dir, exist_ok=True)
    model_max = max(cfg["model_max"], args.context + 256)

    print(f"# Batch x e2e: profile={cfg['profile']} context={args.context} "
          f"batches={batches}", flush=True)
    points = []
    for b in batches:
        out_json = os.path.join(args.out_dir, f"e2e_ctx{args.context}_b{b}.json")
        if args.skip_existing and os.path.exists(out_json):
            try:
                with open(out_json) as f:
                    points.append(json.load(f))
                print(f"\n=== batch={b} === SKIP (reusing {out_json})", flush=True)
                continue
            except (json.JSONDecodeError, OSError):
                print(f"  existing {out_json} unreadable; recomputing", flush=True)
        env = os.environ.copy()
        env.update(INPUT_LEN=str(args.context), AHA_BATCH=str(b),
                   AHA_MODEL_MAX=str(model_max), AHA_GPU_MEM=str(cfg["gpu_mem"]),
                   AHA_E2E_OUT=out_json)
        print(f"\n=== batch={b}  (KV~{128 * args.context * b / 1024**2:.0f}GB) ===",
              flush=True)
        r = subprocess.run([args.python, os.path.join(ROOT, "bench_e2e_decode.py")],
                           env=env, cwd=ROOT)
        if r.returncode != 0 or not os.path.exists(out_json):
            print(f"  FAILED batch={b} (rc={r.returncode}); skipping (OOM?)",
                  flush=True)
            continue
        with open(out_json) as f:
            points.append(json.load(f))

    summary = os.path.join(args.out_dir, f"batch_curve_ctx{args.context}.json")
    with open(summary, "w") as f:
        json.dump(points, f, indent=2)
    print(f"\n{'batch':>6} {'real tok/s':>11} {'full tok/s':>11} "
          f"{'real thru':>10} {'full thru':>10} {'speedup':>8}", flush=True)
    for pt in points:
        b = pt["batch"]
        rt = pt["configs"]["real-gate"]["tok_s"]
        ft = pt["configs"]["all-global"]["tok_s"]
        # tok_s in bench is per-stream (batch/per_step); total throughput = tok_s
        # already includes batch (toks = BATCH/per_step). Report both views.
        print(f"{b:>6} {rt:>11.0f} {ft:>11.0f} {rt:>10.0f} {ft:>10.0f} "
              f"{rt / ft:>7.2f}x", flush=True)
    print(f"\nwrote {summary}", flush=True)


if __name__ == "__main__":
    main()
