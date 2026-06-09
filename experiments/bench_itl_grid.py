#!/usr/bin/env python3
"""ITL/TPOT context x batch matrix driver.

Runs bench_itl.py over a context x batch grid with batch held CONSTANT across
contexts, so the latency-speedup matrix isolates the context effect (down a
column) from the batch effect (across a row). Per-cell gpu_mem is sized so the
KV pool holds that cell's KV while leaving headroom for the long-context
transient prefill/decode workspace (which OOMs at a fixed high gpu_mem). Cells
that still don't fit (e.g. 128K x B=8 on 80GB) are skipped.

Usage (from repo root):
    python experiments/bench_itl_grid.py [--contexts 8192,32768] [--batches 1,2,4]
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from config import get_config, kv_gb  # noqa: E402


def gpu_mem_for(ctx, batch, hbm_gb):
    # pool must hold this cell's KV + weights(~2.5GB) + slack; the rest of HBM
    # stays free for the transient workspace (which scales with context).
    needed = kv_gb(ctx, batch) + 2.5 + 2.0
    return round(min(0.9, max(0.3, needed / hbm_gb)), 2)


def main():
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=str, default="",
                    help="comma list; default = config.amdahl_contexts")
    ap.add_argument("--batches", type=str, default="",
                    help="comma list, held constant across contexts; "
                         "default = config.batches")
    ap.add_argument("--gen-tokens", type=int, default=129)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out-dir", default="results/itl")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    contexts = ([int(x) for x in args.contexts.split(",")] if args.contexts
                else cfg["amdahl_contexts"])
    batches = ([int(x) for x in args.batches.split(",")] if args.batches
               else cfg["batches"])
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"# ITL matrix: profile={cfg['profile']} contexts={contexts} "
          f"batches={batches} (constant across contexts)", flush=True)

    for ctx in contexts:
        for b in batches:
            kv = kv_gb(ctx, b)
            if kv + 3.0 > cfg["hbm_gb"]:          # KV alone won't fit -> skip
                print(f"  SKIP ctx={ctx} B={b}: KV~{kv:.0f}GB > HBM", flush=True)
                continue
            gm = gpu_mem_for(ctx, b, cfg["hbm_gb"])
            out = os.path.join(args.out_dir, f"grid_ctx{ctx}_b{b}.json")
            mm = max(cfg["model_max"], ctx + args.gen_tokens + 64)
            env = os.environ.copy()
            print(f"\n=== ctx={ctx} B={b}  KV~{kv:.0f}GB  gpu_mem={gm} ===", flush=True)
            r = subprocess.run(
                [args.python, os.path.join(ROOT, "experiments/bench_itl.py"),
                 "--ctx", str(ctx), "--batch", str(b), "--gpu-mem", str(gm),
                 "--gen-tokens", str(args.gen_tokens), "--reps", str(args.reps),
                 "--model-max", str(mm), "--out", out],
                env=env, cwd=ROOT)
            if r.returncode != 0:
                print(f"  FAILED ctx={ctx} B={b} (rc={r.returncode}; OOM?)", flush=True)

    print("\n# matrix done -> run `python experiments/make_figures.py` for the table",
          flush=True)


if __name__ == "__main__":
    main()
