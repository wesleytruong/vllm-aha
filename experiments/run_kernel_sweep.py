#!/usr/bin/env python3
"""Kernel-direct decode sweep driver (nsys), GPU-profile aware.

Wraps nsys_sweep_p.py: for each context in the GPU profile, runs the per-context
feasible batch list (config.ctx_batches) across all routing configs, with the
gate-routing probe on (records actual SWA% per cell), then parses the result with
the grid/time-clean decode-direct extractor.

Configs: dense-fi (full-attn ref) / aha-global (forced full) / aha-local (forced
windowed) / aha-half (50%) / aha-flashinfer (real gate). The override now works
with the compile cache on (runtime buffers), so no VLLM_DISABLE_COMPILE_CACHE.

Requires nsys on PATH. Usage (from repo root):
    python experiments/run_kernel_sweep.py [--out-dir results/nsys_cachefix]
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from config import get_config  # noqa: E402

CFGS = ["dense-fi", "aha-global", "aha-local", "aha-half", "aha-flashinfer",
        "local-only"]  # local-only = true native-SWA ceiling (kernel_summary)


def main():
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/nsys_cachefix")
    ap.add_argument("--cfgs", nargs="+", default=CFGS)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    if shutil.which("nsys") is None:
        sys.exit("nsys not on PATH — load the Nsight Systems module (see "
                 "experiments/slurm/env.sh) or run the e2e benchmarks instead.")

    ctx_batches = cfg["ctx_batches"]
    print(f"# kernel sweep: profile={cfg['profile']} "
          f"ctx_batches={ctx_batches}", flush=True)
    env = os.environ.copy()
    env["VLLM_AHA_PROBE_READOUT"] = "1"  # record routing per cell
    env["AHA_PFC_DECODE"] = "1"          # prefix-cache prime -> clean decode-only
    #                                      trace (isolates decode at B>1; prefill
    #                                      otherwise interleaves and contaminates).

    for ctx, batches in ctx_batches.items():
        model_max = max(cfg["model_max"], ctx + 256)
        cmd = [args.python, os.path.join(ROOT, "nsys_sweep_p.py"),
               "--cfgs", *args.cfgs,
               "--contexts", str(ctx),
               "--batches", *[str(b) for b in batches],
               "--max-tokens-pair", "33", "33",   # only mt=33 rep needed
               "--model-max", str(model_max),
               "--gpu-mem", str(cfg["gpu_mem"]),
               "--out-dir", args.out_dir,
               "--cell-timeout", "1800"]
        print(f"\n=== context={ctx} batches={batches} ===", flush=True)
        subprocess.run(cmd, env=env, cwd=ROOT)

    # Parse (grid/time-clean decode-direct)
    print("\n=== parsing decode-direct ===", flush=True)
    subprocess.run([args.python, os.path.join(ROOT, "parse_nsys_decode_direct.py"),
                    "--in-dir", args.out_dir,
                    "--out", "results/nsys_cachefix_direct.json"],
                   env=env, cwd=ROOT)


if __name__ == "__main__":
    main()
