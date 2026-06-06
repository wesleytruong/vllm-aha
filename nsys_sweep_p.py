#!/usr/bin/env python3
"""Driver for the nsys p-verification sweep.

Loops over (cfg, ctx, max_tokens) cells, invoking nsys profile per cell to
capture the graphs-on decode workload defined in nsys_profile_decode.py. Each
cell is captured twice — max_tokens=1 (prefill+1 decode) and max_tokens=N
(prefill+N decodes) — so parse_nsys_p.py can do the long-minus-short subtraction
that isolates pure decode-step attn time and wall time.

NOTE on permissions:
  This box has NVreg_RestrictProfilingToAdminUsers=1 (CUPTI requires root).
  Run THIS driver under sudo so nsys can collect GPU kernel data:
      ! sudo -E .venv/bin/python nsys_sweep_p.py [args]
  Output .nsys-rep files will be root-owned but world-readable. parse_nsys_p.py
  runs as your user against them.

Cell naming: <cfg>_ctx<INPUT_LEN>_mt<MAX_TOKENS>.nsys-rep
Existing .nsys-rep files are skipped (idempotent reruns).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

VALID_CFGS = ["dense", "dense-fi", "aha-global", "aha-half",
              "aha-flashinfer", "aha-local", "local-only"]


def cell_base(cfg: str, ctx: int, mt: int, batch: int = 1) -> str:
    # Keep B=1 names unchanged (back-compat with existing .nsys-rep files);
    # only tag the batch axis when B>1.
    suffix = "" if batch == 1 else f"_b{batch}"
    return f"{cfg}_ctx{ctx}_mt{mt}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", nargs="+", default=VALID_CFGS, choices=VALID_CFGS)
    ap.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument(
        "--max-tokens-pair", type=int, nargs=2, default=[1, 33],
        metavar=("SHORT", "LONG"),
        help="two max_tokens values for the long-minus-short subtraction",
    )
    ap.add_argument("--batches", type=int, nargs="+", default=[1],
                    help="decode batch sizes to sweep (throughput regime)")
    ap.add_argument("--model-max", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.5,
                    help="gpu_memory_utilization (bump to 0.85 for ctx>=64K)")
    ap.add_argument("--out-dir", default="results/nsys")
    ap.add_argument("--harness", default="nsys_profile_decode.py")
    ap.add_argument("--cell-timeout", type=int, default=900,
                    help="seconds per cell")
    args = ap.parse_args()

    if shutil.which("nsys") is None:
        sys.exit("nsys not on PATH. Install with: sudo pacman -S nsight-systems")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Find a venv python (preferred for the harness so torch/vllm/ninja resolve).
    python_bin = sys.executable
    if "/.venv/bin/python" not in python_bin:
        cand = os.path.abspath(".venv/bin/python")
        if os.path.exists(cand):
            python_bin = cand
    venv_bin = os.path.dirname(python_bin)
    base_env = os.environ.copy()
    base_env["PATH"] = venv_bin + ":" + base_env.get("PATH", "")

    cells = [(c, ctx, mt, b)
             for c in args.cfgs
             for ctx in args.contexts
             for b in args.batches
             for mt in args.max_tokens_pair]
    total = len(cells)
    print(f"sweep: {total} cells = "
          f"{len(args.cfgs)} cfgs × {len(args.contexts)} ctxs × "
          f"{len(args.batches)} batches × 2 max_tokens",
          flush=True)
    print(f"  cfgs    : {args.cfgs}", flush=True)
    print(f"  contexts: {args.contexts}", flush=True)
    print(f"  batches : {args.batches}", flush=True)
    print(f"  max_tok : {args.max_tokens_pair}", flush=True)
    print(f"  out_dir : {out_dir}", flush=True)
    if os.geteuid() != 0:
        print("\nWARNING: not running as root. On this box CUPTI requires root "
              "(RmProfilingAdminOnly=1), so nsys captures will lack GPU kernel "
              "data. Re-run with `sudo -E ...`.\n", flush=True)

    ok = skipped = failed = 0
    for i, (cfg, ctx, mt, batch) in enumerate(cells, 1):
        base = cell_base(cfg, ctx, mt, batch)
        out_base = os.path.join(out_dir, base)
        out_rep = out_base + ".nsys-rep"

        if os.path.exists(out_rep):
            print(f"[{i}/{total}] SKIP existing  {base}", flush=True)
            skipped += 1
            continue

        env = base_env.copy()
        env["AHA_CFG"] = cfg
        env["INPUT_LEN"] = str(ctx)
        env["MAX_TOKENS"] = str(mt)
        env["AHA_BATCH"] = str(batch)
        env["AHA_MODEL_MAX"] = str(args.model_max)
        env["AHA_GPU_MEM"] = str(args.gpu_mem)

        cmd = [
            "nsys", "profile",
            "-t", "cuda",
            "--cuda-graph-trace=node",  # CRITICAL: enumerate kernels INSIDE graph
                                        # replays. Default 'graph' traces graphs as a
                                        # whole and hides per-kernel activity inside
                                        # them — which makes the decode attention
                                        # kernel invisible under AHA's full-decode
                                        # cudagraph mode.
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "-o", out_base,  # nsys appends .nsys-rep
            "--",
            python_bin, args.harness,
        ]
        print(f"[{i}/{total}] {base} ...", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=args.cell_timeout)
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"  FAILED rc={r.returncode}  ({dt:.0f}s)", flush=True)
            tail = (r.stderr or r.stdout or "")[-1000:]
            print("  --- tail ---")
            print(tail, flush=True)
            print("  --- end ---")
            failed += 1
            continue
        if not os.path.exists(out_rep):
            print(f"  WARN: no .nsys-rep at {out_rep}  ({dt:.0f}s)", flush=True)
            print((r.stderr or r.stdout or "")[-500:], flush=True)
            failed += 1
            continue
        size_mb = os.path.getsize(out_rep) / 1e6
        # Save cell stdout (carries the VLLM_AHA_PROBE_READOUT routing line).
        with open(out_base + ".log", "w") as lf:
            lf.write(r.stdout or "")
        probe = [ln for ln in (r.stdout or "").splitlines() if "PROBE: overall" in ln]
        tag = f"  {probe[0].split('PROBE:')[1].strip()}" if probe else ""
        print(f"  ok  ({dt:.0f}s, {size_mb:.1f} MB){tag}", flush=True)
        ok += 1

    print(f"\nsweep done: {ok} ok, {skipped} skipped, {failed} failed "
          f"(of {total} cells)", flush=True)
    sys.exit(0 if failed == 0 else 2)


if __name__ == "__main__":
    sys.exit(main())
