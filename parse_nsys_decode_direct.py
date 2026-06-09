#!/usr/bin/env python3
"""Decode-kernel-direct extractor (single-rep, no long-minus-short subtraction).

The subtraction method (parse_nsys_p.py) cancels prefill by differencing the
mt=long and mt=short reps, but run-to-run prefill jitter (~1.6ms) swamps the
~5us decode signal and produces negative/garbage per-step numbers. Instead we
isolate decode launches *within a single mt=long rep* by their duration:

  - The attention kernel (flashinfer::BatchPrefillWithPagedKVCacheKernel) is
    launched once per layer per forward pass. Decode contributes exactly
    n_layers*(mt-1) launches (1/layer/step). Prefill contributes
    n_layers*n_chunks launches -- n_chunks > 1 when the prompt exceeds the
    chunked-prefill budget (e.g. a 32K prompt splits into ~4 chunks -> 64
    prefill launches, not 16). So a fixed "drop top n_layers" is WRONG.
  - Prefill launches process thousands of tokens -> ~1.5-10ms each. Decode
    launches process 1 token/seq -> 5-170us each. The gap is ~10-300x, so the
    n_layers*(mt-1) SMALLEST-duration launches are decode; the rest are prefill
    (robust to any number of prefill chunks).

Per-step decode attention = sum(decode launches) / (mt - 1), plus the split-KV
merge kernel (PersistentVariableLengthMergeStatesKernel) which is decode-only.

Cell file names: <cfg>_ctx<ctx>_mt<mt>[_b<batch>].nsys-rep . We only read the
mt=max rep per (cfg, ctx, batch) -- the short rep is unused here.

Usage:
    python parse_nsys_decode_direct.py --in-dir results/nsys \
        --n-layers 16 --out results/nsys_decode_direct.json
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

CELL_RE = re.compile(
    r"^(?P<cfg>[a-z\-]+)_ctx(?P<ctx>\d+)_mt(?P<mt>\d+)(?:_b(?P<batch>\d+))?\.nsys-rep$"
)

# Decode attention kernel symbol (FlashInfer tensor-core decode == prefill
# kernel) and the split-KV merge kernel. dense (FA2) uses a different name; the
# batch sweep runs only FlashInfer cfgs, so this default covers them all.
ATTN_KERNEL = "flashinfer::BatchPrefillWithPagedKVCacheKernel"
MERGE_KERNEL = "flashinfer::PersistentVariableLengthMergeStatesKernel"


def run_gpu_trace(rep_path: str) -> list[dict]:
    cmd = ["nsys", "stats", "--report", "cuda_gpu_trace",
           "--format", "csv", "--force-export=true", rep_path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    hdr = next(i for i, l in enumerate(lines)
               if "Name" in l and "Duration" in l)
    return list(csv.DictReader(io.StringIO("\n".join(lines[hdr:]))))


def kernel_durs(rows: list[dict], name_substr: str) -> list[int]:
    return [int(r["Duration (ns)"]) for r in rows
            if name_substr in r["Name"]]


def kernel_start_dur(rows: list[dict], name_substr: str) -> list[tuple[int, int]]:
    """(start_ns, dur_ns) for matching launches, sorted by start time."""
    out = [(int(r["Start (ns)"]), int(r["Duration (ns)"]))
           for r in rows if name_substr in r["Name"]]
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/nsys")
    ap.add_argument("--n-layers", type=int, default=16,
                    help="# attention layers (decode launches = n_layers*(mt-1))")
    ap.add_argument("--prefill-threshold-us", type=float, default=1000.0,
                    help="launches >= this (us) are prefill; below are decode")
    ap.add_argument("--out", default="results/nsys_decode_direct.json")
    args = ap.parse_args()

    # Keep only the mt=max rep per (cfg, ctx, batch).
    best: dict[tuple, tuple[int, str]] = {}
    for fn in sorted(os.listdir(args.in_dir)):
        m = CELL_RE.match(fn)
        if not m:
            continue
        cfg, ctx = m["cfg"], int(m["ctx"])
        mt = int(m["mt"])
        batch = int(m["batch"]) if m["batch"] else 1
        key = (cfg, ctx, batch)
        if key not in best or mt > best[key][0]:
            best[key] = (mt, fn)

    results = []
    for (cfg, ctx, batch), (mt, fn) in sorted(best.items()):
        steps = mt - 1
        if steps <= 0:
            print(f"SKIP {fn}: mt={mt} has no decode steps", flush=True)
            continue
        path = os.path.join(args.in_dir, fn)
        try:
            rows = run_gpu_trace(path)
        except Exception as e:
            print(f"WARN {fn}: {e}", flush=True)
            continue

        # Decode is strictly the LAST phase: all B seqs prefill together (chunked
        # -> remainder chunks whose grid AND duration mimic decode), then decode
        # together. So neither grid nor a duration cut separates cleanly. Time
        # does: the last n_layers*(mt-1) attention launches by start time are
        # decode; everything earlier is prefill. Robust to chunked-prefill
        # remainders and to fast (pruned) decode launches.
        sd = kernel_start_dur(rows, ATTN_KERNEL)
        merge = kernel_durs(rows, MERGE_KERNEL)  # split-KV reduction (decode-only)
        expected = args.n_layers * steps
        if len(sd) < expected:
            print(f"WARN {fn}: only {len(sd)} attn launches (< expected decode "
                  f"{expected}); skipping", flush=True)
            continue
        decode = [d for _, d in sd[-expected:]]
        prefill = [d for _, d in sd[:-expected]]
        # Sanity: the decode window must not straddle a prefill chunk (a >1ms
        # launch among the "decode" set means prefill leaked in).
        if decode and max(decode) > 800_000:
            print(f"  ⚠ {fn}: a 'decode' launch is {max(decode)/1e3:.0f}us "
                  f"(>800us) -- prefill may have leaked into the last {expected}",
                  flush=True)
        decode_per_step_ns = sum(decode) / steps
        merge_per_step_ns = sum(merge) / steps if merge else 0.0
        total_per_step_ns = decode_per_step_ns + merge_per_step_ns
        per_launch_us = sum(decode) / len(decode) / 1e3
        prefill_ms = (sum(prefill) / len(prefill) / 1e6) if prefill else 0.0

        print(f"[{cfg:<16} ctx={ctx:>6} b={batch:>3}]  "
              f"decode-attn/step={decode_per_step_ns/1e3:>8.2f}us  "
              f"+merge={merge_per_step_ns/1e3:>5.2f}us  "
              f"= {total_per_step_ns/1e3:>8.2f}us/step  "
              f"(per-launch {per_launch_us:.2f}us, "
              f"n_prefill={len(prefill)}, prefill {prefill_ms:.2f}ms)",
              flush=True)

        results.append({
            "cfg": cfg, "ctx": ctx, "batch": batch, "mt": mt,
            "decode_steps": steps, "rep_file": fn,
            "decode_attn_per_step_us": decode_per_step_ns / 1e3,
            "merge_per_step_us": merge_per_step_ns / 1e3,
            "total_decode_per_step_us": total_per_step_ns / 1e3,
            "per_launch_us": per_launch_us,
            "n_decode_launches": len(decode),
            "n_prefill_launches": len(prefill),
            "prefill_mean_ms": prefill_ms,
        })

    # Cross-cfg ratio per (ctx, batch): baseline = aha-global (forced full attn)
    # if present, else dense-fi. Shows whether pruning separates as B grows.
    by_key = defaultdict(dict)
    for r in results:
        by_key[(r["ctx"], r["batch"])][r["cfg"]] = r
    ratios = []
    print("\n=== decode-attn/step ratio vs baseline (per ctx, batch) ===",
          flush=True)
    for (ctx, batch) in sorted(by_key):
        d = by_key[(ctx, batch)]
        base_cfg = "aha-global" if "aha-global" in d else (
            "dense-fi" if "dense-fi" in d else None)
        if base_cfg is None:
            continue
        base = d[base_cfg]["total_decode_per_step_us"]
        for cfg_d in ("aha-flashinfer", "aha-local", "local-only",
                      "dense-fi", "aha-global"):
            if cfg_d in d and cfg_d != base_cfg:
                v = d[cfg_d]["total_decode_per_step_us"]
                s = v / base if base else None
                ratios.append({"ctx": ctx, "batch": batch, "cfg": cfg_d,
                               "base": base_cfg, "us": v, "base_us": base,
                               "ratio": s})
                print(f"[ctx={ctx:>6} b={batch:>3}]  {cfg_d}/{base_cfg} = "
                      f"{v:>7.2f} / {base:>7.2f} = {s:.3f}x", flush=True)

    out = {"per_cell": results, "ratios": ratios,
           "attn_kernel": ATTN_KERNEL, "merge_kernel": MERGE_KERNEL}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out} ({len(results)} cells)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
