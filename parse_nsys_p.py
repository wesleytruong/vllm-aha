#!/usr/bin/env python3
"""Parse the nsys captures produced by nsys_sweep_p.py into `p` (attn share) and
`s` (kernel-level attn speedup) per cell.

Per `.nsys-rep`:
  - Run `nsys stats --report cuda_gpu_kern_sum --format csv` and parse.
  - attn_total = sum of "Total Time" over kernels matching the per-cfg attention
    regex (audited: matched-kernel names and hit counts are logged so a bad
    regex shows up loudly instead of silently producing a wrong p).
  - gpu_total  = sum of "Total Time" over ALL kernels in the capture (under
    graphs-on this packs tightly, so GPU-busy ≈ wall, which is the denominator
    we want for p).

Per (cfg, ctx), with mt_short=1 and mt_long=N captures:
  attn_per_step_ns = (attn_total[N] − attn_total[1]) / (N − 1)
  gpu_per_step_ns  = (gpu_total[N]  − gpu_total[1])  / (N − 1)
  p                = attn_per_step / gpu_per_step

Per ctx:
  s = attn_per_step[D=aha-flashinfer] / attn_per_step[A=dense]

Runs as your user (parse only; .nsys-rep files were captured under sudo and are
world-readable):
  .venv/bin/python parse_nsys_p.py \\
      --in-dir results/nsys --out results/nsys_p_results.json
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from io import StringIO

# Per-cfg attention-kernel name regexes. nsys reports the C++ symbol name; the
# router/swa flags are encoded in numeric template params, not in plain text —
# so we match by the kernel's namespaced family. The long-minus-short
# subtraction naturally cancels prefill kernel time; the leftover per step is
# the decode attention. AUDIT: parse_nsys_p logs the matched kernel name(s) +
# hit counts per cell so a bad regex shows up loudly.
ATTN_PATTERNS = {
    "aha-global":     [r"flashinfer::BatchPrefillWithPagedKVCacheKernel",
                       r"flashinfer::BatchDecodeWithPagedKVCacheKernel"],
    "aha-half":       [r"flashinfer::BatchPrefillWithPagedKVCacheKernel",
                       r"flashinfer::BatchDecodeWithPagedKVCacheKernel"],
    "aha-local":      [r"flashinfer::BatchPrefillWithPagedKVCacheKernel",
                       r"flashinfer::BatchDecodeWithPagedKVCacheKernel"],
    "aha-flashinfer": [r"flashinfer::BatchPrefillWithPagedKVCacheKernel",
                       r"flashinfer::BatchDecodeWithPagedKVCacheKernel"],
    "local-only":     [r"flashinfer::Batch.*Kernel",
                       r"flash_fwd", r"flash_attn", r"_vllm_fa"],
    "dense":          [r"flash_fwd", r"flash_attn", r"_vllm_fa",
                       r"flashinfer::Batch.*Kernel"],  # fallback if dense ends up on FI
    "dense-fi":       [r"flashinfer::BatchPrefillWithPagedKVCacheKernel",
                       r"flashinfer::BatchDecodeWithPagedKVCacheKernel"],
}

CELL_RE = re.compile(
    r"^(?P<cfg>[a-z\-]+)_ctx(?P<ctx>\d+)_mt(?P<mt>\d+)(?:_b(?P<batch>\d+))?\.nsys-rep$"
)


def run_kern_sum(rep_path: str) -> list[dict]:
    """Returns [{name, total_ns, instances}] from `nsys stats --report cuda_gpu_kern_sum`."""
    # --force-export refreshes the .sqlite if the .nsys-rep is newer; otherwise
    # stale sqlite from a prior run causes nsys stats to error out.
    cmd = ["nsys", "stats", "--report", "cuda_gpu_kern_sum",
           "--force-export=true", "--format", "csv", rep_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"nsys stats failed for {rep_path}: rc={r.returncode}\n"
            f"{(r.stderr or r.stdout or '')[-400:]}"
        )
    # nsys stats prefixes the CSV with a banner; find the real header.
    lines = r.stdout.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines)
         if "Total Time" in ln and ("Name" in ln or "Operation" in ln)),
        None,
    )
    if header_idx is None:
        raise RuntimeError(
            f"could not locate CSV header in nsys stats output for {rep_path}"
        )
    reader = csv.DictReader(StringIO("\n".join(lines[header_idx:])))
    fields = reader.fieldnames or []
    name_key = "Name" if "Name" in fields else "Operation"
    total_key = next((k for k in fields if "Total Time" in k), None)
    inst_key = next((k for k in fields if "Instances" in k), None)
    if total_key is None:
        raise RuntimeError(f"no 'Total Time' column in nsys stats output for {rep_path}")
    rows = []
    for row in reader:
        try:
            total_ns = int(row[total_key].replace(",", "").strip())
        except (KeyError, ValueError):
            continue
        inst = None
        if inst_key and row.get(inst_key):
            try:
                inst = int(row[inst_key].replace(",", "").strip())
            except ValueError:
                pass
        rows.append({"name": row[name_key], "total_ns": total_ns,
                     "instances": inst})
    return rows


def match_attn(rows: list[dict], cfg: str) -> tuple[int, int, list[str]]:
    """Returns (attn_total_ns, matched_hits, sorted_unique_matched_names)."""
    pats = [re.compile(p) for p in ATTN_PATTERNS.get(cfg, [])]
    total = 0
    matched_names = []
    hits = 0
    for row in rows:
        if any(p.search(row["name"]) for p in pats):
            matched_names.append(row["name"])
            total += row["total_ns"]
            hits += row["instances"] if row["instances"] is not None else 1
    return total, hits, sorted(set(matched_names))


def gpu_total(rows: list[dict]) -> int:
    return sum(r["total_ns"] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/nsys")
    ap.add_argument("--out", default="results/nsys_p_results.json")
    args = ap.parse_args()

    if not os.path.isdir(args.in_dir):
        sys.exit(f"in-dir not found: {args.in_dir}")

    # cells[(cfg, ctx, batch)][mt] = per-rep summary
    cells: dict[tuple[str, int, int], dict[int, dict]] = defaultdict(dict)
    n_reps = 0
    for fn in sorted(os.listdir(args.in_dir)):
        m = CELL_RE.match(fn)
        if not m:
            continue
        cfg, ctx, mt = m["cfg"], int(m["ctx"]), int(m["mt"])
        batch = int(m["batch"]) if m["batch"] else 1
        path = os.path.join(args.in_dir, fn)
        try:
            rows = run_kern_sum(path)
        except Exception as e:
            print(f"WARN {fn}: {e}", flush=True)
            continue
        n_reps += 1
        attn_ns, hits, matched = match_attn(rows, cfg)
        g_ns = gpu_total(rows)
        print(f"{fn}: {len(rows)} kernels  "
              f"attn={attn_ns/1e6:>7.2f}ms ({hits} hits, "
              f"{len(matched)} unique name(s))  "
              f"gpu={g_ns/1e6:>7.2f}ms", flush=True)
        if not matched:
            print(f"  WARN no attention kernel matched for cfg={cfg!r}. "
                  f"Top 5 kernels by total time:", flush=True)
            for r in sorted(rows, key=lambda r: -r["total_ns"])[:5]:
                print(f"    {r['total_ns']/1e6:>8.2f} ms  {r['name'][:140]}",
                      flush=True)
        cells[(cfg, ctx, batch)][mt] = {
            "rep_file": fn, "n_kernels": len(rows),
            "attn_ns": attn_ns, "attn_hits": hits,
            "matched_kernels": matched, "gpu_ns": g_ns,
        }

    if n_reps == 0:
        sys.exit(f"no .nsys-rep files matched {CELL_RE.pattern} in {args.in_dir}")

    # Long-minus-short subtraction per (cfg, ctx, batch).
    per_cell = []
    for (cfg, ctx, batch), mts in sorted(cells.items()):
        mt_vals = sorted(mts.keys())
        if len(mt_vals) < 2:
            print(f"SKIP (cfg={cfg}, ctx={ctx}, b={batch}): need 2 max_tokens "
                  f"values, got {mt_vals}", flush=True)
            continue
        mt_short, mt_long = mt_vals[0], mt_vals[-1]
        steps = mt_long - mt_short
        if steps <= 0:
            continue
        a_short = mts[mt_short]["attn_ns"]; a_long = mts[mt_long]["attn_ns"]
        g_short = mts[mt_short]["gpu_ns"];  g_long = mts[mt_long]["gpu_ns"]
        attn_per_step_ns = (a_long - a_short) / steps
        gpu_per_step_ns  = (g_long - g_short) / steps
        p = (attn_per_step_ns / gpu_per_step_ns) if gpu_per_step_ns else None
        positive = attn_per_step_ns > 0 and gpu_per_step_ns > 0
        flag = "" if positive else "  ⚠ NON-POSITIVE SUBTRACTION"
        print(f"\n[{cfg:<16} ctx={ctx:>6} b={batch:>3}]  "
              f"attn/step={attn_per_step_ns/1e3:>7.2f}us  "
              f"gpu/step={gpu_per_step_ns/1e3:>8.2f}us  p={p:.3f}{flag}",
              flush=True)
        per_cell.append({
            "cfg": cfg, "ctx": ctx, "batch": batch,
            "mt_short": mt_short, "mt_long": mt_long, "decode_steps": steps,
            "attn_per_step_us": attn_per_step_ns / 1e3,
            "gpu_per_step_us":  gpu_per_step_ns / 1e3,
            "p": p, "positive": positive,
            "matched_kernels": mts[mt_long]["matched_kernels"],
        })

    # Cross-cfg s per (ctx, batch): s = attn(D) / attn(A=baseline).
    # Baseline is dense if present, else aha-global (forced full attention),
    # so batch sweeps that only run FlashInfer cfgs still get a ratio.
    by_key = defaultdict(dict)
    for c in per_cell:
        by_key[(c["ctx"], c["batch"])][c["cfg"]] = c
    s_per_ctx = []
    for (ctx, batch) in sorted(by_key):
        d = by_key[(ctx, batch)]
        base_cfg = "dense" if "dense" in d else (
            "aha-global" if "aha-global" in d else None)
        if base_cfg is None:
            continue
        for cfg_d in ("aha-flashinfer", "aha-local", "aha-global", "dense-fi"):
            if cfg_d in d and cfg_d != base_cfg:
                aD = d[cfg_d]["attn_per_step_us"]
                aA = d[base_cfg]["attn_per_step_us"]
                s = aD / aA if aA else None
                s_per_ctx.append({"ctx": ctx, "batch": batch, "cfg_D": cfg_d,
                                  "base": base_cfg,
                                  "attn_D_us": aD, "attn_A_us": aA, "s": s})
                print(f"[ctx={ctx:>6} b={batch:>3}]  s({cfg_d}/{base_cfg}) = "
                      f"{aD:>7.2f} / {aA:>7.2f} = {s:.3f}x", flush=True)

    out = {"per_cell": per_cell, "s_per_ctx": s_per_ctx,
           "attn_patterns": ATTN_PATTERNS}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
