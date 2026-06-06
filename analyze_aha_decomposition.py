#!/usr/bin/env python3
"""Turn a decomposition sweep into the A/C/D table, waterfall data, and the
Amdahl ceiling curve.

Consumes the JSON written by sweep_aha_vs_olmo2.py. All ratios are computed in
*time* space (>1 = slower) regardless of whether the sweep metric was a time
(decode_ms_per_token) or a throughput (output_tok_s).

Configs (rename with flags if your sweep used different variant labels):
    A = dense (olmo2)              baseline
    C = aha-global                 AHA machinery, gate forced global (no pruning)
    D = aha-flashinfer             real AHA
    L = local-only                 speed ceiling

Decomposition (per input_len x n):
    r_C/A = t_C / t_A   AHA machinery overhead (>1 = the routing costs you)
    r_D/C = t_D / t_C   net pruning benefit     (<1 = pruning helps)
    r_D/A = t_D / t_A   bottom line
    identity: r_C/A * r_D/C should equal r_D/A to within noise (self-check)
    realized speedup = t_A / t_D  (how many x faster real AHA is than dense)

Amdahl ceiling (optional, needs profiler-derived numbers):
    ceiling = 1 / ((1 - p) + p * s)
        p = attention fraction of decode time on dense (from profile_aha.py)
        s = kernel-level attention speedup ratio, D vs dense (t_D_attn / t_A_attn)
    Pass --p / --s (constants) to overlay realized speedup vs ceiling.

Usage:
    python analyze_aha_decomposition.py /tmp/aha_decomp_sweep.json \\
        --p 0.25 --s 0.86 --out-prefix /tmp/aha_decomp
"""

import argparse
import csv
import json
import sys


def load_cells(path):
    with open(path) as f:
        payload = json.load(f)
    return payload["config"], payload["cells"]


def to_time(best, lower_is_better):
    """Normalize a cell's 'best' value to a time-like quantity (smaller=faster)."""
    if best is None:
        return None
    if lower_is_better:
        return best                       # already a time (ms/token)
    return 1.0 / best if best else None   # throughput -> time


def index_cells(cells, lower_is_better):
    """(input_len, n, variant) -> time-like best value."""
    idx = {}
    for c in cells:
        t = to_time(c["best"], lower_is_better)
        idx[(c["input_len"], c["n"], c["variant"])] = {
            "t": t, "flagged": c["flagged"], "cv": c["stats"].get("cv"),
            "raw_best": c["best"],
        }
    return idx


def ratio(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den


def fmt(x, nd=3, suffix=""):
    return f"{x:.{nd}f}{suffix}" if x is not None else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("--A", default="olmo2", help="dense baseline variant label")
    ap.add_argument("--C", default="aha-global", help="machinery-only label")
    ap.add_argument("--D", default="aha-flashinfer", help="real-AHA label")
    ap.add_argument("--L", default="local-only", help="ceiling label")
    ap.add_argument("--p", type=float, default=None,
                    help="attention fraction of dense decode (for Amdahl)")
    ap.add_argument("--s", type=float, default=None,
                    help="kernel attention speedup D/dense (for Amdahl)")
    ap.add_argument("--out-prefix", default="/tmp/aha_decomp")
    args = ap.parse_args()

    config, cells = load_cells(args.sweep_json)
    lower_is_better = config["lower_is_better"]
    idx = index_cells(cells, lower_is_better)

    input_lens = sorted({c["input_len"] for c in cells})
    ns = sorted({c["n"] for c in cells})

    print(f"metric={config['metric']}  repeats={config['repeats']}  "
          f"(time-space ratios; >1 = slower)\n")
    header = (f"{'ctx':>7}{'N':>4} | {'t_A':>8}{'t_C':>8}{'t_D':>8}{'t_L':>8} | "
              f"{'rC/A':>7}{'rD/C':>7}{'rD/A':>7}{'ident':>7}{'spdup':>7} | flags")
    print(header)
    print("-" * len(header))

    table_rows = []
    waterfall_rows = []
    for il in input_lens:
        for n in ns:
            def g(label):
                return idx.get((il, n, label), {})
            cA, cC, cD, cL = g(args.A), g(args.C), g(args.D), g(args.L)
            tA, tC, tD, tL = (cA.get("t"), cC.get("t"),
                              cD.get("t"), cL.get("t"))

            rCA = ratio(tC, tA)
            rDC = ratio(tD, tC)
            rDA = ratio(tD, tA)
            ident = (rCA * rDC) if (rCA is not None and rDC is not None) else None
            ident_resid = (abs(ident - rDA) / rDA
                           if ident is not None and rDA else None)
            speedup = ratio(tA, tD)  # x faster

            flags = []
            for lbl, c in (("A", cA), ("C", cC), ("D", cD), ("L", cL)):
                if c.get("flagged"):
                    flags.append(lbl)
            flag_str = ("CV>thr:" + ",".join(flags)) if flags else ""
            if ident_resid is not None and ident_resid > 0.05:
                flag_str += f" ident_off={ident_resid*100:.0f}%"

            print(f"{il:>7}{n:>4} | "
                  f"{fmt(tA):>8}{fmt(tC):>8}{fmt(tD):>8}{fmt(tL):>8} | "
                  f"{fmt(rCA,2,'x'):>7}{fmt(rDC,2,'x'):>7}{fmt(rDA,2,'x'):>7}"
                  f"{fmt(ident,2,'x'):>7}{fmt(speedup,2,'x'):>7} | {flag_str}")

            table_rows.append({
                "input_len": il, "n": n,
                "t_A": tA, "t_C": tC, "t_D": tD, "t_L": tL,
                "r_C_over_A": rCA, "r_D_over_C": rDC, "r_D_over_A": rDA,
                "identity_product": ident, "identity_resid": ident_resid,
                "realized_speedup": speedup,
                "flagged": flag_str,
            })

            # Waterfall (time space): dense -> +machinery -> -pruning -> AHA
            if tA is not None and tC is not None and tD is not None:
                waterfall_rows.append({
                    "input_len": il, "n": n,
                    "dense_time": tA,
                    "machinery_overhead": tC - tA,   # +slower
                    "pruning_saving": tD - tC,       # -faster (negative)
                    "aha_time": tD,
                    "ceiling_time": tL,
                })

    # ── Amdahl ceiling overlay ────────────────────────────────────────────────
    if args.p is not None and args.s is not None:
        ceiling = 1.0 / ((1.0 - args.p) + args.p * args.s)
        print(f"\nAmdahl ceiling (p={args.p}, s={args.s}): "
              f"max e2e speedup = {ceiling:.3f}x")
        print(f"{'ctx':>7}{'N':>4} | {'realized':>10}{'ceiling':>10}"
              f"{'% of ceiling':>14}")
        print("-" * 46)
        for row in table_rows:
            sp = row["realized_speedup"]
            if sp is None:
                continue
            frac = ((sp - 1.0) / (ceiling - 1.0) * 100.0
                    if ceiling > 1.0 else None)
            print(f"{row['input_len']:>7}{row['n']:>4} | "
                  f"{sp:>9.3f}x{ceiling:>9.3f}x"
                  f"{(fmt(frac,0,'%') if frac is not None else 'n/a'):>14}")

    # ── chart-ready outputs ────────────────────────────────────────────────────
    table_csv = f"{args.out_prefix}_table.csv"
    with open(table_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    waterfall_csv = f"{args.out_prefix}_waterfall.csv"
    if waterfall_rows:
        with open(waterfall_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(waterfall_rows[0].keys()))
            w.writeheader()
            w.writerows(waterfall_rows)

    summary_json = f"{args.out_prefix}_summary.json"
    with open(summary_json, "w") as f:
        json.dump({
            "config": config,
            "labels": {"A": args.A, "C": args.C, "D": args.D, "L": args.L},
            "amdahl": {"p": args.p, "s": args.s},
            "table": table_rows,
            "waterfall": waterfall_rows,
        }, f, indent=2)

    print(f"\nWrote:\n  {table_csv}\n  "
          f"{waterfall_csv if waterfall_rows else '(no waterfall: need A,C,D)'}"
          f"\n  {summary_json}")


if __name__ == "__main__":
    sys.exit(main())
