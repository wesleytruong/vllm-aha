#!/usr/bin/env python3
"""Turn the result JSONs into presentation-ready CSV tables (+ PNGs if
matplotlib is available). CSVs are the portable deliverable — plot them in
whatever tool the slides use.

Reads whatever exists under results/:
  results/amdahl/amdahl_curve_b*.json     -> amdahl_curve.csv      (e2e speedup vs context)
  results/batch_e2e/batch_curve_*.json    -> batch_e2e.csv         (e2e speedup vs batch)
  results/nsys_cachefix_direct.json       -> decode_speedup.csv    (kernel speedup vs routing/batch)

Usage: python experiments/make_figures.py [--out-dir results/figures]
"""
import argparse
import csv
import glob
import json
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def amdahl(out_dir):
    files = sorted(glob.glob("results/amdahl/amdahl_curve_b*.json"))
    if not files:
        return
    rows = []
    for fp in files:
        for pt in json.load(open(fp)):
            c = pt["configs"]
            rows.append([pt["context"], pt["batch"],
                         round(pt["e2e_speedup"], 3),
                         round(c["all-local"]["step_ms"] and
                               c["all-global"]["step_ms"] / c["all-local"]["step_ms"], 3),
                         round(c["real-gate"]["tok_s"], 1),
                         round(c["all-global"]["tok_s"], 1),
                         round(c["real-gate"]["swa_pct"], 1)])
    rows.sort(key=lambda r: (r[1], r[0]))
    write_csv(os.path.join(out_dir, "amdahl_curve.csv"),
              ["context", "batch", "e2e_speedup", "alllocal_ceiling",
               "real_tok_s", "full_tok_s", "real_swa_pct"], rows)
    if HAVE_MPL and rows:
        for b in sorted({r[1] for r in rows}):
            xs = [r[0] for r in rows if r[1] == b]
            ys = [r[2] for r in rows if r[1] == b]
            plt.plot(xs, ys, "o-", label=f"real gate B={b}")
        plt.xscale("log", base=2); plt.xlabel("context (tokens)")
        plt.ylabel("e2e decode speedup vs full attention")
        plt.axhline(1.0, color="gray", ls=":"); plt.legend(); plt.grid(alpha=.3)
        plt.title("AHA e2e speedup grows with context")
        plt.savefig(os.path.join(out_dir, "amdahl_curve.png"), dpi=130,
                    bbox_inches="tight"); plt.clf()
        print(f"  wrote {out_dir}/amdahl_curve.png")


def batch_e2e(out_dir):
    files = sorted(glob.glob("results/batch_e2e/batch_curve_*.json"))
    if not files:
        return
    rows = []
    for fp in files:
        for pt in json.load(open(fp)):
            c = pt["configs"]
            rows.append([pt["context"], pt["batch"],
                         round(c["real-gate"]["tok_s"], 1),
                         round(c["all-global"]["tok_s"], 1),
                         round(c["real-gate"]["tok_s"] / c["all-global"]["tok_s"], 3)])
    rows.sort(key=lambda r: (r[0], r[1]))
    write_csv(os.path.join(out_dir, "batch_e2e.csv"),
              ["context", "batch", "real_tok_s", "full_tok_s", "speedup"], rows)


def decode_speedup(out_dir):
    fp = "results/nsys_cachefix_direct.json"
    if not os.path.exists(fp):
        return
    d = json.load(open(fp))
    cells = {(c["cfg"], c["ctx"], c["batch"]): c["total_decode_per_step_us"]
             for c in d["per_cell"]}
    rows = []
    for (cfg, ctx, b), v in sorted(cells.items()):
        g = cells.get(("aha-global", ctx, b))
        spd = round(g / v, 3) if g else ""
        rows.append([cfg, ctx, b, round(v, 1), spd])
    write_csv(os.path.join(out_dir, "decode_speedup.csv"),
              ["cfg", "context", "batch", "decode_us_per_step", "speedup_vs_global"],
              rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"# figures/tables -> {args.out_dir}  (matplotlib={'yes' if HAVE_MPL else 'no, CSV only'})")
    amdahl(args.out_dir)
    batch_e2e(args.out_dir)
    decode_speedup(args.out_dir)


if __name__ == "__main__":
    main()
