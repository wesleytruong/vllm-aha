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
import re

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
    kernel_summary(out_dir, cells)


def _swa_pct(sweep_dir, ctx, b):
    """Measured routing % from the aha-flashinfer cell log (probe readout)."""
    suf = "" if b == 1 else f"_b{b}"
    fp = os.path.join(sweep_dir, f"aha-flashinfer_ctx{ctx}_mt33{suf}.log")
    if os.path.exists(fp):
        m = re.search(r"SWA\(local\)=([0-9.]+)%", open(fp).read())
        if m:
            return float(m.group(1))
    return None


def kernel_summary(out_dir, cells, sweep_dir="results/nsys_cachefix"):
    """Canonical L2 summary, one row per (ctx, batch), identical across GPUs.

    Baseline = TRUE base FlashInfer (dense-fi: stock model, full attention, no
    router). Ceiling = TRUE native sliding window (local-only: vLLM per-layer
    128-window, no router). real-gate speedup = base / real. SWA ceiling =
    base / native-SWA (the theoretical max from pure windowing; not reachable by
    per-head MIXED routing, but the absolute floor). aha-global shown for the
    router-overhead check (≈ base FI).
    """
    keys = sorted({(c, b) for (_, c, b) in cells})
    rows = []
    for ctx, b in keys:
        base = cells.get(("dense-fi", ctx, b))      # true base FlashInfer (full)
        glob_ = cells.get(("aha-global", ctx, b))   # our router kernel, all global
        real = cells.get(("aha-flashinfer", ctx, b))
        swa = cells.get(("local-only", ctx, b))      # true native-SWA floor
        if base is None or real is None:
            continue
        rows.append([
            ctx, b, _swa_pct(sweep_dir, ctx, b) or "",
            round(base, 1), round(glob_, 1) if glob_ else "",
            round(real, 1), round(swa, 1) if swa else "",
            round(base / real, 3),
            round(base / swa, 1) if swa else ""])
    write_csv(os.path.join(out_dir, "kernel_summary.csv"),
              ["context", "batch", "swa_pct", "base_fi_us", "aha_global_us",
               "real_us", "native_swa_us", "real_speedup", "swa_ceiling"], rows)
    # print the presentation layout
    print(f"\n  {'ctx':>5} {'B':>3} {'SWA%':>5} {'baseFI':>7} {'real':>6} "
          f"{'natSWA':>7} {'speedup':>8} {'SWA ceil':>9}")
    for ctx, b, swa, ba, gl, re_, sw, sp, cl in rows:
        print(f"  {ctx // 1024:>4}K {b:>3} {(f'{swa:.0f}%' if swa != '' else '-'):>5} "
              f"{ba:>7.0f} {re_:>6.0f} {(f'{sw:.0f}' if sw != '' else '-'):>7} "
              f"{sp:>7.2f}× {(f'{cl:.1f}×' if cl != '' else '-'):>9}")


def box_table(headers, rows, aligns):
    """Render a unicode box-drawing table (separator between every row)."""
    w = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
         for i in range(len(headers))]

    def _line(l, m, r):
        return l + m.join("─" * (w[i] + 2) for i in range(len(w))) + r

    def _row(c):
        return "│" + "│".join(
            " " + (c[i].ljust(w[i]) if aligns[i] == "<" else c[i].rjust(w[i])) + " "
            for i in range(len(w))) + "│"

    out = [_line("┌", "┬", "┐"), _row(headers)]
    for r in rows:
        out += [_line("├", "┼", "┤"), _row(r)]
    out.append(_line("└", "┴", "┘"))
    return "\n".join(out)


def itl_summary(out_dir):
    """Read the ITL/TPOT matrix (experiments/bench_itl_grid.py) -> itl_summary.csv
    + the boxed context x batch table. TPOT = inter-token latency (ms/token)."""
    files = sorted(glob.glob("results/itl/grid_ctx*.json"))
    if not files:
        return
    pts = [json.load(open(f)) for f in files]
    pts.sort(key=lambda p: (p["ctx"], p["batch"]))
    csv_rows, box_rows = [], []
    for p in pts:
        mo = p["modes"]
        r, f, l = mo["real-gate"], mo["all-global"], mo["all-local"]
        csv_rows.append([p["ctx"], p["batch"], round(r["swa_pct"], 1),
                         round(f["tpot_ms"], 2), round(r["tpot_ms"], 2),
                         round(r["speedup_vs_full"], 3), round(r["total_tok_s"], 1),
                         round(l["tpot_ms"], 2)])
        box_rows.append([f"{p['ctx']//1024}K", str(p["batch"]),
                         f"{r['swa_pct']:.0f}%", f"{f['tpot_ms']:.2f}",
                         f"{r['tpot_ms']:.2f}", f"{r['speedup_vs_full']:.2f}×",
                         f"{r['total_tok_s']:.0f}", f"{l['tpot_ms']:.2f}"])
    write_csv(os.path.join(out_dir, "itl_summary.csv"),
              ["context", "batch", "swa_pct", "full_tpot_ms", "real_tpot_ms",
               "latency_speedup", "real_total_tok_s", "alllocal_floor_ms"], csv_rows)
    hdr = ["ctx", "B", "SWA%", "full TPOT (ms)", "real TPOT (ms)",
           "latency speedup", "real total tok/s", "all-local floor (ms)"]
    print(box_table(hdr, box_rows, ["<", ">", ">", ">", ">", ">", ">", ">"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"# figures/tables -> {args.out_dir}  (matplotlib={'yes' if HAVE_MPL else 'no, CSV only'})")
    amdahl(args.out_dir)
    batch_e2e(args.out_dir)
    decode_speedup(args.out_dir)
    itl_summary(args.out_dir)


if __name__ == "__main__":
    main()
