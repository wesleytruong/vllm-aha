#!/usr/bin/env python3
"""Isolated FlashInfer attention-kernel microbench → kernel-level s and A/C/D ratios.

Calls flashinfer.BatchDecodeWithPagedKVCacheWrapper.plan/.run directly on synthetic
paged KV inputs. No engine, no actual CUDA-graph capture, no profiler.

DIAGNOSTIC FINDING (recorded for the eval narrative):
  Using use_cuda_graph=False + use_router=True (standard plan) reproduces the same
  CUDA illegal-memory-access seen in `profile_aha --enforce-eager`. So the eager
  FlashInfer crash is rooted in the FlashInfer fork's router wrapper for the
  non-cudagraph code path itself, NOT in vllm's fast_plan_decode shortcut or in
  how aha_flashinfer.py populates inputs.

This microbench therefore uses use_cuda_graph=True (the production-validated path)
with persistent CSR buffers — we don't actually capture a graph, but the wrapper's
buffer-management code path is the one that works. CUDA-event timing of
wrapper.run() is unaffected by the cudagraph flag.

Three modes per (context, B):
  A (dense proxy): wrapper with use_router=False, full attention (window_left=-1)
  C (all-global):  use_router=True, router = ones uint8       — AHA machinery, no pruning
  D (real AHA):    use_router=True, router = Bernoulli(0.5)   — AHA with pruning

Reports:
  us_A, us_C, us_D       (p10 of CUDA-event-timed kernel duration)
  r_CA = us_C / us_A     (machinery overhead within FlashInfer family)
  r_DC = us_D / us_C     (pure pruning benefit, overhead held constant)
  s = r_DA = us_D / us_A (kernel-level AHA-fork vs dense proxy)

NOTE: A here is a *FlashInfer no-router* proxy for dense, not the true vLLM FA2
kernel that stock OLMo2 uses. Production dense-vs-AHA is captured at the engine
level by sweep_aha_vs_olmo2.py; this script gives the kernel-level decomposition
that feeds the Amdahl ceiling alongside it.

Run with the venv's bin on PATH so FlashInfer's JIT can find ninja:
  PATH="$PWD/.venv/bin:$PATH" .venv/bin/python microbench_aha_attn.py \\
      --contexts 8192 32768 --batches 1 --out results/microbench_s.json
"""

import argparse
import json
import os
import sys

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

# OLMo-2-0425-1B / AHA-OLMO2 shape (verified against HF config + olmo2_aha.py asserts).
NUM_QO_HEADS = 16
NUM_KV_HEADS = 16  # MHA: num_queries_per_kv == 1, see olmo2_aha.py:1352 assert
HEAD_DIM = 128     # hidden_size 2048 / 16 heads
PAGE_SIZE = 16
LOCAL_WINDOW = 128
KV_LAYOUT = "HND"  # vllm/v1/attention/backends/flashinfer.py:381
DTYPE = torch.float16
DEVICE = "cuda"


def time_call(fn, iters: int, warmup: int) -> list[float]:
    """CUDA-event-timed durations of fn() in microseconds, sorted ascending."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = max(0, min(len(sorted_vals) - 1, int(q * len(sorted_vals))))
    return sorted_vals[idx]


def setup_cell(context_len: int, batch_size: int):
    """Build paged-KV tensors and query for a single-token decode at (context, B)."""
    pages_per_seq = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch_size * pages_per_seq
    last_page_len = ((context_len - 1) % PAGE_SIZE) + 1

    # Production plumbing keeps indptr/last_page_len on CPU and indices on GPU
    # (see fast_plan_decode call in aha_flashinfer.py:171–185).
    indptr_cpu = torch.arange(
        0, batch_size + 1, dtype=torch.int32
    ) * pages_per_seq
    indices_gpu = torch.arange(0, total_pages, dtype=torch.int32, device=DEVICE)
    last_page_lens_cpu = torch.full(
        (batch_size,), last_page_len, dtype=torch.int32
    )

    # HND layout: (num_pages, num_kv_heads, page_size, head_dim) per K and V.
    key_cache = torch.randn(
        total_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    value_cache = torch.randn(
        total_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )

    # Single-token decode query: (B, num_qo_heads, head_dim).
    q = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    return indptr_cpu, indices_gpu, last_page_lens_cpu, key_cache, value_cache, q


def run_mode(
    mode: str,
    context_len: int,
    batch_size: int,
    iters: int,
    warmup: int,
) -> float:
    """mode in {A, C, D}. Returns p10 microseconds."""
    indptr, indices, last_page_lens, key_cache, value_cache, q = setup_cell(
        context_len, batch_size
    )
    pages_per_seq = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch_size * pages_per_seq

    use_router = mode in ("C", "D")
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    # Persistent CSR buffers for the use_cuda_graph=True wrapper path. We do NOT
    # actually capture a graph here — this flag just selects the production-
    # validated buffer-management code path inside FlashInfer. The non-cudagraph
    # path (use_cuda_graph=False + None buffers) crashes for use_router=True; see
    # the DIAGNOSTIC FINDING in the file header.
    #
    # Production (aha_flashinfer.py:94–96) sizes these for the max possible batch /
    # pages and passes slices each step. We mirror that by oversizing the
    # persistent buffers vs the per-step actuals (some FlashInfer code paths
    # appear to assume extra headroom in the persistent indices buffer).
    PAGES_HEADROOM = 4
    indptr_buf = torch.empty(batch_size + 1, dtype=torch.int32, device=DEVICE)
    indices_buf = torch.empty(
        total_pages * PAGES_HEADROOM, dtype=torch.int32, device=DEVICE
    )
    last_page_buf = torch.empty(batch_size, dtype=torch.int32, device=DEVICE)

    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        KV_LAYOUT,
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf,
        paged_kv_last_page_len_buffer=last_page_buf,
        use_router=use_router,
        use_tensor_cores=True,
    )

    # window_left:
    #   A (dense proxy): -1 (no sliding window)
    #   C/D: LOCAL_WINDOW-1 (matches production template; run() forces -1 at runtime
    #        for router internally, see flashinfer/decode.py run path)
    window_left = (LOCAL_WINDOW - 1) if use_router else -1
    wrapper.plan(
        indptr, indices, last_page_lens,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
        pos_encoding_mode="NONE",
        window_left=window_left,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )

    if mode == "C":
        router = torch.ones(
            batch_size, NUM_QO_HEADS, dtype=torch.uint8, device=DEVICE
        )
    elif mode == "D":
        router = (
            torch.rand(batch_size, NUM_QO_HEADS, device=DEVICE) > 0.5
        ).to(torch.uint8)
    else:
        router = None

    if use_router:
        run_fn = lambda: wrapper.run(  # noqa: E731
            q, (key_cache, value_cache), router=router
        )
    else:
        run_fn = lambda: wrapper.run(q, (key_cache, value_cache))  # noqa: E731

    samples = time_call(run_fn, iters=iters, warmup=warmup)
    return percentile(samples, 0.1)


def _fmt(v, nd=2, suf=""):
    return f"{v:.{nd}f}{suf}" if v is not None else "    n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--batches", type=int, nargs="+", default=[1])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="results/microbench_s.json")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for D's Bernoulli router (reproducibility)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cells = []
    print(f"{'context':>8}{'B':>4}{'us_A':>10}{'us_C':>10}{'us_D':>10}"
          f"{'r_CA':>9}{'r_DC':>9}{'s=r_DA':>10}")
    print("-" * 68)
    for ctx in args.contexts:
        for B in args.batches:
            row = {"context": ctx, "B": B}
            for mode in ("A", "C", "D"):
                try:
                    us = run_mode(mode, ctx, B, args.iters, args.warmup)
                except Exception as e:
                    print(f"{ctx:>8}{B:>4}  {mode} FAILED: "
                          f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                    us = None
                row[f"us_{mode}"] = us

            us_A, us_C, us_D = row["us_A"], row["us_C"], row["us_D"]
            row["r_CA"] = (us_C / us_A) if us_A and us_C else None
            row["r_DC"] = (us_D / us_C) if us_C and us_D else None
            row["s_eq_r_DA"] = (us_D / us_A) if us_A and us_D else None
            cells.append(row)

            print(f"{ctx:>8}{B:>4}{_fmt(us_A):>10}{_fmt(us_C):>10}{_fmt(us_D):>10}"
                  f"{_fmt(row['r_CA'], 3, 'x'):>9}{_fmt(row['r_DC'], 3, 'x'):>9}"
                  f"{_fmt(row['s_eq_r_DA'], 3, 'x'):>10}", flush=True)

    payload = {
        "config": {
            "contexts": args.contexts, "batches": args.batches,
            "iters": args.iters, "warmup": args.warmup, "seed": args.seed,
            "model_shape": {
                "num_qo_heads": NUM_QO_HEADS, "num_kv_heads": NUM_KV_HEADS,
                "head_dim": HEAD_DIM, "page_size": PAGE_SIZE,
                "local_window": LOCAL_WINDOW, "kv_layout": KV_LAYOUT,
                "dtype": str(DTYPE),
            },
            "modes": {
                "A": "FlashInfer no-router, window_left=-1 (dense proxy)",
                "C": "FlashInfer router=ones (all-global, machinery overhead)",
                "D": "FlashInfer router=Bernoulli(0.5) (real AHA)",
            },
        },
        "cells": cells,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {len(cells)} cells to {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
