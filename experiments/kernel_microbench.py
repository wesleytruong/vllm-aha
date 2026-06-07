#!/usr/bin/env python3
"""Level-1 standalone FlashInfer router-decode kernel microbench (no vLLM).

Isolates the pure kernel: times one paged tensor-core decode launch under a
captured CUDA graph, sweeping the routing fraction {all-global, ~real-gate,
all-local} across contexts. Uses vLLM's exact plan params (NHD, page=16, 16/16
heads, head_dim=128, fp16, use_router, scheduler window=-1 via use_router,
runtime window=128). x n_layers ~= the in-engine decode-attn/step, but excludes
the split-KV merge kernel and all engine overhead, so it is the kernel CEILING
for a UNIFORM routing fraction — the real (heterogeneous) gate realizes less
(see run_kernel_sweep.py / BASELINE).

This is the mechanism slide: windowing makes the kernel context-FLAT while full
attention scales with context.

Usage (needs the modified FlashInfer importable):
    python experiments/kernel_microbench.py [--contexts 8192,32768,131072]
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import get_config  # noqa: E402

import torch  # noqa: E402
import flashinfer  # noqa: E402

NH, HD, PAGE = 16, 128, 16          # AHA-OLMO2: 16 KV-heads, head_dim 128
N_LAYERS = 16
DEV, DT = "cuda:0", torch.float16
FRACTIONS = [("all-global", 0.0), ("real-gate~70%local", 0.70),
             ("all-local", 1.0)]


def measure(ws, ctx, frac_local, iters=100):
    pages = math.ceil(ctx / PAGE)
    indptr = torch.zeros(2, dtype=torch.int32, device=DEV); indptr[1] = pages
    indices = torch.arange(pages, dtype=torch.int32, device=DEV)
    last = torch.zeros(1, dtype=torch.int32, device=DEV); last[0] = (ctx - 1) % PAGE + 1
    k = torch.randn(pages, PAGE, NH, HD, dtype=DT, device=DEV)
    v = torch.randn(pages, PAGE, NH, HD, dtype=DT, device=DEV)
    q = torch.randn(1, NH, HD, dtype=DT, device=DEV)
    router = torch.zeros(1, NH, dtype=torch.uint8, device=DEV)  # 1=local/SWA
    router[:, :round(frac_local * NH)] = 1
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, "NHD", use_cuda_graph=True, use_tensor_cores=True, use_router=True,
        paged_kv_indptr_buffer=indptr, paged_kv_indices_buffer=indices,
        paged_kv_last_page_len_buffer=last)
    w.plan(indptr, indices, last, NH, NH, HD, PAGE, pos_encoding_mode="NONE",
           window_left=128, q_data_type=DT, kv_data_type=DT)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            w.run(q, (k, v), router=router)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        w.run(q, (k, v), router=router)
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record(); torch.cuda.synchronize()
    t = a.elapsed_time(b) / iters * 1000.0  # us/launch
    del k, v, q, router, w, g
    torch.cuda.empty_cache()
    return t


def main():
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=str, default="",
                    help="comma list; default = config.amdahl_contexts")
    ap.add_argument("--out", default="results/kernel_microbench.json")
    args = ap.parse_args()
    contexts = ([int(x) for x in args.contexts.split(",")] if args.contexts
                else cfg["amdahl_contexts"])

    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=DEV)
    print(f"# Level-1 kernel microbench (us/launch; x{N_LAYERS} ~= decode-attn/step)")
    hdr = "routing".rjust(20) + "".join(f"{c // 1024}K".rjust(11) for c in contexts)
    print(hdr); print("-" * len(hdr))
    table = {}
    for name, fl in FRACTIONS:
        row = [measure(ws, c, fl) for c in contexts]
        table[name] = dict(zip(contexts, row))
        print(name.rjust(20) + "".join(f"{t:.1f}".rjust(11) for t in row))
    print("\nspeedup vs all-global (per-launch):")
    for c in contexts:
        g = table["all-global"][c]; r = table["real-gate~70%local"][c]
        print(f"  {c // 1024}K: real-gate {g / r:.2f}x   all-local {g / table['all-local'][c]:.2f}x")

    rec = {"profile": cfg["profile"], "contexts": contexts,
           "us_per_launch": {k: {str(c): v for c, v in d.items()}
                             for k, d in table.items()}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
