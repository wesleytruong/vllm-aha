# AHA baseline results — RTX 5090 (32 GB)

Reference numbers from the dev box, to compare against the H100/H200 runs.
Model: `xuan-luo/AHA-OLMO2` (16 layers, 16 KV-heads, head_dim 128, MHA, fp16).
Gate routes ~72–82% of heads to local (128-token window); prefill is full
attention (gate is decode-only). Reproduce with `experiments/slurm/run_all.sbatch`.

Measured at three scopes — pure kernel → kernel-in-engine → end-to-end. They are
**not directly apples-to-apples** (see "Why the ratios differ" below).

## Level 1 — Standalone kernel microbench (pure FlashInfer, no engine)
µs per decode launch, B=1, cudagraph, uniform routing fraction, vLLM's exact plan
params (the L1 microbench now lives in the FlashInfer fork:
`../flashinfer/benchmarks/bench_router_*.py`). ×16 layers ≈ decode-attn/step;
excludes the split-KV merge + engine overhead, so this is the kernel CEILING for
a *uniform* fraction.

| routing | 8K | 32K |
|---|--:|--:|
| all-global (0% local) | 18.5 | 190.5 |
| ~real-gate (70% local) | 14.4 | 30.8 |
| all-local (100%) | 8.3 | **8.3 (context-flat)** |

per-launch speedup vs full: real-gate **1.3× @8K, ~6× @32K**; all-local 2.2× / **23×**.
Mechanism: windowing makes the kernel context-flat; full attention scales 18→190 µs.

## Level 2 — Kernel-in-engine (nsys decode-direct, µs/step, real gate)
`experiments/run_kernel_sweep.py`. Baseline = true base FlashInfer (`dense-fi`,
stock model, full attention); `dense-fi ≈ aha-global` (≤4%) validates the
full-attention control. **Decode is isolated by priming the prompt into the
prefix cache before the profiler window** (`AHA_PFC_DECODE=1`) — the captured
`generate` is a pure cache hit, so the trace holds only decode-shaped launches at
ANY batch (no chunked-prefill interleave to separate). Speedup = baseFI ÷ real.

| ctx | B | baseFI | **real gate** | native-SWA | **speedup** |
|---|--:|--:|--:|--:|--|
| 8K | 1 | 754 | **259** | 104 | **2.92×** |
| 8K | 4 | 2651 | **765** | 159 | **3.46×** |
| 8K | 8 | 5200 | **1583** | 196 | **3.28×** |
| 8K | 16 | 10316 | **3111** | 271 | **3.32×** |
| 16K | 1 | 1398 | **515** | 104 | **2.71×** |
| 16K | 4 | 5196 | **1925** | 160 | **2.70×** |
| 16K | 8 | 10313 | **3494** | 196 | **2.95×** |
| 32K | 1 | 2715 | **955** | 104 | **2.84×** |
| 32K | 2 | 5934 | **1806** | 148 | **3.29×** |
| 32K | 4 | 11710 | **3672** | 157 | **3.19×** |

Routing: 79% local @8K, 68% @32K. Real-gate speedup is **flat ~2.7–3.5× across
batch** (both baseFI and real scale ~linearly with B, so the ratio holds) — the
earlier "erodes with batch" reading was a chunked-prefill contamination artifact
in the B>1 trace, removed by the prefix-cache isolation above. native-SWA is the
pure-windowing floor (unreachable by mixed per-head routing) — its ratio to base
balloons with B (it stays ~context-flat), so it is a ceiling, not a target.

## Level 3 — End-to-end (tokens/s)
`experiments/bench_amdahl.py`, `experiments/bench_batch_e2e.py`.

vs context (B=1): **1.21× @8K → 1.46× @16K → 1.59× @32K** (all-local ceiling 1.19/1.56/2.00×).

| ctx | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|--:|--:|--:|--:|--:|
| 8K | 1.31× | 1.32× | 1.54× | 1.79× | **2.35×** |
| 32K | 1.25× | 2.04× | **2.10×** | — | — |
| 64K | **2.02×** | — | — | — | — |

(64K KV = 8 GB/seq; needs `AHA_GPU_MEM=0.6` so the 64K transient prefill/decode
workspace fits — at 0.8 it OOMs, at 0.5 the tight pool inflates `all-global`.
B≥2 at 64K is near the 32 GB card's limit; the H100's 80 GB lifts this.) B=1
e2e carries ~20% run-to-run noise (the fixed per-step overhead), so read the
trend, not the third digit: real-gate e2e grows with context — ~1.3× @8K →
~1.3–1.6× @32K → **~2.0× @64K** — as attention takes over the decode step.

e2e throughput speedup **grows** with batch: the kernel advantage is ~flat in
batch (Level 2), while the fixed ~2.2 ms/step CPU/scheduling overhead amortizes
as B rises, so its Amdahl drag shrinks and more of the kernel win lands. AHA is a
throughput-regime win at e2e.

## Why the ratios differ across levels (state this explicitly)
- **L1 → L2:** L1 uses a *uniform* 70%-local pattern and excludes the merge
  kernel + engine overhead (overstates: ~6× @32K). L2 uses the *real,
  heterogeneous* gate (some layers route mostly global, dragging the average) and
  includes merge — so the honest realized number is ~2.7–3.5× (2.84× @32K B=1).
  **L2 is the product number.**
- **L2 → L3:** decode-attention is only part of the step (≈26% @8K, ≈53% @32K);
  Amdahl dilutes 2.8–3.1× → 1.2–1.6× e2e at B=1; batching amortizes the fixed
  overhead back up to ~2.3×.

## Headline
Decode is HBM-bandwidth-bound — full attention at 32K reads ~4 GiB of KV per step
and hits ~90% of the 5090's 1.79 TB/s peak. AHA wins by moving ~3× fewer KV
bytes: **~2.8–3.1× decode-attention** in-engine → **1.2–1.6× e2e single-stream,
up to ~2.3× at batch**, growing with context toward the kernel ceiling. On H100
(3.35 TB/s, 80 GB) push contexts to 64K–128K+ and larger batches. The earlier
"0% / kernel-to-engine gap" was a torch.compile cache confound (gate override
baked + served stale), since fixed.
