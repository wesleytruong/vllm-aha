# AHA baseline results — RTX 5090 (32 GB)

Reference numbers from the dev box, to compare against the H100/H200 runs.
Model: `xuan-luo/AHA-OLMO2` (16 layers, 16 KV-heads, head_dim 128, MHA, fp16).
Gate routes ~72–82% of heads to local (128-token window); prefill is full
attention (gate is decode-only). Reproduce with `experiments/slurm/run_all.sbatch`.

Measured at three scopes — pure kernel → kernel-in-engine → end-to-end. They are
**not directly apples-to-apples** (see "Why the ratios differ" below).

## Level 1 — Standalone kernel microbench (pure FlashInfer, no engine)
`experiments/kernel_microbench.py`. µs per decode launch, B=1, cudagraph, uniform
routing fraction, vLLM's exact plan params. ×16 layers ≈ decode-attn/step;
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
`experiments/run_kernel_sweep.py`. Monotonic in routing fraction;
`dense-fi ≈ aha-global` validates the full-attention control.

| ctx | all-global | half | **real gate** | all-local | batch-scaling (real/full) |
|---|--:|--:|--:|--:|--|
| 8K | 809 | 436 | **258 (3.13×)** | 114 | 2.96 / 2.48 / 2.08× @ B=4/8/16 |
| 32K | 2656 | 1396 | **957 (2.78×)** | 115 | 2.38 / 2.01× @ B=2/4 |

Routing: 79% local @8K, 68% @32K. Speedup **erodes with batch** at the kernel level.

## Level 3 — End-to-end (tokens/s)
`experiments/bench_amdahl.py`, `experiments/bench_batch_e2e.py`.

vs context (B=1): **1.21× @8K → 1.46× @16K → 1.59× @32K** (all-local ceiling 1.19/1.56/2.00×).

| ctx | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|--:|--:|--:|--:|--:|
| 8K | 1.31× | 1.32× | 1.54× | 1.79× | **2.35×** |
| 32K | 1.25× | 2.04× | **2.10×** | — | — |

Unlike the kernel, e2e throughput speedup **grows** with batch (the fixed
~2.2 ms/step CPU/scheduling overhead amortizes faster than the kernel advantage
erodes). AHA is a throughput-regime win at e2e.

## Why the ratios differ across levels (state this explicitly)
- **L1 → L2:** L1 uses a *uniform* 70%-local pattern and excludes the merge
  kernel + engine overhead (overstates: ~6× @32K). L2 uses the *real,
  heterogeneous* gate (some layers route mostly global, dragging the average) and
  includes merge — so the honest realized number is 2.78×. **L2 is the product number.**
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
