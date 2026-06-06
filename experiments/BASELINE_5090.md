# AHA baseline results — RTX 5090 (32 GB)

Reference numbers from the dev box, to compare against the H100/H200 runs.
Model: `xuan-luo/AHA-OLMO2` (16 layers, 16 KV-heads, head_dim 128, MHA, fp16).
Gate routes ~72–82% of heads to local (128-token window); prefill is full
attention (gate is decode-only). Reproduce with `experiments/slurm/run_all.sbatch`.

## Decode-attention kernel speedup (nsys decode-direct, µs/step, B=1)
Real gate vs forced all-global (full attention); monotonic in routing fraction.

| ctx | all-global | half | **real gate** | all-local |
|----:|-----------:|-----:|--------------:|----------:|
| 8K  | 809 | 436 | **258 (3.1×)** | 114 |
| 32K | 2656 | 1396 | **957 (2.8×)** | 115 |

`dense-fi ≈ aha-global` within 0.2% (control). Speedup **erodes with batch** at
the kernel level (8K: 3.1×→2.1× for B=1→16).

## e2e decode speedup vs context (Amdahl curve, B=1, tok/s)
Grows with context as attention's share of the step rises.

| ctx | full tok/s | real tok/s | **e2e speedup** | all-local ceiling |
|----:|-----------:|-----------:|----------------:|------------------:|
| 8K  | 322 | 391 | **1.21×** | 1.19× |
| 16K | 210 | 306 | **1.46×** | 1.56× |
| 32K | 198 | 315 | **1.59×** | 2.00× |

## e2e throughput speedup vs batch (tok/s)
Unlike the kernel, e2e throughput speedup **grows** with batch — the fixed
~2.2 ms/step CPU/scheduling overhead (heavy at B=1) amortizes faster than the
kernel advantage erodes. AHA is a throughput-regime win at e2e.

| ctx | B=1 | B=2 | B=4 | B=8 | B=16 |
|----:|----:|----:|----:|----:|-----:|
| 8K  | 1.31× | — | 1.54× | 1.79× | **2.35×** |
| 32K | 1.25× | 2.04× | **2.10×** | — | — |

## Headline
Real-gate AHA: **~2.8–3.1× decode-attention** kernel speedup → **1.2–1.6× e2e at
B=1**, rising to **2.1–2.35× e2e at higher batch**, and growing with context. On
H100 (80 GB) push contexts to 64K–128K+ and larger batches, where both curves
extend toward the kernel ceiling. The earlier "0% / kernel-to-engine gap" was a
torch.compile cache confound (gate override baked + served stale), now fixed.
