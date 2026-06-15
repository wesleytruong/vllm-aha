# AHA-in-vLLM experiments

Systems-engineering evaluation of **All-or-Here Attention (AHA)** — a learned
per-head, per-token gate that routes each decode head between global (full) and
local (128-token sliding-window) attention — integrated into vLLM v1 with a
FlashInfer router decode kernel. These scripts measure where the kernel-level
speedup goes end-to-end, and scale from a 32 GB RTX 5090 to an 80 GB+ H100/H200
via a GPU-aware config (no per-run editing).

> **Headline (5090, B=1):** the real gate prunes ~70–80% of heads to local and
> delivers **~2.8–3.1× decode-attention** speedup, which dilutes by Amdahl to
> **~1.2× e2e @8K → ~1.6× @32K**, growing toward the kernel ceiling at longer
> context (the H100 experiment). All monotonic in routing fraction; controls
> validated.

## 0. Prerequisites (the two modified forks)

This repo (`vllm-aha-wt`) is an **editable install of a modified vLLM**, and it
depends on a **modified FlashInfer** fork (`../flashinfer`). Both carry AHA
changes and must be present and installed editable on the cluster. The key
edits:

- vLLM: `vllm/model_executor/models/olmo2_aha.py` (the AHA model + cache-safe gate
  override buffers), `vllm/v1/attention/backends/aha_flashinfer.py` (router decode
  plan), `vllm/v1/attention/backends/flashinfer.py` (router scheduler path),
  `vllm/config/compilation.py` + `vllm/transformers_utils/...` (config/registry).
- FlashInfer: `flashinfer/decode.py`, `include/flashinfer/attention/prefill.cuh`
  (per-head router/window in the tensor-core decode kernel).

On the cluster, clone/sync the same forks and install editable:

```bash
cd vllm-aha-wt     && pip install -e . --no-build-isolation
cd ../flashinfer   && pip install -e . --no-build-isolation   # JIT-compiles kernels on first use (needs CUDA + ninja)
```

`nsys` (Nsight Systems) is needed only for the kernel-direct sweep; the e2e and
routing benchmarks are pure timing and need no profiler.

## 1. Data

The gate is content-dependent, so benchmarks use real PG-19 prose from
`.benchmark_datasets/pg19-test_*.jsonl` (~22 MB). Either **rsync** the existing
caches from the dev box (recommended, deterministic):

```bash
rsync -a <devbox>:.../vllm-aha-wt/.benchmark_datasets/ ./.benchmark_datasets/
```

or **regenerate** from Hugging Face:

```bash
python experiments/prepare_data.py --min-tokens 600000
```

## 2. GPU profile (auto-detected)

`experiments/config.py` maps detected HBM → feasible contexts/batches/memory.
Check what it picks (and override via `AHA_GPU_PROFILE`, `AHA_CONTEXTS`,
`AHA_BATCHES`, `AHA_GPU_MEM`, `AHA_MODEL_MAX`):

```bash
python experiments/config.py
# profile=h100 ... amdahl_contexts=[8192,16384,32768,65536,131072] ...
```

## 3. Run everything

```bash
sbatch experiments/slurm/run_all.sbatch      # edit env.sh first (repo path, modules, venv)
```

or individually (from the repo root):

| Experiment | Command | Output |
|---|---|---|
| **Kernel microbench** (Level 1) — *in the [FlashInfer fork](https://github.com/wesleytruong/flashinfer-aha/tree/aha-router/benchmarks)* | `cd ../flashinfer && python benchmarks/bench_router_context_sweep.py` (+ `…_batch_sweep.py`) | `../flashinfer/benchmarks/sweep_logs/*.txt` |
| **Amdahl curve** (e2e speedup vs context) | `python experiments/bench_amdahl.py --batch 1` | `results/amdahl/` |
| **Batch × e2e** throughput | `python experiments/bench_batch_e2e.py` | `results/batch_e2e/` |
| **ITL / TPOT** matrix (inter-token latency) | `python experiments/bench_itl_grid.py` | `results/itl/` |
| **Kernel-direct** decode sweep (nsys) + overrides | `python experiments/run_kernel_sweep.py` | `results/nsys_cachefix/`, `…_direct.json` |
| **Routing** fraction vs context | `python probe_aha_gate_vs_seqlen.py` | `/tmp/aha_gate_vs_seqlen.csv` |
| **Figures/CSVs** | `python experiments/make_figures.py` | `results/figures/` |

## 4. What each experiment shows

### Level 1 — Kernel microbench (in the FlashInfer fork)

- **Router KV-skip microbench** — lives in the [FlashInfer fork](https://github.com/wesleytruong/flashinfer-aha/tree/aha-router/benchmarks):
  `benchmarks/bench_router_kv_skip.py` (harness), driven by
  `bench_router_context_sweep.py` (seq_len → 1M) and `bench_router_batch_sweep.py`
  (batch → 256), plus a window sweep. It times the paged tensor-core decode kernel
  at routing fractions {`plain_full`, `all-global`, `mix@50/70/90`, `all-local`}
  and verifies the design claim directly: when a head routes local, out-of-window
  KV tiles are **skipped** (predicated `cp_async` loads in `decode.cuh`), so DRAM
  reads drop. So windowing makes the kernel cost
  **context-flat** while full attention scales with context, and the per-launch
  speedup is the kernel **ceiling** for that fraction (the real heterogeneous gate
  realizes less — see L2). Result tables: `../flashinfer/benchmarks/sweep_logs/*.txt`.

### Level 2 — Kernel-direct (the kernel inside live vLLM, nsys)

- **Kernel-direct** (`run_kernel_sweep.py`) — isolates the decode attention kernel
  per step. The prompt is primed into the prefix cache *before* the profiler
  window (`AHA_PFC_DECODE=1`), so the captured `generate` is a pure cache hit and
  the trace contains only decode-shaped launches — clean decode isolation at ANY
  batch (no chunked-prefill interleave to subtract). Real-gate decode speedup is
  a flat ~2.7–3.5× across B=1→16; `dense-fi ≈ aha-global` validates the
  full-attention control. Each cell logs measured SWA% (routing).
  The parsed per-cell numbers live in `results/nsys_cachefix_direct.json` and
  `results/figures/decode_speedup.csv` (committed). The raw `.nsys-rep` captures
  (~77 MB) are **not** in git — download them from the
  [GH200 nsys captures release](https://github.com/wesleytruong/vllm-aha/releases/tag/gh200-nsys-captures).

### Level 3 — End-to-end (what the user actually sees)

- **ITL / TPOT** (`bench_itl_grid.py`) — inter-token latency (TPOT = ms per output
  token), the user-facing decode latency, swept over a context×batch grid with
  batch held constant across contexts (a column isolates the context effect, a row
  the batch effect). Real-gate vs full: the latency speedup **grows with both
  context and batch**, from ~1.06× (8K, B=1) to ~2.0× (64K B=8 / 128K B=4) — e.g.
  at 128K B=1 it cuts TPOT 7.2 → 4.4 ms (1.64×). `itl_summary.csv`: context,
  batch, swa%, full/real TPOT ms, latency_speedup, tok/s, all-local floor.
- **Amdahl curve** (`bench_amdahl.py`) — measured e2e decode speedup (real gate vs
  full attention) rises with context as attention's share of the step grows,
  toward the kernel ceiling. `amdahl_curve.csv` columns: context, batch,
  e2e_speedup, alllocal_ceiling, real/full tok/s, routing %.
- **Batch × e2e** (`bench_batch_e2e.py`) — fixed per-step CPU/scheduling overhead
  amortizes with batch, so e2e gain *grows* with batch; the decode-attention
  kernel speedup itself is roughly flat in batch (see Kernel-direct), not eroding.

### Mechanism

- **Routing** (`probe_aha_gate_vs_seqlen.py`) — the gate's local-routing fraction
  vs sequence length (per-layer); ~70–85% local on PG-19 prose, the input that
  drives every speedup above.

## Validation

**Kernel correctness — validated locally (unit test).** The router decode/prefill
kernel is checked against a *blended* full + sliding-window reference in the
FlashInfer fork's `tests/attention/test_router_attention.py`: per head, the router
output must equal **full attention** where the gate routes global (`router==0`)
and **sliding-window attention** where it routes local (`router==1`). All-global
and all-local routers reproduce standard full / SWA attention exactly, and a mixed
router matches each head's respective baseline — for both the decode and prefill
wrappers, across batch sizes, context lengths, window sizes, and the tensor-core
path (`rtol/atol = 1e-3`). A CUDA-graph regression test additionally flips the
router in place across replays and asserts the output follows the *current*
routing (the fix for the stale-buffer windowing bug; it also pins the polarity
`1 == SWA`).

**Task accuracy — separate suite.** End-task quality is evaluated by running the
**AHA paper's own benchmarks** through this vLLM integration via
`lm-evaluation-harness` (`--model vllm`, `pretrained=xuan-luo/AHA-OLMO2,...`), real
gate vs forced `all-global`. The integration gates **decode only** (prefill = full
attention) while the paper gates *all* tokens — so only the paper's **generation**
tasks (GSM8K, MBPP, MultiNews) exercise our gate; the multiple-choice tasks (MMLU,
HellaSwag, CSQA) are loglikelihood/prefill-scored and read as full attention (a
no-regression check).

## 5. Notes / gotchas

- **Gate override is cache-safe.** `VLLM_AHA_GATE_OVERRIDE=global|local|half`
  forces routing for the ablations; it is applied via runtime buffers, so the
  torch.compile cache is correct with no flag. (Historical bug: it used to be
  baked into the compiled graph and served stale from cache — fixed.) The
  benches flip it in-process via `collective_rpc` for clean A/B with no reload.
- **Run from the repo root** so scripts find `.benchmark_datasets/` and the root
  harness scripts.
- **OOM at large context×batch** — lower `AHA_GPU_MEM` or trim `AHA_BATCHES`;
  KV = 128 KB/token, so context×batch must fit `~gpu_mem·HBM − weights`.
- The model is **AHA-OLMO2** (`xuan-luo/AHA-OLMO2`): 16 layers, 16 KV-heads,
  head_dim 128, MHA, fp16.
