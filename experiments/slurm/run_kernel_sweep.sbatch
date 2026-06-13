#!/bin/bash
#SBATCH --job-name=aha-kernsweep
#SBATCH --partition=ghx4
#SBATCH --account=bhdo-dtai-gh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#
# In-vLLM kernel-direct decode sweep (nsys). Isolates the DECODE ATTENTION KERNEL
# per step inside the real vLLM engine, across the h100-profile context x batch
# grid, for all 5 routing configs (dense-fi / aha-global / aha-local / aha-half /
# aha-flashinfer[real gate]). This is the kernel-level companion to the e2e sweep.
#
# Prereq verified (probe_nsys.sbatch): RmProfilingAdminOnly=0 on ghx4, so non-root
# nsys captures DO contain GPU kernel data. The sweep is idempotent (skips existing
# .nsys-rep), so if it hits the wall time just resubmit to continue.
# Output: results/nsys_cachefix/*.nsys-rep + results/nsys_cachefix_direct.json
set -uo pipefail

REPO=/u/wtruong/AHA-Research/vllm-aha
cd "$REPO"
mkdir -p logs results results/nsys_cachefix

source "$REPO/.venv/bin/activate"
export PATH="$REPO/.venv/bin:$PATH"   # so nsys + ninja (FlashInfer JIT) resolve
module load cuda/12.9 2>/dev/null || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "[env] node=$(hostname) python=$(which python) nsys=$(which nsys)"
nvidia-smi -L
echo "[gpu-profile] (decides the ctx x batch grid)"
python experiments/config.py

echo "===== Kernel-direct decode sweep (nsys) ====="
python experiments/run_kernel_sweep.py --out-dir results/nsys_cachefix

echo "===== Rebuild figures/CSVs (adds decode_speedup.csv) ====="
python experiments/make_figures.py || echo "WARN: make_figures failed (non-fatal)"

echo "===== DONE.  results/nsys_cachefix_direct.json + results/figures/decode_speedup.csv ====="
