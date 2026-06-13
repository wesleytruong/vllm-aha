#!/bin/bash
#SBATCH --job-name=aha-itl
#SBATCH --partition=ghx4
#SBATCH --account=bhdo-dtai-gh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#
# Inter-token-latency (ITL / TPOT) e2e benchmark over the ctx x batch matrix.
# Measures steady-state ms/token directly via prefix caching (no prefill
# subtraction). The grid driver sizes gpu_mem per cell and auto-skips cells whose
# KV won't fit (e.g. 128K x B=8), so no manual batch cap is needed.
# Output: results/itl/grid_ctx*_b*.json  +  results/figures/itl_summary.csv
set -uo pipefail

REPO=/u/wtruong/AHA-Research/vllm-aha
cd "$REPO"
mkdir -p logs results results/itl

source "$REPO/.venv/bin/activate"
export PATH="$REPO/.venv/bin:$PATH"
module load cuda/12.9 2>/dev/null || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "[env] node=$(hostname) python=$(which python)"
nvidia-smi -L
echo "[gpu-profile] (decides the ctx x batch grid)"
python experiments/config.py

echo "===== ITL / TPOT matrix (inter-token latency, e2e) ====="
python experiments/bench_itl_grid.py --skip-existing   # reuse the 5 done cells, run the rest

echo "===== Canonical ITL summary table ====="
python experiments/make_figures.py || echo "WARN: make_figures failed (non-fatal)"

echo "===== DONE.  results/itl/ + results/figures/itl_summary.csv ====="
