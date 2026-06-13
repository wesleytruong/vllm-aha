#!/bin/bash
#SBATCH --job-name=aha-batch64k
#SBATCH --partition=ghx4
#SBATCH --account=bhdo-dtai-gh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#
# Follow-up high-batch sweep. The main run's batch sweep is pinned to 131072 ctx,
# where B>=8 OOMs on this ~96GB GH200 (B=8 = 128GB KV). Re-run the batch sweep at
# 65536 ctx, where B=8 = 64GB KV fits, to capture the high-batch throughput point.
# Writes results/batch_e2e/e2e_ctx65536_b{1,2,4,8}.json + batch_curve_ctx65536.json;
# make_figures globs batch_curve_*.json so both contexts land in batch_e2e.csv.
set -uo pipefail

REPO=/u/wtruong/AHA-Research/vllm-aha
cd "$REPO"
mkdir -p logs results results/batch_e2e

source "$REPO/.venv/bin/activate"
export PATH="$REPO/.venv/bin:$PATH"
module load cuda/12.9 2>/dev/null || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "[env] node=$(hostname) python=$(which python)"
nvidia-smi -L

echo "===== BATCH sweep @ 64K  (e2e throughput vs batch, includes B=8) ====="
python experiments/bench_batch_e2e.py --context 65536 --batches 1,2,4,8 --skip-existing

echo "===== Rebuild figures/CSVs (now spans 64K + 128K batch curves) ====="
python experiments/make_figures.py || echo "WARN: make_figures failed (non-fatal)"

echo "===== DONE. ====="
