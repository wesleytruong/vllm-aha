#!/bin/bash
#SBATCH --job-name=aha-microbench
#SBATCH --partition=ghx4
#SBATCH --account=bhdo-dtai-gh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#
# Isolated FlashInfer decode-kernel microbench (microbench_aha_attn.py): times the
# kernel directly on synthetic paged-KV (no engine/model), decomposing dense (A) vs
# all-global router (C) vs real-AHA router=Bernoulli(0.5) (D). Cheap GPU job.
# Same context/batch grid as the e2e sweeps. Output: results/microbench_s.json
set -uo pipefail

REPO=/u/wtruong/AHA-Research/vllm-aha
cd "$REPO"
mkdir -p logs results

source "$REPO/.venv/bin/activate"
export PATH="$REPO/.venv/bin:$PATH"   # so FlashInfer JIT finds ninja
module load cuda/12.9 2>/dev/null || true

echo "[env] node=$(hostname) python=$(which python)"
nvidia-smi -L

echo "===== FlashInfer decode-kernel microbench ====="
python microbench_aha_attn.py \
    --contexts 8192 16384 32768 65536 131072 \
    --batches 1 2 4 8 \
    --out results/microbench_s.json

echo "===== DONE.  results/microbench_s.json ====="
