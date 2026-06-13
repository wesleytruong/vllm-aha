#!/bin/bash
#SBATCH --job-name=aha-sweep
#SBATCH --partition=ghx4               # Delta GH200 partition (nvidia_gh200_120gb)
#SBATCH --account=bhdo-dtai-gh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#
# Context / batch / routing sweep for AHA (vLLM main + FlashInfer aha-router).
# Self-contained (does NOT source the stock env.sh, whose default repo path is
# wrong and whose `config.py | head` pipe can trip `set -e`). Submit with:
#     sbatch experiments/slurm/run_sweep.sbatch
# Outputs: results/{amdahl,batch_e2e,routing,figures}/
set -uo pipefail

REPO=/u/wtruong/AHA-Research/vllm-aha
cd "$REPO"
mkdir -p logs results results/routing

# --- Python env (uv venv with editable vllm + flashinfer-aha) ---------------
source "$REPO/.venv/bin/activate"
export PATH="$REPO/.venv/bin:$PATH"

# --- CUDA for FlashInfer JIT (nvcc); hpc_sdk CUDA is already on PATH ---------
module load cuda/12.9 2>/dev/null || true

# --- Offline: model pre-staged in ~/.cache/huggingface; GH200 nodes have no net
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# --- GH200 "120GB" reports ~96GB HBM, so config.py picks the h100 (80GB)
# profile. Its default batch sweep [1,2,4,8] @ 131072 ctx OOMs at B=8 (128GB KV >
# ~86GB usable @ gpu_mem 0.9); B=2 (32GB) and B=4 (64GB) fit. Cap to drop B=8.
# Promote to a real gh200 profile in config.py later if this becomes the default.
export AHA_BATCHES=1,2,4

echo "[env] node=$(hostname) python=$(which python)"
nvidia-smi -L
echo "[gpu-profile] (decides the context/batch grid)"
python experiments/config.py

echo "===== 1/3  CONTEXT sweep  (Amdahl: e2e speedup vs context, batch=1) ====="
python experiments/bench_amdahl.py --batch 1 --skip-existing

echo "===== 2/3  BATCH sweep  (e2e throughput vs batch) ====="
python experiments/bench_batch_e2e.py --skip-existing

echo "===== 3/3  ROUTING sweep  (local-route % vs sequence length) ====="
python probe_aha_gate_vs_seqlen.py
cp -f /tmp/aha_gate_vs_seqlen.csv results/routing/ 2>/dev/null || true

echo "===== Figures / CSV tables ====="
python experiments/make_figures.py || echo "WARN: make_figures failed (non-fatal)"

echo "===== DONE.  Results in results/{amdahl,batch_e2e,routing,figures}/ ====="
