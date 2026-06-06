#!/bin/bash
# Shared environment setup for AHA experiments on a SLURM cluster.
# EDIT the marked lines for your cluster, then the sbatch scripts source this.
set -euo pipefail

# --- EDIT: absolute path to the vllm-aha-wt repo on the cluster --------------
export AHA_REPO="${AHA_REPO:-$HOME/AHA-research/vllm-aha-wt}"

# --- EDIT: how to activate the Python env (venv or conda) --------------------
# venv example:
source "$AHA_REPO/.venv/bin/activate"
# conda example (comment the venv line above, uncomment these):
# module load anaconda
# conda activate aha

# --- EDIT: load CUDA + Nsight Systems (nsys) modules for your cluster --------
module load cuda || true          # provides nvcc/CUDA runtime for FlashInfer JIT
module load nsight-systems || true  # provides `nsys` for the kernel-direct sweep

# Put the venv bin first so `ninja` (FlashInfer JIT) and `nsys` resolve.
export PATH="$AHA_REPO/.venv/bin:$PATH"

# AHA: the gate-override compile-cache bug is FIXED (override is a runtime
# buffer), so the cache is safe to leave on. If running an older checkout where
# it is not fixed, uncomment to force correct override behavior:
# export VLLM_DISABLE_COMPILE_CACHE=1

cd "$AHA_REPO"
echo "[env] repo=$AHA_REPO python=$(which python) nsys=$(which nsys 2>/dev/null || echo MISSING)"
python experiments/config.py | head -1   # prints detected GPU profile
