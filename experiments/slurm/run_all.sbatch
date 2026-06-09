#!/bin/bash
#SBATCH --job-name=aha-all
#SBATCH --gres=gpu:1                 # EDIT: e.g. gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# EDIT: add --partition / --account / --constraint for your cluster.
#
# Runs the full AHA experiment suite end-to-end on one GPU. The GPU profile
# (contexts/batches/memory) is auto-detected from HBM size (experiments/config.py),
# so this same script scales 5090 -> H100 -> H200 without edits.
#
# Submit:  sbatch experiments/slurm/run_all.sbatch
# Outputs: results/{amdahl,batch_e2e,nsys_cachefix,accuracy,figures}/

source "$(dirname "$0")/env.sh"
mkdir -p logs results

echo "===== 1/4  Data prep (PG-19) ====="
python experiments/prepare_data.py --min-tokens 600000 || \
  echo "WARN: data prep failed; ensure .benchmark_datasets/ is rsync'd"

echo "===== 2/4  Kernel microbench (Level 1) + Amdahl curve (e2e vs context) ====="
python experiments/kernel_microbench.py || echo "WARN: kernel microbench failed (flashinfer?)"
python experiments/bench_amdahl.py --batch 1

echo "===== 3/4  Batch x e2e throughput + ITL/TPOT matrix + online serving ====="
python experiments/bench_batch_e2e.py
python experiments/bench_itl_grid.py   # context x batch inter-token-latency matrix
# Online serving (Poisson arrivals against a live vLLM server). Bigger GPUs can
# push --input-len higher; --num-prompts bounds runtime. Skipped if it can't bind.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python experiments/bench_serving.py --input-len 8192 --request-rates "4,inf" \
  --num-prompts 100 --gpu-mem 0.6 --max-num-seqs 64 || echo "WARN: serving bench failed"

echo "===== 4/4  Kernel-direct decode sweep (nsys) + overrides + figures ====="
# nsys + override sweep. Contexts/batches come from the GPU profile.
python experiments/run_kernel_sweep.py || echo "WARN: kernel sweep failed (nsys?)"
python experiments/make_figures.py

echo "===== DONE.  Tables/figures in results/figures/ ====="
