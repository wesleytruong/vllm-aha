#!/bin/bash
#SBATCH --job-name=aha-nsysprobe
#SBATCH --partition=ghx4
#SBATCH --account=bhdo-dtai-gh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#
# Cheap capability probe: can a NON-root user collect GPU kernel data with nsys on
# this node? The kernel-direct sweep (run_kernel_sweep.py) is useless if not.
# Checks the driver's profiling-restriction flag AND does a tiny real nsys capture.
set -uo pipefail

REPO=/u/wtruong/AHA-Research/vllm-aha
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PATH="$REPO/.venv/bin:$PATH"
module load cuda/12.9 2>/dev/null || true

echo "[env] node=$(hostname)  nsys=$(which nsys)"
nvidia-smi -L

echo; echo "=== driver profiling-restriction flag (0 = users allowed, 1 = admin-only) ==="
grep -i "RmProfilingAdminOnly\|RestrictProfiling" /proc/driver/nvidia/params 2>&1 || \
  echo "(param not found)"

echo; echo "=== tiny nsys capture of a real CUDA kernel ==="
cat > /tmp/_nsys_probe.py <<'PY'
import torch
x = torch.randn(2048, 2048, device="cuda")
torch.cuda.synchronize()
for _ in range(50):
    x = x @ x
    x = x / x.norm()
torch.cuda.synchronize()
print("kernels launched")
PY
nsys profile -t cuda --force-overwrite=true -o /tmp/_nsys_probe \
  python /tmp/_nsys_probe.py 2>&1 | tail -5

echo; echo "=== does the capture contain GPU kernel rows? (the decisive test) ==="
nsys stats --report cuda_gpu_kern_sum /tmp/_nsys_probe.nsys-rep 2>&1 | head -20

echo; echo "=== VERDICT ==="
echo "If the table above lists kernels with Total Time > 0, profiling WORKS for us"
echo "and the kernel sweep is viable. If it says no CUDA kernel data / empty, it is"
echo "admin-restricted and the nsys sweep cannot produce kernel numbers."
