#!/usr/bin/env python3
"""GPU-aware experiment configuration.

Maps the detected GPU HBM size to feasible (context, batch) grids and memory
settings, so the same experiment scripts scale from a 32GB RTX 5090 to an 80GB
H100 (or larger) without hand-editing each run.

Model fact (AHA-OLMO2): per-token KV = 128 KB (16 layers x 16 KV-heads x 128
head_dim x 2[k,v] x 2 bytes fp16). So KV bytes = 128KB * context * batch. Keep
context*batch below ~(usable_HBM / 128KB) with headroom for weights+activations.

Override anything via env: AHA_GPU_PROFILE (5090|h100|h200|auto), AHA_GPU_MEM,
AHA_MODEL_MAX, AHA_CONTEXTS (comma list), AHA_BATCHES (comma list).
"""
import os

PER_TOKEN_KV_KB = 128  # AHA-OLMO2, fp16

# Profiles keyed by approximate HBM (GiB). Contexts in tokens; batches per ctx.
PROFILES = {
    "5090": {  # 32 GB
        "hbm_gb": 32, "gpu_mem": 0.80, "model_max": 33792,
        "amdahl_contexts": [8192, 16384, 32768],
        "batch_context": 32768,
        "batches": [1, 2, 4],
        # per-context feasible batch lists for the kernel/e2e batch sweep
        "ctx_batches": {8192: [1, 4, 8, 16], 16384: [1, 2, 4, 8], 32768: [1, 2, 4]},
    },
    "h100": {  # 80 GB
        "hbm_gb": 80, "gpu_mem": 0.90, "model_max": 263168,
        "amdahl_contexts": [8192, 16384, 32768, 65536, 131072],
        "batch_context": 131072,
        "batches": [1, 2, 4, 8],
        "ctx_batches": {8192: [1, 8, 16, 32], 32768: [1, 4, 8, 16],
                        65536: [1, 2, 4, 8], 131072: [1, 2, 4]},
    },
    "h200": {  # 141 GB
        "hbm_gb": 141, "gpu_mem": 0.90, "model_max": 525312,
        "amdahl_contexts": [8192, 32768, 65536, 131072, 262144],
        "batch_context": 131072,
        "batches": [1, 4, 8, 16],
        "ctx_batches": {32768: [1, 8, 16, 32], 131072: [1, 4, 8, 16],
                        262144: [1, 2, 4]},
    },
}


def _detect_profile() -> str:
    env = os.environ.get("AHA_GPU_PROFILE", "auto")
    if env != "auto":
        return env
    # Use nvidia-smi (NOT torch.cuda) — importing this module must not initialize
    # a CUDA context in the parent process, or vLLM's worker spawn breaks.
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        gb = int(out.strip().splitlines()[0]) / 1024
    except Exception:
        return "5090"
    if gb < 48:
        return "5090"
    if gb < 110:
        return "h100"
    return "h200"


def get_config() -> dict:
    prof = _detect_profile()
    cfg = dict(PROFILES.get(prof, PROFILES["5090"]))
    cfg["profile"] = prof
    # env overrides
    if "AHA_GPU_MEM" in os.environ:
        cfg["gpu_mem"] = float(os.environ["AHA_GPU_MEM"])
    if "AHA_MODEL_MAX" in os.environ:
        cfg["model_max"] = int(os.environ["AHA_MODEL_MAX"])
    if "AHA_CONTEXTS" in os.environ:
        cfg["amdahl_contexts"] = [int(x) for x in os.environ["AHA_CONTEXTS"].split(",")]
    if "AHA_BATCHES" in os.environ:
        cfg["batches"] = [int(x) for x in os.environ["AHA_BATCHES"].split(",")]
    return cfg


def kv_gb(context: int, batch: int) -> float:
    return PER_TOKEN_KV_KB * context * batch / (1024**2)


if __name__ == "__main__":
    c = get_config()
    print(f"profile={c['profile']} hbm={c['hbm_gb']}GB gpu_mem={c['gpu_mem']} "
          f"model_max={c['model_max']}")
    print(f"amdahl_contexts={c['amdahl_contexts']}")
    print(f"batch sweep: ctx={c['batch_context']} batches={c['batches']}")
    print("feasibility (KV GB) at batch_context:")
    for b in c["batches"]:
        print(f"  B={b}: {kv_gb(c['batch_context'], b):.1f} GB KV")
