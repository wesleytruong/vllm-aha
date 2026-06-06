#!/usr/bin/env python3
"""End-to-end decode throughput: real gate vs full attention vs all-local.

The decode-direct nsys numbers isolate the attention KERNEL. This measures the
WALL-CLOCK e2e decode step (attention + MLP + norms + sampling + scheduling),
which is what tokens/s actually depends on (the attention speedup is diluted by
Amdahl). Per-step decode time = (t(mt_long) - t(mt_short)) / (mt_long-mt_short),
which cancels prefill. tokens/s = batch / per_step_time.

The gate override is now a runtime BUFFER (cache-safe), so we load ONCE and flip
between real-gate / all-global / all-local by writing the buffers via
collective_rpc — identical conditions, no reload, no recompile.

Env: INPUT_LEN, AHA_BATCH, AHA_MODEL_MAX, AHA_GPU_MEM.
"""
import json
import os
import pickle
import sys
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))
try:
    from config import get_config
    _CFG = get_config()
except Exception:
    _CFG = {"gpu_mem": 0.80, "model_max": 33792}

INPUT_LEN = int(os.environ.get("INPUT_LEN", "8192"))
BATCH = int(os.environ.get("AHA_BATCH", "1"))
MODEL_MAX = int(os.environ.get("AHA_MODEL_MAX", str(_CFG["model_max"])))
GPU_MEM = float(os.environ.get("AHA_GPU_MEM", str(_CFG["gpu_mem"])))
OUT_JSON = os.environ.get("AHA_E2E_OUT", "")
MT_SHORT, MT_LONG = 1, 65  # 64 decode steps isolated by subtraction

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def _set_override(self, keep, force_mode):
    """Runtime gate override via buffers (cache-safe). force_mode: 'zero'|'one'|'half'."""
    from vllm.model_executor.models.olmo2_aha import _AHAFlashInferAttention
    for _, m in self.model_runner.model.named_modules():
        if isinstance(m, _AHAFlashInferAttention):
            m._gate_keep.fill_(keep)
            if force_mode == "half":
                m._gate_force.zero_(); m._gate_force[::2] = 1.0
            elif force_mode == "one":
                m._gate_force.fill_(1.0)
            else:
                m._gate_force.fill_(0.0)
    return True


def _read_swa(self):
    from vllm.model_executor.models.olmo2_aha import _AHAFlashInferAttention
    s = t = 0
    for _, m in self.model_runner.model.named_modules():
        if isinstance(m, _AHAFlashInferAttention):
            s += int(m._probe_swa_sum.item()); t += int(m._probe_total.item())
            m._probe_swa_sum.zero_(); m._probe_total.zero_()
    return (s, t)


from pg19_loader import load_pg19_prompt  # noqa: E402
prompt_ids = load_pg19_prompt(INPUT_LEN, "xuan-luo/AHA-OLMO2")
inputs = [{"prompt_token_ids": prompt_ids[i:] + prompt_ids[:i]} for i in range(BATCH)]

llm = LLM(model="xuan-luo/AHA-OLMO2", dtype="half", trust_remote_code=True,
          gpu_memory_utilization=GPU_MEM, max_model_len=MODEL_MAX,
          hf_overrides={"attention_implementation": "flashinfer",
                        "max_position_embeddings": MODEL_MAX},
          enforce_eager=False, enable_prefix_caching=False,
          max_num_seqs=max(BATCH, 4), disable_log_stats=True)


def time_gen(mt, reps=3):
    sp = SamplingParams(temperature=0.0, max_tokens=mt, ignore_eos=True)
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        llm.generate(inputs, sp, use_tqdm=False)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


# Warmup
llm.generate([{"prompt_token_ids": prompt_ids[:128]}],
             SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
             use_tqdm=False)

CONFIGS = [
    ("real-gate", 1.0, "zero"),   # keep=1 → no override
    ("all-global", 0.0, "one"),   # full attention
    ("all-local", 0.0, "zero"),
    ("half", 0.0, "half"),
]
print(f"\n### e2e decode throughput  ctx={INPUT_LEN} batch={BATCH} "
      f"({MT_LONG - MT_SHORT} decode steps isolated) ###", flush=True)
print(f"{'config':>12} {'SWA%':>6} {'ms/step':>9} {'tok/s':>9} {'vs full':>8}", flush=True)
results = {}
for name, keep, force in CONFIGS:
    llm.collective_rpc(_set_override, args=(keep, force))
    llm.collective_rpc(_read_swa)  # reset probe
    t_short = time_gen(MT_SHORT)
    t_long = time_gen(MT_LONG)
    swa = llm.collective_rpc(_read_swa)[0]
    swa_pct = 100.0 * swa[0] / max(swa[1], 1)
    per_step = (t_long - t_short) / (MT_LONG - MT_SHORT)
    toks = BATCH / per_step
    results[name] = (per_step, toks, swa_pct)

full_step = results["all-global"][0]
local_step = results["all-local"][0]
for name, _, _ in CONFIGS:
    per_step, toks, swa_pct = results[name]
    spd = full_step / per_step
    print(f"{name:>12} {swa_pct:>5.0f}% {per_step*1e3:>8.2f} {toks:>9.0f} "
          f"{spd:>7.2f}x", flush=True)

# Amdahl decomposition (no profiler needed): all-local's attention is ~0, so
# attention's share of the full-attention step p = (step_global - step_local)/
# step_global, and the kernel attention speedup s = attn_real/attn_global where
# attn_x = step_x - step_local.
p_attn = (full_step - local_step) / full_step
attn_global = full_step - local_step
attn_real = results["real-gate"][0] - local_step
s_kernel = attn_real / attn_global if attn_global > 0 else float("nan")
e2e_speedup = full_step / results["real-gate"][0]
print(f"### Amdahl: p_attn(full)={p_attn:.3f}  s_kernel(real/global)={s_kernel:.3f}  "
      f"e2e_speedup(real)={e2e_speedup:.3f}x ###", flush=True)

if OUT_JSON:
    rec = {
        "context": INPUT_LEN, "batch": BATCH, "profile": _CFG.get("profile"),
        "mt_decode_steps": MT_LONG - MT_SHORT,
        "p_attn": p_attn, "s_kernel": s_kernel, "e2e_speedup": e2e_speedup,
        "configs": {n: {"step_ms": results[n][0] * 1e3, "tok_s": results[n][1],
                        "swa_pct": results[n][2]} for n, _, _ in CONFIGS},
    }
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"### wrote {OUT_JSON} ###", flush=True)
print("### done ###", flush=True)
