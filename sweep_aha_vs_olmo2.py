#!/usr/bin/env python3
"""Decomposition sweep: dense / AHA configs / local-only on PG-19 prose.

Each cell (variant x context-length x batch) runs in its own subprocess (vLLM
keeps too much GPU state for back-to-back LLM() creation in one process) and is
repeated --repeats times so we can report a coefficient of variation (CV) and
flag noisy cells. PG-19 token slices are cached per context length so every run
sees the same prompts.

Variants (for the A/C/D decomposition + ladder, see the eval plan):
    olmo2          Dense baseline (stock OLMo2, full attention)         [A]
    aha-global     AHA flashinfer, gate forced global (VLLM_AHA_GATE_OVERRIDE=global)
                   -> AHA machinery, zero pruning                       [C]
    aha-flashinfer AHA flashinfer, real learned gate                    [D]
    aha-local      AHA flashinfer, gate forced local (lower-bound via AHA path)
    local-only     Pure sliding-window model (speed ceiling)
    aha-dual / aha-routed / aha-greedy   variant ladder (other impls)

Usage:
    .venv/bin/python sweep_aha_vs_olmo2.py \\
        --input-lens 8192 32768 --ns 1 16 --repeats 3 --measure-phases \\
        --variants olmo2 aha-global aha-flashinfer local-only \\
        --out /tmp/aha_decomp_sweep.json
"""

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile

MODEL_OLMO2 = "allenai/OLMo-2-0425-1B"
MODEL_AHA = "xuan-luo/AHA-OLMO2"
MAX_MODEL_LEN = 32000

# CV (std/mean) above this fraction marks a cell as noisy / untrustworthy.
CV_FLAG_THRESHOLD = 0.10


def build_pg19_prompts(num_prompts: int, input_len: int, tokenizer_path: str,
                       cache_path: str) -> list[list[int]]:
    """Tokenize PG-19 test books until we have num_prompts * input_len tokens,
    slice into distinct prompts, and cache to disk for reuse."""
    key = (num_prompts, input_len, tokenizer_path)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == list(key):
            return cached["prompts"]

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    ds = load_dataset(
        "deepmind/pg19", split="test", streaming=True, trust_remote_code=True
    )

    needed = num_prompts * input_len + 2000
    buf: list[int] = []
    for rec in ds:
        buf.extend(tok(rec["text"], add_special_tokens=False).input_ids)
        if len(buf) >= needed:
            break
    if len(buf) < needed:
        raise RuntimeError(f"PG-19 only gave {len(buf)} tokens, need {needed}")

    prompts = [
        buf[i * input_len : (i + 1) * input_len] for i in range(num_prompts)
    ]
    with open(cache_path, "wb") as f:
        pickle.dump({"key": list(key), "prompts": prompts}, f)
    return prompts


CHILD_TEMPLATE = r"""
import json, os, pickle, sys, time
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
if os.environ.get("VLLM_LOCAL_ONLY_ENV") == "1":
    os.environ["VLLM_LOCAL_ONLY"] = "1"

import torch
from vllm import LLM, SamplingParams

with open(sys.argv[1], "rb") as f:
    cfg = pickle.load(f)

prompts = cfg["prompts"]
output_len = cfg["output_len"]
model = cfg["model"]
hf_overrides = cfg["hf_overrides"]
gpu_mem = cfg["gpu_mem"]
max_model_len = cfg["max_model_len"]
trust = cfg["trust_remote_code"]
max_num_seqs = cfg["max_num_seqs"]
measure_phases = cfg["measure_phases"]

llm = LLM(
    model=model,
    dtype="half",
    trust_remote_code=trust,
    gpu_memory_utilization=gpu_mem,
    max_model_len=max_model_len,
    hf_overrides=hf_overrides or None,
    disable_log_stats=True,
    max_num_seqs=max_num_seqs,
)

inputs = [{"prompt_token_ids": p} for p in prompts]

# warmup (small, to trigger any first-time compile that would otherwise
# pollute the timing)
warm_sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
llm.generate([inputs[0]], warm_sp, use_tqdm=False)


def timed_generate(max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(inputs, sp, use_tqdm=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, outs


result = {}

if measure_phases:
    # Calibration pass with max_tokens=1 isolates prefill (+1 decode step);
    # the difference from the full run gives the decode-only time.
    t_calib, _ = timed_generate(1)
    t_total, outputs = timed_generate(output_len)
    t_decode = max(t_total - t_calib, 0.0)
    elapsed = t_total
else:
    elapsed, outputs = timed_generate(output_len)
    t_calib = t_decode = None

total_in = sum(len(o.prompt_token_ids) for o in outputs)
total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
result.update({
    "elapsed_s": elapsed,
    "total_prompt_tokens": total_in,
    "total_output_tokens": total_out,
    "req_per_s": len(outputs) / elapsed,
    "total_tok_s": (total_in + total_out) / elapsed,
    "output_tok_s": total_out / elapsed,
})
if measure_phases:
    result["prefill_ms_per_token"] = t_calib * 1000.0 / total_in if total_in else None
    result["decode_ms_per_token"] = t_decode * 1000.0 / total_out if total_out else None

with open(sys.argv[2], "w") as f:
    json.dump(result, f)
"""


def run_child(prompts, output_len, model, hf_overrides, gpu_mem,
              max_num_seqs, trust_remote_code, measure_phases,
              local_only=False, extra_env=None):
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        cfg = {
            "prompts": prompts,
            "output_len": output_len,
            "model": model,
            "hf_overrides": hf_overrides,
            "gpu_mem": gpu_mem,
            "max_model_len": MAX_MODEL_LEN,
            "trust_remote_code": trust_remote_code,
            "max_num_seqs": max_num_seqs,
            "measure_phases": measure_phases,
        }
        pickle.dump(cfg, f)
        cfg_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "")
    if local_only:
        env["VLLM_LOCAL_ONLY_ENV"] = "1"
    if extra_env:
        env.update(extra_env)

    child = subprocess.run(
        [sys.executable, "-c", CHILD_TEMPLATE, cfg_path, out_path],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    try:
        with open(out_path) as f:
            result = json.load(f)
    except Exception:
        result = {"error": f"rc={child.returncode}", "tail": child.stderr[-1500:]}
    finally:
        for p in (cfg_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass
    return result


# variant -> how to launch it. extra_env carries the gate override.
VARIANT_CFGS = {
    "olmo2": {  # [A] dense baseline
        "model": MODEL_OLMO2, "trust": False,
        "overrides": {"max_position_embeddings": MAX_MODEL_LEN},
        "local_only": False, "extra_env": None,
    },
    "aha-global": {  # [C] AHA machinery, gate forced global (no pruning)
        "model": MODEL_AHA, "trust": True,
        "overrides": {"attention_implementation": "flashinfer",
                      "max_position_embeddings": MAX_MODEL_LEN},
        "local_only": False, "extra_env": {"VLLM_AHA_GATE_OVERRIDE": "global"},
    },
    "aha-flashinfer": {  # [D] real AHA
        "model": MODEL_AHA, "trust": True,
        "overrides": {"attention_implementation": "flashinfer",
                      "max_position_embeddings": MAX_MODEL_LEN},
        "local_only": False, "extra_env": None,
    },
    "aha-local": {  # AHA machinery, gate forced local (lower bound via AHA path)
        "model": MODEL_AHA, "trust": True,
        "overrides": {"attention_implementation": "flashinfer",
                      "max_position_embeddings": MAX_MODEL_LEN},
        "local_only": False, "extra_env": {"VLLM_AHA_GATE_OVERRIDE": "local"},
    },
    "local-only": {  # pure sliding-window model (speed ceiling)
        "model": MODEL_AHA, "trust": True,
        "overrides": {"attention_implementation": "flashinfer",
                      "max_position_embeddings": MAX_MODEL_LEN},
        "local_only": True, "extra_env": None,
    },
    "aha-dual": {
        "model": MODEL_AHA, "trust": True,
        "overrides": {"attention_implementation": "dual",
                      "max_position_embeddings": MAX_MODEL_LEN},
        "local_only": False, "extra_env": None,
    },
    "aha-routed": {
        "model": MODEL_AHA, "trust": True,
        "overrides": {"attention_implementation": "routed",
                      "max_position_embeddings": MAX_MODEL_LEN},
        "local_only": False, "extra_env": None,
    },
    "aha-greedy": {
        "model": MODEL_AHA, "trust": True,
        "overrides": {"attention_implementation": "greedy",
                      "max_position_embeddings": MAX_MODEL_LEN},
        "local_only": False, "extra_env": None,
    },
}


def _stats(values: list[float]) -> dict:
    """mean / std / cv / min / max over the per-repeat values (noise-robust)."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return {"mean": None, "std": None, "cv": None, "min": None,
                "max": None, "n": 0}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    cv = (std / mean) if mean else None
    return {"mean": mean, "std": std, "cv": cv,
            "min": min(vals), "max": max(vals), "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 16],
                    help="batch sizes (num concurrent sequences)")
    ap.add_argument("--input-lens", type=int, nargs="+", default=[8192],
                    help="context lengths to sweep")
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per cell (for CV / best-of)")
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--measure-phases", action="store_true",
                    help="add a max_tokens=1 calibration pass to split "
                         "prefill vs decode (decode_ms_per_token)")
    ap.add_argument("--variants", nargs="+",
                    default=["olmo2", "aha-global", "aha-flashinfer",
                             "local-only"],
                    choices=list(VARIANT_CFGS.keys()))
    ap.add_argument("--out", default="/tmp/aha_decomp_sweep.json")
    args = ap.parse_args()

    cache_dir = "/tmp/pg19_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Primary metric: decode_ms_per_token (lower=better) when measuring phases,
    # else output_tok_s (higher=better). "best" picks the least-contended run.
    metric = "decode_ms_per_token" if args.measure_phases else "output_tok_s"
    lower_is_better = args.measure_phases

    cells = []  # one dict per (input_len, n, variant) with aggregated stats
    print(f"metric={metric}  repeats={args.repeats}  "
          f"variants={args.variants}", flush=True)

    for input_len in args.input_lens:
        max_n = max(args.ns)
        prompts_all = build_pg19_prompts(
            max_n, input_len, MODEL_AHA,
            os.path.join(cache_dir, f"{max_n}x{input_len}.pkl"),
        )
        print(f"\n[input_len={input_len}] loaded {len(prompts_all)} prompts",
              flush=True)
        print(f"{'variant':<16}{'N':>4}{'best':>12}{'mean':>12}"
              f"{'cv%':>8}{'flag':>6}", flush=True)
        print("-" * 58, flush=True)

        for n in args.ns:
            prompts = prompts_all[:n]
            for variant in args.variants:
                vc = VARIANT_CFGS[variant]
                repeats_raw = []
                for _ in range(args.repeats):
                    r = run_child(
                        prompts=prompts,
                        output_len=args.output_len,
                        model=vc["model"],
                        hf_overrides=vc["overrides"],
                        gpu_mem=args.gpu_mem,
                        max_num_seqs=args.max_num_seqs,
                        trust_remote_code=vc["trust"],
                        measure_phases=args.measure_phases,
                        local_only=vc["local_only"],
                        extra_env=vc["extra_env"],
                    )
                    repeats_raw.append(r)

                errors = [r for r in repeats_raw if "error" in r]
                metric_vals = [r.get(metric) for r in repeats_raw
                               if "error" not in r]
                st = _stats(metric_vals)
                best = (st["min"] if lower_is_better else st["max"])
                cv = st["cv"]
                flagged = cv is not None and cv > CV_FLAG_THRESHOLD

                cell = {
                    "variant": variant, "input_len": input_len, "n": n,
                    "output_len": args.output_len, "metric": metric,
                    "lower_is_better": lower_is_better,
                    "best": best, "stats": st, "flagged": flagged,
                    "n_errors": len(errors),
                    "repeats": repeats_raw,
                }
                cells.append(cell)

                if st["n"] == 0:
                    tail = errors[0].get("tail", "") if errors else ""
                    print(f"{variant:<16}{n:>4}  ALL FAILED", flush=True)
                    if tail:
                        print(tail[-400:], flush=True)
                else:
                    cvp = f"{cv * 100:>7.1f}" if cv is not None else "    n/a"
                    flag = "  WARN" if flagged else ""
                    print(f"{variant:<16}{n:>4}{best:>12.3f}"
                          f"{st['mean']:>12.3f}{cvp}{flag:>6}", flush=True)

    payload = {
        "config": {
            "ns": args.ns, "input_lens": args.input_lens,
            "output_len": args.output_len, "repeats": args.repeats,
            "metric": metric, "lower_is_better": lower_is_better,
            "variants": args.variants, "cv_flag_threshold": CV_FLAG_THRESHOLD,
        },
        "cells": cells,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {len(cells)} cells to {args.out}", flush=True)
    print("Feed this into analyze_aha_decomposition.py for the A/C/D "
          "decomposition + Amdahl ceiling.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
