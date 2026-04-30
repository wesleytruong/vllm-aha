#!/usr/bin/env python3
"""Batch-size sweep: OLMo2 vs AHA flashinfer vs Local-only on PG-19 prose.

Each model runs in its own subprocess (vLLM keeps too much GPU state for
back-to-back LLM() creation in one Python process). Cache the PG-19 token
slices in a tempfile so all runs see the same prompts.

Usage:
    .venv/bin/python sweep_aha_vs_olmo2.py \\
        --ns 8 32 64 96 --input-len 8192 --output-len 4000
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

MODEL_OLMO2 = "allenai/OLMo-2-0425-1B"
MODEL_AHA = "xuan-luo/AHA-OLMO2"
MAX_MODEL_LEN = 32000


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

sp = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
inputs = [{"prompt_token_ids": p} for p in prompts]

# warmup (small, to trigger any first-time compile that would otherwise
# pollute the timing)
warm_sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
llm.generate([inputs[0]], warm_sp, use_tqdm=False)

torch.cuda.synchronize()
t0 = time.perf_counter()
outputs = llm.generate(inputs, sp, use_tqdm=False)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

total_in = sum(len(o.prompt_token_ids) for o in outputs)
total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
result = {
    "elapsed_s": elapsed,
    "total_prompt_tokens": total_in,
    "total_output_tokens": total_out,
    "req_per_s": len(outputs) / elapsed,
    "total_tok_s": (total_in + total_out) / elapsed,
    "output_tok_s": total_out / elapsed,
}
with open(sys.argv[2], "w") as f:
    json.dump(result, f)
"""


def run_child(prompts, output_len, model, hf_overrides, gpu_mem,
              max_num_seqs, trust_remote_code, local_only=False):
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
        }
        pickle.dump(cfg, f)
        cfg_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "")
    if local_only:
        env["VLLM_LOCAL_ONLY_ENV"] = "1"

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[8, 32, 64, 96])
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=4000)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--variants", nargs="+",
                    default=["olmo2", "aha-flashinfer", "local-only"],
                    choices=["olmo2", "aha-flashinfer", "local-only"])
    ap.add_argument("--out", default="/tmp/aha_vllm_gap.csv")
    args = ap.parse_args()

    cache_dir = "/tmp/pg19_cache"
    os.makedirs(cache_dir, exist_ok=True)

    variant_cfgs = {
        "olmo2": {
            "model": MODEL_OLMO2, "trust": False,
            "overrides": {"max_position_embeddings": MAX_MODEL_LEN},
            "local_only": False,
        },
        "aha-flashinfer": {
            "model": MODEL_AHA, "trust": True,
            "overrides": {"attention_implementation": "flashinfer",
                          "max_position_embeddings": MAX_MODEL_LEN},
            "local_only": False,
        },
        "local-only": {
            "model": MODEL_AHA, "trust": True,
            "overrides": {"attention_implementation": "flashinfer",
                          "max_position_embeddings": MAX_MODEL_LEN},
            "local_only": True,
        },
    }

    max_n = max(args.ns)
    print(f"Preparing PG-19 prompts: n={max_n}, input_len={args.input_len}",
          flush=True)
    prompts_all = build_pg19_prompts(
        max_n, args.input_len, MODEL_AHA,
        os.path.join(cache_dir, f"{max_n}x{args.input_len}.pkl"),
    )
    print(f"  loaded {len(prompts_all)} prompts of {args.input_len} tokens each",
          flush=True)

    rows = []
    print(f"\n{'variant':<16}{'N':>4}{'req/s':>8}{'tot tok/s':>12}"
          f"{'out tok/s':>12}{'elapsed':>9}")
    print("-" * 61)
    for n in args.ns:
        prompts = prompts_all[:n]
        for variant in args.variants:
            cfg = variant_cfgs[variant]
            r = run_child(
                prompts=prompts,
                output_len=args.output_len,
                model=cfg["model"],
                hf_overrides=cfg["overrides"],
                gpu_mem=args.gpu_mem,
                max_num_seqs=args.max_num_seqs,
                trust_remote_code=cfg["trust"],
                local_only=cfg["local_only"],
            )
            row = {
                "variant": variant, "n": n,
                "input_len": args.input_len,
                "output_len": args.output_len,
                **r,
            }
            rows.append(row)
            if "error" in r:
                print(f"{variant:<16}{n:>4}  FAILED: {r['error']}")
                if "tail" in r:
                    print(r["tail"][-400:])
            else:
                print(f"{variant:<16}{n:>4}"
                      f"{r['req_per_s']:>8.2f}"
                      f"{r['total_tok_s']:>12.1f}"
                      f"{r['output_tok_s']:>12.1f}"
                      f"{r['elapsed_s']:>9.1f}")

    import csv
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {args.out}")

    # Ratio table
    print(f"\n{'N':<6}{'OLMo2 tok/s':>14}{'AHA-fi tok/s':>14}"
          f"{'Local tok/s':>14}{'aha/olmo':>10}{'local/olmo':>12}")
    print("-" * 72)
    for n in args.ns:
        r_o = next((r for r in rows if r["variant"] == "olmo2" and r["n"] == n), None)
        r_a = next((r for r in rows if r["variant"] == "aha-flashinfer" and r["n"] == n), None)
        r_l = next((r for r in rows if r["variant"] == "local-only" and r["n"] == n), None)
        def _t(r):
            return r.get("output_tok_s", 0) if r and "output_tok_s" in r else float("nan")
        ot, at, lt = _t(r_o), _t(r_a), _t(r_l)
        ao = at / ot if ot else float("nan")
        lo = lt / ot if ot else float("nan")
        print(f"{n:<6}{ot:>14.1f}{at:>14.1f}{lt:>14.1f}"
              f"{ao:>10.2f}x{lo:>11.2f}x")


if __name__ == "__main__":
    sys.exit(main())
