#!/usr/bin/env python3
"""Inter-token latency (ITL / TPOT) benchmark for AHA decode.

Measures the steady-state per-decode-token latency DIRECTLY — no prefill
subtraction — so it avoids the prefill-cancellation noise in bench_e2e_decode.py
(the source of the non-monotonic batch numbers). TPOT/ITL is also the standard
serving latency metric.

Method (vLLM v1 RequestOutput.metrics is None here, so we use prefix caching):
  1. prime: generate(prompt, max_tokens=1) -> the prompt KV is cached.
  2. timed: generate(prompt, max_tokens=N) -> the prompt prefill is a cache hit
     (skipped), so the call is ~one cheap cached-forward + (N-1) decode steps.
     TPOT = decode_time / (N-1). For N>>1 the cached forward is negligible.
The cached prompt KV is full-attention (the gate is decode-only), so it is valid
for every routing mode; we prime once and flip the gate buffers in-process.

Reports TPOT (ms/token, the inter-token latency), per-stream and total decode
throughput, and the real-vs-full latency speedup, for each routing mode.

Usage (from repo root): python experiments/bench_itl.py --ctx 8192 --batch 1
"""
import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import get_config  # noqa: E402
from pg19_loader import load_pg19_prompt  # noqa: E402

MODEL = "xuan-luo/AHA-OLMO2"
# Routing modes switched in-process via the cache-safe gate buffers. (True
# "native SWA" is a different model, olmo2_local_only — measured separately.)
MODES = [("real-gate", 1.0, "zero"), ("all-global", 0.0, "one"),
         ("all-local", 0.0, "zero"), ("half", 0.0, "half")]


def _set_override(self, keep, force_mode):
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


def main():
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--gen-tokens", type=int, default=129, help="decode tokens (N)")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--model-max", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    model_max = args.model_max or max(cfg["model_max"], args.ctx + args.gen_tokens + 64)
    gpu_mem = args.gpu_mem or cfg["gpu_mem"]

    import torch
    from vllm import LLM, SamplingParams

    prompt = load_pg19_prompt(args.ctx, MODEL)
    inputs = [{"prompt_token_ids": prompt[i:] + prompt[:i]} for i in range(args.batch)]

    llm = LLM(model=MODEL, dtype="half", trust_remote_code=True,
              gpu_memory_utilization=gpu_mem, max_model_len=model_max,
              hf_overrides={"attention_implementation": "flashinfer",
                            "max_position_embeddings": model_max},
              enforce_eager=False, enable_prefix_caching=True,  # so prefill is a cache hit
              max_num_seqs=max(args.batch, 4), disable_log_stats=True)

    N = args.gen_tokens
    sp = SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=True)
    prime = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    # prime the prompt KV into the cache (full-attention prefill, valid for all modes)
    llm.generate(inputs, prime, use_tqdm=False)

    print(f"\n### ITL/TPOT  ctx={args.ctx} batch={args.batch} N={N} decode tokens ###",
          flush=True)
    print(f"{'mode':>12} {'SWA%':>5} {'TPOT ms':>8} {'tok/s/seq':>10} "
          f"{'total tok/s':>12} {'speedup':>8}", flush=True)
    res = {}
    for name, keep, force in MODES:
        llm.collective_rpc(_set_override, args=(keep, force))
        llm.collective_rpc(_read_swa)
        llm.generate(inputs, SamplingParams(temperature=0.0, max_tokens=2,
                     ignore_eos=True), use_tqdm=False)  # warm this mode's graph
        tpots = []
        for _ in range(args.reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = llm.generate(inputs, sp, use_tqdm=False)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            n_dec = len(out[0].outputs[0].token_ids) - 1   # 1st token = cached forward
            tpots.append(dt / n_dec)
        swa = llm.collective_rpc(_read_swa)[0]
        swa_pct = 100.0 * swa[0] / max(swa[1], 1)
        tpot = statistics.median(tpots)          # ms/token, inter-token latency
        res[name] = (tpot, swa_pct)

    full = res["all-global"][0]
    rows = []
    for name, _, _ in MODES:
        tpot, swa_pct = res[name]
        per_seq = 1.0 / tpot
        total = args.batch / tpot
        rows.append([name, swa_pct, tpot * 1e3, per_seq, total, full / tpot])
        print(f"{name:>12} {swa_pct:>4.0f}% {tpot*1e3:>8.2f} {per_seq:>10.1f} "
              f"{total:>12.1f} {full/tpot:>7.2f}×", flush=True)

    if args.out:
        rec = {"ctx": args.ctx, "batch": args.batch, "gen_tokens": N,
               "profile": cfg.get("profile"),
               "modes": {r[0]: {"tpot_ms": r[2], "tok_s_seq": r[3],
                                "total_tok_s": r[4], "speedup_vs_full": r[5],
                                "swa_pct": r[1]} for r in rows}}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rec, open(args.out, "w"), indent=2)
        print(f"### wrote {args.out} ###", flush=True)
    print("### done ###", flush=True)


if __name__ == "__main__":
    main()
