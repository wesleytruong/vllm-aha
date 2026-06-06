#!/usr/bin/env python3
"""Probe AHA flashinfer gate routing distribution vs sequence length.

Reads per-layer buffers (_probe_swa_sum, _probe_total) added directly to
_AHAFlashInferAttention in olmo2_aha.py. Because they live inside the
compute graph (not Python side-effects), they survive torch.compile and
CUDA graph replay — unlike forward hooks / monkey-patches / Python prints.

Uses a real PG-19 novel as the prompt source. AHA's gate was trained on
natural language; random token IDs drive routing toward "full" since the
gate has never seen that distribution. Real prose is required to measure
the paper's claimed 90% SWA routing.

Usage:
    .venv/bin/python probe_aha_gate_vs_seqlen.py
"""

import csv
import os
import sys

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.olmo2_aha import _AHAFlashInferAttention

MODEL = "xuan-luo/AHA-OLMO2"
MAX_MODEL_LEN = 33000
GPU_MEM = float(os.environ.get("PROBE_GPU_MEM", "0.55"))
DECODE_TOKENS = 64
PROMPT_LENS = [512, 2048, 8192, 16384, 32768]


def _load_pg19_tokens(tokenizer, min_tokens: int) -> list[int]:
    """Concatenate PG-19 prose from the .benchmark_datasets/ JSONL caches
    (the HF `deepmind/pg19` script-based loader no longer works on recent
    `datasets`)."""
    import glob
    import json
    jsonl_files = sorted(
        glob.glob(".benchmark_datasets/pg19-test_*.jsonl"),
        key=lambda p: -os.path.getsize(p),
    )
    if not jsonl_files:
        raise RuntimeError(
            "no .benchmark_datasets/pg19-test_*.jsonl found; "
            "run benchmark_aha_full.py once to populate the cache."
        )
    buf: list[int] = []
    for path in jsonl_files:
        with open(path) as f:
            for line in f:
                txt = json.loads(line).get("prompt", "")
                if not txt:
                    continue
                buf.extend(tokenizer(txt, add_special_tokens=False).input_ids)
                if len(buf) >= min_tokens:
                    return buf
    if len(buf) < min_tokens:
        raise RuntimeError(
            f"PG-19 JSONL caches exhausted with only {len(buf)} tokens "
            f"(needed {min_tokens})"
        )
    return buf


def _reset_buffers(self):
    for _, module in self.model_runner.model.named_modules():
        if isinstance(module, _AHAFlashInferAttention):
            module._probe_swa_sum.zero_()
            module._probe_total.zero_()
            module._probe_forward_calls.zero_()
            module._probe_decode_calls.zero_()
    return True


def _read_buffers(self):
    out = []
    for name, module in self.model_runner.model.named_modules():
        if isinstance(module, _AHAFlashInferAttention):
            out.append((
                name,
                int(module._probe_swa_sum.item()),
                int(module._probe_total.item()),
                int(module._probe_forward_calls.item()),
                int(module._probe_decode_calls.item()),
            ))
    def _idx(item):
        for p in item[0].split("."):
            if p.isdigit():
                return int(p)
        return -1
    out.sort(key=_idx)
    return out


def main():
    llm = LLM(
        model=MODEL,
        dtype="half",
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_MODEL_LEN,
        hf_overrides={
            "attention_implementation": "flashinfer",
            "max_position_embeddings": MAX_MODEL_LEN,
        },
        disable_log_stats=True,
        enforce_eager=False,  # graphs ON (FI eager path crashes). The probe
                              # accumulation is a side-effecting custom op so it
                              # survives torch.compile + cudagraph replay.
        max_num_seqs=16,
    )

    init_state = llm.collective_rpc(_read_buffers)[0]
    n_layers = len(init_state)
    print(f"Found {n_layers} _AHAFlashInferAttention layers with probe buffers")
    if n_layers == 0:
        print("ERROR: no layers found — are you running the flashinfer variant?")
        return 1

    tokenizer = llm.get_tokenizer()
    print("Loading PG-19 test book(s) until we have enough tokens...", flush=True)
    base_tokens = _load_pg19_tokens(tokenizer, min_tokens=max(PROMPT_LENS))
    print(f"  got {len(base_tokens)} tokens of natural prose")

    all_rows = []
    summary = []

    for plen in PROMPT_LENS:
        llm.collective_rpc(_reset_buffers)

        prompt_ids = base_tokens[:plen]
        sp = SamplingParams(
            temperature=0.0, max_tokens=DECODE_TOKENS, ignore_eos=True
        )
        outputs = llm.generate(
            {"prompt_token_ids": prompt_ids}, sp, use_tqdm=False
        )
        actual_plen = len(outputs[0].prompt_token_ids)
        n_out = len(outputs[0].outputs[0].token_ids)

        st = llm.collective_rpc(_read_buffers)[0]

        per_layer = []
        tot_swa = tot_slot = tot_fwd = tot_dec = 0
        for layer_idx, (name, swa, tot, fwd, dec) in enumerate(st):
            frac = (swa / tot) if tot > 0 else 0.0
            per_layer.append(frac)
            tot_swa += swa
            tot_slot += tot
            tot_fwd += fwd
            tot_dec += dec
            all_rows.append({
                "prompt_len": actual_plen,
                "decode_tokens_generated": n_out,
                "layer_idx": layer_idx,
                "layer_name": name,
                "forward_calls": fwd,
                "decode_calls": dec,
                "swa_sum": swa,
                "total_slots": tot,
                "swa_frac": f"{frac:.4f}",
            })

        overall = (tot_swa / tot_slot) if tot_slot > 0 else 0.0
        summary.append((actual_plen, n_out, overall, per_layer))
        print(
            f"\n[prompt_len={actual_plen:>6}  decode_tokens={n_out}]  "
            f"forward_calls={tot_fwd}  decode_calls={tot_dec}  "
            f"total_slots={tot_slot}  "
            f"overall SWA={overall*100:5.1f}%  "
            f"range=[{min(per_layer)*100:4.1f}% .. {max(per_layer)*100:4.1f}%]"
        )
        for i, frac in enumerate(per_layer):
            bar = "#" * int(frac * 40)
            print(f"  layer {i:2d}: SWA={frac*100:5.1f}%  |{bar:<40}|")

    print("\n" + "=" * 72)
    print("SUMMARY — overall SWA% vs prompt length (PG-19 prose)")
    print("=" * 72)
    print(f"{'prompt_len':>12}  {'decode_n':>9}  "
          f"{'overall_SWA%':>13}  {'min%':>7}  {'max%':>7}")
    for plen, n_out, overall, per_layer in summary:
        print(
            f"{plen:>12}  {n_out:>9}  "
            f"{overall*100:>12.2f}%  "
            f"{min(per_layer)*100:>6.2f}%  {max(per_layer)*100:>6.2f}%"
        )

    out_csv = "/tmp/aha_gate_vs_seqlen.csv"
    if all_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nSaved per-layer rows to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
