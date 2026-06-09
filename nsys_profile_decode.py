#!/usr/bin/env python3
"""Graphs-on decode harness for nsys profiling. One process per cell.

Reads cell config from env:
  AHA_CFG     in {dense, aha-global, aha-flashinfer, local-only}
  INPUT_LEN   prompt length (tokens)
  MAX_TOKENS  decode token count for THIS run (use 1 and N=33 per cell for
              long-minus-short subtraction in parse_nsys_p.py)
  AHA_MODEL_MAX  (optional) override of max_model_len (default 32768)

Bracketed with torch.cuda.profiler.start/stop so nsys
  --capture-range=cudaProfilerApi --capture-range-end=stop
limits capture to the main generate (warmup is excluded).

Invoke via sudo -E (CUPTI permission), e.g.:
  sudo -E nsys profile -t cuda \\
      --capture-range=cudaProfilerApi --capture-range-end=stop \\
      --force-overwrite=true -o results/nsys/cell.nsys-rep \\
      -- .venv/bin/python nsys_profile_decode.py
"""

import os
import sys

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

CFG = os.environ.get("AHA_CFG", "aha-flashinfer")
INPUT_LEN = int(os.environ.get("INPUT_LEN", "8192"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "33"))
BATCH = int(os.environ.get("AHA_BATCH", "1"))
# AHA_PFC_DECODE=1: prime the full prompt into the prefix cache BEFORE the
# profiler window, so the profiled generate is a cache hit (no prefill recompute)
# -> the nsys trace contains ONLY decode-shaped launches. This isolates the
# decode phase cleanly at B>1, where chunked-prefill otherwise interleaves with
# decode and contaminates the decode-direct extraction.
_PFC = os.environ.get("AHA_PFC_DECODE") == "1"
MODEL_MAX = int(os.environ.get("AHA_MODEL_MAX", "32768"))
GPU_MEM = float(os.environ.get("AHA_GPU_MEM", "0.5"))  # bump for long-ctx KV

# Legacy env-flag plumbing (must be set BEFORE importing vllm; model code
# reads them at import time — same pattern as sweep_aha_vs_olmo2.py's CHILD).
if CFG == "local-only":
    os.environ["VLLM_LOCAL_ONLY"] = "1"
if CFG == "aha-global":
    os.environ["VLLM_AHA_GATE_OVERRIDE"] = "global"
if CFG == "aha-half":
    os.environ["VLLM_AHA_GATE_OVERRIDE"] = "half"
if CFG == "aha-local":
    os.environ["VLLM_AHA_GATE_OVERRIDE"] = "local"

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

VARIANT_CFGS = {
    "dense": {
        "model": "allenai/OLMo-2-0425-1B", "trust": False,
        "overrides": {"max_position_embeddings": MODEL_MAX},
        "attention_backend": None,  # vLLM default (FA2 on this box)
    },
    "dense-fi": {
        # Stock OLMo2 forced to the FlashInfer attention backend. Same model
        # code as dense, same backend family as AHA. Comparing this against
        # dense isolates the FA2-vs-FI backend effect; comparing this against
        # AHA configs isolates the AHA model-code + router contribution from
        # the backend swap.
        "model": "allenai/OLMo-2-0425-1B", "trust": False,
        "overrides": {"max_position_embeddings": MODEL_MAX},
        "attention_backend": "FLASHINFER",
    },
    "aha-global": {
        "model": "xuan-luo/AHA-OLMO2", "trust": True,
        "overrides": {
            "attention_implementation": "flashinfer",
            "max_position_embeddings": MODEL_MAX,
        },
        "attention_backend": None,  # AHA model selects FI via attention_implementation
    },
    "aha-flashinfer": {
        "model": "xuan-luo/AHA-OLMO2", "trust": True,
        "overrides": {
            "attention_implementation": "flashinfer",
            "max_position_embeddings": MODEL_MAX,
        },
        "attention_backend": None,
    },
    "aha-half": {
        # AHA router with forced 50% deterministic sparsity (every other head
        # routed local). Tests whether the FlashInfer router kernel rewards
        # per-head pruning, controlled for the learned gate's actual routing
        # rate. Compares directly to the microbench's Bernoulli(0.5) case.
        "model": "xuan-luo/AHA-OLMO2", "trust": True,
        "overrides": {
            "attention_implementation": "flashinfer",
            "max_position_embeddings": MODEL_MAX,
        },
        "attention_backend": None,
    },
    "local-only": {
        # TRUE uniform sliding-window baseline: stock OLMo2 routed to the
        # olmo2_local_only module (via VLLM_LOCAL_ONLY=1, set above), which
        # forces per_layer_sliding_window=128 on ALL layers through vLLM's
        # NATIVE sliding-window support -> the scheduler allocates/gathers only
        # the 128-token window, not the full KV. This is the windowing upper
        # bound the microbench measures. It runs on the stock model (the AHA
        # checkpoint ignores VLLM_LOCAL_ONLY), so it isolates "can vLLM window
        # decode memory at all" from "can AHA's per-head routing use it".
        "model": "allenai/OLMo-2-0425-1B", "trust": False,
        "overrides": {"max_position_embeddings": MODEL_MAX},
        "attention_backend": "FLASHINFER",
    },
    "aha-local": {
        # AHA router, gate forced all-local. Decisive test of router polarity:
        # if only the convention is inverted, router all-0 → kernel all-GLOBAL
        # → slow/context-scaling; if router is ignored, stays ~82us flat.
        "model": "xuan-luo/AHA-OLMO2", "trust": True,
        "overrides": {
            "attention_implementation": "flashinfer",
            "max_position_embeddings": MODEL_MAX,
        },
        "attention_backend": None,
    },
}
if CFG not in VARIANT_CFGS:
    raise SystemExit(f"unknown AHA_CFG={CFG!r}; expected one of {list(VARIANT_CFGS)}")

cfg = VARIANT_CFGS[CFG]
print(
    f"### CFG={CFG} model={cfg['model']} INPUT_LEN={INPUT_LEN} "
    f"MAX_TOKENS={MAX_TOKENS} ###",
    flush=True,
)

llm_kwargs = dict(
    model=cfg["model"],
    dtype="half",
    trust_remote_code=cfg["trust"],
    gpu_memory_utilization=GPU_MEM,
    max_model_len=MODEL_MAX,
    hf_overrides=cfg["overrides"],
    disable_log_stats=True,
    enforce_eager=False,           # graphs ON — production path
    enable_prefix_caching=_PFC,    # PFC mode: prime the prompt so profiled gen is cache-hit
    max_num_seqs=max(BATCH, 4),    # must be >= BATCH so all seqs decode concurrently
)
if cfg.get("attention_backend"):
    llm_kwargs["attention_backend"] = cfg["attention_backend"]
llm = LLM(**llm_kwargs)

# Real-text prompt from PG-19. Synthetic tokens cause the AHA learned gate to
# behave unrepresentatively (it's a content-dependent router), which destroyed
# our earlier measurements. We pull text from the JSONL caches that
# benchmark_aha_full.py already populated under .benchmark_datasets/.
import glob  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402

PG19_TOKEN_CACHE = "/tmp/pg19_token_cache"
os.makedirs(PG19_TOKEN_CACHE, exist_ok=True)


def _load_pg19_prompt(input_len: int) -> list[int]:
    cache_path = os.path.join(PG19_TOKEN_CACHE, f"{input_len}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    # Build from the JSONL files; concatenate prompts until we have enough.
    jsonl_files = sorted(
        glob.glob(".benchmark_datasets/pg19-test_*.jsonl"),
        key=lambda p: -os.path.getsize(p),  # try largest first
    )
    if not jsonl_files:
        raise RuntimeError(
            "no .benchmark_datasets/pg19-test_*.jsonl files found; "
            "run benchmark_aha_full.py once to populate the cache."
        )
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    buf: list[int] = []
    needed = input_len + 200
    for path in jsonl_files:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                txt = rec.get("prompt", "")
                if not txt:
                    continue
                buf.extend(tok(txt, add_special_tokens=False).input_ids)
                if len(buf) >= needed:
                    break
        if len(buf) >= needed:
            break
    if len(buf) < needed:
        raise RuntimeError(
            f"PG-19 cache yielded {len(buf)} tokens, need {needed}"
        )
    prompt = buf[:input_len]
    with open(cache_path, "wb") as f:
        pickle.dump(prompt, f)
    return prompt


print(f"### loading PG-19 prompt of length {INPUT_LEN} (BATCH={BATCH})", flush=True)
prompt_ids = _load_pg19_prompt(INPUT_LEN)
# B distinct sequences (rotate token ids per seq so they don't collapse; prefix
# caching is off so each gets its own KV regardless). Decode kernel processes B
# queries/step — this is the throughput regime where per-block setup amortizes.
inputs_warm = [{"prompt_token_ids": prompt_ids[:128]} for _ in range(BATCH)]
inputs_main = [
    {"prompt_token_ids": prompt_ids[i:] + prompt_ids[:i]}  # rotation = distinct seq
    for i in range(BATCH)
]

sp_warm = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
sp_main = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)

_PROBE = os.environ.get("VLLM_AHA_PROBE_READOUT") == "1"


def _reset_probe(self):
    from vllm.model_executor.models.olmo2_aha import _AHAFlashInferAttention
    for _, m in self.model_runner.model.named_modules():
        if isinstance(m, _AHAFlashInferAttention):
            m._probe_swa_sum.zero_(); m._probe_total.zero_()
            m._probe_decode_calls.zero_()
    return True


def _read_probe(self):
    from vllm.model_executor.models.olmo2_aha import _AHAFlashInferAttention
    out = []
    for name, m in self.model_runner.model.named_modules():
        if isinstance(m, _AHAFlashInferAttention):
            out.append((name, int(m._probe_swa_sum.item()),
                        int(m._probe_total.item()),
                        int(m._probe_decode_calls.item())))
    return out


print("### warmup ###", flush=True)
llm.generate(inputs_warm, sp_warm, use_tqdm=False)
torch.cuda.synchronize()
if _PFC:
    # Prime the full prompt into the prefix cache OUTSIDE the profiler window, so
    # the profiled generate skips prefill recompute (cache hit) -> decode-only trace.
    print("### priming prefix cache (full prompt) ###", flush=True)
    llm.generate(inputs_main, SamplingParams(temperature=0.0, max_tokens=1,
                 ignore_eos=True), use_tqdm=False)
    torch.cuda.synchronize()

if _PROBE:
    llm.collective_rpc(_reset_probe)  # measure only the timed main generate

print("### main generate (profiler ON) ###", flush=True)
torch.cuda.profiler.start()
outs = llm.generate(inputs_main, sp_main, use_tqdm=False)
torch.cuda.synchronize()
torch.cuda.profiler.stop()

ntok = len(outs[0].outputs[0].token_ids)
print(f"### done — generated {ntok} tokens ###", flush=True)

if _PROBE:
    rows = llm.collective_rpc(_read_probe)[0]
    tot_swa = sum(r[1] for r in rows); tot = sum(r[2] for r in rows)
    print(f"### PROBE: overall SWA(local)={100.0*tot_swa/max(tot,1):.1f}%  "
          f"slots={tot}  decode_calls={sum(r[3] for r in rows)} ###", flush=True)
    for name, swa, t, dc in rows:
        if t:
            print(f"###   {name}: SWA={100.0*swa/t:.1f}%", flush=True)
