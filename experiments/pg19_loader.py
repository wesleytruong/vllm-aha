#!/usr/bin/env python3
"""Shared PG-19 prompt loader (real natural prose for the content-dependent gate).

Builds a token prompt of a requested length from the PG-19 test JSONL caches
under .benchmark_datasets/ (the HF `deepmind/pg19` script loader no longer works
on recent `datasets`; see experiments/prepare_data.py to populate the caches).
Results are cached as token pickles so repeated runs are instant.

Used by bench_e2e_decode.py, nsys_profile_decode.py, probe_routing.py.
"""
import glob
import json
import os
import pickle

DATA_GLOB = os.environ.get("AHA_PG19_GLOB", ".benchmark_datasets/pg19-test_*.jsonl")
TOKEN_CACHE = os.environ.get("AHA_PG19_CACHE", "/tmp/pg19_token_cache")


def load_pg19_prompt(input_len: int, model: str) -> list[int]:
    """Return a list of `input_len` token ids of PG-19 prose for `model`."""
    os.makedirs(TOKEN_CACHE, exist_ok=True)
    # cache per (model, length) so different tokenizers don't collide
    key = f"{os.path.basename(model)}_{input_len}.pkl"
    cache_path = os.path.join(TOKEN_CACHE, key)
    legacy = os.path.join(TOKEN_CACHE, f"{input_len}.pkl")  # old AHA-OLMO2 cache
    for p in (cache_path, legacy):
        if os.path.exists(p):
            with open(p, "rb") as f:
                ids = pickle.load(f)
            if len(ids) >= input_len:
                return ids[:input_len]

    files = sorted(glob.glob(DATA_GLOB), key=lambda p: -os.path.getsize(p))
    if not files:
        raise RuntimeError(
            f"no PG-19 JSONL at {DATA_GLOB}. Run: python experiments/prepare_data.py")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    buf: list[int] = []
    needed = input_len + 256
    for path in files:
        with open(path) as f:
            for line in f:
                txt = json.loads(line).get("prompt", "")
                if txt:
                    buf.extend(tok(txt, add_special_tokens=False).input_ids)
                if len(buf) >= needed:
                    break
        if len(buf) >= needed:
            break
    if len(buf) < input_len:
        raise RuntimeError(
            f"PG-19 caches yielded {len(buf)} tokens, need {input_len}. "
            f"Add more books via experiments/prepare_data.py --min-tokens {needed}")
    prompt = buf[:input_len]
    with open(cache_path, "wb") as f:
        pickle.dump(prompt, f)
    return prompt
