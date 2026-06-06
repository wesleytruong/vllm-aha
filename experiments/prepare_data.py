#!/usr/bin/env python3
"""Populate the PG-19 prose caches the benchmarks read from.

The benchmarks pull natural prose from .benchmark_datasets/pg19-test_*.jsonl
(JSON lines with a "prompt" field). AHA's gate is content-dependent and was
trained on long natural text, so synthetic/random tokens give unrepresentative
routing — real prose matters.

Two ways to get the data on a new machine:

  A) rsync the existing caches (recommended, ~22 MB, deterministic):
        rsync -a <source>:.../vllm-aha-wt/.benchmark_datasets/ ./.benchmark_datasets/

  B) regenerate from Hugging Face (this script): streams the PG-19 test split
     and writes books until --min-tokens characters worth are cached. Needs
     network + `datasets`. The HF script loader for deepmind/pg19 is blocked on
     recent `datasets`, so we stream the auto-parquet conversion.

Usage:
    python experiments/prepare_data.py --min-tokens 600000
"""
import argparse
import json
import os


def have_enough(min_chars: int) -> int:
    import glob
    total = 0
    for p in glob.glob(".benchmark_datasets/pg19-test_*.jsonl"):
        total += os.path.getsize(p)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tokens", type=int, default=600_000,
                    help="approx tokens needed (largest context*batch you'll run)")
    ap.add_argument("--out", default=".benchmark_datasets/pg19-test_generated.jsonl")
    args = ap.parse_args()

    # ~5 chars/token of English prose; pad 2x for safety.
    min_chars = args.min_tokens * 5 * 2
    existing = have_enough(min_chars)
    if existing >= min_chars:
        print(f"OK: .benchmark_datasets already has ~{existing/1e6:.0f}MB of PG-19 "
              f"(need ~{min_chars/1e6:.0f}MB). Nothing to do.")
        return
    print(f"Have ~{existing/1e6:.0f}MB, need ~{min_chars/1e6:.0f}MB. "
          f"Streaming PG-19 test from Hugging Face...")
    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(
            f"`datasets` not available ({e}). Use option A (rsync) instead — see "
            f"the module docstring.")
    try:
        ds = load_dataset("deepmind/pg19", split="test", streaming=True)
    except Exception as e:
        raise SystemExit(
            f"HF streaming failed ({e}). Use option A (rsync .benchmark_datasets/).")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = existing
    n = 0
    with open(args.out, "w") as f:
        for row in ds:
            text = row.get("text") or row.get("book_text") or ""
            if not text:
                continue
            f.write(json.dumps({"prompt": text}) + "\n")
            written += len(text)
            n += 1
            if written >= min_chars:
                break
    print(f"Wrote {n} books (~{(written-existing)/1e6:.0f}MB) to {args.out}. "
          f"Total now ~{written/1e6:.0f}MB.")


if __name__ == "__main__":
    main()
