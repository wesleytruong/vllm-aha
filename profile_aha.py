#!/usr/bin/env python3
"""Profile OLMo2 baseline and AHA-flashinfer decode steps with vLLM's torch profiler.

Runs each model on a small batch with profiling enabled. The profiler is
started AFTER prefill and warmup, so the captured trace is dominated by
decode steps. Each run dumps a per-op CUDA-time table to stdout and to
{trace_dir}/profiler_out_0.txt.
"""

import argparse
import gc
import math
import os
import sys
import time

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Trained context length for both OLMo2 and AHA-OLMO2.
TRAINED_MAX_POS = 4096


def apply_length_overrides(
    base_overrides: dict | None,
    max_model_len: int | None,
    rope_scaling: str,
    rope_scaling_factor: float | None,
) -> dict | None:
    """Merge length + RoPE-scaling overrides into an existing hf_overrides dict."""
    overrides: dict = dict(base_overrides) if base_overrides else {}
    if max_model_len is not None:
        overrides["max_position_embeddings"] = max_model_len
    if rope_scaling != "none":
        overrides["rope_parameters"] = {
            "rope_type": rope_scaling,
            "factor": rope_scaling_factor,
            "original_max_position_embeddings": TRAINED_MAX_POS,
            "rope_theta": 500000.0,
        }
    return overrides if overrides else None


def run_one(
    model: str,
    trace_dir: str,
    hf_overrides: dict | None,
    num_prompts: int,
    input_len: int,
    output_len: int,
    profile_steps: int,
    warmup_steps: int,
    gpu_mem: float,
    max_model_len: int | None = None,
    enforce_eager: bool = False,
    no_cudagraph: bool = False,
):
    print(f"\n{'#' * 80}")
    print(f"# Profiling: model={model} hf_overrides={hf_overrides}")
    print(f"# trace_dir={trace_dir}")
    print(f"{'#' * 80}\n", flush=True)

    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": trace_dir,
        "torch_profiler_with_stack": False,
        "torch_profiler_with_flops": False,
        "torch_profiler_with_memory": False,
        "torch_profiler_dump_cuda_time_total": True,
        "delay_iterations": warmup_steps,
        "max_iterations": profile_steps,
    }

    llm = LLM(
        model=model,
        dtype="half",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        hf_overrides=hf_overrides,
        max_model_len=max_model_len,
        profiler_config=profiler_config,
        disable_log_stats=True,
        enforce_eager=enforce_eager,
        compilation_config=(
            {"cudagraph_mode": "NONE"} if no_cudagraph else None
        ),
    )

    prompts = [
        TokensPrompt(prompt_token_ids=[100 + i] * input_len)
        for i in range(num_prompts)
    ]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    print("Warmup generate (no profile)...", flush=True)
    _ = llm.generate(
        prompts[:2],
        sampling_params=SamplingParams(
            temperature=0.0, max_tokens=8, ignore_eos=True
        ),
        use_tqdm=False,
    )

    print(
        f"Profiled generate: {num_prompts} prompts, "
        f"warmup={warmup_steps} steps, profile={profile_steps} steps...",
        flush=True,
    )
    llm.start_profile()
    t0 = time.perf_counter()
    out = llm.generate(
        prompts,
        sampling_params=sampling,
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    llm.stop_profile()

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in out)
    print(
        f"Generated {total_out_tokens} tokens in {elapsed:.2f}s "
        f"({total_out_tokens / elapsed:.1f} tok/s)",
        flush=True,
    )

    # Free engine before next run.
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=4,
        help="Worker steps to skip before starting profile capture.",
    )
    ap.add_argument(
        "--profile-steps",
        type=int,
        default=20,
        help="Worker steps to capture in the profile.",
    )
    ap.add_argument("--gpu-memory", type=float, default=0.8)
    ap.add_argument("--only", choices=["olmo2", "aha_fi"], default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--no-cudagraph", action="store_true")
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Override max context length. When set, also overrides "
            "max_position_embeddings in the HF config."
        ),
    )
    ap.add_argument(
        "--rope-scaling",
        choices=["none", "linear", "yarn"],
        default="none",
        help=(
            "RoPE scaling type to inject via hf_overrides. Required for "
            f"coherent generation beyond the trained length of {TRAINED_MAX_POS}."
        ),
    )
    ap.add_argument(
        "--rope-scaling-factor",
        type=float,
        default=None,
        help=(
            "RoPE scaling factor. Defaults to ceil(max_model_len / "
            f"{TRAINED_MAX_POS} * 2) / 2 when --rope-scaling is set."
        ),
    )
    args = ap.parse_args()

    if args.rope_scaling != "none" and args.rope_scaling_factor is None:
        effective_len = args.max_model_len or TRAINED_MAX_POS
        ratio = max(1.0, effective_len / TRAINED_MAX_POS)
        args.rope_scaling_factor = math.ceil(ratio * 2) / 2

    runs = [
        (
            "olmo2",
            "allenai/OLMo-2-0425-1B",
            None,
            "/tmp/aha_profile/olmo2",
        ),
        (
            "aha_fi",
            "xuan-luo/AHA-OLMO2",
            {"attention_implementation": "flashinfer"},
            "/tmp/aha_profile/aha_fi",
        ),
    ]

    for name, model, overrides, trace_dir in runs:
        if args.only and args.only != name:
            continue
        os.makedirs(trace_dir, exist_ok=True)
        merged_overrides = apply_length_overrides(
            overrides,
            max_model_len=args.max_model_len,
            rope_scaling=args.rope_scaling,
            rope_scaling_factor=args.rope_scaling_factor,
        )
        run_one(
            model=model,
            trace_dir=trace_dir,
            hf_overrides=merged_overrides,
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
            profile_steps=args.profile_steps,
            warmup_steps=args.warmup_steps,
            gpu_mem=args.gpu_memory,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
            no_cudagraph=args.no_cudagraph,
        )


if __name__ == "__main__":
    sys.exit(main())
