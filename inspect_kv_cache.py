#!/usr/bin/env python3
"""Dump vLLM's KV cache allocator state for OLMo2 vs AHA-baseline.

Goal: explain why AHA reports 14x concurrency + 19% peak KV% while OLMo2
reports 5x + 99% with nominally the same KV cache size.

Uses LLM.collective_rpc() to reach into the worker's model_runner and
pull out kv_cache_config (num_blocks, groups, specs), shared_kv_cache_layers,
and the get_max_concurrency_for_kv_cache_config() breakdown.

Zero edits to vLLM source. Safe to delete this file when done.
"""

import gc
import json
import os
import sys

# Must be set before importing vllm: collective_rpc with a callable requires
# pickle-based serialization.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from vllm import LLM

AHA_MODEL = "xuan-luo/AHA-OLMO2"
OLMO2_MODEL = "allenai/OLMo-2-0425-1B"

MAX_MODEL_LEN = 32000
GPU_MEM = 0.8


def _worker_dump(self):
    """Runs inside the worker process. `self` is the Worker object."""
    from vllm.v1.core.kv_cache_utils import (
        get_max_concurrency_for_kv_cache_config,
        max_memory_usage_bytes,
    )
    from vllm.utils.math_utils import cdiv
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention.attention import Attention

    mr = self.model_runner
    cfg = mr.kv_cache_config
    vllm_cfg = mr.vllm_config

    hf_cfg = vllm_cfg.model_config.hf_config
    hf_sliding_window = getattr(hf_cfg, "sliding_window", None)
    cache_sliding_window = vllm_cfg.cache_config.sliding_window
    hf_config_class = f"{type(hf_cfg).__module__}.{type(hf_cfg).__name__}"

    # Sample per-layer sliding_window from the actual Attention modules.
    attn_layers = get_layers_from_vllm_config(vllm_cfg, Attention)
    per_layer_sw = [
        {"name": name, "sliding_window": getattr(layer, "sliding_window", None)}
        for name, layer in list(attn_layers.items())[:6]
    ]

    groups_info = []
    for i, g in enumerate(cfg.kv_cache_groups):
        spec = g.kv_cache_spec
        layer_names = list(g.layer_names)
        try:
            per_layer_bytes = spec.max_memory_usage_bytes(vllm_cfg)
        except Exception as e:
            per_layer_bytes = f"ERR: {e}"
        groups_info.append({
            "group_idx": i,
            "spec_type": type(spec).__name__,
            "num_layers": len(layer_names),
            "first_layers": layer_names[:3],
            "last_layers": layer_names[-3:],
            "block_size": spec.block_size,
            "page_size_bytes": spec.page_size_bytes,
            "max_mem_per_layer_bytes": per_layer_bytes,
            "sliding_window": getattr(spec, "sliding_window", None),
        })

    # Replicate get_max_concurrency_for_kv_cache_config breakdown
    num_layer_per_group = max(
        len(g.layer_names) for g in cfg.kv_cache_groups
    ) if cfg.kv_cache_groups else 0
    try:
        max_mem_per_req = num_layer_per_group * max_memory_usage_bytes(
            vllm_cfg, (g.kv_cache_spec for g in cfg.kv_cache_groups)
        )
        mem_per_block = (
            cfg.kv_cache_groups[0].kv_cache_spec.page_size_bytes
            * num_layer_per_group
        )
        num_blocks_per_req = cdiv(max_mem_per_req, mem_per_block)
        max_conc = cfg.num_blocks / num_blocks_per_req
    except Exception as e:
        max_mem_per_req = mem_per_block = num_blocks_per_req = max_conc = f"ERR: {e}"

    # Also the official helper
    try:
        official_conc = get_max_concurrency_for_kv_cache_config(vllm_cfg, cfg)
    except Exception as e:
        official_conc = f"ERR: {e}"

    return {
        "num_blocks": cfg.num_blocks,
        "num_groups": len(cfg.kv_cache_groups),
        "groups": groups_info,
        "shared_kv_cache_layers": dict(mr.shared_kv_cache_layers),
        "num_shared_layers": len(mr.shared_kv_cache_layers),
        "max_model_len": vllm_cfg.model_config.max_model_len,
        "num_layer_per_group": num_layer_per_group,
        "max_mem_per_req_bytes": max_mem_per_req,
        "mem_per_block_bytes": mem_per_block,
        "num_blocks_per_req": num_blocks_per_req,
        "max_concurrency_manual": max_conc,
        "max_concurrency_official": official_conc,
        "hf_config_class": hf_config_class,
        "hf_config_sliding_window": hf_sliding_window,
        "cache_config_sliding_window": cache_sliding_window,
        "per_layer_sliding_window_sample": per_layer_sw,
    }


def inspect(label: str, model: str, hf_overrides: dict | None, trust: bool):
    print(f"\n{'=' * 70}\n[{label}]  model={model}\n  hf_overrides={hf_overrides}\n{'=' * 70}", flush=True)
    llm = LLM(
        model=model,
        dtype="half",
        trust_remote_code=trust,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_MODEL_LEN,
        hf_overrides=hf_overrides,
        disable_log_stats=True,
    )
    results = llm.collective_rpc(_worker_dump)
    # Single GPU → 1 result
    info = results[0]

    print(f"\n--- {label} KV CACHE STATE ---")
    print(f"hf_config class            : {info['hf_config_class']}")
    print(f"hf_config.sliding_window   : {info['hf_config_sliding_window']}")
    print(f"cache_config.sliding_window: {info['cache_config_sliding_window']}")
    print("per-layer Attention.sliding_window (first 6):")
    for entry in info["per_layer_sliding_window_sample"]:
        print(f"    {entry['name']}: sliding_window={entry['sliding_window']}")
    print(f"num_blocks              : {info['num_blocks']:,}")
    print(f"num_groups              : {info['num_groups']}")
    print(f"num_shared_layers       : {info['num_shared_layers']}  (skipped via kv_sharing_target_layer_name)")
    if info['shared_kv_cache_layers']:
        for src, tgt in list(info['shared_kv_cache_layers'].items())[:3]:
            print(f"    {src} -> {tgt}")
        if info['num_shared_layers'] > 3:
            print(f"    ... and {info['num_shared_layers'] - 3} more")
    print(f"max_model_len           : {info['max_model_len']:,}")
    print(f"num_layer_per_group     : {info['num_layer_per_group']}")
    print(f"max_mem_per_req_bytes   : {info['max_mem_per_req_bytes']}")
    print(f"mem_per_block_bytes     : {info['mem_per_block_bytes']}")
    print(f"num_blocks_per_req      : {info['num_blocks_per_req']}")
    print(f"max_concurrency_manual  : {info['max_concurrency_manual']}")
    print(f"max_concurrency_official: {info['max_concurrency_official']}")

    print(f"\n--- groups ({len(info['groups'])}) ---")
    for g in info['groups']:
        print(f"  group {g['group_idx']}: {g['spec_type']} ({g['num_layers']} layers)")
        print(f"    block_size={g['block_size']}, page_size_bytes={g['page_size_bytes']}")
        print(f"    max_mem_per_layer_bytes={g['max_mem_per_layer_bytes']}")
        print(f"    sliding_window={g['sliding_window']}")
        print(f"    first: {g['first_layers']}")
        print(f"    last : {g['last_layers']}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {"label": label, **info}


def main():
    runs = [
        ("OLMo2", OLMO2_MODEL,
         {"max_position_embeddings": MAX_MODEL_LEN},
         False),
        ("AHA baseline", AHA_MODEL,
         {"attention_implementation": "baseline", "local_window_size": 128,
          "max_position_embeddings": MAX_MODEL_LEN},
         True),
        ("AHA flashinfer", AHA_MODEL,
         {"attention_implementation": "flashinfer", "local_window_size": 128,
          "max_position_embeddings": MAX_MODEL_LEN},
         True),
    ]

    all_results = []
    for label, model, ov, trust in runs:
        try:
            all_results.append(inspect(label, model, ov, trust))
        except Exception as e:
            print(f"[{label}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'#' * 70}\nSIDE-BY-SIDE SUMMARY\n{'#' * 70}")
    keys = ("num_blocks", "num_groups", "num_shared_layers",
            "num_layer_per_group", "max_mem_per_req_bytes",
            "mem_per_block_bytes", "num_blocks_per_req",
            "max_concurrency_manual", "max_concurrency_official")
    label_w = 28
    col_w = 22
    header = f"{'Key':<{label_w}}" + "".join(f"{r['label']:>{col_w}}" for r in all_results)
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:<{label_w}}"
        for r in all_results:
            v = r.get(k, "?")
            if isinstance(v, float):
                vs = f"{v:.3f}"
            elif isinstance(v, int):
                vs = f"{v:,}"
            else:
                vs = str(v)
            row += f"{vs:>{col_w}}"
        print(row)

    out_path = "/tmp/aha_kv_inspect.json"
    # Serialize (groups may contain non-JSON-native types)
    def _default(o):
        return str(o)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_default)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
