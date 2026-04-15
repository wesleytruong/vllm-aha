# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from OLMo2 for All-or-Here Attention (AHA)
# https://huggingface.co/xuan-luo/AHA-OLMO2
# AHA uses a learned per-head gate to route between global and local attention
"""Inference-only FAOlmo model with All-or-Here Attention (AHA).

A single unified file containing all attention implementations:
  - "dual"       : always runs both global+local kernels, blends via gate
  - "baseline"   : alias for "dual" (frozen reference benchmark)
  - "routed"     : per-GQA-group decode loop, skips unneeded attention types
  - "greedy"     : local-first decode (1-2 kernel calls total)
  - "flashinfer" : FlashInfer per-head router kernel (fused 1-kernel decode)

The active implementation is selected from config.attention_implementation.
The local sliding-window size is read from config.local_window_size (default 128).
"""

import json
import os
from collections.abc import Iterable
from functools import partial
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.distributed.utils import split_tensor_along_last_dim
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    reshape_and_cache_flash,
)

logger = init_logger(__name__)

# ── Gate statistics logging ──────────────────────────────────────────────────
_GATE_STATS_ENABLED = bool(os.environ.get("VLLM_AHA_GATE_STATS"))
_gate_stats_counter = 0
_GATE_STATS_INTERVAL = 100

# ── Phase timing ─────────────────────────────────────────────────────────────
_TIMING_ENABLED = bool(os.environ.get("VLLM_AHA_TIMING"))
_TIMING_LOG_INTERVAL = 100

# Greedy timing globals
_timing_stats: dict[str, float] = {
    "kv_update": 0.0, "local_attn": 0.0, "global_attn": 0.0,
}
_timing_totals: dict[str, float] = {
    "kv_update": 0.0, "local_attn": 0.0, "global_attn": 0.0,
}
_timing_count = 0
_local_only_calls = 0
_global_calls = 0
_total_global_groups = 0
_TIMING_OUTPUT_FILE_GREEDY = os.environ.get(
    "VLLM_AHA_TIMING_FILE", "/tmp/aha_timing_stats.json"
)

# Routed timing globals
_routed_timing_stats: dict[str, float] = {
    "kv_update": 0.0, "local_attn": 0.0, "global_attn": 0.0,
}
_routed_timing_totals: dict[str, float] = {
    "kv_update": 0.0, "local_attn": 0.0, "global_attn": 0.0,
}
_routed_timing_count = 0
_all_local_groups = 0
_all_global_groups = 0
_mixed_groups = 0
_TIMING_OUTPUT_FILE_ROUTED = os.environ.get(
    "VLLM_AHA_TIMING_FILE", "/tmp/aha_timing_stats_routed.json"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Private attention implementations
# ═══════════════════════════════════════════════════════════════════════════════

class _AHADualAttention(nn.Module):
    """
    All-or-Here Attention — dual kernel implementation.

    Always runs both global (full) and local (sliding window) attention,
    then blends per-head via a learned gate:
        output = gate * global + (1 - gate) * local
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        hidden_size = self.config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = self.config.num_attention_heads

        assert hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = (
            self.config.num_key_value_heads or self.total_num_heads
        )
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = self.config.max_position_embeddings
        self.scaling = self.head_dim**-0.5
        self.local_window_size = getattr(self.config, "local_window_size", 128)

        self.tp_rank = get_tensor_model_parallel_rank()

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim + self.total_num_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.q_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.k_norm = RMSNorm(
            self.total_num_kv_heads * self.head_dim, eps=self.config.rms_norm_eps
        )

        rope_parameters = self.config.rope_parameters
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        # Global attention: full context, owns the KV cache
        self.global_attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=None,
            prefix=f"{prefix}.global_attn",
        )
        # Local attention: sliding window, shares KV cache with global
        self.local_attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=self.local_window_size,
            prefix=f"{prefix}.local_attn",
            kv_sharing_target_layer_name=f"{prefix}.global_attn",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(
                split_tensor_along_last_dim, num_partitions=self.tp_size
            )
            q = splitter(q)[self.tp_rank].contiguous()
            k = splitter(k)[self.tp_rank].contiguous()
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_with_gate, _ = self.q_proj(hidden_states)
        q, gate = q_with_gate.split([self.q_size, self.num_heads], dim=-1)
        q = q.contiguous()
        gate = gate.contiguous()

        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)

        gate_sigmoid = torch.sigmoid(gate)
        gate_hard = (gate_sigmoid > 0.5).to(hidden_states.dtype)

        global_output = self.global_attn(q, k, v)
        local_output = self.local_attn(q, None, None)

        num_tokens = global_output.shape[0]
        global_out = global_output.view(num_tokens, self.num_heads, self.head_dim)
        local_out = local_output.view(num_tokens, self.num_heads, self.head_dim)
        gate_hard = gate_hard.unsqueeze(-1)

        blended = global_out * gate_hard + local_out * (1.0 - gate_hard)
        blended = blended.view(num_tokens, -1)

        output, _ = self.o_proj(blended)
        return output


class _AHARoutedAttention(nn.Module):
    """
    All-or-Here Attention — per-GQA-group routed decode.

    During decode: iterates over GQA groups, skipping the attention type
    that no head in the group needs. ~90% of groups route all-local.

    During prefill: full attention only — no gate, no blending.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        hidden_size = self.config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = self.config.num_attention_heads

        assert hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = (
            self.config.num_key_value_heads or self.total_num_heads
        )
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = self.config.max_position_embeddings
        self.scaling = self.head_dim**-0.5
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.local_window_size = getattr(self.config, "local_window_size", 128)

        self.tp_rank = get_tensor_model_parallel_rank()

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim + self.total_num_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.q_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.k_norm = RMSNorm(
            self.total_num_kv_heads * self.head_dim, eps=self.config.rms_norm_eps
        )

        rope_parameters = self.config.rope_parameters
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        # Single Attention layer — owns the KV cache, used for prefill
        self.kv_cache_attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=None,
            prefix=f"{prefix}.kv_cache_attn",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._prefix = prefix

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(
                split_tensor_along_last_dim, num_partitions=self.tp_size
            )
            q = splitter(q)[self.tp_rank].contiguous()
            k = splitter(k)[self.tp_rank].contiguous()
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_with_gate, _ = self.q_proj(hidden_states)
        q, gate = q_with_gate.split([self.q_size, self.num_heads], dim=-1)
        q = q.contiguous()
        gate = gate.contiguous()

        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)

        attn_metadata, attn_layer, kv_cache = get_attention_context(
            self.kv_cache_attn.layer_name
        )

        if attn_metadata is None or attn_metadata.max_query_len > 1:
            return self._forward_prefill_full(q, k, v)
        else:
            gate_sigmoid = torch.sigmoid(gate)
            gate_hard = (gate_sigmoid > 0.5).to(hidden_states.dtype)

            if _GATE_STATS_ENABLED:
                self._log_gate_stats(gate_hard)

            return self._forward_decode_routed(
                q, k, v, gate_hard, attn_metadata, attn_layer, kv_cache
            )

    def _forward_prefill_full(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = self.kv_cache_attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    @torch.compiler.disable
    def _forward_decode_routed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_hard: torch.Tensor,
        attn_metadata,
        attn_layer,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        global _routed_timing_stats, _routed_timing_totals, _routed_timing_count
        global _all_local_groups, _all_global_groups, _mixed_groups

        N = q.shape[0]
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        # Phase 1: KV cache update
        forward_ctx = get_forward_context()
        slot_mapping = forward_ctx.slot_mapping
        if isinstance(slot_mapping, dict):
            layer_slot_mapping = slot_mapping.get(self.kv_cache_attn.layer_name)
        else:
            layer_slot_mapping = slot_mapping

        key_cache, value_cache = kv_cache.unbind(0)

        if layer_slot_mapping is not None:
            if _TIMING_ENABLED:
                _t0s = torch.cuda.Event(enable_timing=True)
                _t0e = torch.cuda.Event(enable_timing=True)
                _t0s.record()
            reshape_and_cache_flash(
                k, v,
                key_cache, value_cache,
                layer_slot_mapping,
                attn_layer.kv_cache_dtype,
                attn_layer._k_scale,
                attn_layer._v_scale,
            )
            if _TIMING_ENABLED:
                _t0e.record()

        # Phase 2: Per-GQA-group routed attention
        fa_version = attn_layer.impl.vllm_flash_attn_version

        if attn_layer.kv_cache_dtype.startswith("fp8"):
            from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
            fp8_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                attn_layer.kv_cache_dtype
            )
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)

        output = torch.empty(
            N, self.num_heads, self.head_dim, dtype=q.dtype, device=q.device,
        )

        if _TIMING_ENABLED:
            _local_ms = 0.0
            _global_ms = 0.0
            _n_local = _n_global = _n_mixed = 0

        for g in range(self.num_kv_heads):
            q_start = g * self.num_queries_per_kv
            q_end = q_start + self.num_queries_per_kv

            q_group = q[:, q_start:q_end, :].contiguous()
            group_gate = gate_hard[:, q_start:q_end]

            has_any_global = group_gate.any().item()
            has_any_local = (~group_gate.bool()).any().item()

            k_cache_g = key_cache[:, :, g:g+1, :].contiguous()
            v_cache_g = value_cache[:, :, g:g+1, :].contiguous()

            descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, 1)
            q_descale = attn_layer._q_scale.expand(descale_shape)
            k_descale = attn_layer._k_scale.expand(descale_shape)
            v_descale = attn_layer._v_scale.expand(descale_shape)

            if has_any_global and not has_any_local:
                if _TIMING_ENABLED:
                    _n_global += 1
                    _ts = torch.cuda.Event(enable_timing=True)
                    _te = torch.cuda.Event(enable_timing=True)
                    _ts.record()
                group_out = torch.empty(
                    N, self.num_queries_per_kv, self.head_dim,
                    dtype=q.dtype, device=q.device,
                )
                flash_attn_varlen_func(
                    q=q_group, k=k_cache_g, v=v_cache_g, out=group_out,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scaling, causal=True,
                    window_size=[-1, -1],
                    block_table=attn_metadata.block_table,
                    fa_version=fa_version,
                    q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                )
                if _TIMING_ENABLED:
                    _te.record()
                    torch.cuda.synchronize()
                    _global_ms += _ts.elapsed_time(_te)
                output[:, q_start:q_end, :] = group_out

            elif not has_any_global and has_any_local:
                if _TIMING_ENABLED:
                    _n_local += 1
                    _ts = torch.cuda.Event(enable_timing=True)
                    _te = torch.cuda.Event(enable_timing=True)
                    _ts.record()
                group_out = torch.empty(
                    N, self.num_queries_per_kv, self.head_dim,
                    dtype=q.dtype, device=q.device,
                )
                flash_attn_varlen_func(
                    q=q_group, k=k_cache_g, v=v_cache_g, out=group_out,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scaling, causal=True,
                    window_size=[self.local_window_size - 1, 0],
                    block_table=attn_metadata.block_table,
                    fa_version=fa_version,
                    q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                )
                if _TIMING_ENABLED:
                    _te.record()
                    torch.cuda.synchronize()
                    _local_ms += _ts.elapsed_time(_te)
                output[:, q_start:q_end, :] = group_out

            else:
                if _TIMING_ENABLED:
                    _n_mixed += 1
                    _tgs = torch.cuda.Event(enable_timing=True)
                    _tge = torch.cuda.Event(enable_timing=True)
                    _tls = torch.cuda.Event(enable_timing=True)
                    _tle = torch.cuda.Event(enable_timing=True)
                global_out = torch.empty(
                    N, self.num_queries_per_kv, self.head_dim,
                    dtype=q.dtype, device=q.device,
                )
                local_out = torch.empty(
                    N, self.num_queries_per_kv, self.head_dim,
                    dtype=q.dtype, device=q.device,
                )
                if _TIMING_ENABLED:
                    _tgs.record()
                flash_attn_varlen_func(
                    q=q_group, k=k_cache_g, v=v_cache_g, out=global_out,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scaling, causal=True,
                    window_size=[-1, -1],
                    block_table=attn_metadata.block_table,
                    fa_version=fa_version,
                    q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                )
                if _TIMING_ENABLED:
                    _tge.record()
                    _tls.record()
                flash_attn_varlen_func(
                    q=q_group, k=k_cache_g, v=v_cache_g, out=local_out,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scaling, causal=True,
                    window_size=[self.local_window_size - 1, 0],
                    block_table=attn_metadata.block_table,
                    fa_version=fa_version,
                    q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                )
                if _TIMING_ENABLED:
                    _tle.record()
                    torch.cuda.synchronize()
                    _global_ms += _tgs.elapsed_time(_tge)
                    _local_ms += _tls.elapsed_time(_tle)

                gate_expanded = group_gate.unsqueeze(-1).to(q.dtype)
                output[:, q_start:q_end, :] = (
                    global_out * gate_expanded + local_out * (1.0 - gate_expanded)
                )

        if _TIMING_ENABLED:
            torch.cuda.synchronize()
            kv_ms = _t0s.elapsed_time(_t0e) if layer_slot_mapping is not None else 0.0
            self._accumulate_routed_timing(
                kv_ms, _local_ms, _global_ms, _n_local, _n_global, _n_mixed
            )

        attn_output = output.view(N, -1)
        output_final, _ = self.o_proj(attn_output)
        return output_final

    def _accumulate_routed_timing(
        self,
        kv_ms: float,
        local_ms: float,
        global_ms: float,
        n_local: int,
        n_global: int,
        n_mixed: int,
    ) -> None:
        global _routed_timing_stats, _routed_timing_totals, _routed_timing_count
        global _all_local_groups, _all_global_groups, _mixed_groups

        _routed_timing_stats["kv_update"] += kv_ms
        _routed_timing_stats["local_attn"] += local_ms
        _routed_timing_stats["global_attn"] += global_ms

        _routed_timing_totals["kv_update"] += kv_ms
        _routed_timing_totals["local_attn"] += local_ms
        _routed_timing_totals["global_attn"] += global_ms

        _all_local_groups += n_local
        _all_global_groups += n_global
        _mixed_groups += n_mixed
        _routed_timing_count += 1

        if _routed_timing_count % _TIMING_LOG_INTERVAL == 0:
            n = _TIMING_LOG_INTERVAL
            kv_avg = _routed_timing_stats["kv_update"] / n
            local_avg = _routed_timing_stats["local_attn"] / n
            glob_avg = _routed_timing_stats["global_attn"] / n
            total_ms = kv_avg + local_avg + glob_avg
            logger.info(
                "Timing [%s] step=%d (avg over %d decode calls): "
                "kv_update=%.3f ms (%.1f%%), "
                "local_attn=%.3f ms (%.1f%%), "
                "global_attn=%.3f ms (%.1f%%), "
                "total=%.3f ms",
                self._prefix, _routed_timing_count, n,
                kv_avg, 100 * kv_avg / total_ms if total_ms else 0,
                local_avg, 100 * local_avg / total_ms if total_ms else 0,
                glob_avg, 100 * glob_avg / total_ms if total_ms else 0,
                total_ms,
            )
            _routed_timing_stats["kv_update"] = 0.0
            _routed_timing_stats["local_attn"] = 0.0
            _routed_timing_stats["global_attn"] = 0.0

            try:
                total_groups = _all_local_groups + _all_global_groups + _mixed_groups
                with open(_TIMING_OUTPUT_FILE_ROUTED, "w") as _f:
                    json.dump({
                        "totals": _routed_timing_totals,
                        "count": _routed_timing_count,
                        "all_local_groups": _all_local_groups,
                        "all_global_groups": _all_global_groups,
                        "mixed_groups": _mixed_groups,
                        "total_groups": total_groups,
                        "num_kv_heads": self.num_kv_heads,
                    }, _f)
            except OSError:
                pass

    def _log_gate_stats(self, gate_hard: torch.Tensor) -> None:
        global _gate_stats_counter
        _gate_stats_counter += 1
        if _gate_stats_counter % _GATE_STATS_INTERVAL != 0:
            return

        N, H = gate_hard.shape
        gate_by_group = gate_hard.view(N, self.num_kv_heads, self.num_queries_per_kv)

        all_global_count = all_local_count = mixed_count = 0
        for g in range(self.num_kv_heads):
            group_gate = gate_by_group[:, g, :]
            has_any_global = group_gate.any().item()
            has_any_local = (~group_gate.bool()).any().item()
            if has_any_global and not has_any_local:
                all_global_count += 1
            elif not has_any_global and has_any_local:
                all_local_count += 1
            else:
                mixed_count += 1

        logger.info(
            "Gate stats [%s] step=%d: "
            "all_local=%d (%.1f%%), all_global=%d (%.1f%%), mixed=%d (%.1f%%)",
            self._prefix, _gate_stats_counter,
            all_local_count, all_local_count / self.num_kv_heads * 100,
            all_global_count, all_global_count / self.num_kv_heads * 100,
            mixed_count, mixed_count / self.num_kv_heads * 100,
        )


class _AHAGreedyAttention(nn.Module):
    """
    All-or-Here Attention — greedy local-first decode.

    During decode: computes local attention for ALL heads in one kernel call,
    then runs global attention only for the ~10% of GQA groups that need it
    (1-2 kernel calls total).

    During prefill: full attention only — no gate, no blending.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        hidden_size = self.config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = self.config.num_attention_heads

        assert hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = (
            self.config.num_key_value_heads or self.total_num_heads
        )
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = self.config.max_position_embeddings
        self.scaling = self.head_dim**-0.5
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.local_window_size = getattr(self.config, "local_window_size", 128)

        self.tp_rank = get_tensor_model_parallel_rank()

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim + self.total_num_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.q_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.k_norm = RMSNorm(
            self.total_num_kv_heads * self.head_dim, eps=self.config.rms_norm_eps
        )

        rope_parameters = self.config.rope_parameters
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.kv_cache_attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=None,
            prefix=f"{prefix}.kv_cache_attn",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._prefix = prefix

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(
                split_tensor_along_last_dim, num_partitions=self.tp_size
            )
            q = splitter(q)[self.tp_rank].contiguous()
            k = splitter(k)[self.tp_rank].contiguous()
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_with_gate, _ = self.q_proj(hidden_states)
        q, gate = q_with_gate.split([self.q_size, self.num_heads], dim=-1)
        q = q.contiguous()
        gate = gate.contiguous()

        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)

        attn_metadata, attn_layer, kv_cache = get_attention_context(
            self.kv_cache_attn.layer_name
        )

        if attn_metadata is None or attn_metadata.max_query_len > 1:
            return self._forward_prefill_full(q, k, v)
        else:
            gate_sigmoid = torch.sigmoid(gate)
            gate_hard = (gate_sigmoid > 0.5).to(hidden_states.dtype)

            if _GATE_STATS_ENABLED:
                self._log_gate_stats(gate_hard)

            return self._forward_decode_greedy(
                q, k, v, gate_hard, attn_metadata, attn_layer, kv_cache
            )

    def _forward_prefill_full(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = self.kv_cache_attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_decode_greedy(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_hard: torch.Tensor,
        attn_metadata,
        attn_layer,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        global _timing_stats, _timing_totals, _timing_count
        global _local_only_calls, _global_calls, _total_global_groups

        N = q.shape[0]
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        if _TIMING_ENABLED:
            t0_start = torch.cuda.Event(enable_timing=True)
            t0_end = torch.cuda.Event(enable_timing=True)
            t1_start = torch.cuda.Event(enable_timing=True)
            t1_end = torch.cuda.Event(enable_timing=True)
            t3_start = torch.cuda.Event(enable_timing=True)
            t3_end = torch.cuda.Event(enable_timing=True)

        # Phase 0: KV cache update
        forward_ctx = get_forward_context()
        slot_mapping = forward_ctx.slot_mapping
        if isinstance(slot_mapping, dict):
            layer_slot_mapping = slot_mapping.get(self.kv_cache_attn.layer_name)
        else:
            layer_slot_mapping = slot_mapping

        key_cache, value_cache = kv_cache.unbind(0)

        if layer_slot_mapping is not None:
            if _TIMING_ENABLED:
                t0_start.record()
            reshape_and_cache_flash(
                k, v,
                key_cache, value_cache,
                layer_slot_mapping,
                attn_layer.kv_cache_dtype,
                attn_layer._k_scale,
                attn_layer._v_scale,
            )
            if _TIMING_ENABLED:
                t0_end.record()

        fa_version = attn_layer.impl.vllm_flash_attn_version

        if attn_layer.kv_cache_dtype.startswith("fp8"):
            from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
            fp8_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                attn_layer.kv_cache_dtype
            )
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)

        descale_shape = (
            attn_metadata.query_start_loc.shape[0] - 1,
            self.num_kv_heads,
        )
        q_descale = attn_layer._q_scale.expand(descale_shape)
        k_descale = attn_layer._k_scale.expand(descale_shape)
        v_descale = attn_layer._v_scale.expand(descale_shape)

        # Phase 1: Local attention for ALL heads (1 kernel call)
        output = torch.empty(
            N, self.num_heads, self.head_dim, dtype=q.dtype, device=q.device,
        )

        if _TIMING_ENABLED:
            t1_start.record()
        flash_attn_varlen_func(
            q=q, k=key_cache, v=value_cache, out=output,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scaling, causal=True,
            window_size=[self.local_window_size - 1, 0],
            block_table=attn_metadata.block_table,
            fa_version=fa_version,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        )
        if _TIMING_ENABLED:
            t1_end.record()

        # Phase 2: Identify GQA groups needing global attention
        gate_by_group = gate_hard.view(N, self.num_kv_heads, self.num_queries_per_kv)
        group_needs_global = gate_by_group.any(dim=0).any(dim=-1)
        global_group_indices = group_needs_global.nonzero(as_tuple=False).squeeze(-1)

        if _TIMING_ENABLED:
            n_global = global_group_indices.numel()
            if n_global > 0:
                _global_calls += 1
                _total_global_groups += n_global
            else:
                _local_only_calls += 1

        # Phase 3: Global attention for subset (0 or 1 kernel call)
        if _TIMING_ENABLED:
            t3_start.record()

        if global_group_indices.numel() > 0:
            num_global_kv = global_group_indices.numel()
            nqpkv = self.num_queries_per_kv
            q_head_indices = torch.cat([
                torch.arange(g * nqpkv, (g + 1) * nqpkv, device=q.device)
                for g in global_group_indices
            ])

            q_global = q[:, q_head_indices, :].contiguous()
            k_global = key_cache[:, :, global_group_indices, :].contiguous()
            v_global = value_cache[:, :, global_group_indices, :].contiguous()

            global_output = torch.empty(
                N, q_head_indices.numel(), self.head_dim,
                dtype=q.dtype, device=q.device,
            )

            global_descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                num_global_kv,
            )
            q_descale_g = attn_layer._q_scale.expand(global_descale_shape)
            k_descale_g = attn_layer._k_scale.expand(global_descale_shape)
            v_descale_g = attn_layer._v_scale.expand(global_descale_shape)

            flash_attn_varlen_func(
                q=q_global, k=k_global, v=v_global, out=global_output,
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scaling, causal=True,
                window_size=[-1, -1],
                block_table=attn_metadata.block_table,
                fa_version=fa_version,
                q_descale=q_descale_g, k_descale=k_descale_g,
                v_descale=v_descale_g,
            )

            gate_expanded = gate_hard[:, q_head_indices].unsqueeze(-1)
            output[:, q_head_indices, :] = (
                gate_expanded * global_output
                + (1.0 - gate_expanded) * output[:, q_head_indices, :]
            )

        if _TIMING_ENABLED:
            t3_end.record()
            self._accumulate_greedy_timing(
                t0_start, t0_end, t1_start, t1_end, t3_start, t3_end,
                layer_slot_mapping is not None,
            )

        attn_output = output.view(N, -1)
        output_final, _ = self.o_proj(attn_output)
        return output_final

    def _accumulate_greedy_timing(
        self,
        t0_start, t0_end,
        t1_start, t1_end,
        t3_start, t3_end,
        has_kv_update: bool,
    ) -> None:
        global _timing_stats, _timing_totals, _timing_count
        torch.cuda.synchronize()
        kv_ms = t0_start.elapsed_time(t0_end) if has_kv_update else 0.0
        local_ms = t1_start.elapsed_time(t1_end)
        global_ms = t3_start.elapsed_time(t3_end)

        _timing_stats["kv_update"] += kv_ms
        _timing_stats["local_attn"] += local_ms
        _timing_stats["global_attn"] += global_ms

        _timing_totals["kv_update"] += kv_ms
        _timing_totals["local_attn"] += local_ms
        _timing_totals["global_attn"] += global_ms

        _timing_count += 1

        if _timing_count % _TIMING_LOG_INTERVAL == 0:
            n = _TIMING_LOG_INTERVAL
            kv_avg = _timing_stats["kv_update"] / n
            local_avg = _timing_stats["local_attn"] / n
            glob_avg = _timing_stats["global_attn"] / n
            total_ms = kv_avg + local_avg + glob_avg
            logger.info(
                "Timing [%s] step=%d (avg over %d decode calls): "
                "kv_update=%.3f ms (%.1f%%), "
                "local_attn=%.3f ms (%.1f%%), "
                "global_attn=%.3f ms (%.1f%%), "
                "total=%.3f ms",
                self._prefix, _timing_count, n,
                kv_avg, 100 * kv_avg / total_ms if total_ms else 0,
                local_avg, 100 * local_avg / total_ms if total_ms else 0,
                glob_avg, 100 * glob_avg / total_ms if total_ms else 0,
                total_ms,
            )
            _timing_stats["kv_update"] = 0.0
            _timing_stats["local_attn"] = 0.0
            _timing_stats["global_attn"] = 0.0

            try:
                with open(_TIMING_OUTPUT_FILE_GREEDY, "w") as _f:
                    json.dump({
                        "totals": _timing_totals,
                        "count": _timing_count,
                        "local_only_calls": _local_only_calls,
                        "global_calls": _global_calls,
                        "total_global_groups": _total_global_groups,
                        "num_kv_heads": self.num_kv_heads,
                    }, _f)
            except OSError:
                pass

    def _log_gate_stats(self, gate_hard: torch.Tensor) -> None:
        global _gate_stats_counter
        _gate_stats_counter += 1
        if _gate_stats_counter % _GATE_STATS_INTERVAL != 0:
            return

        N, H = gate_hard.shape
        gate_by_group = gate_hard.view(N, self.num_kv_heads, self.num_queries_per_kv)

        all_global_count = all_local_count = mixed_count = 0
        for g in range(self.num_kv_heads):
            group_gate = gate_by_group[:, g, :]
            has_any_global = group_gate.any().item()
            has_any_local = (~group_gate.bool()).any().item()
            if has_any_global and not has_any_local:
                all_global_count += 1
            elif not has_any_global and has_any_local:
                all_local_count += 1
            else:
                mixed_count += 1

        group_needs_global = gate_by_group.any(dim=0).any(dim=-1)
        num_global_groups = group_needs_global.sum().item()

        logger.info(
            "Gate stats [%s] step=%d: "
            "all_local=%d (%.1f%%), all_global=%d (%.1f%%), mixed=%d (%.1f%%), "
            "global_kernel_groups=%d/%d",
            self._prefix, _gate_stats_counter,
            all_local_count, all_local_count / self.num_kv_heads * 100,
            all_global_count, all_global_count / self.num_kv_heads * 100,
            mixed_count, mixed_count / self.num_kv_heads * 100,
            num_global_groups, self.num_kv_heads,
        )


class _AHAFlashInferAttention(nn.Module):
    """
    All-or-Here Attention — FlashInfer router kernel.

    Uses FlashInfer's per-head router paged-attention kernel (use_router=True)
    to fuse full-attention and SWA routing into a single kernel call.

    Currently requires MHA (num_heads == num_kv_heads). The per-(token, q-head)
    gate is passed straight through as the per-(seq, kv-head) router tensor.
    GQA support would need a reduction rule across Q-heads in each group.

    During prefill: full attention only — no routing (matches routed/greedy).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        hidden_size = self.config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = self.config.num_attention_heads

        assert hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = (
            self.config.num_key_value_heads or self.total_num_heads
        )
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = self.config.max_position_embeddings
        self.scaling = self.head_dim**-0.5
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.local_window_size = getattr(self.config, "local_window_size", 128)

        self.tp_rank = get_tensor_model_parallel_rank()

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim + self.total_num_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.q_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.k_norm = RMSNorm(
            self.total_num_kv_heads * self.head_dim, eps=self.config.rms_norm_eps
        )

        rope_parameters = self.config.rope_parameters
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.kv_cache_attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=None,
            prefix=f"{prefix}.kv_cache_attn",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._prefix = prefix
        self._decode_wrapper = None
        self._prefill_wrapper = None
        self._workspace_buffer = None
        self._kv_layout = None

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(
                split_tensor_along_last_dim, num_partitions=self.tp_size
            )
            q = splitter(q)[self.tp_rank].contiguous()
            k = splitter(k)[self.tp_rank].contiguous()
        return q, k

    def _ensure_wrappers(self, device: torch.device) -> None:
        if self._decode_wrapper is not None:
            return
        import flashinfer
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
        self._kv_layout = get_kv_cache_layout()
        self._workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device,
        )
        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace_buffer, self._kv_layout, use_router=True,
        )
        self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace_buffer, self._kv_layout,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_with_gate, _ = self.q_proj(hidden_states)
        q, gate = q_with_gate.split([self.q_size, self.num_heads], dim=-1)
        q = q.contiguous()
        gate = gate.contiguous()

        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)

        attn_metadata, attn_layer, kv_cache = get_attention_context(
            self.kv_cache_attn.layer_name
        )

        if attn_metadata is None or attn_metadata.max_query_len > 1:
            return self._forward_prefill_full(q, k, v)
        else:
            gate_sigmoid = torch.sigmoid(gate)
            gate_hard = (gate_sigmoid > 0.5).to(hidden_states.dtype)

            if _GATE_STATS_ENABLED:
                self._log_gate_stats(gate_hard)

            return self._forward_decode_flashinfer(
                q, k, v, gate_hard, attn_metadata, attn_layer, kv_cache
            )

    def _forward_prefill_full(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = self.kv_cache_attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    @torch.compiler.disable
    def _forward_decode_flashinfer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_hard: torch.Tensor,
        attn_metadata,
        attn_layer,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        N = q.shape[0]
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        # Phase 0: KV cache update
        forward_ctx = get_forward_context()
        slot_mapping = forward_ctx.slot_mapping
        if isinstance(slot_mapping, dict):
            layer_slot_mapping = slot_mapping.get(self.kv_cache_attn.layer_name)
        else:
            layer_slot_mapping = slot_mapping

        key_cache, value_cache = kv_cache.unbind(0)

        if layer_slot_mapping is not None:
            reshape_and_cache_flash(
                k, v,
                key_cache, value_cache,
                layer_slot_mapping,
                attn_layer.kv_cache_dtype,
                attn_layer._k_scale,
                attn_layer._v_scale,
            )

        # Phase 1: Build flashinfer paged-KV indptr/indices/last_page_len
        # from vLLM's block_table + seq_lens.
        self._ensure_wrappers(q.device)
        block_size = key_cache.shape[1]
        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        batch_size = seq_lens.shape[0]

        num_blocks_per_seq = (seq_lens + block_size - 1) // block_size
        kv_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=q.device,
        )
        kv_indptr[1:] = num_blocks_per_seq.to(torch.int32).cumsum(0)

        max_blocks = int(num_blocks_per_seq.max().item())
        row_idx = torch.arange(max_blocks, device=q.device).unsqueeze(0)
        mask = row_idx < num_blocks_per_seq.unsqueeze(1)
        kv_indices = block_table[:, :max_blocks][mask].to(torch.int32).contiguous()

        kv_last_page_len = ((seq_lens - 1) % block_size + 1).to(torch.int32)

        # Phase 2: MHA — gate is already shape [N, num_kv_heads] since
        # num_heads == num_kv_heads. Pass through without reduction so the
        # kernel sees the raw per-head routing signal.
        assert self.num_queries_per_kv == 1, (
            "FlashInfer AHA variant currently requires MHA "
            "(num_heads == num_kv_heads). Got num_queries_per_kv="
            f"{self.num_queries_per_kv}. GQA support requires a reduction rule."
        )
        router = gate_hard.to(torch.uint8).contiguous()

        # Phase 3: Plan and run.
        self._decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            block_size,
            window_left=self.local_window_size - 1,
            q_data_type=q.dtype,
            kv_data_type=key_cache.dtype,
        )
        output = self._decode_wrapper.run(
            q, (key_cache, value_cache), router=router,
        )

        attn_output = output.view(N, -1)
        output_final, _ = self.o_proj(attn_output)
        return output_final

    def _log_gate_stats(self, gate_hard: torch.Tensor) -> None:
        global _gate_stats_counter
        _gate_stats_counter += 1
        if _gate_stats_counter % _GATE_STATS_INTERVAL != 0:
            return
        swa_frac = gate_hard.float().mean().item()
        logger.info(
            "FlashInfer router stats [%s] step=%d: SWA=%.1f%%, full=%.1f%%",
            self._prefix, _gate_stats_counter,
            swa_frac * 100, (1 - swa_frac) * 100,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch table
# ═══════════════════════════════════════════════════════════════════════════════

_ATTENTION_IMPLEMENTATIONS: dict[str, type[nn.Module]] = {
    "dual":       _AHADualAttention,
    "baseline":   _AHADualAttention,   # same impl, used as frozen reference
    "routed":     _AHARoutedAttention,
    "greedy":     _AHAGreedyAttention,
    "flashinfer": _AHAFlashInferAttention,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Shared model components (single copy used by all variants)
# ═══════════════════════════════════════════════════════════════════════════════

class FAOlmoMLP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class FAOlmoDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        impl = getattr(config, "attention_implementation", "greedy")
        if impl not in _ATTENTION_IMPLEMENTATIONS:
            raise ValueError(
                f"Unknown AHA attention_implementation: {impl!r}. "
                f"Choose from {list(_ATTENTION_IMPLEMENTATIONS)}"
            )
        self.self_attn = _ATTENTION_IMPLEMENTATIONS[impl](
            vllm_config=vllm_config, prefix=f"{prefix}.self_attn"
        )

        self.mlp = FAOlmoMLP(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile
class FAOlmoModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: FAOlmoDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            assert isinstance(hidden_states, torch.Tensor)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


class FAOlmoForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    """FAOlmo (AHA) for causal language modeling.

    The attention implementation is selected from
    config.attention_implementation: "dual" | "routed" | "greedy" | "baseline".
    """

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = FAOlmoModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head.weight"] if self.config.tie_word_embeddings else None
            ),
        )
        return loader.load_weights(weights)
