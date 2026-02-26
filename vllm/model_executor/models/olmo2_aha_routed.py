# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from OLMo2 for All-or-Here Attention (AHA)
# https://huggingface.co/xuan-luo/AHA-OLMO2
# AHA uses a learned per-head gate to route between global and local attention
"""Inference-only OLMo2-AHA model with routed per-head attention.

This implementation only computes the attention type each (token, head) pair
is actually routed to, batching efficiently via per-GQA-group kernel calls.
During decode (~90% local), this skips the expensive global attention read
of the full KV cache for most heads.

Prefill uses full (global) attention only — no dual computation, no gate.
"""

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

# AHA Configuration
LOCAL_WINDOW_SIZE = 128

# Gate statistics logging
_GATE_STATS_ENABLED = bool(os.environ.get("VLLM_AHA_GATE_STATS"))
_gate_stats_counter = 0
_GATE_STATS_INTERVAL = 100  # Log every N forward passes

# Phase timing (VLLM_AHA_TIMING=1)
_TIMING_ENABLED = bool(os.environ.get("VLLM_AHA_TIMING"))
_TIMING_OUTPUT_FILE = os.environ.get("VLLM_AHA_TIMING_FILE", "/tmp/aha_timing_stats_routed.json")
_timing_stats: dict[str, float] = {   # per-interval (resets after logging)
    "kv_update": 0.0,
    "local_attn": 0.0,
    "global_attn": 0.0,
}
_timing_totals: dict[str, float] = {  # lifetime totals (never reset)
    "kv_update": 0.0,
    "local_attn": 0.0,
    "global_attn": 0.0,
}
_timing_count = 0
# Per-group routing counts (lifetime, never reset)
_all_local_groups  = 0
_all_global_groups = 0
_mixed_groups      = 0
_TIMING_LOG_INTERVAL = 100


class Olmo2AHARoutedAttention(nn.Module):
    """
    All-or-Here Attention (AHA) with per-GQA-group routed decode.

    During decode: only computes the attention type each head needs.
    ~90% of heads route to local attention, skipping expensive global reads.

    During prefill: uses full (global) attention only — no gate, no blending.
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
        self.local_window_size = LOCAL_WINDOW_SIZE

        self.tp_rank = get_tensor_model_parallel_rank()

        # Q projection WITH GATE (matches FAOlmo weight shape)
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim + self.total_num_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.q_proj",
        )

        # Separate K, V projections
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

        # QK normalization
        self.q_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.k_norm = RMSNorm(
            self.total_num_kv_heads * self.head_dim,
            eps=self.config.rms_norm_eps,
        )

        # Rotary embeddings
        rope_parameters = self.config.rope_parameters
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        # SINGLE Attention layer — owns the KV cache, used for prefill
        # and as the source of KV cache/metadata for routed decode
        self.kv_cache_attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=None,  # Full attention (no window)
            prefix=f"{prefix}.kv_cache_attn",
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Layer name for logging
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
        # === Project Q with gate ===
        q_with_gate, _ = self.q_proj(hidden_states)
        q, gate = q_with_gate.split([self.q_size, self.num_heads], dim=-1)
        q = q.contiguous()
        gate = gate.contiguous()

        # === Project K, V ===
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # === Apply QK normalization ===
        q, k = self._apply_qk_norm(q, k)

        # === Apply rotary embeddings ===
        q, k = self.rotary_emb(positions, q, k)

        # === Detect decode vs prefill ===
        attn_metadata, attn_layer, kv_cache = get_attention_context(
            self.kv_cache_attn.layer_name
        )

        if attn_metadata is None or attn_metadata.max_query_len > 1:
            # Prefill path: full attention only, no gate
            return self._forward_prefill_full(q, k, v)
        else:
            # Decode path: routed per-head attention
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
        """Prefill: use standard Attention layer for full attention + KV update."""
        attn_output = self.kv_cache_attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

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
        """Decode: per-GQA-group routed attention with skip optimization."""
        N = q.shape[0]

        # Reshape for attention: [N, num_heads, head_dim]
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        # === Phase 1: KV Cache Update (decoupled from attention) ===
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

        # === Phase 2: Per-GQA-Group Routed Attention ===
        fa_version = attn_layer.impl.vllm_flash_attn_version

        # Handle FP8 KV cache
        if attn_layer.kv_cache_dtype.startswith("fp8"):
            from vllm.v1.attention.backends.flash_attn import (
                FlashAttentionBackend,
            )
            fp8_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                attn_layer.kv_cache_dtype
            )
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)

        output = torch.empty(
            N, self.num_heads, self.head_dim,
            dtype=q.dtype, device=q.device,
        )

        if _TIMING_ENABLED:
            _local_ms  = 0.0
            _global_ms = 0.0
            _n_local = _n_global = _n_mixed = 0

        for g in range(self.num_kv_heads):
            q_start = g * self.num_queries_per_kv
            q_end = q_start + self.num_queries_per_kv

            # Q slice for this group: [N, nqpkv, D]
            q_group = q[:, q_start:q_end, :].contiguous()

            # Gate for this group: [N, nqpkv]
            group_gate = gate_hard[:, q_start:q_end]

            # Determine routing for the group
            has_any_global = group_gate.any().item()
            has_any_local = (~group_gate.bool()).any().item()

            # Slice KV cache to this single KV head
            # key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            k_cache_g = key_cache[:, :, g:g+1, :].contiguous()
            v_cache_g = value_cache[:, :, g:g+1, :].contiguous()

            # FP8 descale shapes — 1 KV head
            descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, 1)
            q_descale = attn_layer._q_scale.expand(descale_shape)
            k_descale = attn_layer._k_scale.expand(descale_shape)
            v_descale = attn_layer._v_scale.expand(descale_shape)

            if has_any_global and not has_any_local:
                # === ALL GLOBAL: single call, no sliding window ===
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
                    q=q_group,
                    k=k_cache_g,
                    v=v_cache_g,
                    out=group_out,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scaling,
                    causal=True,
                    window_size=[-1, -1],
                    block_table=attn_metadata.block_table,
                    fa_version=fa_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                if _TIMING_ENABLED:
                    _te.record()
                    torch.cuda.synchronize()
                    _global_ms += _ts.elapsed_time(_te)
                output[:, q_start:q_end, :] = group_out

            elif not has_any_global and has_any_local:
                # === ALL LOCAL: single call, sliding window ===
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
                    q=q_group,
                    k=k_cache_g,
                    v=v_cache_g,
                    out=group_out,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scaling,
                    causal=True,
                    window_size=[self.local_window_size - 1, 0],
                    block_table=attn_metadata.block_table,
                    fa_version=fa_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                if _TIMING_ENABLED:
                    _te.record()
                    torch.cuda.synchronize()
                    _local_ms += _ts.elapsed_time(_te)
                output[:, q_start:q_end, :] = group_out

            else:
                # === MIXED: both calls needed, per-head blend ===
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
                    q=q_group, k=k_cache_g, v=v_cache_g,
                    out=global_out,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scaling, causal=True,
                    window_size=[-1, -1],
                    block_table=attn_metadata.block_table,
                    fa_version=fa_version,
                    q_descale=q_descale, k_descale=k_descale,
                    v_descale=v_descale,
                )
                if _TIMING_ENABLED:
                    _tge.record()
                    _tls.record()
                flash_attn_varlen_func(
                    q=q_group, k=k_cache_g, v=v_cache_g,
                    out=local_out,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scaling, causal=True,
                    window_size=[self.local_window_size - 1, 0],
                    block_table=attn_metadata.block_table,
                    fa_version=fa_version,
                    q_descale=q_descale, k_descale=k_descale,
                    v_descale=v_descale,
                )
                if _TIMING_ENABLED:
                    _tle.record()
                    torch.cuda.synchronize()
                    _global_ms += _tgs.elapsed_time(_tge)
                    _local_ms  += _tls.elapsed_time(_tle)

                # Per-head blend: gate=1 selects global, gate=0 selects local
                gate_expanded = group_gate.unsqueeze(-1).to(q.dtype)
                output[:, q_start:q_end, :] = (
                    global_out * gate_expanded
                    + local_out * (1.0 - gate_expanded)
                )

        if _TIMING_ENABLED:
            torch.cuda.synchronize()
            kv_ms = _t0s.elapsed_time(_t0e) if layer_slot_mapping is not None else 0.0
            self._accumulate_timing(kv_ms, _local_ms, _global_ms,
                                    _n_local, _n_global, _n_mixed)

        # === Phase 3: Flatten and output projection ===
        attn_output = output.view(N, -1)
        output_final, _ = self.o_proj(attn_output)
        return output_final

    def _accumulate_timing(
        self,
        kv_ms: float,
        local_ms: float,
        global_ms: float,
        n_local: int,
        n_global: int,
        n_mixed: int,
    ) -> None:
        global _timing_stats, _timing_totals, _timing_count
        global _all_local_groups, _all_global_groups, _mixed_groups

        _timing_stats["kv_update"]  += kv_ms
        _timing_stats["local_attn"] += local_ms
        _timing_stats["global_attn"] += global_ms

        _timing_totals["kv_update"]  += kv_ms
        _timing_totals["local_attn"] += local_ms
        _timing_totals["global_attn"] += global_ms

        _all_local_groups  += n_local
        _all_global_groups += n_global
        _mixed_groups      += n_mixed
        _timing_count += 1

        if _timing_count % _TIMING_LOG_INTERVAL == 0:
            n = _TIMING_LOG_INTERVAL
            kv_ms_avg    = _timing_stats["kv_update"]  / n
            local_ms_avg = _timing_stats["local_attn"] / n
            glob_ms_avg  = _timing_stats["global_attn"] / n
            total_ms     = kv_ms_avg + local_ms_avg + glob_ms_avg
            logger.info(
                "Timing [%s] step=%d (avg over %d decode calls): "
                "kv_update=%.3f ms (%.1f%%), "
                "local_attn=%.3f ms (%.1f%%), "
                "global_attn=%.3f ms (%.1f%%), "
                "total=%.3f ms",
                self._prefix, _timing_count, n,
                kv_ms_avg,    100 * kv_ms_avg    / total_ms if total_ms else 0,
                local_ms_avg, 100 * local_ms_avg / total_ms if total_ms else 0,
                glob_ms_avg,  100 * glob_ms_avg  / total_ms if total_ms else 0,
                total_ms,
            )
            _timing_stats["kv_update"]  = 0.0
            _timing_stats["local_attn"] = 0.0
            _timing_stats["global_attn"] = 0.0

            import json
            try:
                total_groups = _all_local_groups + _all_global_groups + _mixed_groups
                with open(_TIMING_OUTPUT_FILE, "w") as _f:
                    json.dump({
                        "totals": _timing_totals,
                        "count": _timing_count,
                        "all_local_groups":  _all_local_groups,
                        "all_global_groups": _all_global_groups,
                        "mixed_groups":      _mixed_groups,
                        "total_groups":      total_groups,
                        "num_kv_heads":      self.num_kv_heads,
                    }, _f)
            except OSError:
                pass

    def _log_gate_stats(self, gate_hard: torch.Tensor) -> None:
        """Log per-layer routing statistics."""
        global _gate_stats_counter
        _gate_stats_counter += 1
        if _gate_stats_counter % _GATE_STATS_INTERVAL != 0:
            return

        N, H = gate_hard.shape
        # Reshape gate to [N, num_kv_heads, num_queries_per_kv]
        gate_by_group = gate_hard.view(N, self.num_kv_heads, self.num_queries_per_kv)

        all_global_count = 0
        all_local_count = 0
        mixed_count = 0

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

        pct_local = all_local_count / self.num_kv_heads * 100
        pct_global = all_global_count / self.num_kv_heads * 100
        pct_mixed = mixed_count / self.num_kv_heads * 100
        logger.info(
            "Gate stats [%s] step=%d: "
            "all_local=%d (%.1f%%), all_global=%d (%.1f%%), mixed=%d (%.1f%%)",
            self._prefix, _gate_stats_counter,
            all_local_count, pct_local,
            all_global_count, pct_global,
            mixed_count, pct_mixed,
        )


# === MLP, DecoderLayer, Model, ForCausalLM — identical to olmo2_aha.py ===


class Olmo2AHARoutedMLP(nn.Module):
    """MLP block (same as standard OLMo2)."""

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


class Olmo2AHARoutedDecoderLayer(nn.Module):
    """Transformer decoder layer with routed AHA attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.self_attn = Olmo2AHARoutedAttention(
            vllm_config=vllm_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = Olmo2AHARoutedMLP(
            vllm_config=vllm_config, prefix=f"{prefix}.mlp"
        )
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
        # Attention block
        residual = hidden_states
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # MLP block
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile
class Olmo2AHARoutedModel(nn.Module):
    """OLMo2-AHA model backbone with routed attention."""

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
            lambda prefix: Olmo2AHARoutedDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states"], self.config.hidden_size
            )
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

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
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
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


class Olmo2AHARoutedForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    """OLMo2 with routed AHA for causal language modeling."""

    packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = Olmo2AHARoutedModel(
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
                ["lm_head.weight"]
                if self.config.tie_word_embeddings
                else None
            ),
        )
        return loader.load_weights(weights)
