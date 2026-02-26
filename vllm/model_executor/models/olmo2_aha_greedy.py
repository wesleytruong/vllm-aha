# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from OLMo2 for All-or-Here Attention (AHA)
# https://huggingface.co/xuan-luo/AHA-OLMO2
# AHA uses a learned per-head gate to route between global and local attention
"""Inference-only OLMo2-AHA model with greedy local-first attention.

This implementation computes local (sliding window) attention for ALL heads
in a single kernel call, then runs a second kernel call for only the ~10%
of GQA groups that need global attention. This reduces kernel launch overhead
from O(num_kv_heads) to O(1).

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
_TIMING_OUTPUT_FILE = os.environ.get("VLLM_AHA_TIMING_FILE", "/tmp/aha_timing_stats.json")
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
_local_only_calls = 0   # decode calls where global kernel was skipped entirely
_global_calls = 0       # decode calls where global kernel ran for ≥1 GQA group
_total_global_groups = 0  # cumulative GQA groups routed to global across all calls
_TIMING_LOG_INTERVAL = 100  # Log every N decode steps


class Olmo2AHAGreedyAttention(nn.Module):
    """
    All-or-Here Attention (AHA) with greedy local-first decode.

    During decode: computes local attention for ALL heads in one kernel call,
    then runs global attention only for the ~10% of GQA groups that need it.

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
        # and as the source of KV cache/metadata for greedy decode
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
            # Decode path: greedy local-first attention
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
        """Prefill: use standard Attention layer for full attention + KV update."""
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
        """Decode: local-first greedy attention — 1-2 kernel calls total."""
        N = q.shape[0]

        # Reshape for attention: [N, num_heads, head_dim]
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

        # === Phase 0: KV Cache Update (decoupled from attention) ===
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

        # FP8 descale shapes — all KV heads
        descale_shape = (
            attn_metadata.query_start_loc.shape[0] - 1,
            self.num_kv_heads,
        )
        q_descale = attn_layer._q_scale.expand(descale_shape)
        k_descale = attn_layer._k_scale.expand(descale_shape)
        v_descale = attn_layer._v_scale.expand(descale_shape)

        # === Phase 1: Local attention for ALL heads (1 kernel call) ===
        output = torch.empty(
            N, self.num_heads, self.head_dim,
            dtype=q.dtype, device=q.device,
        )

        if _TIMING_ENABLED:
            t1_start.record()
        flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            out=output,
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
            t1_end.record()

        # === Phase 2: Identify GQA groups needing global attention ===
        # gate_hard: [N, num_heads] — reshape to [N, num_kv_heads, nqpkv]
        gate_by_group = gate_hard.view(N, self.num_kv_heads, self.num_queries_per_kv)
        # A group needs global if ANY (token, q-head) in the group has gate=1
        group_needs_global = gate_by_group.any(dim=0).any(dim=-1)  # [num_kv_heads]
        global_group_indices = group_needs_global.nonzero(as_tuple=False).squeeze(-1)

        if _TIMING_ENABLED:
            global _local_only_calls, _global_calls, _total_global_groups
            n_global = global_group_indices.numel()
            if n_global > 0:
                _global_calls += 1
                _total_global_groups += n_global
            else:
                _local_only_calls += 1

        # === Phase 3: Global attention for subset (0 or 1 kernel call) ===
        if _TIMING_ENABLED:
            t3_start.record()

        if global_group_indices.numel() > 0:
            num_global_kv = global_group_indices.numel()

            # Build Q head indices for the global groups
            # For each KV group g, Q heads are [g*nqpkv, ..., (g+1)*nqpkv)
            nqpkv = self.num_queries_per_kv
            q_head_indices = torch.cat([
                torch.arange(
                    g * nqpkv, (g + 1) * nqpkv,
                    device=q.device,
                )
                for g in global_group_indices
            ])

            # Gather Q, K, V for global groups
            q_global = q[:, q_head_indices, :].contiguous()
            k_global = key_cache[:, :, global_group_indices, :].contiguous()
            v_global = value_cache[:, :, global_group_indices, :].contiguous()

            global_output = torch.empty(
                N, q_head_indices.numel(), self.head_dim,
                dtype=q.dtype, device=q.device,
            )

            # FP8 descale for subset
            global_descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                num_global_kv,
            )
            q_descale_g = attn_layer._q_scale.expand(global_descale_shape)
            k_descale_g = attn_layer._k_scale.expand(global_descale_shape)
            v_descale_g = attn_layer._v_scale.expand(global_descale_shape)

            flash_attn_varlen_func(
                q=q_global,
                k=k_global,
                v=v_global,
                out=global_output,
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scaling,
                causal=True,
                window_size=[-1, -1],  # Full attention
                block_table=attn_metadata.block_table,
                fa_version=fa_version,
                q_descale=q_descale_g,
                k_descale=k_descale_g,
                v_descale=v_descale_g,
            )

            # === Phase 4: Selective replacement ===
            # gate=1 → use global result; gate=0 → keep local result
            gate_expanded = gate_hard[:, q_head_indices].unsqueeze(-1)  # [N, global_q_heads, 1]
            output[:, q_head_indices, :] = (
                gate_expanded * global_output
                + (1.0 - gate_expanded) * output[:, q_head_indices, :]
            )

        if _TIMING_ENABLED:
            t3_end.record()
            self._accumulate_timing(t0_start, t0_end, t1_start, t1_end, t3_start, t3_end,
                                    layer_slot_mapping is not None)

        # === Phase 5: Output projection ===
        attn_output = output.view(N, -1)
        output_final, _ = self.o_proj(attn_output)
        return output_final

    def _accumulate_timing(
        self,
        t0_start, t0_end,
        t1_start, t1_end,
        t3_start, t3_end,
        has_kv_update: bool,
    ) -> None:
        """Accumulate phase timings and log periodically."""
        global _timing_stats, _timing_totals, _timing_count
        torch.cuda.synchronize()
        kv_ms     = t0_start.elapsed_time(t0_end) if has_kv_update else 0.0
        local_ms  = t1_start.elapsed_time(t1_end)
        global_ms = t3_start.elapsed_time(t3_end)

        _timing_stats["kv_update"]  += kv_ms
        _timing_stats["local_attn"] += local_ms
        _timing_stats["global_attn"] += global_ms

        _timing_totals["kv_update"]  += kv_ms
        _timing_totals["local_attn"] += local_ms
        _timing_totals["global_attn"] += global_ms

        _timing_count += 1

        if _timing_count % _TIMING_LOG_INTERVAL == 0:
            n = _TIMING_LOG_INTERVAL
            kv_ms    = _timing_stats["kv_update"]  / n
            local_ms = _timing_stats["local_attn"] / n
            glob_ms  = _timing_stats["global_attn"] / n
            total_ms = kv_ms + local_ms + glob_ms
            logger.info(
                "Timing [%s] step=%d (avg over %d decode calls): "
                "kv_update=%.3f ms (%.1f%%), "
                "local_attn=%.3f ms (%.1f%%), "
                "global_attn=%.3f ms (%.1f%%), "
                "total=%.3f ms",
                self._prefix, _timing_count, n,
                kv_ms,    100 * kv_ms    / total_ms if total_ms else 0,
                local_ms, 100 * local_ms / total_ms if total_ms else 0,
                glob_ms,  100 * glob_ms  / total_ms if total_ms else 0,
                total_ms,
            )
            _timing_stats["kv_update"]  = 0.0
            _timing_stats["local_attn"] = 0.0
            _timing_stats["global_attn"] = 0.0

            # Flush lifetime totals to file so the parent process can read them
            import json
            try:
                with open(_TIMING_OUTPUT_FILE, "w") as _f:
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
        """Log per-layer routing statistics."""
        global _gate_stats_counter
        _gate_stats_counter += 1
        if _gate_stats_counter % _GATE_STATS_INTERVAL != 0:
            return

        N, H = gate_hard.shape
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

        # Greedy-specific: count groups needing global kernel
        group_needs_global = gate_by_group.any(dim=0).any(dim=-1)
        num_global_groups = group_needs_global.sum().item()

        pct_local = all_local_count / self.num_kv_heads * 100
        pct_global = all_global_count / self.num_kv_heads * 100
        pct_mixed = mixed_count / self.num_kv_heads * 100
        logger.info(
            "Gate stats [%s] step=%d: "
            "all_local=%d (%.1f%%), all_global=%d (%.1f%%), mixed=%d (%.1f%%), "
            "global_kernel_groups=%d/%d",
            self._prefix, _gate_stats_counter,
            all_local_count, pct_local,
            all_global_count, pct_global,
            mixed_count, pct_mixed,
            num_global_groups, self.num_kv_heads,
        )


# === MLP, DecoderLayer, Model, ForCausalLM — identical structure, renamed ===


class Olmo2AHAGreedyMLP(nn.Module):
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


class Olmo2AHAGreedyDecoderLayer(nn.Module):
    """Transformer decoder layer with greedy AHA attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.self_attn = Olmo2AHAGreedyAttention(
            vllm_config=vllm_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = Olmo2AHAGreedyMLP(
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
class Olmo2AHAGreedyModel(nn.Module):
    """OLMo2-AHA model backbone with greedy local-first attention."""

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
            lambda prefix: Olmo2AHAGreedyDecoderLayer(
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


class Olmo2AHAGreedyForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    """OLMo2 with greedy local-first AHA for causal language modeling."""

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
        self.model = Olmo2AHAGreedyModel(
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
