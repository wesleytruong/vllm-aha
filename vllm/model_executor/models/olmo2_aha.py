# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from OLMo2 for All-or-Here Attention (AHA)
# https://huggingface.co/xuan-luo/AHA-OLMO2
# AHA uses a learned per-head gate to route between global and local attention
"""Inference-only OLMo2-AHA model with All-or-Here Attention."""

from collections.abc import Iterable
from functools import partial
from itertools import islice

import torch
from torch import nn
from transformers import Olmo2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.distributed.utils import split_tensor_along_last_dim
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
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

# AHA Configuration
LOCAL_WINDOW_SIZE = 128


class Olmo2AHAAttention(nn.Module):
    """
    All-or-Here Attention (AHA) block.

    This attention computes both global (full) and local (sliding window)
    attention, then uses a learned per-head gate to blend the outputs:

    output = gate * global_attn + (1 - gate) * local_attn

    For inference, we use hard gating (gate > 0.5) to select between outputs.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        # FAOlmo uses a custom config, not Olmo2Config
        # assert isinstance(self.config, Olmo2Config)

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

        self.tp_rank = get_tensor_model_parallel_rank()

        # Q projection WITH GATE (matches FAOlmo weight shape)
        # Output: [Q (num_heads * head_dim), gate (num_heads)]
        # Note: FAOlmo uses total dimensions, ColumnParallelLinear will shard
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim + self.total_num_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.q_proj",
        )

        # Separate K, V projections (matches FAOlmo structure)
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

        # QK normalization (same as OLMo2)
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

        # === DUAL ATTENTION LAYERS ===
        # Global attention: full context, owns the KV cache
        self.global_attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=None,  # Full attention
            prefix=f"{prefix}.global_attn",
        )

        # Local attention: sliding window, SHARES KV cache with global
        self.local_attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=LOCAL_WINDOW_SIZE,
            prefix=f"{prefix}.local_attn",
            kv_sharing_target_layer_name=f"{prefix}.global_attn",  # Share KV cache
        )

        # Output projection
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
        # === Project Q with gate ===
        q_with_gate, _ = self.q_proj(hidden_states)
        # Split Q and gate
        # After TP sharding: q_size = num_heads * head_dim, gate_size = num_heads
        q, gate = q_with_gate.split([self.q_size, self.num_heads], dim=-1)
        # Make tensors contiguous after split to ensure proper memory layout
        q = q.contiguous()
        gate = gate.contiguous()

        # === Project K, V ===
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # === Apply QK normalization ===
        q, k = self._apply_qk_norm(q, k)

        # === Apply rotary embeddings ===
        q, k = self.rotary_emb(positions, q, k)

        # === Global attention (updates KV cache) ===
        global_output = self.global_attn(q, k, v)

        # === Local attention (uses shared KV cache, applies sliding window) ===
        # Pass None for k, v since we're sharing the KV cache
        local_output = self.local_attn(q, None, None)

        # === Hard gate routing ===
        gate_sigmoid = torch.sigmoid(gate)  # [num_tokens, num_heads]
        gate_hard = (gate_sigmoid > 0.5).to(hidden_states.dtype)

        # === Blend outputs per-head ===
        num_tokens = global_output.shape[0]
        global_out = global_output.view(num_tokens, self.num_heads, self.head_dim)
        local_out = local_output.view(num_tokens, self.num_heads, self.head_dim)
        gate_hard = gate_hard.unsqueeze(-1)  # [num_tokens, num_heads, 1]

        # gate=1 -> global, gate=0 -> local
        blended = global_out * gate_hard + local_out * (1.0 - gate_hard)
        blended = blended.view(num_tokens, -1)

        # === Output projection ===
        output, _ = self.o_proj(blended)
        return output


class Olmo2AHAMLP(nn.Module):
    """
    MLP block (same as standard OLMo2).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        # FAOlmo uses a custom config
        # assert isinstance(config, Olmo2Config)
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        # Feed-forward input projection
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        # Activation function
        self.act_fn = SiluAndMul()

        # Feed-forward output projection
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


class Olmo2AHADecoderLayer(nn.Module):
    """
    Transformer decoder layer with AHA attention.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        # FAOlmo uses a custom config
        # assert isinstance(config, Olmo2Config)

        # AHA Attention block
        self.self_attn = Olmo2AHAAttention(
            vllm_config=vllm_config, prefix=f"{prefix}.self_attn"
        )

        # MLP block
        self.mlp = Olmo2AHAMLP(vllm_config=vllm_config, prefix=f"{prefix}.mlp")

        # LayerNorm
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
class Olmo2AHAModel(nn.Module):
    """
    OLMo2-AHA model backbone.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        # FAOlmo uses a custom config, not Olmo2Config
        # assert isinstance(self.config, Olmo2Config)

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: Olmo2AHADecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
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

        # Apply decoder layers
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # FAOlmo uses separate Q, K, V projections (not fused QKV)
        # Only MLP gate_up_proj needs stacking
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue

            # Check for stacked params (MLP gate_up_proj)
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
                # Direct loading for Q, K, V, O projections and norms
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


class Olmo2AHAForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    """
    OLMo2 with All-or-Here Attention (AHA) for causal language modeling.
    """

    # No QKV fusion for AHA - only MLP gate_up_proj is packed
    packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        # FAOlmo uses a custom config
        # assert isinstance(config, Olmo2Config)
        self.config = config
        self.model = Olmo2AHAModel(
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
