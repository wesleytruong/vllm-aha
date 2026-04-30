# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-side config for the FAOlmo (AHA) model family.

Compatible with the HuggingFace FAOlmo config (model_type = "faolmo").
Adds two AHA-specific fields:
  - attention_implementation: "dual" | "routed" | "greedy" | "baseline"
  - local_window_size: sliding-window size for local attention heads (int)

The active implementation can be overridden at runtime via the env var
VLLM_AHA_IMPL (takes precedence over the config-file value).
"""

import os
from typing import Any

from transformers.configuration_utils import PretrainedConfig

_VALID_IMPLS = ("dual", "routed", "greedy", "baseline")


class FAOlmoConfig(PretrainedConfig):
    model_type = "faolmo"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-06,
        use_cache: bool = True,
        pad_token_id: int = 100277,
        bos_token_id: int | None = None,
        eos_token_id: int = 100257,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # AHA-specific fields:
        attention_implementation: str = "greedy",
        local_window_size: int = 128,
        **kwargs,
    ):
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["FAOlmoForCausalLM"]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        # rope_parameters handling — mirrors FlexOlmoConfig for BC
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = rope_scaling or rope_parameters or {"rope_type": "default"}
        rope_theta = kwargs.pop("rope_theta", 500000.0)
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        self.rope_parameters = rope_parameters
        if self.rope_parameters is not None and "type" in self.rope_parameters:
            self.rope_parameters["rope_type"] = self.rope_parameters["type"]

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # AHA-specific: env var overrides config-file value for backward compat
        env_impl = os.environ.get("VLLM_AHA_IMPL")
        self.attention_implementation = env_impl if env_impl else attention_implementation
        self.local_window_size = local_window_size
