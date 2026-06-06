# SPDX-License-Identifier: Apache-2.0
"""FlashInfer AHA backend: plans a router-enabled decode wrapper in build().

Subclasses FlashInferBackend / FlashInferMetadataBuilder so that the
router wrapper's plan() runs in the metadata phase (outside the model
forward and Dynamo's view), matching vLLM's native FlashInfer integration.

Supports full-decode CUDA graph capture by maintaining a per-batch-size
dict of router wrappers planned against the parent's persistent CSR
buffers (mirrors FlashInferMetadataBuilder._get_decode_wrapper).
"""

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flashinfer import (
    FIDecode,
    FlashInferBackend,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
    fast_plan_decode,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import AttentionSpec


class AHAFlashInferBackend(FlashInferBackend):
    @staticmethod
    def get_name() -> str:
        # Preserve the FlashInfer backend identity so vLLM's existing
        # FlashInfer-specific initialization, warmup, and workspace-sharing
        # paths still apply. The AHA behavior comes from the custom builder.
        return "FLASHINFER"

    @staticmethod
    def get_builder_cls() -> type["AHAFlashInferMetadataBuilder"]:
        return AHAFlashInferMetadataBuilder


class AHAFlashInferMetadataBuilder(FlashInferMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self._local_window_size = getattr(
            vllm_config.model_config.hf_config, "local_window_size", 128
        )

        # Separate workspace from the parent's standard wrapper to avoid
        # serialization on the scratch buffer when both wrappers run in the
        # same step.
        self._router_workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )

        # Non-cudagraph fallback wrapper, used for batches above the cudagraph
        # max batch size, mixed prefill/decode batches, or when cudagraph mode
        # is disabled entirely.
        self._router_decode_wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None

        if self.enable_cuda_graph:
            self._router_decode_wrappers_cudagraph: dict[
                int, BatchDecodeWithPagedKVCacheWrapper
            ] = {}
            self._router_decode_cudagraph_max_bs = self._decode_cudagraph_max_bs

    def _get_router_decode_wrapper(
        self, batch_size: int, use_cudagraph: bool
    ) -> BatchDecodeWithPagedKVCacheWrapper:
        if use_cudagraph:
            wrapper = self._router_decode_wrappers_cudagraph.get(batch_size)
        else:
            wrapper = self._router_decode_wrapper

        if wrapper is None:
            if use_cudagraph:
                # Share the parent's CSR GPU buffers as persistent capture
                # buffers. Sizes: max_num_reqs+1, max_num_pages, max_num_reqs.
                # These are populated by super().build() each step, so replay
                # picks up fresh CSR data without re-recording the graph.
                indptr_buf = self.paged_kv_indptr.gpu[: batch_size + 1]
                indices_buf = self.paged_kv_indices.gpu
                last_page_buf = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                indptr_buf = None
                indices_buf = None
                last_page_buf = None

            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._router_workspace,
                get_kv_cache_layout(),
                use_cuda_graph=use_cudagraph,
                paged_kv_indptr_buffer=indptr_buf,
                paged_kv_indices_buffer=indices_buf,
                paged_kv_last_page_len_buffer=last_page_buf,
                use_router=True,
                use_tensor_cores=True,
            )

            if use_cudagraph:
                self._router_decode_wrappers_cudagraph[batch_size] = wrapper
            else:
                self._router_decode_wrapper = wrapper

        return wrapper

    @override
    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Allow full-decode cudagraph capture; without this, the runner's
        # min-cg-support resolver (gpu_model_runner.py) downgrades the
        # configured FULL_AND_PIECEWISE mode to PIECEWISE, which forces a
        # Python-side launch per layer's attention call.
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    @override
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )

        is_pure_decode = (
            attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
            and isinstance(attn_metadata.decode, FIDecode)
        )

        if is_pure_decode:
            num_decode_tokens = attn_metadata.num_decode_tokens
            num_input_tokens = num_decode_tokens

            use_cudagraph = (
                self.enable_cuda_graph
                and num_input_tokens <= self._router_decode_cudagraph_max_bs
            )

            router_wrapper = self._get_router_decode_wrapper(
                num_input_tokens, use_cudagraph
            )

            indptr_cpu = self.paged_kv_indptr.cpu[: num_input_tokens + 1]
            num_actual_pages = int(self.paged_kv_indptr.np[num_input_tokens])
            indices_gpu = self.paged_kv_indices.gpu[:num_actual_pages]
            last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_input_tokens]
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_input_tokens]

            fast_plan_decode(
                router_wrapper,
                indptr_cpu,
                indices_gpu,
                last_page_len_cpu,
                seq_lens_cpu,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                pos_encoding_mode="NONE",
                window_left=self._local_window_size - 1,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_cache_dtype,
            )
            attn_metadata.router_decode_wrapper = router_wrapper  # type: ignore[attr-defined]

            import os as _os
            if _os.environ.get("VLLM_AHA_DUMP_PLAN") == "1":
                rw = router_wrapper
                pi = getattr(rw, "_plan_info", None)
                try:
                    pi_repr = list(pi) if pi is not None else None
                except TypeError:
                    pi_repr = repr(pi)
                _sl = int(seq_lens_cpu[0]) if len(seq_lens_cpu) else -1
                if _sl > 1000:  # only the real long-context decode
                    from vllm.v1.attention.backends.utils import (
                        get_kv_cache_layout as _gkl,
                    )
                    print(
                        f"[aha plan dump] seq_len={_sl} backend={getattr(rw, '_backend', '?')} "
                        f"layout={_gkl()} use_cudagraph={use_cudagraph} bs={num_input_tokens} "
                        f"qo={self.num_qo_heads} kv={self.num_kv_heads} hd={self.head_dim} "
                        f"page={self.page_size} runtime_win={getattr(rw, '_window_left', '?')} "
                        f"use_tc={getattr(rw, '_use_tensor_cores', '?')} "
                        f"use_router={getattr(rw, '_use_router', '?')} "
                        f"num_pages={num_actual_pages} plan_info={pi_repr}",
                        flush=True,
                    )
        else:
            attn_metadata.router_decode_wrapper = None  # type: ignore[attr-defined]

        return attn_metadata
