# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 KV cache layout planner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.deepseek_compressor import CompressorBackend
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

_DEEPSEEK_V4_ALIGNMENT = 576
_DEEPSEEK_V4_KV_BYTES_PER_TOKEN = 584


class DeepseekV4CacheType(str, Enum):
    """Logical DeepSeek V4 KV cache roles handled by this planner."""

    MAIN_MLA = "main_mla"
    SWA = "swa"
    COMPRESSOR_STATE = "compressor_state"
    INDEXER_K = "indexer_k"


@dataclass(frozen=True)
class DeepseekV4SpecInfo:
    """Normalized metadata for one DeepSeek V4 KV cache spec."""

    layer_name: str
    cache_type: DeepseekV4CacheType
    spec: KVCacheSpec
    compress_ratio: int
    bytes_per_token: int
    alignment: int | None


@dataclass(frozen=True)
class DeepseekV4CanonicalBuckets:
    """Canonical page-size buckets shared by DeepSeek V4 KV cache layers."""

    main_block_size: int
    page_sizes: tuple[int, ...]
    page_size_by_layer: dict[str, int]


SupportedBlockSizes = Sequence[int | MultipleOf]


def _supports_block_size(
    block_size: int, supported_block_sizes: SupportedBlockSizes
) -> bool:
    """Return whether a concrete block size satisfies backend constraints."""

    return any(
        block_size == supported
        if isinstance(supported, int)
        else block_size % supported.base == 0
        for supported in supported_block_sizes
    )


def _format_supported_block_sizes(supported_block_sizes: SupportedBlockSizes) -> str:
    """Format backend block-size constraints for error messages."""

    return repr(
        [
            supported
            if isinstance(supported, int)
            else f"MultipleOf({supported.base})"
            for supported in supported_block_sizes
        ]
    )


def _aligned_page_size(page_size: int, alignment: int | None) -> int:
    """Return page size padded to alignment, or unchanged when alignment is None."""

    if alignment is None:
        return page_size
    return round_up(page_size, alignment)


def infer_block_size_against_buckets(
    *,
    bytes_per_token: int,
    canonical_page_sizes: Sequence[int],
    alignment: int | None,
    supported_block_sizes: SupportedBlockSizes,
    extra_constraints: Sequence[Callable[[int], bool]] = (),
    candidate_key: Callable[[int, int, int], tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Infer a block size whose aligned page size fits a canonical bucket.

    Returns a ``(block_size, padded_page_size)`` pair. By default, the smallest
    usable bucket is chosen, with larger block sizes preferred inside the same
    bucket.
    """

    def feasible(block_size: int) -> tuple[int, int, int] | None:
        """Return the bucket assignment for a block size, if it is valid."""

        if block_size <= 0:
            return None
        if not _supports_block_size(block_size, supported_block_sizes):
            return None
        if not all(constraint(block_size) for constraint in extra_constraints):
            return None
        aligned_page_size = _aligned_page_size(
            block_size * bytes_per_token, alignment
        )
        for bucket in canonical_page_sizes:
            if bucket >= aligned_page_size:
                return block_size, bucket, aligned_page_size
        return None

    if candidate_key is None:
        # NOTE(Mengqing): the priority for selecting block size:
        # 1. align the current spec with the nearest page_size of MLAAttentionSpec
        # 2. use the largest block_size when 1 is met
        candidate_key = lambda block_size, bucket, _: (bucket, -block_size)

    max_bucket = max(canonical_page_sizes)
    max_candidate_block_size = max_bucket // bytes_per_token
    candidates = []
    for block_size in range(1, max_candidate_block_size + 1):
        if candidate := feasible(block_size):
            candidates.append(candidate)
    if not candidates:
        raise ValueError(
            "No valid DeepSeek V4 auxiliary KV cache block size found: "
            f"bytes_per_token={bytes_per_token}, alignment={alignment}, "
            f"canonical_buckets={list(canonical_page_sizes)}, "
            "backend_supported_block_sizes="
            f"{_format_supported_block_sizes(supported_block_sizes)}."
        )

    block_size, bucket, _ = min(candidates, key=lambda item: candidate_key(*item))
    return block_size, bucket


class DeepseekV4KVCachePlanner:
    """Finalize DeepSeek V4 KV cache layouts before generic grouping.

    DeepSeek V4 has several auxiliary caches whose natural page sizes differ
    from the main MLA cache. This planner assigns compatible block sizes and
    padded page sizes so the generic KV cache grouping logic can place all
    related layers into canonical page-size buckets.
    """

    def __init__(self, vllm_config: VllmConfig):
        """Create a planner bound to the engine cache configuration."""

        # main_block_size refers to the block_size of MLAAttentionSpec
        self.main_block_size = vllm_config.cache_config.block_size
        self.supported_compress_ratios = (
            set(vllm_config.model_config.hf_config.compress_ratios) | {1}
        )

    def plan(self, kv_cache_spec: dict[str, KVCacheSpec]) -> dict[str, KVCacheSpec]:
        """Return KV cache specs with DeepSeek V4 block/page sizes finalized."""

        spec_infos = self._classify_specs(kv_cache_spec)
        canonical_buckets = self._build_canonical_buckets(spec_infos)
        inferred_layouts = self._infer_sliding_win_mla_spec_layouts(
            spec_infos, canonical_buckets
        )
        finalized = self._finalize_specs(
            kv_cache_spec, spec_infos, canonical_buckets, inferred_layouts
        )
        return finalized

    def _classify_specs(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> list[DeepseekV4SpecInfo]:
        """Classify all recognizable DeepSeek V4 specs in a cache spec map."""

        spec_infos: list[DeepseekV4SpecInfo] = []

        for layer_name, spec in kv_cache_specs.items():
            assert isinstance(spec, MLAAttentionSpec | SlidingWindowMLASpec), \
                f"Unsupported DeepSeek V4 KV cache spec type: {type(spec).__name__}"
            assert spec.compress_ratio in self.supported_compress_ratios, \
                f"DeepSeek V4 KV cache spec {layer_name!r} has unsupported " \
                f"compress_ratio={spec.compress_ratio}; expected one of " \
                f"{sorted(self.supported_compress_ratios)}."
            if layer_name.endswith(".compressor.state_cache"):
                assert spec.dtype == torch.float32, (
                    f"DeepSeek V4 compressor state cache {layer_name!r} "
                    f"must use torch.float32, got {spec.dtype}."
                )
                cache_type = DeepseekV4CacheType.COMPRESSOR_STATE
            elif layer_name.endswith(".indexer.k_cache"):
                assert isinstance(spec, MLAAttentionSpec), (
                    f"DeepSeek V4 indexer K cache {layer_name!r} must use "
                    f"MLAAttentionSpec, got {type(spec).__name__}."
                )
                assert spec.dtype == torch.uint8, (
                    f"DeepSeek V4 indexer K cache {layer_name!r} must use "
                    f"torch.uint8, got {spec.dtype}."
                )
                cache_type = DeepseekV4CacheType.INDEXER_K
            elif layer_name.endswith(".swa_cache"):
                assert isinstance(spec, SlidingWindowMLASpec), (
                    f"DeepSeek V4 SWA cache {layer_name!r} must use "
                    f"SlidingWindowMLASpec, got {type(spec).__name__}."
                )
                cache_type = DeepseekV4CacheType.SWA
            elif isinstance(spec, MLAAttentionSpec):
                cache_type = DeepseekV4CacheType.MAIN_MLA
            spec_infos.append(
                self._create_info(
                    layer_name, cache_type, spec
                )
            )
        return spec_infos

    def _create_info(
        self, layer_name: str, cache_type: DeepseekV4CacheType, spec: KVCacheSpec
    ) -> DeepseekV4SpecInfo:
        """Build normalized metadata for a validated DeepSeek V4 cache spec."""
        bytes_per_token = self._bytes_per_token(cache_type, spec)
        return DeepseekV4SpecInfo(
            layer_name=layer_name,
            cache_type=cache_type,
            spec=spec,
            compress_ratio=spec.compress_ratio,
            bytes_per_token=bytes_per_token,
            alignment=spec.alignment,
        )

    def _bytes_per_token(
        self,
        cache_type: DeepseekV4CacheType,
        spec: MLAAttentionSpec | SlidingWindowMLASpec,
    ) -> int:
        """Return the physical bytes stored for one logical cache slot."""

        if cache_type in (DeepseekV4CacheType.MAIN_MLA, DeepseekV4CacheType.SWA):
            return _DEEPSEEK_V4_KV_BYTES_PER_TOKEN
        return spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)

    def _build_canonical_buckets(
        self, spec_infos: list[DeepseekV4SpecInfo]
    ) -> DeepseekV4CanonicalBuckets:
        """Build canonical page-size buckets from main MLA-like cache layers."""

        # TODO(Mengqing): get the supported block sizes from the actual backend 
        # implementations instead of hardcoding them here
        supported = DeepseekV4FlashMLASparseBackend.get_supported_kernel_block_sizes()
        if not _supports_block_size(self.main_block_size, supported):
            raise ValueError(
                "DeepSeek V4 main MLA backend supports block sizes "
                f"{_format_supported_block_sizes(supported)}, but "
                f"cache_config.block_size={self.main_block_size}. Current "
                "DeepSeek V4 FlashMLA sparse backend supports only "
                "block_size=256."
            )

        page_size_by_layer: dict[str, int] = {}
        for spec_info in spec_infos:
            if spec_info.cache_type not in (
                DeepseekV4CacheType.MAIN_MLA,
                DeepseekV4CacheType.INDEXER_K,
            ):
                # Only main MLA and indexer K caches contribute to canonical buckets, and
                # auxiliary caches can adjust their block sizes to fit into these buckets.
                continue
            if self.main_block_size % spec_info.compress_ratio != 0:
                raise ValueError(
                    "DeepSeek V4 requires cache_config.block_size to be "
                    f"divisible by compress_ratio={spec_info.compress_ratio} for "
                    f"layer {spec_info.layer_name}, got "
                    f"block_size={self.main_block_size}."
                )
            # the actual block size should be divided by the compress ratio for compressed caches
            storage_block_size = self.main_block_size // spec_info.compress_ratio
            page_size_by_layer[spec_info.layer_name] = _aligned_page_size(
                storage_block_size * spec_info.bytes_per_token, spec_info.alignment
            )

        if not page_size_by_layer:
            raise ValueError("DeepSeek V4 canonical KV cache buckets are empty.")

        return DeepseekV4CanonicalBuckets(
            main_block_size=self.main_block_size,
            page_sizes=tuple(sorted(set(page_size_by_layer.values()))),
            page_size_by_layer=page_size_by_layer,
        )

    def _infer_sliding_win_mla_spec_layouts(
        self,
        spec_infos: list[DeepseekV4SpecInfo],
        canonical: DeepseekV4CanonicalBuckets,
    ) -> dict[str, tuple[int, int]]:
        """Infer ``(block_size, padded_page_size)`` layouts for sliding window mla caches."""

        inferred: dict[str, tuple[int, int]] = {}
        swa_block_size: tuple[int, int] | None = None
        for spec_info in spec_infos:
            if spec_info.cache_type == DeepseekV4CacheType.SWA:
                if swa_block_size is None:
                    supported = (
                        DeepseekSparseSWABackend.get_supported_kernel_block_sizes()
                    )
                    swa_block_size = infer_block_size_against_buckets(
                        bytes_per_token=spec_info.bytes_per_token,
                        canonical_page_sizes=canonical.page_sizes,
                        alignment=spec_info.alignment,
                        supported_block_sizes=supported,
                    )
                inferred[spec_info.layer_name] = swa_block_size
            elif spec_info.cache_type == DeepseekV4CacheType.COMPRESSOR_STATE:
                spec = spec_info.spec
                assert isinstance(spec, SlidingWindowMLASpec)
                supported = CompressorBackend.get_supported_kernel_block_sizes()
                candidate_key = None
                if "indexer" not in spec_info.layer_name:
                    # NOTE(Mengqing): for the normal cache, we constraint it to be 
                    # aligned to the nearest smaller bucket, so that it can minimizes
                    # the pad. Ideally the other state caches should
                    # also be aligned to their nearest smaller bucket, but currently
                    # we observe that the block size inferred with this strategy is
                    # too small. This mainly because the compressor state cache has
                    # a smaller bytes_per_token. If all the state caches following
                    # this strategy, there will be a larger wastey memory due to padding
                    # introduced by kv cache grouping, which causes more number of groups
                    # and thus more padding.
                    candidate_key = (
                        lambda block_size, bucket: (
                            -block_size,
                            bucket,
                        )
                    )
                inferred[spec_info.layer_name] = infer_block_size_against_buckets(
                    bytes_per_token=spec_info.bytes_per_token,
                    canonical_page_sizes=canonical.page_sizes,
                    alignment=spec_info.alignment,
                    supported_block_sizes=supported,
                    extra_constraints=(
                        lambda block_size, window=spec.sliding_window: (
                            window % block_size == 0
                        ),
                        lambda block_size, window=spec.sliding_window: (
                            block_size <= window
                        ),
                    ),
                    candidate_key=candidate_key,
                )
        return inferred

    def _finalize_specs(
        self,
        kv_cache_specs: dict[str, KVCacheSpec],
        spec_infos: list[DeepseekV4SpecInfo],
        canonical: DeepseekV4CanonicalBuckets,
        auxiliary_layouts: dict[str, tuple[int, int]],
    ) -> dict[str, KVCacheSpec]:
        """Apply inferred DeepSeek V4 block/page sizes to the original specs."""

        finalized_kv_cache_specs: dict[str, KVCacheSpec] = {}
        spec_info_by_name = {spec_info.layer_name: spec_info for spec_info in spec_infos}
        for layer_name, spec_info in spec_info_by_name.items():
            if spec_info.cache_type in (
                DeepseekV4CacheType.MAIN_MLA,
                DeepseekV4CacheType.INDEXER_K,
            ):
                finalized_kv_cache_specs[layer_name] = replace(
                    spec_info.spec,
                    block_size=self.main_block_size,
                    page_size_padded=canonical.page_size_by_layer[layer_name],
                )
            elif spec_info.cache_type in (
                DeepseekV4CacheType.SWA,
                DeepseekV4CacheType.COMPRESSOR_STATE,
            ):
                block_size, page_size = auxiliary_layouts[layer_name]
                finalized_kv_cache_specs[layer_name] = replace(
                    spec_info.spec,
                    block_size=block_size,
                    page_size_padded=page_size,
                )
        res = {
            layer_name: finalized_kv_cache_specs.get(layer_name, spec)
            for layer_name, spec in kv_cache_specs.items()
        }
        print("Finalized DeepSeek V4 KV cache specs:")
        for layer_name, spec in res.items():
            print(f"  {layer_name}: {spec}")

        return res
