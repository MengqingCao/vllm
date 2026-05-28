# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 KV cache layout planner."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.models.deepseek_v4.compressor import CompressorBackend
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.core.kv_cache_planner import KVCachePlanner

_DEEPSEEK_V4_ALIGNMENT = 576
_DEEPSEEK_V4_KV_BYTES_PER_TOKEN = 584

logger = init_logger(__name__)


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


_SupportedBlockSizes = Sequence[int | MultipleOf]


def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding."""

    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


def _supports_block_size(
    block_size: int, supported_block_sizes: _SupportedBlockSizes
) -> bool:
    """Return whether a concrete block size satisfies backend constraints."""

    return any(
        block_size == supported
        if isinstance(supported, int)
        else block_size % supported.base == 0
        for supported in supported_block_sizes
    )


def _format_supported_block_sizes(supported_block_sizes: _SupportedBlockSizes) -> str:
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


def _infer_block_size_against_buckets(
    *,
    bytes_per_token: int,
    canonical_page_sizes: Sequence[int],
    alignment: int | None,
    supported_block_sizes: _SupportedBlockSizes,
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


class DeepseekV4KVCachePlanner(KVCachePlanner):
    """Plan DeepSeek V4 KV cache layouts and groups.

    DeepSeek V4 has several auxiliary caches whose natural page sizes differ
    from the main MLA cache. This planner assigns compatible block sizes and
    padded page sizes, then builds KV cache groups around canonical page-size
    buckets shared by the main MLA layers.
    """

    def __init__(self, vllm_config: VllmConfig):
        """Create a planner bound to the engine cache configuration."""

        super().__init__(vllm_config)
        # main_block_size refers to the block_size of MLAAttentionSpec
        self.main_block_size = vllm_config.cache_config.block_size
        supported = DeepseekV4FlashMLASparseBackend.get_supported_kernel_block_sizes()
        if (
            not _supports_block_size(self.main_block_size, supported)
            and not self.cache_config.user_specified_block_size
        ):
            self.main_block_size = (
                DeepseekV4FlashMLASparseBackend.get_preferred_block_size(
                    self.main_block_size
                )
            )
            self.cache_config.block_size = self.main_block_size
        self.supported_compress_ratios = set(
            vllm_config.model_config.hf_config.compress_ratios
        ) | {1}

    def get_kv_cache_configs(
        self,
        kv_cache_specs: list[dict[str, KVCacheSpec]],
        available_memory: list[int],
    ) -> list[KVCacheConfig]:
        """Build per-worker KV cache configs for DeepSeek V4."""

        merged_specs = self._merge_worker_specs(kv_cache_specs)
        finalized_specs = self._post_process_kv_cache_specs(merged_specs)
        global_groups = self._get_kv_cache_groups(finalized_specs)
        worker_groups = [
            self._project_groups_to_worker(global_groups, worker_spec)
            for worker_spec in kv_cache_specs
        ]

        available_memory = self._apply_num_blocks_override(
            worker_groups, available_memory
        )
        self._maybe_auto_fit_max_model_len(worker_groups, available_memory)
        for groups, memory in zip(worker_groups, available_memory):
            if groups:
                self._check_model_len_capacity(groups, memory)

        kv_cache_configs = [
            self._build_config_from_groups(groups, memory)
            for groups, memory in zip(worker_groups, available_memory)
        ]
        self._shrink_to_min_num_blocks(kv_cache_configs)
        for config in kv_cache_configs:
            if config.kv_cache_groups:
                self._report_config(config)
        return kv_cache_configs

    def _post_process_kv_cache_specs(
        self, kv_cache_spec: dict[str, KVCacheSpec]
    ) -> dict[str, KVCacheSpec]:
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

    def get_kv_cache_groups(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> list[KVCacheGroupSpec]:
        """Finalize and group one DeepSeek V4 KV cache spec map."""

        _finalize_specs = self._post_process_kv_cache_specs(kv_cache_specs)
        return self._get_kv_cache_groups(
            _finalize_specs
        )

    def get_kv_cache_config_from_groups(
        self, kv_cache_groups: list[KVCacheGroupSpec], available_memory: int
    ) -> KVCacheConfig:
        """Build one DeepSeek V4 KV cache config from planned groups."""

        return self._build_config_from_groups(kv_cache_groups, available_memory)

    def _get_kv_cache_groups(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> list[KVCacheGroupSpec]:
        """Build DeepSeek V4 KV cache groups from finalized specs."""

        grouped_specs = self._group_and_unify_specs(kv_cache_specs)
        kv_cache_groups = self._get_kv_cache_groups_from_uniform_groups(grouped_specs)
        self._annotate_eagle_groups(kv_cache_specs, kv_cache_groups)
        return kv_cache_groups

    def get_max_model_len_capacity(
        self, kv_cache_groups: list[KVCacheGroupSpec], available_memory: int
    ) -> int:
        """Return the largest model length supported by this KV layout."""

        original_max_len = self.vllm_config.model_config.max_model_len

        def fits(model_len: int) -> bool:
            self.vllm_config.model_config.max_model_len = model_len
            return self._max_memory_usage_bytes(kv_cache_groups) <= available_memory

        try:
            left, right = 1, original_max_len
            if not fits(left):
                return 0
            capacity = 1
            while left <= right:
                mid = (left + right) // 2
                if fits(mid):
                    capacity = mid
                    left = mid + 1
                else:
                    right = mid - 1
            return capacity
        finally:
            self.vllm_config.model_config.max_model_len = original_max_len

    def _merge_worker_specs(
        self, kv_cache_specs: list[dict[str, KVCacheSpec]]
    ) -> dict[str, KVCacheSpec]:
        """Merge per-worker specs into one global DeepSeek V4 spec map."""

        merged: dict[str, KVCacheSpec] = {}
        for worker_specs in kv_cache_specs:
            for layer_name, layer_spec in worker_specs.items():
                if layer_name not in merged:
                    merged[layer_name] = layer_spec
                else:
                    assert merged[layer_name] == layer_spec, (
                        "The KV cache specs for the same layer are different "
                        "across workers. This is not supported yet."
                    )
        return merged

    def _project_groups_to_worker(
        self,
        global_groups: list[KVCacheGroupSpec],
        worker_specs: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        """Filter global groups to the layers owned by one worker."""

        projected: list[KVCacheGroupSpec] = []
        for group in global_groups:
            layer_names = [
                layer_name
                for layer_name in group.layer_names
                if layer_name in worker_specs
            ]
            group_spec = group.kv_cache_spec
            if layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
                group_spec = UniformTypeKVCacheSpecs(
                    block_size=group_spec.block_size,
                    kv_cache_specs={
                        layer_name: group_spec.kv_cache_specs[layer_name]
                        for layer_name in layer_names
                    },
                )
            projected.append(
                KVCacheGroupSpec(
                    layer_names,
                    group_spec,
                    is_eagle_group=group.is_eagle_group and bool(layer_names),
                )
            )

        assert sum(len(group.layer_names) for group in projected) == len(worker_specs), (
            "Some DeepSeek V4 KV cache layers are not assigned to any group."
        )
        return projected

    def _apply_num_blocks_override(
        self,
        worker_groups: list[list[KVCacheGroupSpec]],
        available_memory: list[int],
    ) -> list[int]:
        """Translate num block override into the equivalent memory budget."""

        override = self.cache_config.num_gpu_blocks_override
        if override is None:
            return available_memory

        adjusted_memory: list[int] = []
        for groups, memory in zip(worker_groups, available_memory):
            if not groups:
                adjusted_memory.append(memory)
                continue
            bytes_per_block = self._pool_bytes_per_block(groups)
            logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                memory // bytes_per_block,
                override,
            )
            adjusted_memory.append(override * bytes_per_block)
        return adjusted_memory

    def _maybe_auto_fit_max_model_len(
        self,
        worker_groups: list[list[KVCacheGroupSpec]],
        available_memory: list[int],
    ) -> None:
        """Auto-fit max_model_len from per-worker DeepSeek V4 capacity."""

        original_max_len = self.vllm_config.model_config.max_model_len
        if self.vllm_config.model_config.original_max_model_len != -1:
            return
        if all(not groups for groups in worker_groups):
            logger.info_once(
                "Auto-fit max_model_len: attention-free model, "
                "using derived max_model_len=%d",
                original_max_len,
            )
            return

        capacity = original_max_len
        limiting_memory = available_memory[0]
        for groups, memory in zip(worker_groups, available_memory):
            if not groups:
                continue
            worker_capacity = self.get_max_model_len_capacity(groups, memory)
            if worker_capacity < capacity:
                capacity = worker_capacity
                limiting_memory = memory

        if capacity <= 0:
            raise ValueError(
                "Cannot auto-fit max_model_len: not enough GPU memory available "
                "to serve even a single token. Try increasing "
                "`gpu_memory_utilization`."
            )
        if capacity < original_max_len:
            self.vllm_config.model_config.max_model_len = capacity
            logger.info_once(
                "Auto-fit max_model_len: reduced from %d to %d to fit in "
                "available GPU memory (%s GiB available for KV cache)",
                original_max_len,
                capacity,
                format_gib(limiting_memory),
            )
        else:
            logger.info_once(
                "Auto-fit max_model_len: full model context length %d fits in "
                "available GPU memory",
                original_max_len,
            )

    def _check_model_len_capacity(
        self, kv_cache_groups: list[KVCacheGroupSpec], available_memory: int
    ) -> None:
        """Raise if current max_model_len exceeds DeepSeek V4 KV capacity."""

        if available_memory <= 0:
            raise ValueError(
                "No available memory for the cache blocks. Try increasing "
                "`gpu_memory_utilization` when initializing the engine."
            )

        needed_memory = self._max_memory_usage_bytes(kv_cache_groups)
        if needed_memory <= available_memory:
            return

        capacity = self.get_max_model_len_capacity(kv_cache_groups, available_memory)
        estimated_msg = ""
        if capacity > 0:
            estimated_msg = (
                "Based on the available memory, the estimated maximum model "
                f"length is {capacity}. "
            )
        raise ValueError(
            "To serve at least one request with the model's max seq len "
            f"({self.vllm_config.model_config.max_model_len}), "
            f"({format_gib(needed_memory)} GiB KV cache is needed, which is "
            "larger than the available KV cache memory "
            f"({format_gib(available_memory)} GiB). {estimated_msg}"
            "Try increasing `gpu_memory_utilization` or decreasing "
            "`max_model_len` when initializing the engine."
        )

    def _build_config_from_groups(
        self, kv_cache_groups: list[KVCacheGroupSpec], available_memory: int
    ) -> KVCacheConfig:
        """Create DeepSeek V4 KV cache tensors for one worker."""

        if not kv_cache_groups:
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
            )

        full_mla_spec = kv_cache_groups[0].kv_cache_spec
        assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
        page_sizes = sorted(full_mla_spec.get_page_sizes())
        bytes_per_block = self._pool_bytes_per_block(kv_cache_groups)
        num_blocks = max(available_memory // bytes_per_block, 0)
        num_blocks = self._may_override_num_blocks(num_blocks)

        bucketed_groups: list[dict[int, list[str]]] = []
        for group in kv_cache_groups:
            assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            specs = group.kv_cache_spec.kv_cache_specs
            buckets: dict[int, list[str]] = defaultdict(list)
            for name in group.layer_names:
                buckets[specs[name].page_size_bytes].append(name)
            bucketed_groups.append(buckets)

        num_layer_tuples = max(
            len(layers)
            for buckets in bucketed_groups
            for layers in buckets.values()
        )
        kv_cache_tensors: list[KVCacheTensor] = []
        for tuple_idx in range(num_layer_tuples):
            for page_size in page_sizes:
                shared_by: list[str] = []
                for buckets in bucketed_groups:
                    bucket = buckets.get(page_size)
                    if bucket is not None and tuple_idx < len(bucket):
                        shared_by.append(bucket[tuple_idx])
                kv_cache_tensors.append(
                    KVCacheTensor(
                        size=page_size * num_blocks,
                        shared_by=shared_by,
                    )
                )

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

    def _shrink_to_min_num_blocks(
        self, kv_cache_configs: list[KVCacheConfig]
    ) -> None:
        """Use one num_blocks value across workers."""

        min_num_blocks = min(config.num_blocks for config in kv_cache_configs)
        for config in kv_cache_configs:
            old_num_blocks = config.num_blocks
            config.num_blocks = min_num_blocks
            for tensor in config.kv_cache_tensors:
                assert tensor.size % old_num_blocks == 0
                tensor.size = tensor.size // old_num_blocks * min_num_blocks

    def _may_override_num_blocks(self, num_blocks: int) -> int:
        override = self.cache_config.num_gpu_blocks_override
        return override if override is not None else num_blocks

    def _pool_bytes_per_block(self, kv_cache_groups: list[KVCacheGroupSpec]) -> int:
        """Bytes consumed by one DeepSeek V4 shared KV block."""

        full_mla_spec = kv_cache_groups[0].kv_cache_spec
        assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
        layer_tuple_page_bytes = sum(full_mla_spec.get_page_sizes())
        num_layer_tuples = max(
            cast_group.kv_cache_spec.get_num_layer_tuples()
            for cast_group in kv_cache_groups
            if isinstance(cast_group.kv_cache_spec, UniformTypeKVCacheSpecs)
        )
        return layer_tuple_page_bytes * num_layer_tuples

    def _max_memory_usage_bytes(
        self, kv_cache_groups: list[KVCacheGroupSpec]
    ) -> int:
        """Return bytes needed to hold one max_model_len request."""

        if not kv_cache_groups:
            return 0
        full_mla_spec = kv_cache_groups[0].kv_cache_spec
        assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
        layer_tuple_bytes = sum(full_mla_spec.get_page_sizes())
        num_layer_tuples = max(
            group.kv_cache_spec.get_num_layer_tuples()
            for group in kv_cache_groups
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        )

        total = 0
        for group in kv_cache_groups:
            if not group.layer_names:
                continue
            group_spec = group.kv_cache_spec
            assert isinstance(group_spec, UniformTypeKVCacheSpecs)
            pages = group_spec.max_memory_usage_pages(self.vllm_config)
            total += num_layer_tuples * pages * layer_tuple_bytes
        return total

    def _max_concurrency(self, kv_cache_config: KVCacheConfig) -> float:
        bytes_per_block = self._pool_bytes_per_block(kv_cache_config.kv_cache_groups)
        blocks_per_request = cdiv(
            self._max_memory_usage_bytes(kv_cache_config.kv_cache_groups),
            bytes_per_block,
        )
        return kv_cache_config.num_blocks / blocks_per_request

    def _report_config(self, kv_cache_config: KVCacheConfig) -> None:
        max_model_len = self.vllm_config.model_config.max_model_len
        max_concurrency = self._max_concurrency(kv_cache_config)
        logger.info_once(
            "GPU KV cache size: %s tokens",
            f"{int(max_concurrency * max_model_len):,}",
        )
        logger.info_once(
            "Maximum concurrency for %s tokens per request: %.2fx",
            f"{max_model_len:,}",
            max_concurrency,
        )

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
            else:
                raise AssertionError(
                    "Unsupported DeepSeek V4 sliding-window MLA cache name: "
                    f"{layer_name!r}"
                )
            spec_infos.append(self._create_info(layer_name, cache_type, spec))
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
                    swa_block_size = _infer_block_size_against_buckets(
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
                        lambda block_size, bucket, _: (
                            -block_size,
                            bucket,
                        )
                    )
                inferred[spec_info.layer_name] = _infer_block_size_against_buckets(
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
        return res

    def _group_and_unify_specs(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> list[UniformTypeKVCacheSpecs]:
        """Group finalized DeepSeek V4 specs into uniform-type spec groups."""

        mla_specs: dict[str, KVCacheSpec] = {}
        grouped_swa_mla_specs: dict[tuple[int, int], dict[str, KVCacheSpec]] = (
            defaultdict(dict)
        )
        # Group SWA layers by (block_size, sliding_window), separating SWA,
        # C4I+C4A, and C128A layers.
        for name, spec in kv_cache_specs.items():
            if isinstance(spec, SlidingWindowMLASpec):
                grouped_swa_mla_specs[(spec.block_size, spec.sliding_window)][name] = (
                    spec
                )
            elif isinstance(spec, MLAAttentionSpec):
                mla_specs[name] = spec

        assert len(mla_specs) > 0
        mla_uniform_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)
        assert mla_uniform_spec is not None

        swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
        for spec_dict in grouped_swa_mla_specs.values():
            uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
            assert uniform_spec is not None
            swa_uniform_specs.append(uniform_spec)

        return [mla_uniform_spec, *swa_uniform_specs]

    def _get_kv_cache_groups_from_uniform_groups(
        self, grouped_specs: list[UniformTypeKVCacheSpecs]
    ) -> list[KVCacheGroupSpec]:
        """Generate DeepSeek V4 KV cache groups from uniform-type specs."""

        assert len(grouped_specs) > 0 and all(
            isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs
        )
        # The first group is the full MLA group whose page sizes define the
        # canonical buckets for the auxiliary cache groups.
        full_mla_spec = grouped_specs[0]
        assert all(
            isinstance(spec, MLAAttentionSpec)
            for spec in full_mla_spec.kv_cache_specs.values()
        )
        full_mla_group = KVCacheGroupSpec(
            layer_names=list(full_mla_spec.kv_cache_specs.keys()),
            kv_cache_spec=full_mla_spec,
        )

        num_layer_tuples_per_group: list[int] = [
            g_spec.get_num_layer_tuples() for g_spec in grouped_specs
        ]
        num_layer_tuples = _approximate_gcd(
            num_layer_tuples_per_group,
            lower_bound=num_layer_tuples_per_group[0],
        )

        swa_mla_specs = grouped_specs[1:]
        assert all(
            isinstance(spec, SlidingWindowMLASpec)
            for group in swa_mla_specs
            for spec in group.kv_cache_specs.values()
        )

        all_page_sizes = full_mla_spec.get_page_sizes()
        swa_mla_groups = []
        for sm_spec in swa_mla_specs:
            sm_page_sizes = sm_spec.get_page_sizes()
            layers_per_size: dict[int, list[str]] = defaultdict(list)
            assert max(sm_page_sizes) <= max(all_page_sizes)

            for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
                current_size = layer_spec.page_size_bytes
                assert current_size in all_page_sizes, (
                    f"DeepSeek V4 KV cache layer {layer_name} has page_size "
                    f"{current_size}, which is not in canonical MLA buckets "
                    f"{all_page_sizes}."
                )
                layers_per_size[current_size].append(layer_name)

            assert len(set(len(layers) for layers in layers_per_size.values())) == 1
            num_layers_per_size = len(next(iter(layers_per_size.values())))

            num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
            layer_tuples = list(zip(*layers_per_size.values()))
            for i in range(num_tuple_groups):
                group_layer_tuples = layer_tuples[i::num_tuple_groups]
                group_layer_names = [
                    name for layer_tuple in group_layer_tuples for name in layer_tuple
                ]
                group_layer_specs = {
                    name: sm_spec.kv_cache_specs[name] for name in group_layer_names
                }
                sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
                assert sub_sm_spec is not None
                swa_mla_groups.append(
                    KVCacheGroupSpec(
                        layer_names=group_layer_names,
                        kv_cache_spec=sub_sm_spec,
                    )
                )

        return [full_mla_group, *swa_mla_groups]

    def _annotate_eagle_groups(
        self,
        kv_cache_specs: dict[str, KVCacheSpec],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        """Mark the DeepSeek V4 EAGLE/MTP group when speculative decoding uses it."""

        spec_config = self.vllm_config.speculative_config
        if spec_config is None or not spec_config.use_eagle():
            return
        # DeepSeek V4's MTP attention layer is always the last layer.
        # FIXME(yifan): avoid/generalize this hacky check.
        last_layer = next(reversed(kv_cache_specs))
        for group in kv_cache_groups:
            if last_layer in group.layer_names:
                group.is_eagle_group = True
                break
