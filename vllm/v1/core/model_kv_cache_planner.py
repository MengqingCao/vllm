# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default model KV cache planner."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_planner import KVCachePlanner
from vllm.v1.core.kv_cache_utils import (
    _check_enough_kv_cache_memory,
    create_kv_cache_group_specs,
    is_kv_cache_spec_uniform,
    max_memory_usage_bytes,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)


class ModelKVCachePlanner(KVCachePlanner):
    """Default KV cache planner used by models without a custom planner."""

    def get_kv_cache_configs(
        self,
        kv_cache_specs: list[dict[str, KVCacheSpec]],
        available_memory: list[int],
    ) -> list[KVCacheConfig]:
        """
        Generates the KV cache configurations for a model.

        Since we use a shared centralized controller for all workers, we need
        the `kv_cache_config` to be consistent across all workers to make sure
        the KV cache allocation can be applied to all workers.
        """

        merged_specs = self._merge_worker_specs(kv_cache_specs)
        global_groups = self.get_kv_cache_groups(merged_specs)
        worker_groups = [
            self._project_groups_to_worker(global_groups, worker_spec)
            for worker_spec in kv_cache_specs
        ]

        available_memory = self._apply_num_blocks_override(
            worker_groups, available_memory
        )

        if self.vllm_config.model_config.original_max_model_len == -1:
            self._auto_fit_max_model_len(worker_groups, available_memory)

        for groups, memory in zip(worker_groups, available_memory):
            if groups:
                self._check_model_len_capacity(groups, memory)

        kv_cache_configs: list[KVCacheConfig] = []
        for groups, worker_spec, memory in zip(
            worker_groups, kv_cache_specs, available_memory
        ):
            assert sum(len(group.layer_names) for group in groups) == len(
                worker_spec
            ), "Some layers are not assigned to any group."
            kv_cache_configs.append(
                self.get_kv_cache_config_from_groups(groups, memory)
            )

        self._shrink_to_min_num_blocks(kv_cache_configs)
        for kv_cache_config in kv_cache_configs:
            if kv_cache_config.kv_cache_groups:
                self._report_config(kv_cache_config)
        return kv_cache_configs

    def get_kv_cache_groups(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> list[KVCacheGroupSpec]:
        """Split the layers in the model into KV cache groups."""

        if self.vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            self._unify_hybrid_kv_cache_specs(kv_cache_specs)

        if self._is_kv_cache_type_attention_free(kv_cache_specs):
            return []

        if is_kv_cache_spec_uniform(kv_cache_specs):
            return create_kv_cache_group_specs(
                kv_cache_specs, [list(kv_cache_specs.keys())]
            )
        if uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_specs):
            return [
                KVCacheGroupSpec(
                    list(uniform_spec.kv_cache_specs.keys()), uniform_spec
                )
            ]

        hidden_specs = {
            k: v
            for k, v in kv_cache_specs.items()
            if isinstance(v, HiddenStateCacheSpec)
        }
        filtered_specs = {
            k: v
            for k, v in kv_cache_specs.items()
            if not isinstance(v, HiddenStateCacheSpec)
        }

        filtered_specs = self._unify_kv_cache_spec_page_size(filtered_specs)
        groups = self._get_kv_cache_groups_uniform_page_size(filtered_specs)

        if hidden_specs:
            common_page = self._get_uniform_page_size(
                [group.kv_cache_spec for group in groups]
            )
            for name, spec in hidden_specs.items():
                per_token = (
                    spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
                )
                new_block_size = max(common_page // per_token, 1)
                aligned = replace(
                    spec,
                    block_size=new_block_size,
                    page_size_padded=common_page,
                )
                groups.append(KVCacheGroupSpec([name], aligned))

        return groups

    def get_kv_cache_config_from_groups(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        """Generate one KV cache config from planned groups."""

        if not kv_cache_groups:
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
            )

        if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            num_blocks = (
                available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
            )
            num_blocks = self._may_override_num_blocks(num_blocks)
            per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
            kv_cache_tensors = [
                KVCacheTensor(
                    size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                    shared_by=[layer_name],
                )
                for layer_name in kv_cache_groups[0].layer_names
            ]
        else:
            group_size = max(len(group.layer_names) for group in kv_cache_groups)
            page_size = self._get_uniform_page_size(
                [group.kv_cache_spec for group in kv_cache_groups]
            )
            assert group_size > 0, "group_size must be greater than 0"
            num_blocks = self._get_num_blocks(
                group_size, available_memory, page_size
            )
            kv_cache_tensors = []
            for i in range(group_size):
                shared_by = []
                for group in kv_cache_groups:
                    if i < len(group.layer_names):
                        shared_by.append(group.layer_names[i])
                kv_cache_tensors.append(
                    KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
                )

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

    def get_max_model_len_capacity(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
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
        """Merge per-worker specs into one global spec map."""

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
        worker_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        """Filter global groups to the layers owned by one worker."""

        projected_groups: list[KVCacheGroupSpec] = []
        for group in global_groups:
            worker_layer_names = [
                layer_name for layer_name in group.layer_names
                if layer_name in worker_spec
            ]
            group_spec = group.kv_cache_spec
            if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
                group_spec = UniformTypeKVCacheSpecs(
                    block_size=group_spec.block_size,
                    kv_cache_specs={
                        layer_name: group_spec.kv_cache_specs[layer_name]
                        for layer_name in worker_layer_names
                    },
                )
            projected_groups.append(
                KVCacheGroupSpec(
                    worker_layer_names,
                    group_spec,
                    is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
                )
            )
        return projected_groups

    def _apply_num_blocks_override(
        self,
        worker_groups: list[list[KVCacheGroupSpec]],
        available_memory: list[int],
    ) -> list[int]:
        """Adjust memory budgets when num_gpu_blocks_override is set."""

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

    def _auto_fit_max_model_len(
        self,
        worker_groups: list[list[KVCacheGroupSpec]],
        available_memory: list[int],
    ) -> None:
        """Auto-fit max_model_len to the available KV cache memory."""

        original_max_len = self.vllm_config.model_config.max_model_len
        if all(not groups for groups in worker_groups):
            logger.info_once(
                "Auto-fit max_model_len: attention-free model, "
                "using derived max_model_len=%d",
                original_max_len,
            )
            return

        auto_fit_max = original_max_len
        limiting_worker_mem = available_memory[0]
        for groups, memory in zip(worker_groups, available_memory):
            if not groups:
                continue
            worker_max = self.get_max_model_len_capacity(groups, memory)
            if worker_max < auto_fit_max:
                auto_fit_max = worker_max
                limiting_worker_mem = memory

        if auto_fit_max <= 0:
            raise ValueError(
                "Cannot auto-fit max_model_len: not enough GPU memory available "
                "to serve even a single token. Try increasing "
                "`gpu_memory_utilization`."
            )

        if auto_fit_max >= original_max_len:
            logger.info_once(
                "Auto-fit max_model_len: full model context length %d fits in "
                "available GPU memory",
                original_max_len,
            )
        else:
            self.vllm_config.model_config.max_model_len = auto_fit_max
            logger.info_once(
                "Auto-fit max_model_len: reduced from %d to %d to fit in "
                "available GPU memory (%s GiB available for KV cache)",
                original_max_len,
                auto_fit_max,
                format_gib(limiting_worker_mem),
            )

    def _check_model_len_capacity(
        self, kv_cache_groups: list[KVCacheGroupSpec], available_memory: int
    ) -> None:
        """Validate that at least one max-length request fits."""

        _check_enough_kv_cache_memory(
            available_memory,
            lambda: self._max_memory_usage_bytes(kv_cache_groups),
            self.vllm_config.model_config.max_model_len,
            lambda memory: self.get_max_model_len_capacity(
                kv_cache_groups, memory
            ),
        )

    def _max_memory_usage_bytes(
        self, kv_cache_groups: list[KVCacheGroupSpec]
    ) -> int:
        """Calculate maximum memory usage in bytes from KV cache groups."""

        if not kv_cache_groups:
            return 0

        if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
            return sum(
                spec.max_memory_usage_bytes(self.vllm_config)
                for spec in per_layer_specs.values()
            )

        group_size = max(len(group.layer_names) for group in kv_cache_groups)
        page_size = self._get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        blocks_needed = sum(
            cdiv(
                group.kv_cache_spec.max_memory_usage_bytes(self.vllm_config),
                page_size,
            )
            for group in kv_cache_groups
        )
        return group_size * page_size * blocks_needed

    def _pool_bytes_per_block(
        self, kv_cache_groups: list[KVCacheGroupSpec]
    ) -> int:
        """Return bytes consumed by one block in a worker KV cache pool."""

        if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            return kv_cache_groups[0].kv_cache_spec.page_size_bytes
        group_size = max(len(group.layer_names) for group in kv_cache_groups)
        page_size = self._get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        return page_size * group_size

    def _get_num_blocks(
        self, num_layers: int, available_memory: int, page_size: int
    ) -> int:
        """Get the number of KV cache blocks."""

        num_blocks = int(available_memory // page_size // num_layers)
        return self._may_override_num_blocks(max(num_blocks, 0))

    def _may_override_num_blocks(self, num_blocks: int) -> int:
        """Apply num_gpu_blocks_override when configured."""

        if self.cache_config.num_gpu_blocks_override is not None:
            return self.cache_config.num_gpu_blocks_override
        return num_blocks

    def _shrink_to_min_num_blocks(
        self, kv_cache_configs: list[KVCacheConfig]
    ) -> None:
        """Make all ranks use the smallest block count."""

        min_num_blocks = min(
            kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
        )
        for kv_cache_config in kv_cache_configs:
            num_blocks_old = kv_cache_config.num_blocks
            kv_cache_config.num_blocks = min_num_blocks
            for tensor in kv_cache_config.kv_cache_tensors:
                assert tensor.size % num_blocks_old == 0
                tensor.size = tensor.size // num_blocks_old * min_num_blocks

    def _report_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Log resolved KV cache configuration."""

        max_model_len = self.vllm_config.model_config.max_model_len
        max_concurrency = self._max_concurrency(kv_cache_config)
        num_tokens = int(max_concurrency * max_model_len)
        logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}")
        logger.info_once(
            "Maximum concurrency for %s tokens per request: %.2fx",
            f"{max_model_len:,}",
            max_concurrency,
        )

    def _max_concurrency(self, kv_cache_config: KVCacheConfig) -> float:
        """Get the maximum concurrency for the given KV cache configuration."""

        num_layer_per_group = max(
            len(group.layer_names) for group in kv_cache_config.kv_cache_groups
        )
        max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(
            self.vllm_config,
            (group.kv_cache_spec for group in kv_cache_config.kv_cache_groups),
        )
        memory_per_block = (
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
            * num_layer_per_group
        )
        num_block_per_request = cdiv(
            max_memory_usage_per_request, memory_per_block
        )
        return kv_cache_config.num_blocks / num_block_per_request

    def _get_kv_cache_groups_uniform_page_size(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> list[KVCacheGroupSpec]:
        """Generate groups for hybrid models with uniform page size."""

        same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
        for layer_name, layer_spec in kv_cache_specs.items():
            same_type_layers[layer_spec].append(layer_name)

        min_num_layers = min(len(layers) for layers in same_type_layers.values())
        group_size = min_num_layers
        max_num_layers = max(len(layers) for layers in same_type_layers.values())
        if max_num_layers < min_num_layers * 1.5:
            group_size = max_num_layers

        grouped_layers = []
        for layers in same_type_layers.values():
            num_padding_layers = group_size - len(layers) % group_size
            if num_padding_layers != group_size:
                logger.warning(
                    "Add %d padding layers, may waste at most %.2f%% KV cache "
                    "memory",
                    num_padding_layers,
                    num_padding_layers / len(layers) * 100,
                )
            num_groups = cdiv(len(layers), group_size)
            for i in range(num_groups):
                grouped_layers.append(layers[i::num_groups])
        return create_kv_cache_group_specs(kv_cache_specs, grouped_layers)

    def _unify_hybrid_kv_cache_specs(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> None:
        """Convert hybrid KV specs to one type when hybrid manager is disabled."""

        if (
            is_kv_cache_spec_uniform(kv_cache_specs)
            or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_specs)
        ):
            return

        logger.warning(
            "Hybrid KV cache manager is disabled for this hybrid model, "
            "This means we do not enable any optimizations for saving KV "
            "cache memory (e.g., dropping the KV cache outside the sliding "
            "window). The compute of layers like sliding window is still "
            "saved."
        )

        has_full_attention = any(
            isinstance(spec, FullAttentionSpec) for spec in kv_cache_specs.values()
        )
        has_sliding_window = any(
            isinstance(spec, SlidingWindowSpec) for spec in kv_cache_specs.values()
        )
        has_chunked_local_attention = any(
            isinstance(spec, ChunkedLocalAttentionSpec)
            for spec in kv_cache_specs.values()
        )
        has_swa_mla = any(
            isinstance(spec, SlidingWindowMLASpec)
            for spec in kv_cache_specs.values()
        )

        uniform_block_size: int | None = None
        if has_swa_mla:
            assert has_full_attention
            any_full_spec = next(
                spec
                for spec in kv_cache_specs.values()
                if isinstance(spec, FullAttentionSpec)
            )
            uniform_block_size = any_full_spec.block_size

        if has_full_attention and (
            has_sliding_window or has_chunked_local_attention
        ):
            for layer_name, spec in kv_cache_specs.items():
                if isinstance(spec, SlidingWindowMLASpec):
                    kv_cache_specs[layer_name] = MLAAttentionSpec(
                        block_size=uniform_block_size
                        if uniform_block_size is not None
                        else spec.block_size,
                        num_kv_heads=spec.num_kv_heads,
                        head_size=spec.head_size,
                        dtype=spec.dtype,
                        page_size_padded=spec.page_size_padded,
                        cache_dtype_str=spec.cache_dtype_str,
                        alignment=spec.alignment,
                        compress_ratio=spec.compress_ratio,
                        model_version=spec.model_version,
                    )
                elif isinstance(spec, SlidingWindowSpec):
                    kv_cache_specs[layer_name] = FullAttentionSpec(
                        block_size=spec.block_size,
                        num_kv_heads=spec.num_kv_heads,
                        head_size=spec.head_size,
                        head_size_v=spec.head_size_v,
                        dtype=spec.dtype,
                        kv_quant_mode=spec.kv_quant_mode,
                        sliding_window=spec.sliding_window,
                        page_size_padded=spec.page_size_padded,
                    )
                elif isinstance(spec, ChunkedLocalAttentionSpec):
                    kv_cache_specs[layer_name] = FullAttentionSpec(
                        block_size=spec.block_size,
                        num_kv_heads=spec.num_kv_heads,
                        head_size=spec.head_size,
                        dtype=spec.dtype,
                        attention_chunk_size=spec.attention_chunk_size,
                        page_size_padded=spec.page_size_padded,
                    )

        if not (
            is_kv_cache_spec_uniform(kv_cache_specs)
            or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_specs)
        ):
            raise ValueError(
                "Hybrid KV cache manager is disabled but failed to convert "
                "the KV cache specs to one unified type."
            )

    def _unify_kv_cache_spec_page_size(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> dict[str, KVCacheSpec]:
        """Unify page size by increasing block size where possible."""

        page_sizes = {layer.page_size_bytes for layer in kv_cache_specs.values()}
        if len(page_sizes) <= 1:
            return kv_cache_specs

        max_page_size = max(page_sizes)
        new_kv_cache_specs = {}
        for layer_name, layer_spec in kv_cache_specs.items():
            if layer_spec.page_size_bytes == max_page_size:
                new_kv_cache_specs[layer_name] = layer_spec
                continue
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_spec = replace(layer_spec, block_size=layer_spec.block_size * ratio)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_specs[layer_name] = new_spec
        return new_kv_cache_specs

    @staticmethod
    def _get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
        """Get the uniform page size of the KV cache specs."""

        page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
        assert len(page_sizes) == 1
        return page_sizes.pop()

    @staticmethod
    def _is_kv_cache_type_attention_free(
        kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        """Return whether the model has no KV cache."""

        return not kv_cache_specs
