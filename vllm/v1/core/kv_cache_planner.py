# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)


class KVCachePlanner(ABC):
    """Plan model-specific KV cache layouts.

    Most models use ``ModelKVCachePlanner``. A model should declare a custom
    planner only when its KV cache layout cannot be expressed by the default
    grouping and tensor allocation logic. The planner owns the model-specific
    path end-to-end: spec normalization, grouping, capacity checks, and KV
    tensor planning.
    """

    def __init__(self, vllm_config: VllmConfig):
        """Create a planner bound to one engine configuration."""

        self.vllm_config = vllm_config
        self.cache_config = vllm_config.cache_config

    @abstractmethod
    def get_kv_cache_configs(
        self,
        kv_cache_specs: list[dict[str, KVCacheSpec]],
        available_memory: list[int],
    ) -> list[KVCacheConfig]:
        """Return per-worker KV cache configs.

        This is the main planner entry point used by engine initialization
        after workers have reported their KV cache specs and available KV
        memory. Implementations should return one ``KVCacheConfig`` for each
        worker, in the same order as ``kv_cache_specs`` and
        ``available_memory``.

        Args:
            kv_cache_specs: Per-worker mappings from layer name to KV cache
                spec. Pipeline-parallel workers may own different layer names.
            available_memory: Per-worker memory budgets, in bytes, available
                for KV cache allocation.

        Returns:
            Per-worker KV cache configs used by workers and the scheduler.
        """
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_groups(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> list[KVCacheGroupSpec]:
        """Return planned KV cache groups for one worker spec map.

        This entry point is used by worker-side profiling paths that need a
        temporary/minimal KV cache before the final per-worker memory budgets
        are known. Implementations should apply the same model-specific spec
        normalization and grouping rules used by ``get_kv_cache_configs``.

        Args:
            kv_cache_specs: Mapping from local layer name to KV cache spec.

        Returns:
            KV cache groups for the supplied spec map.
        """
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_config_from_groups(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        """Return one KV cache config from already-planned groups.

        This entry point is paired with ``get_kv_cache_groups`` for
        profiling/minimal-cache initialization. It must use the same tensor
        layout rules as the final configs produced by ``get_kv_cache_configs``.

        Args:
            kv_cache_groups: KV cache groups produced by this planner.
            available_memory: Memory budget in bytes. Callers may temporarily
                set ``num_gpu_blocks_override`` to request a small profiling
                cache; implementations should honor the override consistently
                with final planning.

        Returns:
            KV cache config for a single worker.
        """
        raise NotImplementedError

    @abstractmethod
    def get_max_model_len_capacity(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> int:
        """Return the maximum model length supported by a KV cache layout.

        This answers the capacity question for one worker: given planned KV
        cache groups and a memory budget, what is the largest
        ``model_config.max_model_len`` that can fit at least one request?
        Implementations may temporarily mutate ``model_config.max_model_len``
        while estimating capacity, but must restore it before returning.

        Args:
            kv_cache_groups: KV cache groups produced by this planner.
            available_memory: Memory budget in bytes for the worker.

        Returns:
            Maximum supported model length in tokens. Return ``0`` if even a
            one-token request cannot fit.
        """
        raise NotImplementedError
