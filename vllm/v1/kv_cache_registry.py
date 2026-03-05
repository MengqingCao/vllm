# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Registry for KVCacheSpec types and their associated managers.

This module provides a pluggable architecture for registering custom KVCacheSpec
subclasses without modifying vLLM core code. Out-of-tree platforms can define
custom specs and managers by using the @register_kv_cache_spec decorator.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from vllm.v1.core.single_type_kv_cache_manager import (
        SingleTypeKVCacheManager)
    from vllm.v1.kv_cache_interface import KVCacheSpec


@dataclass
class KVCacheSpecMetadata:
    """Metadata for a registered KVCacheSpec."""

    spec_class: Type["KVCacheSpec"]
    manager_class: Type["SingleTypeKVCacheManager"]
    # The base spec class for grouping compatibility checks.
    # Custom specs that inherit from FullAttentionSpec should set this to
    # FullAttentionSpec so they're treated as "full attention" for grouping.
    grouping_base_class: Type["KVCacheSpec"]


class KVCacheSpecRegistry:
    """Global registry for KVCacheSpec types and their associated managers."""

    _registry: Dict[Type["KVCacheSpec"], KVCacheSpecMetadata] = {}

    @classmethod
    def register(
        cls,
        spec_class: Type["KVCacheSpec"],
        manager_class: Type["SingleTypeKVCacheManager"],
        grouping_base_class: Type["KVCacheSpec"] | None = None,
    ) -> None:
        """
        Register a KVCacheSpec class with its manager and grouping base.

        Args:
            spec_class: The KVCacheSpec subclass to register
            manager_class: The SingleTypeKVCacheManager to use for this spec
            grouping_base_class: The base spec class for grouping compatibility.
                If None, defaults to spec_class itself (for built-in base specs).
        """
        if grouping_base_class is None:
            grouping_base_class = spec_class

        cls._registry[spec_class] = KVCacheSpecMetadata(
            spec_class=spec_class,
            manager_class=manager_class,
            grouping_base_class=grouping_base_class,
        )

    @classmethod
    def get_manager_class(
        cls, spec: "KVCacheSpec"
    ) -> Type["SingleTypeKVCacheManager"]:
        """
        Get the manager class for a given spec instance.

        Walks up the MRO to find a registered base class, so custom specs
        that inherit from registered specs automatically get the right manager.

        Args:
            spec: A KVCacheSpec instance

        Returns:
            The SingleTypeKVCacheManager class to use for this spec

        Raises:
            ValueError: If no manager is registered for this spec type
        """
        spec_class = type(spec)

        # Walk up the MRO to find a registered base class
        for base in spec_class.__mro__:
            if base in cls._registry:
                return cls._registry[base].manager_class

        raise ValueError(
            f"No manager registered for spec type {spec_class}. "
            f"Please register it using KVCacheSpecRegistry.register() or "
            f"the @register_kv_cache_spec decorator."
        )

    @classmethod
    def get_grouping_base_class(
        cls, spec: "KVCacheSpec"
    ) -> Type["KVCacheSpec"]:
        """
        Get the base spec class for grouping compatibility checks.

        For example, a custom spec that inherits from FullAttentionSpec
        will return FullAttentionSpec, so it's grouped with other full
        attention layers.

        Args:
            spec: A KVCacheSpec instance

        Returns:
            The base KVCacheSpec class for grouping

        Raises:
            ValueError: If no grouping base class is registered for this spec
        """
        spec_class = type(spec)

        # Walk up the MRO to find a registered base class
        for base in spec_class.__mro__:
            if base in cls._registry:
                return cls._registry[base].grouping_base_class

        raise ValueError(
            f"No grouping base class registered for spec type {spec_class}."
        )


def register_kv_cache_spec(
    manager_class: Type["SingleTypeKVCacheManager"],
    grouping_base_class: Type["KVCacheSpec"] | None = None,
):
    """
    Decorator to register a custom KVCacheSpec class.

    Usage:
        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            grouping_base_class=FullAttentionSpec
        )
        @dataclass(frozen=True, kw_only=True)
        class MyCustomFullAttentionSpec(FullAttentionSpec):
            custom_alignment: int = 64

            @property
            def page_size_bytes(self) -> int:
                # Custom page size calculation with alignment
                base_size = super().page_size_bytes
                return ((base_size + self.custom_alignment - 1)
                        // self.custom_alignment * self.custom_alignment)

    Args:
        manager_class: The SingleTypeKVCacheManager to use for this spec
        grouping_base_class: The base spec class for grouping compatibility.
            If None, the spec is treated as a new base type.
    """

    def decorator(spec_class: Type["KVCacheSpec"]) -> Type["KVCacheSpec"]:
        KVCacheSpecRegistry.register(
            spec_class=spec_class,
            manager_class=manager_class,
            grouping_base_class=grouping_base_class,
        )
        return spec_class

    return decorator
