# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test cases for the pluggable KVCacheSpec architecture.

Tests cover:
1. Registry registration and lookup
2. Custom specs with existing managers
3. Custom specs with custom managers
4. Grouping logic with custom specs
5. Backward compatibility with built-in specs
"""

import torch
import pytest
from dataclasses import dataclass

from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_registry import (
    KVCacheSpecRegistry,
    register_kv_cache_spec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_spec(**kwargs) -> FullAttentionSpec:
    defaults = dict(block_size=16, num_kv_heads=8, head_size=128,
                    dtype=torch.bfloat16)
    defaults.update(kwargs)
    return FullAttentionSpec(**defaults)


def _sw_spec(**kwargs) -> SlidingWindowSpec:
    defaults = dict(block_size=16, num_kv_heads=8, head_size=128,
                    dtype=torch.bfloat16, sliding_window=4096)
    defaults.update(kwargs)
    return SlidingWindowSpec(**defaults)


@dataclass(frozen=True)
class _TrulyUnregisteredSpec(KVCacheSpec):
    """
    A spec that inherits directly from KVCacheSpec with no registered
    ancestor in the MRO.  Used to test that the registry correctly raises
    when no entry can be found.
    """

    @property
    def page_size_bytes(self) -> int:
        return self.block_size * 128

    def max_memory_usage_bytes(self, _) -> int:
        return 0


# ---------------------------------------------------------------------------
# 1. Core Registry Functionality
# ---------------------------------------------------------------------------

class TestKVCacheSpecRegistry:
    """Test the core registry functionality."""

    def test_builtin_full_attention_registered(self):
        """FullAttentionSpec maps to FullAttentionManager."""
        assert (
            KVCacheSpecRegistry.get_manager_class(_full_spec())
            is FullAttentionManager
        )

    def test_builtin_sliding_window_registered(self):
        """SlidingWindowSpec maps to SlidingWindowManager."""
        assert (
            KVCacheSpecRegistry.get_manager_class(_sw_spec())
            is SlidingWindowManager
        )

    def test_custom_spec_registration(self):
        """A decorated custom spec resolves to the declared manager."""

        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            grouping_base_class=FullAttentionSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class _CustomFullSpec(FullAttentionSpec):
            custom_param: int = 42

        spec = _CustomFullSpec(block_size=16, num_kv_heads=8, head_size=128,
                               dtype=torch.bfloat16, custom_param=100)

        assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
        assert (
            KVCacheSpecRegistry.get_grouping_base_class(spec)
            is FullAttentionSpec
        )

    def test_unregistered_spec_no_registered_parent_raises(self):
        """
        A spec whose entire MRO contains no registered class raises ValueError.
        Subclasses of registered specs intentionally *do not* raise — they
        inherit their parent's manager via MRO walking.
        """
        spec = _TrulyUnregisteredSpec(block_size=16)

        with pytest.raises(ValueError, match="No manager registered"):
            KVCacheSpecRegistry.get_manager_class(spec)

        with pytest.raises(ValueError, match="No grouping base class"):
            KVCacheSpecRegistry.get_grouping_base_class(spec)

    def test_unregistered_subclass_inherits_parent_manager(self):
        """
        An unregistered subclass of a registered spec resolves via MRO
        to its parent's manager — this is intentional registry behaviour.
        """

        @dataclass(frozen=True, kw_only=True)
        class _ImplicitlyInheritedSpec(FullAttentionSpec):
            pass

        spec = _ImplicitlyInheritedSpec(block_size=16, num_kv_heads=8,
                                        head_size=128, dtype=torch.bfloat16)

        # MRO walk finds FullAttentionSpec → FullAttentionManager
        assert (
            KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
        )


# ---------------------------------------------------------------------------
# 2. Custom Specs Reusing Existing Managers
# ---------------------------------------------------------------------------

class TestCustomSpecWithExistingManager:
    """Custom specs that only change memory characteristics."""

    def test_hardware_aligned_spec_page_size(self):
        """page_size_bytes is rounded up to the declared alignment."""

        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            grouping_base_class=FullAttentionSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class _HardwareAlignedSpec(FullAttentionSpec):
            page_alignment: int = 4096

            @property
            def page_size_bytes(self) -> int:
                base = super().page_size_bytes
                return (
                    (base + self.page_alignment - 1)
                    // self.page_alignment
                    * self.page_alignment
                )

        spec = _HardwareAlignedSpec(block_size=16, num_kv_heads=8,
                                    head_size=128, dtype=torch.bfloat16,
                                    page_alignment=4096)

        assert spec.page_size_bytes % 4096 == 0
        assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager

    def test_compressed_cache_spec_page_size(self):
        """page_size_bytes is halved when compression is active."""

        @register_kv_cache_spec(
            manager_class=SlidingWindowManager,
            grouping_base_class=SlidingWindowSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class _CompressedSlidingSpec(SlidingWindowSpec):
            compression_ratio: int = 2

            @property
            def page_size_bytes(self) -> int:
                return super().page_size_bytes // self.compression_ratio

        spec = _CompressedSlidingSpec(block_size=16, num_kv_heads=8,
                                      head_size=128, dtype=torch.bfloat16,
                                      sliding_window=4096, compression_ratio=2)
        base = _sw_spec()

        assert spec.page_size_bytes == base.page_size_bytes // 2
        assert (
            KVCacheSpecRegistry.get_manager_class(spec) is SlidingWindowManager
        )


# ---------------------------------------------------------------------------
# 3. Custom Specs with Custom Managers
# ---------------------------------------------------------------------------

class TestCustomSpecWithCustomManager:
    """Custom specs backed by a brand-new SingleTypeKVCacheManager."""

    def test_custom_manager_is_returned(self):
        """Registering a custom manager means the registry returns it."""

        class _CustomManager(SingleTypeKVCacheManager):
            pass

        @register_kv_cache_spec(
            manager_class=_CustomManager,
            grouping_base_class=None,  # new base type, incompatible with existing
        )
        @dataclass(frozen=True, kw_only=True)
        class _CustomSpec(FullAttentionSpec):
            custom_feature: bool = True

        spec = _CustomSpec(block_size=16, num_kv_heads=8, head_size=128,
                           dtype=torch.bfloat16, custom_feature=True)

        assert KVCacheSpecRegistry.get_manager_class(spec) is _CustomManager
        # grouping_base_class=None → registry defaults to spec_class itself
        assert (
            KVCacheSpecRegistry.get_grouping_base_class(spec) is _CustomSpec
        )


# ---------------------------------------------------------------------------
# 4. Grouping Logic with Custom Specs
# ---------------------------------------------------------------------------

class TestGroupingWithCustomSpecs:
    """Grouping decisions respect registered grouping_base_class."""

    def test_custom_spec_groups_with_its_base(self):
        """
        A mix of FullAttentionSpec and a custom subclass with
        grouping_base_class=FullAttentionSpec is treated as uniform type.
        """

        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            grouping_base_class=FullAttentionSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class _CustomFullSpec2(FullAttentionSpec):
            tag: int = 0

        specs = {
            "layer.0": _full_spec(),
            "layer.1": _CustomFullSpec2(block_size=16, num_kv_heads=8,
                                        head_size=128, dtype=torch.bfloat16,
                                        tag=1),
            "layer.2": _full_spec(),
        }

        assert UniformTypeKVCacheSpecs.is_uniform_type(specs)

    def test_different_base_types_are_not_uniform(self):
        """Custom specs with different grouping_base_class values are not uniform."""

        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            grouping_base_class=FullAttentionSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class _CFA(FullAttentionSpec):
            pass

        @register_kv_cache_spec(
            manager_class=SlidingWindowManager,
            grouping_base_class=SlidingWindowSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class _CSW(SlidingWindowSpec):
            pass

        specs = {
            "layer.0": _CFA(block_size=16, num_kv_heads=8, head_size=128,
                             dtype=torch.bfloat16),
            "layer.1": _CSW(block_size=16, num_kv_heads=8, head_size=128,
                             dtype=torch.bfloat16, sliding_window=4096),
        }

        assert not UniformTypeKVCacheSpecs.is_uniform_type(specs)

    def test_truly_unregistered_spec_in_grouping_raises(self):
        """
        A spec with no registered ancestor raises NotImplementedError
        inside is_uniform_type, with a helpful message.
        """
        specs = {
            "layer.0": _full_spec(),
            "layer.1": _TrulyUnregisteredSpec(block_size=16),
        }

        with pytest.raises(
            NotImplementedError,
            match="Please register it using @register_kv_cache_spec",
        ):
            UniformTypeKVCacheSpecs.is_uniform_type(specs)


# ---------------------------------------------------------------------------
# 5. Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Existing code must continue to work without any changes."""

    def test_builtin_specs_resolve_managers(self):
        assert (
            KVCacheSpecRegistry.get_manager_class(_full_spec())
            is FullAttentionManager
        )
        assert (
            KVCacheSpecRegistry.get_manager_class(_sw_spec())
            is SlidingWindowManager
        )

    def test_all_full_attention_is_uniform(self):
        specs = {"layer.0": _full_spec(), "layer.1": _full_spec()}
        assert UniformTypeKVCacheSpecs.is_uniform_type(specs)

    def test_mixed_builtin_types_are_not_uniform(self):
        specs = {"layer.0": _full_spec(), "layer.1": _sw_spec()}
        assert not UniformTypeKVCacheSpecs.is_uniform_type(specs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
