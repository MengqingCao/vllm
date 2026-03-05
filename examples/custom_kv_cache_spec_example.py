# Example: Custom KVCacheSpec with Hardware-Specific Alignment
#
# This example demonstrates how to create a custom KVCacheSpec for
# out-of-tree platforms without modifying vLLM core code.

from dataclasses import dataclass

from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_registry import register_kv_cache_spec


# Example 1: Custom spec that reuses existing manager
@register_kv_cache_spec(
    manager_class=FullAttentionManager,
    grouping_base_class=FullAttentionSpec,
)
@dataclass(frozen=True, kw_only=True)
class HardwareAlignedFullAttentionSpec(FullAttentionSpec):
    """
    Custom FullAttentionSpec with hardware-specific page alignment.

    This spec is useful for platforms that require specific memory alignment
    (e.g., 4KB pages for certain accelerators).
    """

    page_alignment: int = 4096  # 4KB alignment

    @property
    def page_size_bytes(self) -> int:
        """Round up page size to alignment boundary."""
        base_size = super().page_size_bytes
        return (
            (base_size + self.page_alignment - 1)
            // self.page_alignment
            * self.page_alignment
        )


# Example 2: Using the custom spec in an attention layer
# (This would go in your custom model implementation)
"""
class MyCustomAttention(nn.Module):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return HardwareAlignedFullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            page_alignment=4096,  # Custom parameter
        )
"""

# That's it! The custom spec will automatically:
# - Be grouped with other FullAttentionSpec layers
# - Use FullAttentionManager for allocation/eviction
# - Work with prefix caching and all other vLLM features
# - No modifications to vLLM core code required!
