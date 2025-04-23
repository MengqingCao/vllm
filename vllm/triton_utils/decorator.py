# SPDX-License-Identifier: Apache-2.0
"""
Decorators used to prevent invoking into triton when HAS_TRITON is False.
Nothing should be done in all dummy decorator.
"""

from typing import TypeVar

from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON

logger = init_logger(__name__)
T = TypeVar("T")

if HAS_TRITON:
    import triton
    triton_jit_decorator = triton.jit
    triton_autotune_decorator = triton.autotune
    triton_heuristics_decorator = triton.heuristics
else:
    logger.warning_once(
        "Triton is not found in current env. Decorators like @triton.jit will "
        "be replaced with dummy decorators. Run `pip install triton` to enable it."
    )

    def make_dummy_decorator(name: str):
        def wrapper(*args, **kwargs):
            def inner(fn):
                logger.warning_once(
                    f"Using dummy decorator '{name}' because Triton is not available."
                )
                return fn
            if args and callable(args[0]):
                return inner(args[0])  # @decorator
            return inner  # @decorator(...)
        return wrapper

    triton_jit_decorator = make_dummy_decorator("triton.jit")
    triton_autotune_decorator = make_dummy_decorator("triton.autotune")
    triton_heuristics_decorator = make_dummy_decorator("triton.heuristics")
