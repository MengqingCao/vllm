# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec
import types

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

HAS_TRITON = (
    find_spec("triton") is not None
    and not current_platform.is_xpu()  # Not compatible
)

if not HAS_TRITON:
    logger.info("Triton not installed or not compatible; certain GPU-related"
                " functions will not be available.")

class TritonLangPlaceholder(types.ModuleType):
    """Placeholder for `triton.language` when it's not installed."""

    def __init__(self):
        super().__init__("triton.language")
        self._warned_attrs = set()
        logger.warning_once(
            "triton.language is not installed. Please run `pip install triton` if needed."
        )

    def __getattr__(self, name):
        if name not in self._warned_attrs:
            logger.warning_once(f"Accessed triton.language.{name}, but it's not available. Returning dummy.")
            self._warned_attrs.add(name)

        def dummy_func(*args, **kwargs):
            return None

        return dummy_func


class TritonPlaceholder(types.ModuleType):
    """A placeholder module for `triton` when it's not installed."""

    def __init__(self):
        super().__init__("triton")
        self._warned_attrs = set()
        self.constexpr = None

        # Add dummy decorators
        self.jit = self._dummy_decorator("jit")
        self.autotune = self._dummy_decorator("autotune")
        self.heuristics = self._dummy_decorator("heuristics")

        # Add TritonLangPlaceholder as submodule for triton.language
        self.language = TritonLangPlaceholder()

        logger.warning_once(
            "Triton is not installed. Using dummy decorators. "
            "Install it via `pip install triton` to enable kernel compilation."
        )

    def _dummy_decorator(self, name):
        def decorator(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func
        return decorator

    def __getattr__(self, name):
        if name not in self._warned_attrs:
            logger.warning_once(f"Accessed triton.{name}, but triton is not installed. Returning dummy object.")
            self._warned_attrs.add(name)

        # Return dummy function or object
        def dummy_func(*args, **kwargs):
            return None

        return dummy_func


def optional_import_triton():
    try:
        import triton
    except ImportError:
        triton = TritonPlaceholder()

    return triton

def optional_import_triton_language():
    try:
        import triton.language as tl
    except ImportError:
        tl = TritonLangPlaceholder()

    return tl
