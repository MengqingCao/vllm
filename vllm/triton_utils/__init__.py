# SPDX-License-Identifier: Apache-2.0

from vllm.triton_utils.importing import HAS_TRITON, optional_import_triton, optional_import_triton_language

triton = optional_import_triton()
tl = optional_import_triton_language()

__all__ = [
    "HAS_TRITON", "triton", "tl"
]
