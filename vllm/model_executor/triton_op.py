# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class TritonOpBase:
    """
    Base class for Triton custom ops.
    Dispatches the forward method to the appropriate hardware backend.
    Similar to CustomOp but specifically for Triton-based operations.
    """

    def __new__(cls, *args, **kwargs):
        try:
            op_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@TritonOpBase.register, or it's the TritonOpBase base class "
                f"itself."
            ) from None

        if op_name not in cls.op_registry_oot:
            op_cls_to_instantiate = cls
        else:
            op_cls_to_instantiate = cls.op_registry_oot[op_name]
            logger.debug(
                "Instantiating triton op: %s using %s",
                op_name,
                str(op_cls_to_instantiate),
            )
        return super().__new__(op_cls_to_instantiate)

    def __init__(self):
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        # By default, we assume that XPU ops are compatible with CUDA ops.
        # NOTE: This is a placeholder for future extensions.
        return self.forward_cuda(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with CUDA ops.
        # NOTE: This is a placeholder for future extensions.
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with CUDA ops.
        # NOTE: This is a placeholder for future extensions.
        return self.forward_cuda(*args, **kwargs)

    def forward_oot(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with CUDA ops.
        # NOTE: This is a placeholder for future extensions.
        return self.forward_cuda(*args, **kwargs)

    def dispatch_forward(self):
        enabled = self.enabled()
        if not enabled:
            raise RuntimeError("Triton is not available.")

        if current_platform.is_rocm():
            return self.forward_hip
        elif current_platform.is_cpu():
            return self.forward_cpu
        elif current_platform.is_tpu():
            return self.forward_tpu
        elif current_platform.is_xpu():
            return self.forward_xpu
        elif current_platform.is_out_of_tree():
            return self.forward_oot
        else:
            return self.forward_cuda

    @classmethod
    def enabled(cls) -> bool:
        """Returns True if Triton is available."""
        # Import here to avoid circular imports
        from vllm.triton_utils import HAS_TRITON

        return HAS_TRITON

    # Dictionary of all triton ops (classes, indexed by registered name).
    op_registry: dict[str, type["TritonOpBase"]] = {}
    op_registry_oot: dict[str, type["TritonOpBase"]] = {}

    # Decorator to register triton ops.
    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls

        return decorator

    # Decorator to register out-of-tree(oot) triton ops.
    # Example:
    # - @TritonOp.register_oot
    #   class OOTTritonOp(TritonOp)
    # or
    # - TritonOpBase.register_oot(name="TritonOp")
    @classmethod
    def register_oot(cls, _decorated_op_cls=None, name: str | None = None):
        def decorator(op_cls):
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in cls.op_registry_oot, f"Duplicate op name: {reg_name}"
            op_cls.name = reg_name
            cls.op_registry_oot[reg_name] = op_cls
            return op_cls

        if _decorated_op_cls is None:
            # Called with parentheses: @TritonOpBase.register_oot()
            # or @TritonOpBase.register_oot(name="...")
            return decorator
        elif isinstance(_decorated_op_cls, type):
            # Called without parentheses: @TritonOp.register_oot
            return decorator(_decorated_op_cls)
        else:
            raise TypeError("Decorator can only be applied to classes.")
