"""Device-backend compatibility layer.

Everything in this subpackage changes the *runtime* (``torch``), not Megatron.
Keeping it separate makes the split obvious: ``backends`` makes MUSA look like
CUDA, ``patches`` fixes Megatron's NVIDIA-shaped assumptions.
"""

from __future__ import annotations

from . import torch_cuda

__all__ = ["torch_cuda", "musa_available"]


def musa_available() -> bool:
    """Query the current runtime without importing torchada or caching state."""
    import torch

    musa = getattr(torch, "musa", None)
    available = getattr(musa, "is_available", None)
    return callable(available) and bool(available())
