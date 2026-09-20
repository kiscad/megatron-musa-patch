"""Repair DCP CPU staging on MUSA.

DCP's CUDA availability probe sees the emulated API rather than the tensor's
actual MUSA device. Correct its local device selection so the existing loader
performs and synchronizes the device-to-host copy before serialization.

This touches neither the checkpoint format nor Megatron's own writer: the outer
async-save path (``async_utils.DynamicAsyncCaller``) and the bucket writer in
``filesystem_async.py`` keep upstream behaviour. Review those paths separately
on stack upgrades.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from .._engine import HookPatch

__all__ = ["PATCHES"]

_dcp_device_owned: tuple[Any, Any, Any] | None = None


def _install_dcp_device() -> bool:
    """Correct only DCP's imported device selector, after Megatron activation."""
    global _dcp_device_owned
    import importlib
    import sys

    torch = sys.modules.get("torch")
    musa = getattr(torch, "musa", None)
    if _dcp_device_owned is not None or torch is None or musa is None or not musa.is_available():
        return False
    filesystem = importlib.import_module("torch.distributed.checkpoint.filesystem")
    original = getattr(filesystem, "_get_available_device_type", None)
    if original is None:
        return False

    @wraps(original)
    def actual_device_type():
        device_type = original()
        # CUDA API availability is emulated by torchada, but tensor.device.type
        # remains 'musa'. DCP compares these strings before staging to CPU.
        # Inspect the actual stream at call time, independent of hook ordering.
        if device_type == "cuda" and torch.cuda.current_stream().device.type == "musa":
            return "musa"
        return device_type

    filesystem._get_available_device_type = actual_device_type  # type: ignore[attr-defined]
    _dcp_device_owned = (filesystem, original, actual_device_type)
    return True


def _uninstall_dcp_device() -> None:
    global _dcp_device_owned
    if _dcp_device_owned is not None:
        module, original, replacement = _dcp_device_owned
        if getattr(module, "_get_available_device_type", None) is replacement:
            module._get_available_device_type = original
        _dcp_device_owned = None


PATCHES = (
    HookPatch(
        id="megatron.dist-ckpt.musa-cpu-staging",
        trigger="megatron",
        run=_install_dcp_device,
        undo=_uninstall_dcp_device,
        rationale=(
            "With CUDA API emulation, torch 2.7 DCP selects cuda while tensor devices "
            "remain musa. _OverlappingCpuLoader skips the CPU copy and the writer "
            "fails assert tensor.is_cpu, including FSDP DTensor checkpoints."
        ),
        strategy=(
            "Own only filesystem's imported device selector. When its cuda selection "
            "resolves to an actual MUSA stream, return musa. Reuse upstream staging, "
            "stream synchronization, planners and serialization without changing "
            "checkpoint keys or sharding. No additional synchronous-copy fallback."
        ),
        upstream="PyTorch torch/distributed/checkpoint/filesystem.py:_OverlappingCpuLoader",
        remove_when=(
            "Remove when disabling this hook passes tiny multi-rank DTensor model/optimizer "
            "save/load and original FSDP checkpoint tests under CUDA API emulation. "
            "Review the private filesystem selector on PyTorch upgrades."
        ),
    ),
)
