"""Synthetic NVIDIA-scale values for Megatron's architecture comparisons.

MUSA capability numbers are not NVIDIA compute capabilities. The default 8.3
passes Megatron's >=8 grouped-GEMM gate and stays below its >=10 architecture
checks. It neither describes real hardware nor proves that a gated kernel is
supported. In particular, overriding ARCH can enable additional NVIDIA paths.
The torch.cuda override is process-wide once Megatron triggers it; undo restores
it only while we still own the attribute. Real MUSA capability queries should
use the backend's native API.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .. import _env
from .._engine import AttrPatch, HookPatch

__all__ = ["PATCHES", "arch_tuple", "arch_major"]

logger = logging.getLogger("megatron_musa_patch")

_DEFAULT_ARCH = (8, 3)
_MISSING = object()
_capability_override: tuple[Any, Any, Any] | None = None


def arch_tuple() -> tuple[int, int]:
    raw = _env.value("ARCH")
    if raw is None or not raw.strip():
        return _DEFAULT_ARCH
    value = raw.strip()
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
        major, _, minor = value.partition(".")
        return int(major), int(minor or 0)
    logger.warning(
        "%s_ARCH=%r is not a nonnegative <major>[.<minor>]; falling back to %d.%d",
        _env.ENV_PREFIX,
        raw,
        *_DEFAULT_ARCH,
    )
    return _DEFAULT_ARCH


def arch_major() -> int:
    return arch_tuple()[0]


def _install_torch_capability() -> bool | None:
    global _capability_override
    if _capability_override is not None:
        return False

    import torch

    capability = arch_tuple()

    def get_device_capability(device=None):
        return capability

    owner = torch.cuda
    # Some compatibility namespaces resolve/cache attributes in __getattr__.
    # Journal the direct attribute only; a lookup would itself mutate the owner.
    original = vars(owner).get("get_device_capability", _MISSING)
    owner.get_device_capability = get_device_capability
    _capability_override = (owner, original, get_device_capability)
    return None  # applied; the hook contract distinguishes decline (False)


def _uninstall_torch_capability() -> None:
    global _capability_override
    if _capability_override is None:
        return
    owner, original, replacement = _capability_override
    if vars(owner).get("get_device_capability", _MISSING) is replacement:
        if original is _MISSING:
            delattr(owner, "get_device_capability")
        else:
            owner.get_device_capability = original
    _capability_override = None


def _replace_arch_version(original: Any) -> Any:
    major = arch_major()

    def get_device_arch_version():
        return major

    return get_device_arch_version


PATCHES = (
    HookPatch(
        id="torch.cuda.device-capability.nvidia-scale",
        trigger="megatron",
        run=_install_torch_capability,
        undo=_uninstall_torch_capability,
        rationale=(
            "Megatron arguments.py compares torch.cuda.get_device_capability "
            "against NVIDIA's >=8 grouped-GEMM threshold; MUSA's native numbering "
            "is not comparable. A numeric gate is not a backend kernel probe."
        ),
        strategy=(
            "On Megatron import substitute the configurable NVIDIA-scale pair "
            "(default 8.3) on torch.cuda only; record ownership for reversible undo. "
            "This is a process-wide policy, not a claim of Ampere compatibility."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/training/arguments.py; "
            "pytorch/pytorch torch/cuda/__init__.py"
        ),
        remove_when=(
            "Review capability consumers on every Megatron and torch_musa/backend "
            "upgrade; remove when Megatron uses backend feature probes. Before "
            "removal test grouped GEMM and architecture-gated paths on target hardware."
        ),
    ),
    AttrPatch(
        id="megatron.training.get-device-arch-version.nvidia-scale",
        target="megatron.training.utils:get_device_arch_version",
        replace=_replace_arch_version,
        rationale=(
            "The upstream helper reads CUDA device properties.major and documents "
            "NVIDIA architecture numbers. MUSA's native major would be interpreted "
            "as a NVIDIA generation in connection-limit and stream-priority checks."
        ),
        strategy=(
            "Return the same synthetic major as the capability hook without "
            "constructing or querying a CUDA device. Default 8 stays below 10; "
            "this does not bypass every NVIDIA-specific code path."
        ),
        upstream="NVIDIA/Megatron-LM megatron/training/utils.py; arguments.py",
        remove_when=(
            "Review on Megatron and backend upgrades; remove with the capability "
            "hook when architecture checks become backend-aware, after TP/CP/FSDP "
            "connection-limit and priority-stream initialization tests."
        ),
    ),
)
