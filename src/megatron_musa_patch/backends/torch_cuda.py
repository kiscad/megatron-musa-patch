"""Lazy CUDA-to-MUSA adaptation for Megatron.

``torchada`` owns the general adapter: CUDA namespaces, device strings, tensor
factories, distributed backends, and more. This module owns only four gaps:
MUSA availability, CUDA tensor type names, the graph-class alias, and tensor
subclass transfers. No torch or accelerator package is imported at module load.

Mutation boundary
-----------------
Importing ``torch_musa`` and ``torchada`` has process-wide side effects. In
particular, torchada may replace ``torch.cuda`` and entries in ``sys.modules``;
there is no supported general undo API for those changes. We do not copy or
reverse torchada's implementation. :func:`unapply` restores only our attribute
overrides to their *post-torchada* state, and failed activation rolls those same
overrides back. Use a fresh process to remove the external adapters completely.

Our overrides operate in place on the objects exposed by the adapter. Undo is
identity-checked so a subsequent third-party replacement is not overwritten.
References captured elsewhere while the layer was active cannot be revoked.
"""

from __future__ import annotations

import functools
import logging
import sys
from threading import RLock

from typing import Any

from .. import _compat

__all__ = ["apply", "unapply", "is_applied", "torchada_version"]

logger = logging.getLogger("megatron_musa_patch")

_APPLIED = False
_LOCK = RLock()
_MISSING = object()
# owner, attribute, previous direct binding, our replacement. Direct bindings
# matter: getattr() would populate torchada's caching proxy during bookkeeping.
_OVERRIDES: list[tuple[Any, str, Any, Any]] = []


def is_applied() -> bool:
    """Whether this project's overrides (not just torchada) are installed."""
    return _APPLIED


def torchada_version() -> str:
    """Read the adapter version without importing it or torch."""
    import importlib.metadata as md

    try:
        return md.version("torchada")
    except md.PackageNotFoundError:  # pragma: no cover - declared dependency
        return "unknown"


def _set_attr(owner: Any, name: str, replacement: Any) -> None:
    previous = vars(owner).get(name, _MISSING)
    if previous is replacement:
        return
    _OVERRIDES.append((owner, name, previous, replacement))
    setattr(owner, name, replacement)


def _restore_overrides() -> None:
    while _OVERRIDES:
        owner, name, previous, replacement = _OVERRIDES[-1]
        if vars(owner).get(name, _MISSING) is not replacement:
            logger.debug("Leaving subsequently replaced backend attribute %s", name)
        elif previous is _MISSING:
            delattr(owner, name)
        else:
            setattr(owner, name, previous)
        # Keep a failing restoration in the journal so unapply can be retried.
        _OVERRIDES.pop()


def _refresh_transformers_device_constants() -> None:
    """Invalidate only cached probes affected by the CUDA/MUSA aliases.

    Do not import transformers or clear unrelated package-availability caches.
    Refresh both after activation and after rollback/unapply, so the cached
    result always reflects the currently owned availability binding.
    """
    module = sys.modules.get("transformers.utils.import_utils")
    if module is None:
        return
    blocked, detail = _compat.check_version_gate("transformers >=5")
    if blocked:
        # The frozen-constant mechanism this refresh targets appeared in
        # transformers 5.x; older versions re-evaluate on every call.
        logger.debug("transformers device-constant refresh skipped (%s)", detail)
        return
    refreshed = False
    names = (
        "is_torch_cuda_available", "is_torch_bf16_gpu_available",
        "is_torch_fp16_available_on_device", "is_torch_bf16_available_on_device",
        "is_torch_tf32_available", "is_flash_attn_2_available", "is_flash_attn_3_available",
    )
    for name in names:
        fn = vars(module).get(name)
        clear = getattr(fn, "cache_clear", None)
        if not callable(clear):
            continue
        clear()
        refreshed = True
    if refreshed:
        # The curated list above is bound to the transformers freeze
        # mechanism (lru_cache + compile constants, transformers 5.x);
        # re-check it when transformers changes how it freezes probes.
        logger.debug(
            "transformers device-availability caches refreshed (transformers %s)",
            _compat.transformers_version())


def _alias_availability(torch: Any, musa: Any) -> None:
    """Megatron's CUDA assertion must query MUSA, not report a constant True."""
    _set_attr(torch.cuda, "is_available", musa.is_available)


def _alias_tensor_type_names(torch: Any) -> None:
    """Translate Tensor.type() queries; leave conversions and CPU names alone."""
    original_type = torch.Tensor.type

    @functools.wraps(original_type)
    def _type(self, *args, **kwargs):
        result = original_type(self, *args, **kwargs)
        if isinstance(result, str) and result.startswith("torch.musa."):
            return "torch.cuda." + result[len("torch.musa.") :]
        return result

    _set_attr(torch.Tensor, "type", _type)


def _alias_graph_class(torch: Any, musa: Any) -> None:
    """Expose MUSAGraph under the CUDA spelling, including the graphs module."""
    graph_cls = getattr(musa, "MUSAGraph", None)
    if graph_cls is None:  # Some backend releases do not expose graph support.
        return

    _set_attr(torch.cuda, "CUDAGraph", graph_cls)
    graphs = sys.modules.get("torch.cuda.graphs", getattr(musa, "graphs", None))
    if graphs is not None:
        _set_attr(graphs, "CUDAGraph", graph_cls)


def _fix_tensor_musa_for_subclasses(torch: Any) -> None:
    """Use aten .to() for subclasses affected by torch_musa's C dispatch shim.

    TransformerEngine FP8 Float8Tensor is one such subclass. Keep ordinary
    tensors on the original fast path and preserve transfer options on the
    subclass path. Remove this workaround when torch_musa fixes its C shim.

    The same shim also re-enters this wrapper from inside a torch function
    mode (e.g. ``with torch.device(...)``): the C implementation re-dispatches
    ``Tensor.musa`` with the real tensor shifted into ``*args`` and a
    non-Tensor placeholder in ``self``, so a naive ``type(self) is not Tensor``
    check would treat the placeholder as a tensor subclass and pass it to
    ``torch.device``.
    """
    original_musa = torch.Tensor.musa

    def _resolve_musa_device(device):
        if device is None:
            return torch.device("musa")
        if isinstance(device, int):
            return torch.device("musa", device)
        device = torch.device(device)
        return device

    def _move_subclass(
        self, device=None, non_blocking=False, *, memory_format=torch.preserve_format
    ):
        device = _resolve_musa_device(device)
        if device.type != "musa":
            raise RuntimeError(f"Invalid device, must be musa device: {device}")
        return self.to(
            device=device, non_blocking=non_blocking, memory_format=memory_format
        )

    @functools.wraps(original_musa)
    def _musa(self, *args, **kwargs):
        if type(self) is torch.Tensor:
            return original_musa(self, *args, **kwargs)
        if isinstance(self, torch.Tensor):
            return _move_subclass(self, *args, **kwargs)
        # C-shim re-entry: the real tensor arrives as args[0] while ``self``
        # is a non-Tensor placeholder. Route the tensor through the same
        # subclass transfer path instead of treating it as a device argument.
        if args and isinstance(args[0], torch.Tensor):
            return _move_subclass(args[0], *args[1:], **kwargs)
        return original_musa(self, *args, **kwargs)

    _set_attr(torch.Tensor, "musa", _musa)


def apply() -> None:
    """Install the adapter and our overrides, once per apply/unapply cycle.

    A visible MUSA device is required. Check it before importing torchada to
    avoid installing that adapter in an unusable environment. Dependencies may
    mutate global state during import; only our own changes are transactional.
    """
    global _APPLIED
    with _LOCK:
        if _APPLIED:
            return
        # A prior failed activation may itself have encountered a failing undo.
        # Finish that cleanup before taking a new post-adapter baseline.
        _restore_overrides()

        from .._errors import MusaUnavailable

        try:
            import torch
            import torch_musa  # noqa: F401
        except ImportError as exc:
            raise MusaUnavailable(
                "megatron-musa-patch needs the MUSA build of PyTorch (torch_musa), "
                "which is not importable. Disable the patch package with "
                "MEGATRON_MUSA_PATCH=0 to bypass it."
            ) from exc

        musa = getattr(torch, "musa", None)
        available = getattr(musa, "is_available", None)
        if not callable(available) or not available():
            raise MusaUnavailable(
                "torch_musa is importable but no MUSA device is available. "
                "Check the MUSA runtime and MUSA_VISIBLE_DEVICES, or disable "
                "the patch package with MEGATRON_MUSA_PATCH=0."
            )

        try:
            import torchada
        except ImportError as exc:
            raise MusaUnavailable(
                "megatron-musa-patch requires torchada for its torch.cuda "
                "compatibility layer, but torchada is not importable. Install it "
                "with `pip install torchada`, or disable the patch package with "
                "MEGATRON_MUSA_PATCH=0."
            ) from exc

        # torchada considers its CPU/CUDA no-op branches 'patched' too. Use its
        # public platform probe when present; do not depend on private internals.
        is_musa_platform = getattr(torchada, "is_musa_platform", None)
        if callable(is_musa_platform) and not is_musa_platform():
            raise MusaUnavailable(
                "torchada did not select MUSA. Check TORCHADA_PLATFORM and "
                "initialize the MUSA runtime before importing torchada."
            )
        if not torchada.is_patched():
            torchada.apply_patches()
        if not torchada.is_patched():
            raise MusaUnavailable("torchada did not finish installing its MUSA adapter.")

        try:
            _alias_availability(torch, musa)
            _alias_tensor_type_names(torch)
            _alias_graph_class(torch, musa)
            _fix_tensor_musa_for_subclasses(torch)
            _refresh_transformers_device_constants()
        except BaseException:
            _restore_overrides()
            _refresh_transformers_device_constants()
            raise

        _APPLIED = True
        logger.debug(
            "torch.cuda compatibility installed (torchada %s + project overrides)",
            getattr(torchada, "__version__", "unknown"),
        )


def unapply() -> None:
    """Restore project-owned overrides; leave external adapter effects intact.

    Idempotent, and safe before apply() or after a failed apply(). Subsequent
    apply() wraps the restored baseline once, without stacking old wrappers.
    """
    global _APPLIED
    with _LOCK:
        _restore_overrides()
        _refresh_transformers_device_constants()
        _APPLIED = False
