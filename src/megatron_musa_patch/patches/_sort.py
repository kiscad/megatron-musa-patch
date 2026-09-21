"""Sorting with boolean keys, which the MUDNN radix sort has no kernel for.

``muDNN(v3107) NOT_SUPPORTED in Sort::Run: radix sort kernel launch fail`` ->
``RuntimeError: SortCall MUDNN failed in: Sort run kernel failed``. The measured
failure matrix on this stack is exactly one dtype: ``torch.bool`` fails for
every combination of ``stable``, ``descending``, rank and length, while uint8,
int8, int32, int64, float16, bfloat16 and float32 all pass. MoE token
permutation sorts its boolean routing map
(``moe_utils.permute``: ``routing_map.argsort(dim=-1, descending=True,
stable=True)`` and the flattened variant), so every MoE layer that reaches the
reference permutation dies.

``False < True`` maps onto ``0 < 1``, so sorting a ``uint8`` view is the same
comparison: the returned permutation -- including ties under ``stable=True`` --
is identical, and ``sort`` only has to cast its values back. No ordering,
stability or gradient contract changes; nothing is moved off the device.
"""

from __future__ import annotations

import functools
from typing import Any

from .._engine import HookPatch

__all__ = ["PATCHES"]


def _on_musa(tensor: Any) -> bool:
    """Whether this tensor lives on the device whose sort kernel is missing."""
    return tensor.device.type == "musa"


def _bool_keys_on_musa(tensor: Any) -> bool:
    """True when this input is the one MUDNN's sort cannot serve."""
    import torch

    return isinstance(tensor, torch.Tensor) and tensor.dtype is torch.bool and _on_musa(tensor)


def _argsort_via_uint8(original: Any) -> Any:
    """Sort a uint8 view for boolean keys; every other input is untouched."""

    @functools.wraps(original)
    def argsort(input, *args, **kwargs):
        if _bool_keys_on_musa(input):
            import torch

            return original(input.to(torch.uint8), *args, **kwargs)
        return original(input, *args, **kwargs)

    return argsort


def _sort_via_uint8(original: Any) -> Any:
    """Same adaptation for ``sort``, casting the returned values back to bool."""

    @functools.wraps(original)
    def sort(input, *args, **kwargs):
        if not _bool_keys_on_musa(input):
            return original(input, *args, **kwargs)
        import torch

        values, indices = original(input.to(torch.uint8), *args, **kwargs)
        return torch.return_types.sort((values.to(torch.bool), indices))

    return sort


#: (module, attribute) -> original, for the overrides this patch owns.
_owned: dict[tuple[Any, str], Any] = {}

#: ``torch`` is always imported before this package activates, so the overrides
#: are installed from a hook on the Megatron import boundary rather than by an
#: AttrPatch, which would stay pending forever.
_SPELLINGS = (
    ("argsort", _argsort_via_uint8),
    ("sort", _sort_via_uint8),
)


def _install_bool_sort() -> bool | None:
    """Adapt both spellings of both ops, or decline when MUSA is not live."""
    import sys

    torch = sys.modules.get("torch")
    musa = getattr(torch, "musa", None)
    if _owned or torch is None or musa is None or not musa.is_available():
        return False

    for name, factory in _SPELLINGS:
        for owner in (torch, torch.Tensor):
            original = getattr(owner, name, None)
            if original is None:
                continue
            setattr(owner, name, factory(original))
            _owned[(owner, name)] = original
    return None if _owned else False


def _uninstall_bool_sort() -> None:
    """Restore each original while this patch still owns the binding."""
    for (owner, name), original in list(_owned.items()):
        current = getattr(owner, name, None)
        if getattr(current, "__wrapped__", None) is original:
            setattr(owner, name, original)
        _owned.pop((owner, name), None)


PATCHES = (
    HookPatch(
        id="torch.sort.bool-keys",
        trigger="megatron",
        run=_install_bool_sort,
        undo=_uninstall_bool_sort,
        rationale=(
            "The MUDNN radix sort has no boolean kernel: every torch.bool input fails "
            "with 'SortCall MUDNN failed in: Sort run kernel failed' "
            "(muDNN v3107 'NOT_SUPPORTED in Sort::Run'), independent of stable, "
            "descending, rank and length, while uint8/int8/int32/int64/fp16/bf16/fp32 "
            "all pass (measured dtype matrix, not the error text). Megatron's MoE token "
            "permutation sorts its boolean routing map, so 161 of the 1303 failing "
            "unit-test cases in the 2026-09-20 sweep come from this one gap "
            "(megatron/core/transformer/moe/moe_utils.py:431 and :457)."
        ),
        strategy=(
            "For boolean MUSA inputs only, run the vendor's own sort on a uint8 view: "
            "False<True maps onto 0<1, so the comparison, the permutation and tie "
            "handling under stable=True are identical, and sort casts its values back to "
            "bool. Covers all four spellings the callers use (torch.argsort, "
            "Tensor.argsort, torch.sort, Tensor.sort) because one capability is missing, "
            "and owns them as one reversible unit. Every other dtype and device reaches "
            "the original untouched; nothing is moved off the device. Installed from the "
            "Megatron import hook because torch is already imported by then."
        ),
        upstream=(
            "pytorch torch.sort/torch.argsort; NVIDIA/Megatron-LM "
            "megatron/core/transformer/moe/moe_utils.py:permute"
        ),
        remove_when=(
            "MUDNN's sort serves torch.bool: re-run the dtype matrix (bool argsort with "
            "stable/descending on device) and the MoE permutation tests with this patch "
            "disabled (MEGATRON_MUSA_PATCH_DISABLE=torch.sort.bool-keys)."
        ),
    ),
)
