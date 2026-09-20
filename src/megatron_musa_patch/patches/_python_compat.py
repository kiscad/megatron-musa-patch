"""Interpreter backports the MUSA stack's Python version needs.

Megatron 0.19 declares ``requires-python >= 3.12`` and imports
``typing.override`` at module scope (``megatron/training/models/hybrid.py``,
``gpt.py``). The supported MUSA wheels (torch 2.7.1a0 / torch_musa 2.7.1) are
built for CPython 3.10, where that name does not exist, so every importer of
``megatron.training`` dies with ``ImportError: cannot import name 'override'``
before any MUSA code runs. PEP 698 ``override`` is a pure-typing marker with no
runtime behaviour, so publishing the ``typing_extensions`` implementation (or a
minimal equivalent) restores the import without changing semantics.
"""

from __future__ import annotations

from typing import Any, Callable

from .._engine import HookPatch

__all__ = ["PATCHES"]

#: Objects this patch published, so undo only removes what it owns.
_owned: dict[str, Any] = {}


def _fallback_override(method: Callable) -> Callable:
    """PEP 698 ``override`` without typing_extensions: set the marker, return as-is."""
    try:
        method.__override__ = True  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # Slots, builtins and descriptors reject the attribute; the decorator is
        # still required to return the method unchanged.
        pass
    return method


def _install_typing_override() -> bool | None:
    """Publish ``typing.override`` on interpreters older than 3.12."""
    import typing

    if hasattr(typing, "override"):
        # CPython >= 3.12, or another caller already provided it: decline.
        return False
    try:
        from typing_extensions import override
    except ImportError:
        override = _fallback_override
    typing.override = override  # type: ignore[attr-defined]
    _owned["override"] = override
    return None


def _uninstall_typing_override() -> None:
    """Remove ``typing.override`` only while this patch still owns it."""
    import typing

    owned = _owned.pop("override", None)
    if owned is not None and typing.__dict__.get("override") is owned:
        del typing.override  # type: ignore[attr-defined]


PATCHES = (
    HookPatch(
        id="python.typing.override.backport",
        trigger="megatron",
        run=_install_typing_override,
        undo=_uninstall_typing_override,
        rationale=(
            "Megatron 0.19 requires Python >= 3.12 and imports typing.override at module "
            "scope; the supported MUSA wheels are CPython 3.10 builds, so importing "
            "megatron.training raises ImportError before any patch or MUSA code runs. "
            "Observed on Python 3.10.12 with Megatron-LM core_v0.19.0 "
            "(megatron/training/models/hybrid.py:5, gpt.py:5)."
        ),
        strategy=(
            "Before megatron is imported, publish typing_extensions.override as "
            "typing.override, falling back to a local decorator that only sets "
            "__override__ when typing_extensions is absent. PEP 698 override has no "
            "runtime effect beyond that marker, so type-checking semantics are "
            "unchanged and no upstream file is rewritten. Decline when the attribute "
            "already exists and undo only the binding this patch owns."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/training/models/{hybrid,gpt}.py (core_v0.19.0); "
            "CPython typing.override (3.12+, PEP 698)"
        ),
        remove_when=(
            "The MUSA stack runs on CPython >= 3.12, or upstream imports override from "
            "typing_extensions; verify by importing megatron.training with this patch "
            "disabled (MEGATRON_MUSA_PATCH_DISABLE=python.typing.override.backport)."
        ),
    ),
)
