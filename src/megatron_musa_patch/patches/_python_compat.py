"""Standard-library names newer Megatron imports but the MUSA interpreter lacks.

The MUSA vendor stack (torch_musa, MT-TransformerEngine) ships for CPython 3.10,
while Megatron core_v0.19.x declares ``requires-python >= 3.12`` and imports
``typing.override`` in ``megatron/training/models/{gpt,hybrid}.py``.  Nothing
else in that tree needs a newer interpreter -- it compiles under 3.10 -- so the
only import-time blocker is that one name.  The hook fires when ``megatron`` is
first imported, before any Megatron module executes, and is a no-op wherever
``typing.override`` already exists.
"""

from __future__ import annotations

from typing import Any, Callable

from .._engine import HookPatch

__all__ = ["PATCHES"]

_owned: dict[str, Any] = {}


def _fallback_override(method: Callable) -> Callable:
    """PEP 698 runtime behaviour: mark the object and return it unchanged."""
    try:
        method.__override__ = True
    except (AttributeError, TypeError):
        pass
    return method


def _install_typing_override() -> bool | None:
    import typing

    if hasattr(typing, "override"):
        return False
    try:
        from typing_extensions import override
    except ImportError:
        override = _fallback_override
    typing.override = override
    _owned["override"] = override
    return None


def _uninstall_typing_override() -> None:
    import typing

    owned = _owned.pop("override", None)
    # Leave a binding that someone else installed after us untouched.
    if owned is not None and typing.__dict__.get("override") is owned:
        del typing.override


PATCHES = (
    HookPatch(
        id="python.typing.override.backport",
        trigger="megatron",
        run=_install_typing_override,
        undo=_uninstall_typing_override,
        rationale=(
            "The MUSA vendor stack provides CPython 3.10 builds only. Megatron "
            "core_v0.19.x megatron/training/models/gpt.py and hybrid.py run "
            "`from typing import ..., override` (added in Python 3.12), so "
            "importing megatron.training or pretrain_gpt.py fails with ImportError "
            "before any MUSA code path is reached. The rest of that tree compiles "
            "under 3.10."
        ),
        strategy=(
            "Only when typing has no `override`, bind typing_extensions.override "
            "(or an equivalent that sets __override__ and returns its argument) "
            "on the typing module. The decorator has no runtime effect beyond "
            "that marker. This mutates the process-wide typing module; undo "
            "removes the binding only while it is still the one this hook set."
        ),
        upstream=(
            "megatron/training/models/gpt.py, megatron/training/models/hybrid.py "
            "(Megatron-LM core_v0.19.x); CPython typing.override (PEP 698)"
        ),
        remove_when=(
            "The supported MUSA stack runs on Python >= 3.12, or upstream stops "
            "importing override from typing; with this patch disabled, importing "
            "megatron.training.models.gpt and running examples/run_pretrain_smoke.sh "
            "still succeed."
        ),
    ),
)
