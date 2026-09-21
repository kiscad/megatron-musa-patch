"""Interpreter backports the MUSA stack's Python version needs.

Megatron 0.19 declares ``requires-python >= 3.12`` and imports
``typing.override`` at module scope (``megatron/training/models/hybrid.py``,
``gpt.py``). The supported MUSA wheels (torch 2.7.1a0 / torch_musa 2.7.1) are
built for CPython 3.10, where that name does not exist, so every importer of
``megatron.training`` dies with ``ImportError: cannot import name 'override'``
before any MUSA code runs. PEP 698 ``override`` is a pure-typing marker with no
runtime behaviour, so publishing the ``typing_extensions`` implementation (or a
minimal equivalent) restores the import without changing semantics.

The same applies to ``typing.Concatenate[X, ...]``: CPython 3.11 started
accepting a trailing ``Ellipsis`` (PEP 612's "arbitrary remaining parameters"),
while 3.10 rejects it while *evaluating the annotation*. Megatron's declared
optimizer dependency (``emerging_optimizers`` >= 0.2, which core 0.19 imports
eagerly once installed) writes that form at module scope, so on 3.10 the whole
``megatron.core`` import dies inside a type annotation. Both patches only
restore names the interpreter is missing; neither changes runtime behaviour,
and both decline when the interpreter already provides them.
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


class _ConcatenateCompat:
    """``typing.Concatenate`` that accepts 3.11+'s trailing ``Ellipsis``.

    Only subscription differs: a trailing ``Ellipsis`` (meaning "then arbitrary
    parameters") is expressed with a sentinel ``ParamSpec``, which is what 3.10's
    special form accepts, and every other subscription is delegated untouched.
    The object produced is the stdlib's own, so ``get_origin``/``get_args`` and
    annotation evaluation keep working; attribute access falls through to the
    original special form.
    """

    def __init__(self, original: Any, arbitrary_params: Any) -> None:
        self._original = original
        self._arbitrary_params = arbitrary_params

    def __getitem__(self, parameters: Any) -> Any:
        if isinstance(parameters, tuple) and parameters and parameters[-1] is Ellipsis:
            parameters = parameters[:-1] + (self._arbitrary_params,)
        return self._original[parameters]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def __repr__(self) -> str:
        return repr(self._original)


def _install_typing_concatenate() -> bool | None:
    """Accept ``Concatenate[X, ...]`` on interpreters older than 3.11."""
    import typing

    original = getattr(typing, "Concatenate", None)
    if original is None:
        return False  # too old to carry PEP 612 at all: not this patch's business
    try:
        original[int, ...]
    except TypeError:
        pass
    else:
        return False  # CPython >= 3.11, or someone already relaxed it
    compat = _ConcatenateCompat(original, typing.ParamSpec("_MegatronMusaPatchParams"))
    typing.Concatenate = compat  # type: ignore[assignment]
    _owned["Concatenate"] = compat
    return None


def _uninstall_typing_concatenate() -> None:
    """Restore the stdlib special form while this patch still owns the name."""
    import typing

    owned = _owned.pop("Concatenate", None)
    if owned is not None and typing.__dict__.get("Concatenate") is owned:
        typing.Concatenate = owned._original  # type: ignore[assignment]


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
    HookPatch(
        id="python.typing.concatenate.ellipsis",
        trigger="megatron",
        run=_install_typing_concatenate,
        undo=_uninstall_typing_concatenate,
        rationale=(
            "CPython 3.11 started accepting a trailing Ellipsis in "
            "typing.Concatenate (PEP 612); 3.10 raises TypeError while evaluating the "
            "annotation. emerging_optimizers (Megatron 0.19's declared optimizer "
            "dependency, imported eagerly by megatron/core/optimizer/__init__.py once "
            "installed) annotates with Callable[Concatenate[ParamsT, ...], ...] at "
            "module scope, so on the CPython 3.10 MUSA stack the whole megatron.core "
            "import dies inside a type annotation "
            "(emerging_optimizers/registry.py:88, observed with v0.2.0)."
        ),
        strategy=(
            "Before megatron is imported, replace typing.Concatenate with a thin "
            "subscription adapter that expresses a trailing Ellipsis as a sentinel "
            "ParamSpec -- the form 3.10 accepts -- and delegates every other "
            "subscription, attribute access and repr to the stdlib special form. The "
            "objects handed back are the stdlib's own, so get_origin/get_args and "
            "annotation evaluation are unchanged; annotations have no runtime effect "
            "here. Declines when the installed form already accepts Ellipsis, and undo "
            "restores the original only while this patch owns the name."
        ),
        upstream=(
            "NVIDIA-NeMo/Emerging-Optimizers emerging_optimizers/registry.py (v0.2.0); "
            "CPython typing.Concatenate (PEP 612, Ellipsis accepted from 3.11)"
        ),
        remove_when=(
            "The MUSA stack runs on CPython >= 3.11; verify by importing megatron.core "
            "with emerging_optimizers installed and this patch disabled "
            "(MEGATRON_MUSA_PATCH_DISABLE=python.typing.concatenate.ellipsis)."
        ),
    ),
)
