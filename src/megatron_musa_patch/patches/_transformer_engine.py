"""Transformer Engine compatibility policies for the MUSA port.

The MUSA Transformer Engine development tree is not version-compatible with
NVIDIA's public Transformer Engine releases. Comparing its package version to
upstream release thresholds therefore selects the wrong Megatron branches.
"""

from __future__ import annotations

import functools
from typing import Any

from .._engine import AttrPatch

__all__ = ["PATCHES"]

#: Megatron's Transformer Engine extension module: wraps TE and owns the
#: version-gated call shapes this module adapts.
_TE_EXTENSION = "megatron.core.extensions.transformer_engine"


def _te_extension():
    """The extension module being patched, without importing it.

    It is always in ``sys.modules`` when these patches run; importing it here
    would drag Transformer Engine (and, on MUSA, its shared libraries) into
    processes and unit tests that never needed it.
    """
    import sys

    return sys.modules.get(_TE_EXTENSION)


def _ignore_te_min_version(original: Any) -> Any:
    """Ignore NVIDIA Transformer Engine version thresholds on MUSA.

    The MUSA Transformer Engine development version has its own API and feature
    timeline, so its version string cannot be compared with upstream TE
    versions. Megatron's caller still decides whether Transformer Engine is
    importable; this replacement only bypasses the unrelated numeric check.
    """

    @functools.wraps(original)
    def is_te_min_version(version: str, check_equality: bool = True) -> bool:
        del version, check_equality
        return True

    return is_te_min_version


def _cpu_offload_context_by_signature(original: Any) -> Any:
    """Pick TE's CPU-offload call by arity, not by version threshold.

    Megatron chooses between Transformer Engine's three CPU-offload signatures
    with ``is_te_min_version`` thresholds.  The blanket version policy above
    makes the TE >= 2.5 branch (six arguments, ``double_buffering``) win, while
    the MUSA fork still stops at the five-argument (TE ~2.3) shape, so
    ``TransformerBlock.__init__`` -- which calls this unconditionally -- raises
    ``TypeError: ... takes from 0 to 5 positional arguments but 6 were given``
    before the model is built.  Dispatch on the installed function's real
    signature instead; leave upstream alone when it is already right.
    """
    import inspect

    module = _te_extension()
    target = getattr(module, "_get_cpu_offload_context", None) if module else None
    if target is None or original is None:
        return None
    try:
        parameters = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_POSITIONAL for p in parameters):
        return None
    accepted = sum(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters
    )
    if accepted != 5:  # >= 6: upstream's choice is right; fewer: not this fork
        return None

    @functools.wraps(original)
    def get_cpu_offload_context(
        enabled,
        num_layers,
        model_layers,
        activation_offloading,
        weight_offloading,
        double_buffering,
    ):
        """Get CPU offload context and sync function (five-argument TE)."""
        return target(
            enabled, num_layers, model_layers, activation_offloading, weight_offloading
        )

    return get_cpu_offload_context


PATCHES = (
    AttrPatch(
        id="megatron.core.utils.te-version-check.ignore",
        target="megatron.core.utils:is_te_min_version",
        replace=_ignore_te_min_version,
        rationale=(
            "The MUSA Transformer Engine development tree is not a release-compatible "
            "fork of NVIDIA's public Transformer Engine, so upstream version thresholds "
            "do not describe its available APIs."
        ),
        strategy=(
            "Replace Megatron's numeric minimum-version predicate with an unconditional "
            "true result while leaving Transformer Engine importability checks and all "
            "other version predicates unchanged."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/utils.py:is_te_min_version",
        remove_when=(
            "Remove after the MUSA Transformer Engine publishes a versioning and API "
            "compatibility contract that can be mapped reliably to Megatron's upstream "
            "Transformer Engine thresholds."
        ),
    ),
    AttrPatch(
        id="megatron.te.cpu-offload-context.signature-dispatch",
        target=f"{_TE_EXTENSION}:get_cpu_offload_context",
        replace=_cpu_offload_context_by_signature,
        rationale=(
            "TransformerBlock calls get_cpu_offload_context unconditionally, and "
            "Megatron selects TE's six-argument (TE >= 2.5) call from the "
            "is_te_min_version policy above. The MUSA TE fork's "
            "transformer_engine.pytorch.cpu_offload.get_cpu_offload_context still takes "
            "five arguments (no double_buffering), so every run that builds a "
            "TransformerBlock dies with 'takes from 0 to 5 positional arguments but 6 "
            "were given' before the model exists -- on a path that never touches CPU "
            "offloading. Observed on the MT fork of TE 2.0.0 with Megatron core_v0.16.1."
        ),
        strategy=(
            "Keep Megatron's public six-argument wrapper and call the installed TE "
            "function with the five arguments it accepts, chosen by inspecting that "
            "function's own signature instead of a version number. Declines when the "
            "installed function takes six arguments (upstream's choice is then correct) "
            "or when its arity is anything else, so the patch never guesses."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/extensions/transformer_engine.py:"
            "get_cpu_offload_context"
        ),
        remove_when=(
            "Remove once the MUSA Transformer Engine exposes the double_buffering "
            "argument (or the version policy can report the fork's real API level), "
            "after a pretrain run with CPU offloading enabled and disabled has been "
            "re-validated on MUSA."
        ),
    ),
)
