"""Transformer Engine call-shape adapters for the MUSA port.

The MUSA Transformer Engine fork reports NVIDIA TE release numbering without
following NVIDIA's release API timeline, so Megatron's version thresholds do
not describe the installed API.  This module no longer overrides the version
predicate itself: ``megatron.core.utils.is_te_min_version`` keeps its real
comparison so that version-specific branches report the fork honestly.
Call-shape gaps are instead adapted at their own boundary by inspecting the
installed function (see ``_cpu_offload_context_by_signature``), or reported as
unsupported (QK-clip max-logit, ``quantized_model_init``).
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from .._engine import AttrPatch, HookPatch

__all__ = ["PATCHES"]

logger = logging.getLogger("megatron_musa_patch")

#: Megatron's Transformer Engine extension module: wraps TE and owns the
#: version-gated call shapes this module adapts.
_TE_EXTENSION = "megatron.core.extensions.transformer_engine"

#: sys.modules entries this package installs for the MT-TE fork's legacy
#: ``musa_patch`` dependency, and that its undo may remove.
_mem_monitor_owned: dict[str, Any] = {}
_quantized_init_owned: tuple[Any, Any] | None = None


def _install_quantized_model_init() -> bool:
    """Expose only the verified delayed-FP8 subset of the newer TE spelling.

    Runs after Megatron activation, never during patch registration. The
    attribute is owned by this hook, not by a second import/patch system.
    """
    global _quantized_init_owned
    import importlib
    import inspect
    import sys

    if _quantized_init_owned is not None or not _te_fork_needs_mem_monitor():
        return False
    # Device adaptation is a runtime prerequisite, not a reason to initialize
    # the vendor stack inside a CPU-only or synthetic Megatron import.
    torch = sys.modules.get("torch")
    musa = getattr(torch, "musa", None)
    if musa is None or not musa.is_available() or not torch.cuda.is_available():
        return False
    te = importlib.import_module("transformer_engine.pytorch")
    if hasattr(te, "quantized_model_init"):
        return False
    original = getattr(te, "fp8_model_init", None)
    if original is None:
        return False
    signature = inspect.signature(original)
    if not {"enabled", "recipe", "preserve_high_precision_init_val"} <= signature.parameters.keys():
        return False
    from transformer_engine.common.recipe import DelayedScaling

    @functools.wraps(original)
    def quantized_model_init(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        if arguments.arguments["enabled"] and not isinstance(
            arguments.arguments["recipe"], DelayedScaling
        ):
            raise NotImplementedError(
                "MUSA quantized_model_init compatibility requires an explicit "
                "DelayedScaling recipe; other quantization recipes need native TE support"
            )
        return original(*arguments.args, **arguments.kwargs)

    te.quantized_model_init = quantized_model_init
    _quantized_init_owned = (te, quantized_model_init)
    return True


def _uninstall_quantized_model_init() -> None:
    global _quantized_init_owned
    if _quantized_init_owned is not None:
        module, replacement = _quantized_init_owned
        if getattr(module, "quantized_model_init", None) is replacement:
            del module.quantized_model_init
        _quantized_init_owned = None


def _te_extension():
    """The extension module being patched, without importing it.

    It is always in ``sys.modules`` when these patches run; importing it here
    would drag Transformer Engine (and, on MUSA, its shared libraries) into
    processes and unit tests that never needed it.
    """
    import sys

    return sys.modules.get(_TE_EXTENSION)


def _cpu_offload_context_by_signature(original: Any) -> Any:
    """Pick TE's CPU-offload call by arity, not by version threshold.

    Megatron chooses between Transformer Engine's three CPU-offload signatures
    with ``is_te_min_version`` thresholds, so a fork whose reported version and
    real API disagree is called with the wrong argument count:
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


def _te_fork_needs_mem_monitor() -> bool:
    """Is the installed TransformerEngine the MUSA fork that imports musa_patch?"""
    import importlib.metadata as md
    from pathlib import Path

    try:
        distribution = md.distribution("transformer_engine")
        musa_dir = Path(distribution.locate_file("transformer_engine/musa"))
    except Exception:  # noqa: BLE001 - distribution metadata is best effort
        return False
    return musa_dir.exists()


def _install_mem_monitor_shim() -> bool:
    """Bridge the MT-TE fork's legacy ``musa_patch.mem_utils`` import.

    ``transformer_engine/pytorch/module/grouped_linear.py`` imports
    ``musa_patch.mem_utils.MemMonitor`` unconditionally and updates
    ``MemMonitor.max_token_num``; the legacy ``musa_patch`` distribution is not
    installed. Provide the minimal accounting type the fork mutates, owned by
    this hook so undo removes exactly what it added.
    """
    import sys
    import types

    if "musa_patch" in sys.modules or "musa_patch.mem_utils" in sys.modules:
        # A real musa_patch (or another owner) is importable: never shadow it.
        return False
    if not _te_fork_needs_mem_monitor():
        return False

    package = types.ModuleType("musa_patch")
    package.__path__ = []  # mark as a package so submodule imports resolve
    package.__doc__ = "Compatibility shim owned by megatron-musa-patch."

    mem_utils = types.ModuleType("musa_patch.mem_utils")
    mem_utils.__doc__ = "Shim for the MT-TE fork's grouped-linear memory accounting."

    class MemMonitor:
        """Tracks the largest token count seen, as the fork's import expects."""

        max_token_num = 0

    mem_utils.MemMonitor = MemMonitor
    package.mem_utils = mem_utils

    sys.modules["musa_patch"] = package
    sys.modules["musa_patch.mem_utils"] = mem_utils
    _mem_monitor_owned["musa_patch"] = package
    _mem_monitor_owned["musa_patch.mem_utils"] = mem_utils
    return True


def _uninstall_mem_monitor_shim() -> None:
    import sys

    for name in list(_mem_monitor_owned):
        module = _mem_monitor_owned.pop(name)
        if sys.modules.get(name) is module:
            del sys.modules[name]


PATCHES = (
    HookPatch(
        id="megatron.te.quantized-model-init.delayed-compat",
        trigger="megatron",
        run=_install_quantized_model_init,
        undo=_uninstall_quantized_model_init,
        rationale=(
            "MT-TE 2.0 exposes fp8_model_init but FSDP delayed-FP8 callers use "
            "quantized_model_init with preserve_high_precision_init_val."
        ),
        strategy=(
            "After Megatron activation, own the missing TE attribute and delegate "
            "explicit DelayedScaling contexts to the existing native implementation. "
            "Reject other enabled recipes, preserve context nesting and exceptions, "
            "and never overwrite a native or third-party implementation."
        ),
        upstream="TransformerEngine pytorch/fp8.py:fp8_model_init",
        remove_when=(
            "Remove once native quantized_model_init supports delayed FP8 with "
            "high-precision initialization and both original FSDP cases pass."
        ),
    ),
    AttrPatch(
        id="megatron.te.cpu-offload-context.signature-dispatch",
        target=f"{_TE_EXTENSION}:get_cpu_offload_context",
        replace=_cpu_offload_context_by_signature,
        rationale=(
            "TransformerBlock calls get_cpu_offload_context unconditionally, and "
            "Megatron picks TE's six-argument (TE >= 2.5) call from a version "
            "threshold. The MUSA TE fork reports release numbering without "
            "following NVIDIA's API timeline, so its reported version can select "
            "the six-argument call while "
            "transformer_engine.pytorch.cpu_offload.get_cpu_offload_context still "
            "takes five arguments (no double_buffering); every run that builds a "
            "TransformerBlock then dies with 'takes from 0 to 5 positional "
            "arguments but 6 were given' before the model exists -- on a path that "
            "never touches CPU offloading. Observed on the MT fork of TE 2.0.0 "
            "with Megatron core_v0.16.1."
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
            "argument, after a pretrain run with CPU offloading enabled and disabled "
            "has been re-validated on MUSA."
        ),
    ),
    HookPatch(
        id="megatron.te.grouped-linear.mem-monitor-compat",
        trigger="megatron",
        run=_install_mem_monitor_shim,
        undo=_uninstall_mem_monitor_shim,
        rationale=(
            "The MUSA TransformerEngine fork's grouped_linear.py imports "
            "musa_patch.mem_utils.MemMonitor unconditionally inside "
            "TEGroupedLinear._forward and updates MemMonitor.max_token_num, but "
            "the legacy musa_patch distribution is not installed, so every MoE "
            "grouped-GEMM forward raises ModuleNotFoundError before any math. "
            "Observed on MT-TE 2.0.0 with Megatron core_v0.16.1 in "
            "a2a_overlap and dist_checkpointing grouped-expert cases."
        ),
        strategy=(
            "When Megatron is first imported, the import name is free and the "
            "installed TransformerEngine is the MUSA fork, own two sys.modules "
            "entries (musa_patch and musa_patch.mem_utils) carrying the minimal "
            "MemMonitor type with a max_token_num counter, so the fork's "
            "accounting update keeps its behavior instead of becoming a no-op "
            "mock. A pre-existing musa_patch module is never shadowed; a failing "
            "install removes its partial entries; undo deletes only modules this "
            "hook still owns. This is a bridge for one TE import, not a "
            "re-creation of the legacy package."
        ),
        upstream="transformer_engine/pytorch/module/grouped_linear.py:musa_patch import",
        remove_when=(
            "Remove when the MUSA TransformerEngine drops the external "
            "musa_patch import: disable this patch id and re-run the MoE/A2A "
            "grouped-linear forward cases; delete only if they pass without it."
        ),
    ),
)
