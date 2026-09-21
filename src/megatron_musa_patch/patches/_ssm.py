"""Gated delta rule (linear attention) on MUSA: torch-kernels' TileLang kernels behind the FLA seam.

Megatron's ``GatedDeltaNet`` (``megatron.core.ssm.gated_delta_net``) computes the
chunked gated delta rule with flash-linear-attention's ``chunk_gated_delta_rule``
Triton kernels, and the mcore-bridge model layer
(``mcore_bridge.model.modules.gated_delta_net``) re-imports the same symbol for
its context-parallel / sequence-parallel forward, which is the path ms-swift's
``--bridge_backend mcore-bridge`` actually executes for Qwen3.5-style hybrid
models.  On MUSA the FLA path runs through Triton, where torch-kernels'
TileLang GDN kernels are an order of magnitude faster end to end
(see ``torch_kernels/benchmarks/attention/README.md``: 8-10x on MTT S5000).

Both bindings are wrapped with the same dispatcher: a call whose tensors, dtypes
and shapes the TileLang kernels explicitly support (``gdn_dense_supported`` /
``gdn_varlen_supported`` are the operators' own authority) runs on
torch-kernels; every other call is forwarded to FLA unchanged.  FLA therefore
remains the correctness reference and the fallback for everything the MUSA
kernels do not cover (other devices, non-BF16 activations, short sequences,
extra fla-only keyword arguments).
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, Tuple

from .. import _env
from .._engine import AttrPatch

__all__ = ["PATCHES"]

logger = logging.getLogger("megatron_musa_patch")

#: The two modules that bind FLA's chunked gated delta rule for training-time use.
_CORE_GDN = "megatron.core.ssm.gated_delta_net"
_BRIDGE_GDN = "mcore_bridge.model.modules.gated_delta_net"

#: Marks the wrappers this module installs (one dispatcher per binding).
_MARKER = "_megatron_musa_patch_tk_gdn"

#: One warning per reason: a silent per-call demotion would hide a dead
#: fast path, but a per-layer warning would flood the training log.
_warned: set[str] = set()

#: Set on the first successful dispatch, so a training log answers "did the
#: TileLang path run?" without profiler work.
_dispatch_noted = False

#: fla's positional parameters (fla 0.5.x has no ``head_first``; the layout is
#: fixed at ``[B, T, H, D]``, which is also torch-kernels' ``head_first=False``).
_POSITIONAL = ("q", "k", "v", "g", "beta")

#: Keyword arguments the torch-kernels front door understands and this
#: dispatcher is allowed to forward.  ``q/k/v/g/beta`` are included because
#: Megatron passes the decay and beta by keyword; anything else (fla-only
#: activations such as ``use_beta_sigmoid_in_kernel``, CP contexts, ...) stays
#: with FLA.
_KNOWN_KWARGS = frozenset(_POSITIONAL + ("scale", "initial_state", "output_final_state",
                                         "use_qk_l2norm_in_kernel", "cu_seqlens"))


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def _tilelang_stack() -> Optional[Tuple[Callable[..., Any], Callable[..., bool], Callable[..., bool]]]:
    """torch-kernels' GDN front door plus its own shape guards, or ``None``.

    Called from ``replace`` while the target module is being patched, i.e. in a
    process that already runs a real MUSA torch.  Importing torch-kernels here
    (and the TileLang backend it resolves lazily) keeps ``patches/`` importable
    on machines without the stack; a broken install declines the patch instead
    of crashing the first training step.
    """
    try:
        from torch_kernels.attention import gated_delta_net as front_door
        from torch_kernels.attention.gated_delta_net import is_backend_available
        from torch_kernels.attention.tilelang.flash_linear_attention.gdn_shapes import (
            gdn_dense_supported,
            gdn_varlen_supported,
        )
    except Exception as exc:  # ImportError, or a broken native extension
        _warn_once(
            "torch-kernels-missing",
            "megatron-musa-patch: torch-kernels is unavailable (%s: %s); the gated "
            "delta rule keeps running on flash-linear-attention's kernels.",
            type(exc).__name__,
            exc,
        )
        return None
    if not is_backend_available("tilelang"):
        _warn_once(
            "torch-kernels-no-tilelang",
            "megatron-musa-patch: torch-kernels has no tilelang gated-delta-rule "
            "backend registered; keeping flash-linear-attention.",
        )
        return None
    return front_door, gdn_dense_supported, gdn_varlen_supported


def _normalized_call(args: Tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[dict[str, Any]]:
    """fla-shaped arguments as a plain dict, or ``None`` when not translatable.

    The wrapper must stay signature-transparent for FLA: unknown keywords,
    duplicate bindings or missing tensors all take the fallback, because the
    front door would either reject them or attach different semantics.
    """
    if len(args) > len(_POSITIONAL):
        return None
    call = dict(zip(_POSITIONAL, args))
    for key, value in kwargs.items():
        if key not in _KNOWN_KWARGS or key in call:
            return None
        call[key] = value
    if any(call.get(name) is None for name in _POSITIONAL):
        return None
    return call


def _supported_by_tilelang(
    call: dict[str, Any], dense_supported: Callable[..., bool], varlen_supported: Callable[..., bool]
) -> bool:
    """Whether this exact call is inside the TileLang kernels' audited envelope.

    Device/dtype are the kernels' own hard requirements (bf16 q/k/v/beta, fp32
    log-space decay, MUSA tensors); the shape predicates are the operator's
    documented authority ("``gdn_dense_supported`` ... is the single question a
    caller should ask").  Reading ``cu_seqlens`` to the host costs one small
    sync per packed call; the per-sequence two-chunk precondition cannot be
    checked from metadata alone.
    """
    import torch

    tensors = [call[name] for name in _POSITIONAL]
    if not all(torch.is_tensor(t) for t in tensors):
        return False
    q, k, v, g, beta = tensors
    device = q.device
    if device.type != "musa" or any(t.device != device for t in (k, v, g, beta)):
        return False
    if any(t.dtype is not torch.bfloat16 for t in (q, k, v, beta)):
        return False
    if g.dtype not in (torch.float32, torch.float64):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 3 or beta.ndim != 3:
        return False
    if len({t.shape[2] for t in (v, g, beta)}) != 1:  # value heads must agree
        return False

    scale = call.get("scale")
    if scale is not None:
        if isinstance(scale, torch.Tensor) or not isinstance(scale, (int, float)) or not scale > 0:
            return False

    initial_state = call.get("initial_state")
    if initial_state is not None and (
        not torch.is_tensor(initial_state) or initial_state.device != device
    ):
        return False

    heads, key_dim, value_dim = v.shape[2], k.shape[3], v.shape[3]
    cu_seqlens = call.get("cu_seqlens")
    if cu_seqlens is None:
        batch, seq = q.shape[0], q.shape[1]
        # Defensive restatement of the relayout constraint the shape guard
        # leaves implicit: odd head counts outside one tile would fail at
        # kernel compile time, and FLA serves them fine.
        if heads > 8 and heads % 8:
            return False
        return dense_supported(batch, seq, heads, key_dim, value_dim)
    if not torch.is_tensor(cu_seqlens) or cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
        return False
    lengths = [int(n) for n in torch.diff(cu_seqlens.detach()).cpu().tolist()]
    return varlen_supported(lengths, heads, key_dim, value_dim)


def _gated_delta_rule_tilelang(original: Any) -> Any:
    """Same-signature dispatcher: TileLang where supported, FLA everywhere else."""
    if not _env.flag("GDN_TILELANG", True):
        return None
    if original is None or getattr(original, _MARKER, False):
        # FLA missing means Megatron refuses to build GDN layers at all; an
        # already-marked binding must not be stacked a second time.
        return None
    if original.__module__ == __name__:
        return None
    stack = _tilelang_stack()
    if stack is None:
        return None
    front_door, dense_supported, varlen_supported = stack

    @functools.wraps(original)
    def chunk_gated_delta_rule(*args: Any, **kwargs: Any):
        global _dispatch_noted
        call = _normalized_call(args, kwargs)
        if call is None or not _supported_by_tilelang(call, dense_supported, varlen_supported):
            return original(*args, **kwargs)
        if not _dispatch_noted:
            _dispatch_noted = True
            q = call["q"]
            logger.info(
                "megatron-musa-patch: chunked gated delta rule dispatched to "
                "torch-kernels tilelang (B=%d, S=%d, H=%d, D=%d); MEGATRON_MUSA_PATCH_"
                "GDN_TILELANG=0 restores flash-linear-attention.",
                q.shape[0],
                q.shape[1],
                q.shape[2],
                q.shape[3],
            )
        return front_door(
            call["q"],
            call["k"],
            call["v"],
            call["g"],
            call["beta"],
            backend="tilelang",
            scale=call.get("scale"),
            initial_state=call.get("initial_state"),
            output_final_state=call.get("output_final_state", False),
            use_qk_l2norm_in_kernel=call.get("use_qk_l2norm_in_kernel", False),
            cu_seqlens=call.get("cu_seqlens"),
        )

    setattr(chunk_gated_delta_rule, _MARKER, True)
    return chunk_gated_delta_rule


PATCHES = (
    AttrPatch(
        id="megatron.ssm.gated-delta-rule.tilelang",
        target=f"{_CORE_GDN}:chunk_gated_delta_rule",
        replace=_gated_delta_rule_tilelang,
        rationale=(
            "Megatron's GatedDeltaNet computes the chunked gated delta rule with "
            "flash-linear-attention's Triton kernels. On MUSA that Triton path is "
            "the slowest part of Qwen3.5-style hybrid models (24 of 32 layers are "
            "GDN): torch-kernels' TileLang implementation of the identical "
            "operator, signature and [B,T,H,D] layout measures 8-10x end to end "
            "against fla 0.5.2 on MTT S5000 (torch_kernels GDN benchmark)."
        ),
        strategy=(
            "Wrap Megatron's own FLA binding, not the fla package: calls whose "
            "tensors, dtypes and shapes the TileLang kernels document as "
            "supported (MUSA, bf16 q/k/v/beta, fp32 g, gdn_dense_supported / "
            "gdn_varlen_supported) are dispatched to "
            "torch_kernels.attention.gated_delta_net (tilelang backend), which "
            "handles chunk padding, GVA head expansion and autograd itself. "
            "Every other call -- CPU/CUDA tensors, fp16/fp32 activations, short "
            "sequences, fla-only keywords such as use_beta_sigmoid_in_kernel or "
            "cp_context -- is forwarded to FLA with the arguments untouched, so "
            "FLA stays the correctness reference and the fallback. The dispatch "
            "checks shapes per call from tensor metadata only; packed calls "
            "read cu_seqlens once (small sync) to validate the per-sequence "
            "two-chunk precondition. The TileLang kernels JIT-compile on first "
            "use per head count and dense/unpadded specialization (minutes, "
            "cached in ~/.tilelang); on multi-rank runs pre-warm the cache once "
            "with examples/warm_gdn_tilelang.py -- every rank compiling the "
            "same kernels into the shared cache concurrently has crashed runs "
            "(device error / SIGABRT). The TileLang stack is version-bound: "
            "tilelang-musa and the tilelang operator package (torch-kernels) "
            "are built against a specific torch_musa and musa_toolkits "
            "release, so they upgrade only as a matched set with the MUSA "
            "stack, and a stack change also invalidates the ~/.tilelang "
            "kernel cache (drop it and re-warm). GDN_TILELANG=0 declines the "
            "patch and restores upstream's FLA binding entirely."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/ssm/gated_delta_net.py:chunk_gated_delta_rule",
        remove_when=(
            "Remove when flash-linear-attention ships MUSA kernels matching the "
            "TileLang performance, or when Megatron selects a kernel through an "
            "extension seam this package can implement more directly; disable "
            "with MEGATRON_MUSA_PATCH_DISABLE=megatron.ssm.gated-delta-rule.tilelang "
            "and compare the 3-iteration ms-swift Qwen3.5 run and the gdn smoke "
            "before deleting."
        ),
    ),
    AttrPatch(
        id="mcore_bridge.ssm.gated-delta-rule.tilelang",
        target=f"{_BRIDGE_GDN}:chunk_gated_delta_rule",
        rebind_prefixes=("mcore_bridge",),
        replace=_gated_delta_rule_tilelang,
        rationale=(
            "mcore-bridge (the bridge ms-swift selects with --bridge_backend "
            "mcore-bridge) subclasses Megatron's GatedDeltaNet with its own "
            "CP/SP-aware forward and re-imports fla's chunk_gated_delta_rule "
            "into its own module namespace, so patching Megatron's binding alone "
            "never reaches the execution path of a Qwen3.5 mcore-bridge run."
        ),
        strategy=(
            "Install the same dispatcher on the bridge module's binding. It is "
            "an independent patch: selecting only one of the two leaves the "
            "other caller on FLA, and neither inspects the other's state. "
            "Alias repair scans mcore_bridge instead of megatron. The TileLang "
            "stack's version binding and JIT-cache requirements are the ones "
            "recorded on megatron.ssm.gated-delta-rule.tilelang."
        ),
        upstream=(
            "modelscope/mcore-bridge mcore_bridge/model/modules/gated_delta_net.py"
            ":chunk_gated_delta_rule"
        ),
        remove_when=(
            "Remove together with megatron.ssm.gated-delta-rule.tilelang, after "
            "re-verifying that the mcore-bridge forward no longer binds fla's "
            "symbol (or that the MUSA fast path ships where the bridge looks)."
        ),
    ),
)
