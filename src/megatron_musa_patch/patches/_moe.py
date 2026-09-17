"""Module-local MoE adapters for the MUSA stack.

These patches replace names inside ``megatron.core.transformer.moe.moe_utils``
only. The forwarding ``torch`` namespace adapts exactly the kernels the MUSA
stack cannot run (FP64 top-k) and forwards everything else unchanged, so the
upstream routing functions, group-limited top-k and router replay keep their
semantics.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from .._engine import AttrPatch

__all__ = ["PATCHES"]

logger = logging.getLogger("megatron_musa_patch")


def _musa_live() -> bool:
    import torch

    musa = getattr(torch, "musa", None)
    available = getattr(musa, "is_available", None)
    return callable(available) and bool(available())


def _fp64_topk(torch, input, k, dim, largest, sorted):
    """Reference top-k for FP64: MUSA's MuDNN TopK has no DOUBLE kernel.

    Take the indices at the input's own precision on CPU (never demote to
    FP32: near-tied scores could reorder), move the discrete indices back to
    the device and gather from the original tensor, so gradients keep flowing
    through it -- indices are discrete and need no gradient.
    """
    if dim is None:
        dim = -1  # torch.topk's documented default
    _, cpu_indices = torch.topk(
        input.detach().cpu(), k=k, dim=dim, largest=largest, sorted=sorted
    )
    indices = cpu_indices.to(input.device)
    return input.gather(dim, indices), indices


class _MoeTorchProxy:
    """``torch`` namespace for moe_utils that adapts only broken kernels."""

    def __init__(self, torch):
        self._torch = torch

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def topk(self, input, k, dim=None, largest=True, sorted=True, *, out=None):
        if (
            input.dtype == self._torch.float64
            and input.device.type == "musa"
            and _musa_live()
        ):
            logger.debug(
                "MoE topk FP64 reference path: shape=%s k=%s dim=%s",
                tuple(input.shape), k, dim,
            )
            return _fp64_topk(self._torch, input, k, dim, largest, sorted)
        if out is not None:
            return self._torch.topk(
                input, k, dim=dim, largest=largest, sorted=sorted, out=out
            )
        return self._torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)


def _moe_torch_namespace(original: Any) -> Any:
    return _MoeTorchProxy(original)


#: dtypes the MT-TE moe permutation kernel accepts. The kernel's own error text
#: says "Invalid type for 16 bit", but the measured constraint is the opposite:
#: float32 and float64 fail while float16/bfloat16 pass (muDNN
#: ``permutation_mask.mu`` rejects the 32/64-bit key). FP8 subclasses are
#: untested and deliberately left on the fused path.
_FUSED_PERMUTE_BROKEN_DTYPES: tuple[Any, ...] | None = None


def _fused_permute_unsupported(tensor) -> bool:
    """True when the fused permute kernel cannot serve this tensor's dtype."""
    global _FUSED_PERMUTE_BROKEN_DTYPES
    if _FUSED_PERMUTE_BROKEN_DTYPES is None:
        import torch

        _FUSED_PERMUTE_BROKEN_DTYPES = (torch.float32, torch.float64)
    if not _musa_live() or tensor.device.type != "musa":
        return False
    return tensor.dtype in _FUSED_PERMUTE_BROKEN_DTYPES


def _moe_permute_unfused(original: Any) -> Any:
    """Demote the fused MoE permute to upstream's reference implementation.

    ``fused_permute`` (TE ``moe_permute``) aborts in ``nvte_permute_mask`` for
    float32/float64 tokens. Upstream's ``fused=False`` branch is the reference
    implementation with identical semantics, and the paired unpermute patch
    keeps both ends on the same index format.
    """

    @functools.wraps(original)
    def permute(tokens, routing_map, probs=None, num_out_tokens=None, fused=False,
                drop_and_pad=False):
        demote = fused and _fused_permute_unsupported(tokens)
        if demote:
            logger.debug(
                "MoE permute unfused fallback: dtype=%s shape=%s",
                tokens.dtype, tuple(tokens.shape),
            )
        return original(
            tokens, routing_map, probs=probs, num_out_tokens=num_out_tokens,
            fused=False if demote else fused, drop_and_pad=drop_and_pad,
        )

    return permute


def _moe_unpermute_unfused(original: Any) -> Any:
    """Demote the fused MoE unpermute; pairs with the permute fallback."""

    @functools.wraps(original)
    def unpermute(permuted_tokens, sorted_indices, restore_shape, probs=None,
                  routing_map=None, fused=False, drop_and_pad=False):
        demote = fused and _fused_permute_unsupported(permuted_tokens)
        if demote:
            logger.debug(
                "MoE unpermute unfused fallback: dtype=%s", permuted_tokens.dtype
            )
        return original(
            permuted_tokens, sorted_indices, restore_shape, probs=probs,
            routing_map=routing_map, fused=False if demote else fused,
            drop_and_pad=drop_and_pad,
        )

    return unpermute


PATCHES = (
    AttrPatch(
        id="megatron.moe.topk.fp64-reference",
        target="megatron.core.transformer.moe.moe_utils:torch",
        replace=_moe_torch_namespace,
        rationale=(
            "MoE routing calls torch.topk on router scores. With "
            "moe_router_dtype='fp64' the scores are float64 and MuDNN's TopK "
            "kernel rejects that type ('TopkOut MUDNN failed in: Run'; muDNN "
            "logs 'Unsupported in data type: DOUBLE'), so every routing step of "
            "the fp64 discrepancy/aux-loss tests fails. The device-side "
            "alternatives (sort, argsort, max-with-indices) fail for float64 "
            "too, so the fallback needs a host-side index pass. Verified on "
            "torch_musa 2.7.1 / muDNN v3107 with Megatron core_v0.16.1."
        ),
        strategy=(
            "Bind moe_utils's module-local torch global to a forwarding proxy "
            "that adapts only topk: for float64 MUSA inputs it takes indices at "
            "the same precision on CPU, moves the discrete indices back and "
            "gathers values from the original tensor so gradients still flow "
            "through it. dim/largest/sorted and the (values, indices) contract "
            "are preserved, out= is delegated, and every other torch attribute "
            "forwards untouched. Non-float64 inputs and non-MUSA runtimes use "
            "the native kernel."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py "
            "(_compute_topk, group_limited_topk, capacity top-k)"
        ),
        remove_when=(
            "Remove when MuDNN TopK supports float64 (or the router stops "
            "producing fp64 scores on MUSA): disable this patch id and re-run "
            "transformer/moe/test_moe_layer_discrepancy.py, test_aux_loss.py and "
            "the fp64 routing cases; delete only if the native path passes."
        ),
    ),
    AttrPatch(
        id="megatron.moe.permutation.unfused-musa",
        target="megatron.core.transformer.moe.moe_utils:permute",
        replace=_moe_permute_unfused,
        rationale=(
            "The token dispatcher's fused permute calls TE's moe_permute, whose "
            "MUSA kernel (nvte_permute_mask) aborts for float32 tokens with "
            "'Invalid type for 16 bit' despite the 16-bit wording: measured on "
            "torch_musa 2.7.1, float32/float64 fail while float16/bfloat16 pass. "
            "This breaks every MoE a2a-token-dispatcher case that runs with "
            "moe_permute_fusion enabled and non-16-bit activations."
        ),
        strategy=(
            "Demote exactly the fused branch to upstream's own reference "
            "implementation by re-calling the original with fused=False when the "
            "token dtype is a confirmed-broken one on a live MUSA device. "
            "probs/num_out_tokens/drop_and_pad and the (permuted tokens, "
            "permuted probs, sorted indices) contract are unchanged, gradients "
            "flow through upstream's scatter path, and float16/bfloat16/FP8 "
            "inputs keep the fused kernel. Requires the paired unpermute patch "
            "so both ends use the same index format."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py:permute",
        remove_when=(
            "Remove when the MUSA TE permutation kernel supports 32-bit inputs: "
            "disable both permutation patch ids and re-run "
            "transformer/moe/test_a2a_token_dispatcher.py forward/backward; "
            "delete only if the fused path passes."
        ),
    ),
    AttrPatch(
        id="megatron.moe.unpermutation.unfused-musa",
        target="megatron.core.transformer.moe.moe_utils:unpermute",
        replace=_moe_unpermute_unfused,
        requires=("megatron.moe.permutation.unfused-musa",),
        rationale=(
            "The fused unpermute uses the same MUSA permutation kernel family as "
            "the fused permute. Restoring tokens must stay on the same index "
            "format as the permute that produced them: mixing a non-fused "
            "permute with the fused unpermute (or vice versa) silently "
            "scrambles token order."
        ),
        strategy=(
            "Demote the fused branch to upstream's reference implementation under "
            "the same dtype condition as the permute patch, keeping probs, "
            "routing_map, restore_shape and drop_and_pad semantics. Declared as "
            "requiring the permute companion so ONLY/DISABLE can never activate "
            "one end alone."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py:unpermute",
        remove_when=(
            "Remove together with megatron.moe.permutation.unfused-musa after the "
            "MUSA TE permutation kernel supports 32-bit inputs and the a2a "
            "dispatcher forward/backward passes on the fused path."
        ),
    ),
)
