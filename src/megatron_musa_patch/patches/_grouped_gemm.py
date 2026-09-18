"""Device-side grouped GEMM reference for MUSA.

The upstream MoE grouped-GEMM path requires the fanshiqing ``grouped_gemm``
CUDA extension, which has no MUSA build. This module provides a reference
``ops.gmm`` built from per-expert ``torch.matmul`` so ``GroupedMLP`` keeps its
weights, checkpoint keys and sharding exactly as upstream. It is a performance
fallback, not a second model implementation: one GEMM per local expert.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from .._engine import AttrPatch

__all__ = ["PATCHES"]

logger = logging.getLogger("megatron_musa_patch")

_UTIL = "megatron.core.transformer.moe.grouped_gemm_util"


def _grouped_gemm_module():
    import sys

    return sys.modules.get(_UTIL)


def _vendor_grouped_gemm_missing() -> bool:
    module = _grouped_gemm_module()
    return module is not None and getattr(module, "grouped_gemm", None) is None


def gmm(a, b, tokens_per_expert, trans_b=False):
    """Grouped matmul: ``y[e] = a[e] @ b[e]`` (or ``b[e].T`` with trans_b).

    ``a`` is [num_tokens, K] with tokens already grouped by expert, ``b`` is
    [num_experts, K, N] (trans_b=False), and ``tokens_per_expert`` gives each
    expert's row count. Only the token counts cross to the CPU; every matmul
    stays on the activations' device and autograd flows through the split and
    the concatenation, so empty experts still receive zero gradients instead
    of losing the edge.
    """
    import torch

    if trans_b:
        b = b.transpose(-2, -1)
    counts = tokens_per_expert.tolist()
    outputs = []
    start = 0
    for count, weight in zip(counts, b):
        end = start + count
        outputs.append(torch.matmul(a[start:end], weight))
        start = end
    if start != a.size(0):
        raise ValueError(f"tokens_per_expert sums to {start} but a has {a.size(0)} rows")
    if not outputs:
        return a.new_zeros((a.size(0), b.size(-1)))
    return torch.cat(outputs, dim=0)


class _GroupedGemmOps:
    """Minimal stand-in for ``grouped_gemm.ops``."""

    gmm = staticmethod(gmm)


def _grouped_gemm_ops(original: Any) -> Any:
    if not _vendor_grouped_gemm_missing():
        return None
    return _GroupedGemmOps()


def _grouped_gemm_is_available(original: Any) -> Any:
    if not _vendor_grouped_gemm_missing():
        return None

    def grouped_gemm_is_available() -> bool:
        return True

    return grouped_gemm_is_available


def _assert_grouped_gemm_is_available(original: Any) -> Any:
    if not _vendor_grouped_gemm_missing():
        return None

    @functools.wraps(original)
    def assert_grouped_gemm_is_available() -> None:

        module = _grouped_gemm_module()
        check = getattr(module, "grouped_gemm_is_available", None)
        if check is not None and not check():
            raise AssertionError(
                "Grouped GEMM is not available. Please run "
                "`pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4`."
            )

    return assert_grouped_gemm_is_available


PATCHES = (
    AttrPatch(
        id="megatron.moe.grouped-gemm.torch-ops",
        target=f"{_UTIL}:ops",
        replace=_grouped_gemm_ops,
        rationale=(
            "GroupedMLP builds through gg.assert_grouped_gemm_is_available() and "
            "runs gg.ops.gmm, but the fanshiqing grouped_gemm extension is a CUDA "
            "build with no MUSA port, so every grouped-expert construction fails "
            "with 'Grouped GEMM is not available' (72 upstream cases in "
            "dist_checkpointing/models/test_moe_experts.py)."
        ),
        strategy=(
            "When the vendor package is absent, provide a minimal ops namespace "
            "whose gmm splits the activations by tokens_per_expert and runs one "
            "torch.matmul per local expert on the device, honouring trans_b and "
            "concatenating in expert order. Only the token counts cross to the "
            "CPU; autograd flows through the split/cat so empty experts keep "
            "zero (not None) weight gradients, and weight1/weight2 layouts, "
            "checkpoint keys and expert sharding are untouched. Declines when a "
            "real grouped_gemm is importable."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/transformer/moe/grouped_gemm_util.py; "
            "fanshiqing/grouped_gemm ops.gmm"
        ),
        remove_when=(
            "Remove when a validated MUSA grouped_gemm (vendor or community) is "
            "installable: disable this patch id and re-run "
            "dist_checkpointing/models/test_moe_experts.py plus a GroupedMLP "
            "forward/backward comparison; delete only if the vendor path passes."
        ),
    ),
    AttrPatch(
        id="megatron.moe.grouped-gemm.available-flag",
        target=f"{_UTIL}:grouped_gemm_is_available",
        replace=_grouped_gemm_is_available,
        requires=("megatron.moe.grouped-gemm.torch-ops",),
        rationale=(
            "Consumers gate the grouped path on grouped_gemm_is_available(); "
            "with the reference ops installed the flag must tell the truth "
            "about the fallback instead of reporting a missing vendor package."
        ),
        strategy=(
            "Report True only while the torch-ops fallback is actually applied; "
            "the requires declaration keeps the flag from ever advertising a "
            "fallback that is not installed. Declines when the vendor package "
            "is present."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/moe/grouped_gemm_util.py",
        remove_when=(
            "Remove together with megatron.moe.grouped-gemm.torch-ops when a "
            "validated MUSA grouped_gemm is available."
        ),
    ),
    AttrPatch(
        id="megatron.moe.grouped-gemm.assert-noop",
        target=f"{_UTIL}:assert_grouped_gemm_is_available",
        replace=_assert_grouped_gemm_is_available,
        requires=("megatron.moe.grouped-gemm.torch-ops",),
        rationale=(
            "GroupedMLP.__init__ asserts availability before building; with the "
            "fallback installed the assertion must consult the patched flag "
            "rather than fail on the absent vendor package."
        ),
        strategy=(
            "Re-check the module's (patched) grouped_gemm_is_available at call "
            "time and keep the upstream error message for the genuinely "
            "unavailable case; becomes a no-op only while the fallback is live."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/moe/grouped_gemm_util.py",
        remove_when=(
            "Remove together with megatron.moe.grouped-gemm.torch-ops when a "
            "validated MUSA grouped_gemm is available."
        ),
    ),
)
