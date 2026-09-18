"""Install the ``torch.cuda`` compatibility layer, but only for Megatron runs.

The trigger is ``megatron`` rather than ``torch`` on purpose.  ``torch_musa``,
MT-TransformerEngine and ``vllm_musa`` already install overlapping shims at
``import torch`` time, and a package that silently rewrites ``torch.cuda`` for
*every* Python process in the environment -- including ``pip``, ``pytest`` and
every dataloader worker -- is not a patch library anyone wants installed.  The
watcher fires this hook the moment ``megatron`` is first imported, which is
still before a single line of Megatron executes. Its undo restores this
project's overrides only: imported torch_musa/torchada side effects remain.
"""

from __future__ import annotations

from .._engine import HookPatch
from ..backends import torch_cuda

__all__ = ["PATCHES"]


def _install_torch_cuda_compat() -> bool:
    # Do not acquire undo ownership of a layer installed by another caller.
    if torch_cuda.is_applied():
        return False
    torch_cuda.apply()
    return True


PATCHES = (
    HookPatch(
        id="torch.cuda.compat-layer",
        trigger="megatron",
        run=_install_torch_cuda_compat,
        undo=torch_cuda.unapply,
        strategy=(
            "Delegate general CUDA-to-MUSA adaptation to torchada, then apply "
            "identity-tracked availability, tensor type, graph class, and subclass "
            "transfer overrides; undo only project-owned bindings, not external "
            "adapter side effects or another caller's active layer."
        ),
        rationale=(
            "Megatron requires working CUDA APIs on MUSA, a truthful availability "
            "probe, CUDA-spelled Tensor.type() queries and graph classes, and "
            "device transfers that preserve TransformerEngine tensor subclasses."
        ),
        upstream="torchada; torch_musa/core/tensor_attrs.py; Megatron CUDA consumers",
        remove_when=(
            "The supported torchada/torch_musa stack provides CUDA adaptation plus "
            "MUSA availability, CUDA tensor type names, the graphs.CUDAGraph alias, "
            "and subclass-safe transfers with transfer options intact; verify the "
            "contracts in tests/test_torch_cuda.py without these overrides."
        ),
    ),
)
