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
            "identity-tracked availability, tensor type, graph class, subclass "
            "transfer and allocator OOM-observer overrides, and mirror PyTorch's "
            "standard fp32-matmul TF32 switch into torch_musa's own flag; undo only project-owned bindings, not external "
            "adapter side effects or another caller's active layer."
        ),
        rationale=(
            "Megatron requires working CUDA APIs on MUSA, a truthful availability "
            "probe, CUDA-spelled Tensor.type() queries and graph classes, and "
            "device transfers that preserve TransformerEngine tensor subclasses. "
            "Megatron-Bridge attaches an OOM snapshot observer through "
            "torch._C._cuda_attach_out_of_memory_observer, which the MUSA build and "
            "torchada do not provide (5 Bridge 0.4.2 unit tests, 2026-09-21 sweep). "
            "torch_musa ignores torch.backends.cuda.matmul.allow_tf32 and "
            "set_float32_matmul_precision and defaults its own TF32 flag on, so fp32 "
            "matmuls silently lose precision (Qwen3-VL RoPE angles off by 0.35 rad "
            "at 1024 positions); CUDA defaults to full precision."
        ),
        upstream="torchada; torch_musa/core/tensor_attrs.py; Megatron CUDA consumers",
        remove_when=(
            "The supported torchada/torch_musa stack provides CUDA adaptation plus "
            "MUSA availability, CUDA tensor type names, the graphs.CUDAGraph alias, "
            "subclass-safe transfers with transfer options intact, and the "
            "torch._C._cuda_attach_out_of_memory_observer binding, and honours "
            "PyTorch's standard TF32 switches with CUDA's default; verify the "
            "contracts in tests/test_torch_cuda.py without these overrides."
        ),
    ),
)
