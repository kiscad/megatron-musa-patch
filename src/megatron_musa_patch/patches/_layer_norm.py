"""PyTorch normalization fallback for the affected apex/TE integration.

An importable apex package does not establish that its fused_layer_norm_cuda
extension can run on MUSA. The block-level TE norm also hit an allocateSpace
assertion in the bring-up stack. These are version-specific failures, not a
claim that every MUSA apex/TE build is unsupported. TE's fused norm-linear is
patched separately: replacing standalone norms cannot reach its internal norm.
Functional PyTorch ops keep
the module/parameter contract, but kernel choice, rounding and performance can
differ from fused implementations and must be checked on stack upgrades.
"""

from __future__ import annotations

import numbers
from typing import Any

from .. import _env
from .._engine import AttrPatch

__all__ = ["PATCHES"]


def _pure_torch_layer_norm(original: Any) -> Any:
    """Build a fallback without importing torch until the target is available."""
    import torch

    class FusedLayerNorm(torch.nn.Module):
        """Match Megatron's norm constructor and optimizer parameter markers.

        Config is authoritative, as in upstream FusedLayerNorm/TENorm. In
        particular TransformerBlock passes config.normalization but omits the
        normalization keyword, so using that keyword's default corrupts RMSNorm
        models. Legacy constructor keywords remain accepted for compatibility.
        """

        _megatron_musa_patch_fallback = True

        def __init__(
            self,
            config,
            hidden_size,
            eps: float = 1e-5,
            persist_layer_norm: bool = True,
            zero_centered_gamma: bool = False,
            normalization: str = "LayerNorm",
        ):
            super().__init__()
            self.config = config
            self.zero_centered_gamma = config.layernorm_zero_centered_gamma
            self.normalization = getattr(config, "normalization", normalization)
            if self.normalization not in {"LayerNorm", "RMSNorm"}:
                raise ValueError(f"Unsupported normalization: {self.normalization!r}")
            if isinstance(hidden_size, numbers.Integral):
                hidden_size = (hidden_size,)
            self.hidden_size = torch.Size(hidden_size)
            self.eps = eps
            # This fallback never selects apex's persistent FastLayerNorm kernel.
            self.persist_layer_norm = False

            self.weight = torch.nn.Parameter(torch.empty(self.hidden_size))
            if self.normalization == "LayerNorm":
                self.bias = torch.nn.Parameter(torch.empty(self.hidden_size))
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()

            self.sequence_parallel = config.sequence_parallel
            self.weight.sequence_parallel = self.sequence_parallel
            if self.bias is not None:
                self.bias.sequence_parallel = self.sequence_parallel

        def reset_parameters(self) -> None:
            torch.nn.init.constant_(self.weight, 0 if self.zero_centered_gamma else 1)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

        def forward(self, input):
            weight = self.weight + 1 if self.zero_centered_gamma else self.weight
            if self.normalization == "RMSNorm":
                return torch.nn.functional.rms_norm(input, self.hidden_size, weight, self.eps)
            return torch.nn.functional.layer_norm(
                input, self.hidden_size, weight, self.bias, self.eps
            )

    return FusedLayerNorm


def _using_torch_fallback() -> bool:
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    return bool(getattr(FusedLayerNorm, "_megatron_musa_patch_fallback", False))


def _fallback_available_flag(original: Any) -> Any:
    # ONLY/DISABLE may select flags independently; never advertise a fallback
    # if its class replacement was not applied.
    return True if _using_torch_fallback() else None


def _persistent_available_flag(original: Any) -> Any:
    return False if _using_torch_fallback() else None


def _block_layer_norm_impl(original: Any) -> Any:
    """Use a functional norm even when the separate local-class patch is off."""
    if _env.value("BLOCK_LAYERNORM", "local").strip().lower() in {"upstream", "te", "keep"}:
        return None
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    if getattr(FusedLayerNorm, "_megatron_musa_patch_fallback", False):
        return FusedLayerNorm
    return _pure_torch_layer_norm(FusedLayerNorm)


def _unfused_te_layer_norm_linear(original: Any) -> Any:
    """Unfuse LayerNorm only; construct native TE modules for RMSNorm.

    Register norm parameters on the linear itself to retain the fused module's
    checkpoint names and replicated (rather than column-sharded) norm weights.
    """
    if _env.flag("TE_FUSED_LAYERNORM", False):
        return None
    from megatron.core.extensions.transformer_engine import HAVE_TE

    # Megatron uses MagicMock-backed placeholders when TE imports fail. They
    # are not usable module classes and have no sharded_state_dict to reuse.
    # Keep upstream's optional-dependency behavior in that case.
    if not HAVE_TE:
        return None
    import torch
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

    class _NormLinearDispatchMeta(type(TEColumnParallelLinear)):
        def __instancecheck__(cls, instance):
            # Megatron's FP8 parameter gathering identifies column-parallel
            # modules with isinstance(..., TELayerNormColumnParallelLinear).
            return super().__instancecheck__(instance) or isinstance(instance, original)

    class TELayerNormColumnParallelLinear(
        TEColumnParallelLinear, metaclass=_NormLinearDispatchMeta
    ):
        _megatron_musa_patch_fallback = True

        def __new__(cls, *args, **kwargs):
            # Dispatch once at construction. Returning the original instance
            # keeps RMSNorm's fused forward/backward and adds no runtime wrapper.
            # Other normalization values retain upstream validation as well.
            # deepcopy reconstructs modules via __new__(cls) without config.
            config = kwargs.get("config")
            if config is not None and config.normalization != "LayerNorm":
                return original(*args, **kwargs)
            return super().__new__(cls)

        def __init__(self, input_size, output_size, *, config, **kwargs):
            if kwargs.get("is_expert", False):
                raise ValueError("Transformer Engine norm-linear layers do not support MoE")
            super().__init__(input_size, output_size, config=config, **kwargs)
            self.normalization = "LayerNorm"
            self.eps = config.layernorm_epsilon
            self.zero_centered_gamma = config.layernorm_zero_centered_gamma
            self.layer_norm_weight = torch.nn.Parameter(torch.full(
                (input_size,), 0.0 if self.zero_centered_gamma else 1.0,
                dtype=config.params_dtype, device=self.weight.device,
            ))
            self.layer_norm_bias = torch.nn.Parameter(torch.zeros(
                input_size, dtype=config.params_dtype, device=self.weight.device,
            ))
            for parameter in (self.layer_norm_weight, self.layer_norm_bias):
                if parameter is not None:
                    parameter.sequence_parallel = config.sequence_parallel
                    parameter.allreduce = True

        def forward(self, x):
            # TE casts norm inputs/parameters to the activation dtype. Under
            # autocast x can differ from params_dtype; casts retain autograd.
            weight = self.layer_norm_weight.to(x.dtype)
            weight = weight + 1 if self.zero_centered_gamma else weight
            normalized = torch.nn.functional.layer_norm(
                x, (x.shape[-1],), weight, self.layer_norm_bias.to(x.dtype), self.eps,
            )
            return super().forward(normalized)

        # The upstream fused wrapper already handles metadata defaults and
        # shards only weight/bias, leaving layer_norm_* replicated.
        sharded_state_dict = original.sharded_state_dict

    return TELayerNormColumnParallelLinear


PATCHES = (
    AttrPatch(
        id="megatron.te.layer-norm-linear.unfused",
        target="megatron.core.extensions.transformer_engine:TELayerNormColumnParallelLinear",
        replace=_unfused_te_layer_norm_linear,
        rationale=(
            "TE GPT specs fuse QKV/FC1 normalization inside LayerNormLinear; "
            "neither local FusedLayerNorm nor block LayerNormImpl covers its "
            "MUSA allocateSpace assertion. The affected TE port comments out "
            "the LayerNorm workspace query/kernel but still allocates the "
            "uninitialized workspace shape."
        ),
        strategy=(
            "For LayerNorm only, use PyTorch normalization followed by TEColumnParallelLinear, "
            "retaining FP8, TP, norm parameter names and sharding. "
            "RMSNorm constructs the original TE fused module without a forward wrapper. "
            "TE_FUSED_LAYERNORM=1 restores upstream for upgrade validation."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/extensions/transformer_engine.py",
        remove_when=(
            "Remove after upgraded MT-TransformerEngine fused LayerNormLinear "
            "passes LayerNorm FP8 forward/backward, zero-centered gamma, "
            "checkpoint and multi-rank sequence-parallel tests on MUSA."
        ),
    ),
    AttrPatch(
        id="megatron.fusions.fused-layer-norm.pure-torch",
        target="megatron.core.fusions.fused_layer_norm:FusedLayerNorm",
        replace=_pure_torch_layer_norm,
        rationale=(
            "In the affected stack apex imports but its fused_layer_norm_cuda "
            "extension is unavailable, so Megatron's import probe selects a class "
            "that fails on forward instead of a usable backend fallback."
        ),
        strategy=(
            "Replace only the norm class with functional PyTorch LayerNorm/RMSNorm "
            "selected from config, preserving gamma convention, state-dict keys and "
            "sequence-parallel markers; do not promise fused-kernel numerical identity."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/fusions/fused_layer_norm.py",
        remove_when=(
            "Review on apex/MT-TransformerEngine, torch_musa and Megatron upgrades; "
            "remove when upstream selects a usable norm and forward/backward, "
            "low-precision, zero-centered-gamma and checkpoint parity tests pass "
            "on MUSA. Benchmark the replacement before switching."
        ),
    ),
    AttrPatch(
        id="megatron.fusions.fused-layer-norm.have-apex-flag",
        requires=("megatron.fusions.fused-layer-norm.pure-torch",),
        target="megatron.core.fusions.fused_layer_norm:HAVE_FUSED_LAYER_NORM",
        replace=_fallback_available_flag,
        rationale=(
            "Consumers such as bert_lm_head use HAVE_FUSED_LAYER_NORM to decide "
            "whether the local norm class is usable, even after class replacement."
        ),
        strategy=(
            "Advertise the installed fallback as usable; this flag does not "
            "certify that an apex fused kernel is available. Skip if selective "
            "patch filtering left the local class unreplaced."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/fusions/fused_layer_norm.py",
        remove_when=(
            "Remove together with the pure-torch norm patch; on Megatron upgrades "
            "review flag consumers, including the BERT head's norm construction."
        ),
    ),
    AttrPatch(
        id="megatron.fusions.persist-layer-norm.disable",
        requires=("megatron.fusions.fused-layer-norm.pure-torch",),
        target="megatron.core.fusions.fused_layer_norm:HAVE_PERSIST_LAYER_NORM",
        replace=_persistent_available_flag,
        rationale=(
            "The functional fallback does not execute apex's FastLayerNormFN; "
            "leaving its persistent-kernel flag enabled misdescribes the selected path."
        ),
        strategy=(
            "Set the persistent-kernel availability flag false only while the "
            "functional fallback is installed; otherwise leave upstream unchanged."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/fusions/fused_layer_norm.py",
        remove_when=(
            "Remove with the functional fallback after a usable native norm is "
            "validated; review any new consumers of this flag on Megatron upgrades."
        ),
    ),
    AttrPatch(
        id="megatron.transformer-block.layer-norm.impl-local",
        target="megatron.core.transformer.transformer_block:LayerNormImpl",
        replace=_block_layer_norm_impl,
        rationale=(
            "TransformerBlock chooses TENorm whenever TE imports, including for "
            "a local layer spec. The affected TE MUSA norm path aborted in "
            "allocateSpace; importability alone did not prove runtime usability."
        ),
        strategy=(
            "Bind the block's default norm to a functional fallback, reusing the "
            "patched local class when present but not requiring its patch. Explicit "
            "block submodule specs are unchanged; BLOCK_LAYERNORM=upstream keeps "
            "the upstream default for upgrade testing."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/transformer_block.py",
        remove_when=(
            "Review on MT-TransformerEngine/torch_musa and Megatron upgrades; remove "
            "after BLOCK_LAYERNORM=upstream passes LayerNorm and RMSNorm final-block "
            "forward/backward, checkpoint and multi-rank sequence-parallel tests."
        ),
    ),
)
