"""Capability dispatch for MT-TE's hard-coded MUSA flash path.

Reference: Megatron core_v0.16.1 TEDotProductAttention; TransformerEngine
``UnfusedDotProductAttention`` and ``get_full_mask``. No attention math is
implemented here: inputs inside the MUSA flash kernel's capability window run
the native MT-TE flash path; every other eligible input runs TE's own unfused
backend; segmented THD is sliced into per-sequence vendor calls. Only ordinary
tensors, CP=1, vanilla softmax and unrestricted windows are handled.
"""

from __future__ import annotations

import functools
from typing import Any

from .. import _compat
from .._engine import AttrPatch
from ..backends import musa_available as _musa_live

__all__ = ["PATCHES"]
_TEDPA = "megatron.core.extensions.transformer_engine:TEDotProductAttention.forward"

#: MUSA flash SDPA kernel window (torch.ops.aten._scaled_dot_product_attention_
#: flash_musa; MT-TE gates on the same values in flash_attn_varlen_func_variance).
_MIN_FLASH_DIM = 64
_MAX_FLASH_DIM = 192


def _effective_dropout(self) -> float:
    """Dropout the flash kernel would have to apply on this call."""
    training = bool(getattr(self, "training", False))
    return float(getattr(self, "attention_dropout", 0.0) or 0.0) if training else 0.0


def _flash_kernel_unsupported(self, query, key=None, value=None) -> bool:
    """True when the native MT-TE MUSA flash kernel cannot run these inputs.

    The kernel path asserts FP16/BF16 (or Float8Tensor) inputs, supports head
    dimensions 64..192, and traps on the device when ``dropout_p > 0`` is
    passed (flash_attn's own wrapper asserts dropout is unsupported yet).
    Measured on torch_musa 2.7.1 / MT-TE 2.0.0 / flash-attn 2.6.3.
    """
    import torch

    if not _musa_live() or query.device.type != "musa":
        return False
    tensors = [t for t in (query, key, value) if t is not None]
    return (
        any(t.dtype not in (torch.float16, torch.bfloat16) for t in tensors)
        or any(not _MIN_FLASH_DIM <= t.size(-1) <= _MAX_FLASH_DIM for t in tensors)
        or _effective_dropout(self) != 0.0
    )


def _mask_type_name(attn_mask_type) -> str:
    return (getattr(attn_mask_type, "name", None) or str(attn_mask_type)).replace(",", "_")


def _attn_mask_type_value(name):
    """Rebuild an enum-like ``attn_mask_type`` for calls delegated upstream.

    Megatron's ``TEDotProductAttention.forward`` reads ``attn_mask_type.name``,
    so synthesized mask types must carry the caller's own enum value.
    """
    try:
        from megatron.core.transformer.enums import AttnMaskType

        return getattr(AttnMaskType, name)
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(name=name)


def _dispatch_contract(self, query, key, value, packed, num_splits):
    """Do not bypass upstream validation/parallel or quantization protocols."""
    import torch

    if any(type(t) is not torch.Tensor for t in (query, key, value)):
        return False
    if any(
        t.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        for t in (query, key, value)
    ):
        return False
    config = getattr(self, "config", None)
    if any(
        getattr(config, name, False)
        for name in (
            "fp8_dot_product_attention",
            "fp8_multi_head_attention",
            "qk_clip",
            "log_max_attention_logit",
            "apply_query_key_layer_scaling",
        )
    ):
        return False
    if getattr(config, "softmax_type", "vanilla") != "vanilla":
        return False
    if num_splits is not None or getattr(self, "num_splits", None) is not None:
        return False
    window = getattr(self, "window_size", None)
    if window is not None and tuple(window) not in ((-1, -1), (-1, 0)):
        return False
    group = getattr(self, "cp_group", None)
    if packed is not None:
        dynamic_group = getattr(packed, "cp_group", None)
        local_size = getattr(packed, "local_cp_size", None)
        if dynamic_group is not None:
            group = dynamic_group
        elif local_size == 1:
            group = None
        elif local_size is not None:
            return False
    if group is not None and (isinstance(group, (list, tuple)) or group.size() > 1):
        return False
    # The constructor may expose CP only through the config on some forks.
    if group is None and getattr(config, "context_parallel_size", 1) > 1:
        if packed is None or getattr(packed, "local_cp_size", None) != 1:
            return False
    return True


def _te_padding_mask(attention_mask, sq, sk, attention_type="self"):
    """Normalize Megatron padding masks to the shapes TE's ``get_full_mask`` consumes.

    TE derives the combined query|key mask from a single ``[b, 1, 1, sk]``
    tensor for self-attention, or from a tuple of ``[b, 1, 1, s]`` tensors for
    cross-attention. Returns ``None`` for masks this adapter does not claim.
    """
    import torch

    if attention_mask is None:
        return None

    def shaped(mask):
        if mask.dim() == 2:
            return mask[:, None, None, :]
        if mask.dim() == 3 and mask.shape[1] == 1:
            return mask[:, None, :, :]
        if mask.dim() == 4 and mask.shape[1:3] == (1, 1):
            return mask
        return None

    masks = attention_mask if isinstance(attention_mask, tuple) else (attention_mask,)
    if len(masks) not in (1, 2) or any(
        not isinstance(m, torch.Tensor) or m.dtype != torch.bool for m in masks
    ):
        return None
    shaped_masks = tuple(shaped(m) for m in masks)
    if any(m is None for m in shaped_masks):
        return None
    if attention_type == "cross":
        if len(shaped_masks) != 2:
            return None
        q_mask, k_mask = shaped_masks
        if q_mask.shape[-1] != sq or k_mask.shape[-1] != sk or q_mask.shape[0] != k_mask.shape[0]:
            return None
        return q_mask, k_mask
    if attention_type != "self" or sq != sk:
        return None
    if any(m.shape[-1] != sk for m in shaped_masks):
        return None
    if len(shaped_masks) == 2 and not torch.equal(*shaped_masks):
        # One self-attention token mask cannot represent distinct Q/K masks.
        # OR-ing the vectors would silently mask additional valid queries/keys.
        return None
    return shaped_masks[0]


def _unfused_forward(self, query, key, value, attention_mask, mask_name, attention_bias, layout):
    """Run TE's own ``UnfusedDotProductAttention`` backend, or give up with None."""
    backend = getattr(self, "unfused_attention", None)
    if backend is None:
        return None
    if layout == "bshd":
        sq, sk = query.shape[1], key.shape[1]
    else:
        sq, sk = query.shape[0], key.shape[0]
    mask = None
    if "padding" in mask_name:
        mask = _te_padding_mask(attention_mask, sq, sk, getattr(backend, "attention_type", "self"))
        if mask is None:
            return None
    elif attention_mask is not None:
        mask = attention_mask
    return backend(
        query,
        key,
        value,
        qkv_layout=f"{layout}_{layout}_{layout}",
        attn_mask_type=mask_name,
        attention_mask=mask,
        window_size=getattr(self, "window_size", None),
        core_attention_bias_type="post_scale_bias" if attention_bias is not None else "no_bias",
        core_attention_bias=attention_bias,
    )


def _dense_dispatch(
    self,
    query,
    key,
    value,
    attention_mask,
    attn_mask_type,
    mask_name,
    attention_bias,
    layout,
    original,
):
    """Flash when the kernel supports the inputs, TE's unfused backend otherwise.

    ``attn_mask_type`` is forwarded to ``original`` untouched (it must remain
    the caller's enum value); ``mask_name`` is the string form used internally.
    """
    if not _flash_kernel_unsupported(self, query, key, value):
        return original(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=None,
            num_splits=None,
        )
    out = _unfused_forward(
        self, query, key, value, attention_mask, mask_name, attention_bias, layout
    )
    if out is not None:
        return out
    # Nothing claims these inputs: surface the vendor error instead of guessing.
    return original(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias=attention_bias,
        packed_seq_params=None,
        num_splits=None,
    )


def _packed_spans(cumulative, padded, total):
    """Read only O(batch) metadata; keep activations and gradients on device."""
    lengths = cumulative.detach().cpu().tolist()
    offsets = lengths if padded is None else padded.detach().cpu().tolist()
    if (
        len(lengths) < 2
        or len(offsets) != len(lengths)
        or lengths[0] != 0
        or offsets[0] != 0
        or offsets[-1] != total
    ):
        raise ValueError("Invalid packed cumulative lengths/physical offsets")
    spans = []
    for i in range(len(lengths) - 1):
        count, capacity = lengths[i + 1] - lengths[i], offsets[i + 1] - offsets[i]
        if count < 0 or capacity < count:
            raise ValueError("Packed lengths must be monotonic and fit padded storage")
        spans.append((offsets[i], count, capacity))
    return spans


def _packed_forward(self, query, key, value, mask_name, packed, original):
    """Segmented THD via per-sequence vendor calls.

    The MUSA flash wrapper drops ``cu_seqlens`` and yields NaN once physical
    offsets are padded, so each sequence is dispatched separately in ``sbhd``
    layout through the same dense dispatch (native flash or TE's unfused
    backend). Empty sequences keep zero-valued outputs with gradient edges.
    """
    import torch

    q_spans = _packed_spans(
        packed.cu_seqlens_q, getattr(packed, "cu_seqlens_q_padded", None), query.shape[0]
    )
    k_spans = _packed_spans(
        packed.cu_seqlens_kv, getattr(packed, "cu_seqlens_kv_padded", None), key.shape[0]
    )
    if len(q_spans) != len(k_spans) or key.shape[0] != value.shape[0]:
        raise ValueError("Packed query/key/value sequences must match")
    # Spans hold only valid tokens, so the padding qualifier is meaningless
    # per span; retain causal alignment, including rectangular bottom-right.
    span_mask = mask_name.removeprefix("padding_")
    if span_mask == "padding":
        span_mask = "no_mask"
    outputs = []
    for (qs, nq, capacity), (ks, nk, _) in zip(q_spans, k_spans):
        # clone() detaches the span from the packed storage: TE's layout probe
        # rejects non-zero storage offsets, and .contiguous() would be a no-op.
        q, k, v = (
            query[qs : qs + nq].clone(),
            key[ks : ks + nk].clone(),
            value[ks : ks + nk].clone(),
        )
        if nq and nk:
            out = _dense_dispatch(
                self,
                q[:, None],
                k[:, None],
                v[:, None],
                None,
                _attn_mask_type_value(span_mask),
                span_mask,
                None,
                "sbhd",
                original,
            )
            out = out.reshape(nq, q.shape[-2], v.shape[-1])
        else:
            # Retain zero gradient edges for empty sequences too.
            out = q.sum(-1, keepdim=True).expand(nq, q.shape[-2], v.shape[-1]) * 0
            out = out + (k.sum() + v.sum()) * 0
        if capacity > nq:
            out = torch.cat((out, out.new_zeros(capacity - nq, out.shape[1], out.shape[2])))
        outputs.append(out)
    # TE's THD contract flattens heads: [total, h * d].
    return torch.cat(outputs, dim=0).reshape(query.shape[0], -1)


def _tedpa_forward(original: Any) -> Any:
    """Wrap TEDotProductAttention.forward, or decline when not applicable.

    The dispatch exists because MT-TE hard-codes ``use_flash_attention = True``
    in ``transformer_engine/musa/pytorch/attention.py``. When the fingerprint
    is absent this patch declines; that alone does not prove native THD,
    dropout or backward correctness. Revalidate those paths after upgrading.
    """
    hardcoded_flash = _compat.module_source_contains(
        "transformer_engine.musa.pytorch.attention", "use_flash_attention = True"
    )
    if hardcoded_flash is False:
        _compat.logger.info(
            "capability-dispatch declined: transformer_engine no longer "
            "hard-codes use_flash_attention (te=%s)",
            _compat.te_version(),
        )
        return None

    @functools.wraps(original)
    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
        num_splits=None,
    ):
        packed = packed_seq_params
        layout = getattr(packed, "qkv_format", None) or getattr(self, "qkv_format", "sbhd")
        mask_name = _mask_type_name(attn_mask_type)
        allowed_masks = {
            "no_mask",
            "causal",
            "padding",
            "padding_causal",
            "causal_bottom_right",
            "padding_causal_bottom_right",
            "arbitrary",
        }
        eligible = (
            _dispatch_contract(self, query, key, value, packed, num_splits)
            and mask_name in allowed_masks
        )
        if (
            packed is None
            and attention_mask is None
            and ("padding" in mask_name or mask_name == "arbitrary")
        ):
            eligible = False
        if getattr(self, "window_size", None) == (-1, 0) and "causal" not in mask_name:
            eligible = False
        if eligible and _musa_live() and query.device.type == "musa":
            if (
                layout == "thd"
                and packed is not None
                and mask_name != "arbitrary"
                and attention_mask is None
                and attention_bias is None
            ):
                return _packed_forward(self, query, key, value, mask_name, packed, original)
            if packed is None and layout in ("sbhd", "bshd"):
                return _dense_dispatch(
                    self,
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type,
                    mask_name,
                    attention_bias,
                    layout,
                    original,
                )
        return original(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed,
            num_splits=num_splits,
        )

    return forward


PATCHES = (
    AttrPatch(
        id="megatron.te.attention.capability-dispatch",
        target=_TEDPA,
        replace=_tedpa_forward,
        rationale=(
            "MT-TE's DotProductAttention hard-codes use_flash_attention, so "
            "every call reaches the MUSA flash SDPA kernel regardless of "
            "capability: the kernel asserts FP16/BF16 inputs, supports head "
            "dimensions 64..192, traps on the device when dropout_p > 0, and "
            "its wrapper drops cu_seqlens so padded THD batches produce NaN. "
            "Measured on MT-TE 2.0.0 / torch_musa 2.7.1 / flash-attn 2.6.3 "
            "with Megatron core_v0.16.1. External attention interfaces still "
            "cannot fill these gaps: mate varlen (0.2.6 + tilelang_musa "
            "0.1.12) exposes neither dropout nor padded-THD physical offsets, "
            "flash_attn dense asserts dropout_p == 0 and its GQA backward "
            "fails unreproducibly across identical fresh processes, and MuDNN "
            "flash rejects FP32 ('Unsupport Type FLOAT')."
        ),
        strategy=(
            "Dispatch by kernel capability, implementing no attention math: "
            "FP16/BF16 inputs with head_dim in [64,192] and no effective "
            "dropout run the native MT-TE flash path untouched; every other "
            "eligible input runs TE's own UnfusedDotProductAttention backend "
            "(torch softmax fallback on MUSA) with its mask, GQA, bias and "
            "dropout-RNG contracts; segmented THD is sliced into "
            "per-sequence sbhd calls of the same dispatch, keeping padded "
            "offsets and zero-gradient edges for empty sequences. Padding "
            "masks are normalized to the shapes TE's get_full_mask consumes; "
            "unrecognized masks give up so the vendor error surfaces. CP, "
            "FP8 DPA, special softmax, windowed attention and max-logit "
            "remain upstream. The unfused route costs O(S^2) memory; THD "
            "slicing adds one host sync of cu_seqlens metadata. Declines "
            "entirely when the TE hard-coded flash marker is gone (source "
            "probe)."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/extensions/transformer_engine.py:"
            "TEDotProductAttention.forward; TransformerEngine pytorch/attention.py"
        ),
        remove_when=(
            "Remove when the MUSA TransformerEngine selects backends by input "
            "capability, its flash kernels accept the full head-dim range and "
            "dropout, and THD honors cu_seqlens: disable this patch id and "
            "re-run transformer/test_attention.py, "
            "test_multi_latent_attention.py and resharding/test_model_swap.py; "
            "delete only if the native path passes."
        ),
    ),
)
