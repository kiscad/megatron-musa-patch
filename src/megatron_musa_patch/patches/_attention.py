"""Capability dispatch for MT-TE's hard-coded MUSA flash path.

Reference: Megatron core_v0.16.1 TEDotProductAttention; TransformerEngine
``UnfusedDotProductAttention`` and ``get_full_mask``; mate's TileLang flash
(``mate.flash_attn_varlen_func``). No attention math is implemented here:
inputs inside the MUSA flash kernel's measured forward *and* backward
capability window run the native MT-TE flash path; the shapes MuDNN's
backward kernel rejects (MLA's 192/128 mix, 144 and 168..192) run mate's
verified TileLang flash when mate is installed; every other eligible input
runs TE's own unfused backend; segmented THD is sliced into per-sequence
vendor calls. ``MEGATRON_MUSA_PATCH_ATTN_BACKEND`` selects the kernel
order. Only ordinary tensors, CP=1, vanilla softmax and unrestricted
windows are handled.
"""

from __future__ import annotations

import functools
from typing import Any

from .. import _compat, _env
from .._engine import AttrPatch, HookPatch
from ..backends import musa_available as _musa_live

__all__ = ["PATCHES"]
_TEDPA = "megatron.core.extensions.transformer_engine:TEDotProductAttention.forward"

#: MUSA flash SDPA kernel window (torch.ops.aten._scaled_dot_product_attention_
#: flash_musa; MT-TE gates on the same values in flash_attn_varlen_func_variance).
_MIN_FLASH_DIM = 64
_MAX_FLASH_DIM = 192

#: Head dimensions the MuDNN flash *backward* kernel (v3107) actually serves,
#: measured directly against aten._scaled_dot_product_attention_flash_musa:
#: 64/80/96/112/128/160 (and 200/224/256/512, outside the forward window)
#: pass forward+backward, while 144 and 168..192 fail backward with
#: "MuDNNFlashSDPABwd MUDNN failed" even though the forward accepts them, and
#: every measured mixed qk/v head-dim pair (128/192, 192/128) fails backward
#: too. MLA runs exactly such a mix (qk 128+64=192, v 128).
_FLASH_BWD_SAFE_DIMS = frozenset((64, 80, 96, 112, 128, 160))

#: Head-dim shapes mate's TileLang flash kernels are *verified* to serve on
#: this stack, restricted to the shapes MuDNN's flash backward rejects:
#: equal dims 144 and 168..192, plus the measured MLA-style mixed pairs
#: (192/128 and 160/128 pass with bit-sane bf16 output, deterministic
#: backward and honored softmax_scale; 128/64 and 256/128 are rejected by
#: mate's own config table with "Add config for headdim ..."). Equal dims
#: inside the MuDNN backward window never reach mate -- the native kernel
#: stays the default there.
_MATE_EQUAL_DIMS = frozenset((144, 168, 176, 184, 192))
_MATE_MIXED_DIMS = frozenset(((192, 128), (160, 128)))

#: mate's entry point once imported, with a sentinel for "tried and absent".
_MATE_TRIED = False
_MATE_FN = None


def _mate_flash_fn():
    """mate.flash_attn_varlen_func lazily, or None when mate is unusable.

    Imported at call time (never at patch import: ``patches/`` stays
    standard-library only) and cached, because the dispatch runs per forward.
    A missing or broken mate install just removes the fast path; the unfused
    backend remains the correctness reference.
    """
    global _MATE_TRIED, _MATE_FN
    if not _MATE_TRIED:
        _MATE_TRIED = True
        try:
            import mate

            fn = getattr(mate, "flash_attn_varlen_func", None)
            if fn is not None:
                _MATE_FN = fn
        except Exception as exc:  # noqa: BLE001 - any import failure declines
            _compat.logger.info(
                "mate unavailable for the attention dispatch (%s: %s); "
                "MuDNN-unsafe shapes fall back to the unfused backend",
                type(exc).__name__,
                exc,
            )
    return _MATE_FN


def _attn_backend() -> str:
    """User-selected attention backend preference for the MUSA-unsafe shapes.

    ``MEGATRON_MUSA_PATCH_ATTN_BACKEND`` selects ``auto`` (default: native
    MuDNN flash inside its measured backward window, then mate's TileLang
    flash for the shapes MuDNN's backward rejects, then TE's unfused
    backend), ``mudnn`` (native flash whenever the forward accepts it, the
    pre-selection behavior -- backward may still fail), ``mate`` (prefer
    mate wherever its verified shape table applies) or ``unfused`` (always
    TE's unfused backend). Unknown values keep ``auto``.
    """
    raw = (_env.value("ATTN_BACKEND", "auto") or "auto").strip().lower()
    return raw if raw in ("auto", "mudnn", "mate", "unfused") else "auto"


def _effective_dropout(self) -> float:
    """Dropout the flash kernel would have to apply on this call."""
    training = bool(getattr(self, "training", False))
    return float(getattr(self, "attention_dropout", 0.0) or 0.0) if training else 0.0


def _flash_backward_may_run(self, tensors) -> bool:
    """Whether this call can reach the MuDNN flash backward kernel.

    A training-mode module may be recomputed in backward (mcore's checkpoint
    runs the forward under ``no_grad`` and re-runs it with grad enabled), so
    the module's own ``training`` flag counts even when grad is currently
    disabled; grad-enabled calls with grad-requiring inputs count regardless
    of the flag. Pure inference (eval and/or no grad, no grad-requiring
    inputs) never runs backward and keeps the fast forward path.
    """
    import torch

    if bool(getattr(self, "training", False)):
        return True
    return torch.is_grad_enabled() and any(getattr(t, "requires_grad", False) for t in tensors)


def _flash_kernel_unsupported_fwd(self, query, key=None, value=None) -> bool:
    """True when the native MT-TE MUSA flash *forward* cannot run these inputs.

    The kernel path asserts FP16/BF16 (or Float8Tensor) inputs, supports head
    dimensions 64..192 in the forward, and traps on the device when
    ``dropout_p > 0`` is passed (flash_attn's own wrapper asserts dropout is
    unsupported yet). Measured on torch_musa 2.7.1 / MT-TE 2.0.0 /
    flash-attn 2.6.3.
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


def _flash_kernel_unsupported(self, query, key=None, value=None) -> bool:
    """True when the native MT-TE MUSA flash kernel cannot run these inputs.

    Composes the forward window with the *backward* kernel's narrower
    measured support (see ``_FLASH_BWD_SAFE_DIMS``): calls that may reach
    backward -- training, checkpoint recompute, or grad-enabled calls with
    grad-requiring inputs -- only take flash with equal qk/v head dims
    inside the measured backward-safe set. Pure inference keeps the full
    forward window.
    """
    if _flash_kernel_unsupported_fwd(self, query, key, value):
        return True
    tensors = [t for t in (query, key, value) if t is not None]
    if _flash_backward_may_run(self, tensors):
        dims = {t.size(-1) for t in tensors}
        if len(dims) > 1 or not dims <= _FLASH_BWD_SAFE_DIMS:
            return True
    return False


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
    elif mask_name == "arbitrary":
        mask = attention_mask
    # Other mask types (causal, no_mask, causal_bottom_right) drop any mask
    # tensor: TE's own backend table requires attention_mask=None for them,
    # and the flash path this dispatch replaces ignores the tensor entirely.
    # Megatron callers sometimes pass a dummy all-ones mask there, whose TE
    # semantics (True=masked) would blank the whole output once get_full_mask
    # ORs it into the causal mask.
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


def _mate_eligible(self, query, key, value, mask_name, attention_bias, layout) -> bool:
    """Whether mate's verified TileLang flash serves exactly this call.

    Only claims what the capability probes measured on this stack: fp16/bf16
    MUSA tensors of one dtype, no dropout (mate has no dropout parameter), no
    attention bias, plain causal/no-mask semantics (a bottom-right causal
    mask only when query and key lengths match, where it equals top-left),
    equal head dims in the MuDNN-backward-broken set or the verified mixed
    MLA pairs, and q heads divisible by kv heads. Everything else keeps the
    unfused backend.
    """
    import torch

    if not _musa_live() or query.device.type != "musa":
        return False
    tensors = (query, key, value)
    if any(t.dtype not in (torch.float16, torch.bfloat16) for t in tensors):
        return False
    if len({t.dtype for t in tensors}) != 1:
        return False
    if _effective_dropout(self) != 0.0:
        return False
    if attention_bias is not None:
        return False
    if mask_name not in ("no_mask", "causal", "causal_bottom_right"):
        return False
    dqk, dk, dv = query.shape[-1], key.shape[-1], value.shape[-1]
    if dqk != dk:
        return False
    if dqk == dv:
        if dqk not in _MATE_EQUAL_DIMS:
            return False
    elif (dqk, dv) not in _MATE_MIXED_DIMS:
        return False
    if query.shape[-2] % key.shape[-2]:
        return False
    if mask_name == "causal_bottom_right":
        if layout == "bshd":
            if query.shape[1] != key.shape[1]:
                return False
        elif query.shape[0] != key.shape[0]:
            return False
    return True


def _mate_dense_forward(self, query, key, value, mask_name, layout):
    """Run the call on mate's TileLang flash, or decline with None.

    mate consumes and produces ``[batch, seq, heads, dim]``; Megatron's sbhd
    layout is transposed on the way in, and the output is transposed back and
    flattened to TE's dense contract (``[seq, batch, heads * v_dim]`` for
    sbhd, ``[batch, seq, heads * v_dim]`` for bshd -- exactly what
    UnfusedDotProductAttention returns, so downstream linears and the THD
    slicer see one shape contract). softmax_scale and the deterministic flag
    are forwarded from the TE module so MLA's yarn mscale scaling and
    deterministic-mode runs keep their contracts.
    """
    fn = _mate_flash_fn()
    if fn is None:
        return None
    if layout == "bshd":
        q, k, v = query, key, value
    else:
        q, k, v = (t.transpose(0, 1) for t in (query, key, value))
    out = fn(
        q,
        k,
        v,
        causal="causal" in mask_name,
        softmax_scale=getattr(self, "softmax_scale", None),
        deterministic=bool(getattr(self, "deterministic", False)),
    )
    if layout == "bshd":
        return out.reshape(out.shape[0], out.shape[1], -1)
    out = out.transpose(0, 1)
    return out.reshape(out.shape[0], out.shape[1], -1)


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
    """Run the first backend whose measured capability covers the call.

    ``attn_mask_type`` is forwarded to ``original`` untouched (it must remain
    the caller's enum value); ``mask_name`` is the string form used internally.
    The kernel order follows ``MEGATRON_MUSA_PATCH_ATTN_BACKEND``:

    - ``auto`` (default): native MuDNN flash inside its measured
      forward+backward window, then mate's verified TileLang flash, then
      TE's unfused backend;
    - ``mudnn``: native flash whenever the forward accepts it (backward
      may still fail -- the pre-selection behavior), then unfused;
    - ``mate``: mate's TileLang flash first, then the native safe window,
      then unfused;
    - ``unfused``: TE's unfused backend only.
    """
    backend = _attn_backend()

    def native_forward():
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

    if backend == "unfused":
        out = _unfused_forward(
            self, query, key, value, attention_mask, mask_name, attention_bias, layout
        )
        return out if out is not None else native_forward()

    if backend == "mate" and _mate_eligible(
        self, query, key, value, mask_name, attention_bias, layout
    ):
        out = _mate_dense_forward(self, query, key, value, mask_name, layout)
        if out is not None:
            return out

    flash_ok = (
        not _flash_kernel_unsupported_fwd(self, query, key, value)
        if backend == "mudnn"
        else not _flash_kernel_unsupported(self, query, key, value)
    )
    if flash_ok:
        return native_forward()

    if _mate_eligible(self, query, key, value, mask_name, attention_bias, layout):
        out = _mate_dense_forward(self, query, key, value, mask_name, layout)
        if out is not None:
            return out

    out = _unfused_forward(
        self, query, key, value, attention_mask, mask_name, attention_bias, layout
    )
    if out is not None:
        return out
    # Nothing claims these inputs: surface the vendor error instead of guessing.
    return native_forward()


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


_NATIVE_DPA_STATE: dict[str, Any] = {}


def _native_dpa_forward_wrapper(original_forward: Any) -> Any:
    """Wrap TE's own DotProductAttention.forward with the capability dispatch.

    ``te.pytorch.TransformerLayer`` (the direct-TE models of the megatron-FSDP
    suite) runs its attention through TE's native class, not Megatron's
    ``TEDotProductAttention`` wrapper -- Megatron-scope patches never reach it.
    The MUSA port hard-codes the flash backend there, so inputs NVIDIA's
    selector would route to the unfused backend (e.g. FP32 activations from
    the suite's fp32 models) die on "FlashAttention only supports FP16 and
    BF16 data types". The wrapper is installed *in place* on the original
    class: MT-TE's own ``DotProductAttention__init__`` resolves
    ``super(DotProductAttention, self)`` through its module global, so a
    class-binding replacement (or an alias repair into the musa module)
    breaks every construction with "missing 2 required positional arguments".
    """

    @functools.wraps(original_forward)
    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask=None,
        qkv_format=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        attn_mask_type=None,
        window_size=None,
        checkpoint_core_attention=False,
        core_attention_bias_type="no_bias",
        core_attention_bias=None,
        alibi_slopes=None,
        **kwargs,
    ):
        import torch

        layout = qkv_format or getattr(self, "qkv_format", "sbhd")
        mask_name = _mask_type_name(
            attn_mask_type if attn_mask_type is not None else getattr(self, "attn_mask_type", None)
        )
        tensors = (query_layer, key_layer, value_layer)
        if (
            not checkpoint_core_attention
            and layout in ("sbhd", "bshd")
            and cu_seqlens_q is None
            and cu_seqlens_kv is None
            and all(torch.is_tensor(t) and type(t) is torch.Tensor for t in tensors)
            and _musa_live()
            and query_layer.device.type == "musa"
            and _flash_kernel_unsupported(self, query_layer, key_layer, value_layer)
            and mask_name
            in (
                "no_mask",
                "causal",
                "padding",
                "padding_causal",
                "causal_bottom_right",
                "padding_causal_bottom_right",
                "arbitrary",
            )
            and not (
                attention_mask is None and ("padding" in mask_name or mask_name == "arbitrary")
            )
        ):
            out = _unfused_forward(
                self,
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                mask_name,
                core_attention_bias,
                layout,
            )
            if out is not None:
                return out
        return original_forward(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask=attention_mask,
            qkv_format=qkv_format,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            checkpoint_core_attention=checkpoint_core_attention,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
            alibi_slopes=alibi_slopes,
            **kwargs,
        )

    return forward


def _install_native_dpa_dispatch() -> bool:
    """Install the wrapper on TE's DotProductAttention, in place.

    Runs when MT-TE's musa attention module imports, so the class and all of
    the port's own ``replace_attr`` mutations have settled. Idempotent;
    declines (keeping upstream) without a live MUSA runtime or when the TE
    source no longer hard-codes the flash backend.
    """
    import sys

    if not _musa_live():
        return False
    if (
        _compat.module_source_contains(
            "transformer_engine.musa.pytorch.attention", "use_flash_attention = True"
        )
        is False
    ):
        return False
    module = sys.modules.get("transformer_engine.pytorch.attention")
    if module is None:
        return False
    cls = getattr(module, "DotProductAttention", None)
    if cls is None or getattr(cls, "_megatron_musa_patch_dpa_dispatch", False):
        return True
    original_forward = cls.forward
    wrapper = _native_dpa_forward_wrapper(original_forward)
    cls.forward = wrapper
    cls._megatron_musa_patch_dpa_dispatch = True
    _NATIVE_DPA_STATE.clear()
    _NATIVE_DPA_STATE.update(cls=cls, wrapper=wrapper, original=original_forward)
    return True


def _undo_native_dpa_dispatch() -> None:
    """Restore the original forward while this patch still owns the binding."""
    cls = _NATIVE_DPA_STATE.get("cls")
    if cls is None:
        return
    if getattr(cls, "forward", None) is _NATIVE_DPA_STATE.get("wrapper"):
        cls.forward = _NATIVE_DPA_STATE["original"]
        if getattr(cls, "_megatron_musa_patch_dpa_dispatch", False):
            try:
                delattr(cls, "_megatron_musa_patch_dpa_dispatch")
            except AttributeError:
                pass
    _NATIVE_DPA_STATE.clear()


PATCHES = (
    AttrPatch(
        id="megatron.te.attention.capability-dispatch",
        target=_TEDPA,
        replace=_tedpa_forward,
        rationale=(
            "MT-TE's DotProductAttention hard-codes use_flash_attention, so "
            "every call reaches the MUSA flash SDPA kernel regardless of "
            "capability: the kernel asserts FP16/BF16 inputs, supports head "
            "dimensions 64..192 in the forward, traps on the device when "
            "dropout_p > 0, its wrapper drops cu_seqlens so padded THD batches "
            "produce NaN, and its backward kernel is narrower than the forward "
            "-- measured on muDNN v3107, backward serves equal qk/v head dims "
            "in {64,80,96,112,128,160} but fails for 144 and 168..192 and for "
            "every mixed qk/v pair (MLA's 192/128 hits both), surfacing as "
            "'MuDNNFlashSDPABwd MUDNN failed' in the a2a overlap suite. "
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
            "FP16/BF16 inputs with head_dim in [64,192], no effective dropout, "
            "and -- when the call may reach backward (training, checkpoint "
            "recompute, or grad-enabled with grad-requiring inputs) -- equal "
            "qk/v head dims inside the measured backward-safe set, run the "
            "native MT-TE flash path untouched; inference-only calls keep "
            "flash for the full forward window. The shapes MuDNN's backward "
            "rejects (equal 144 and 168..192; mixed pairs such as MLA's "
            "192/128) run mate's TileLang flash when mate is installed and "
            "the call is inside mate's verified envelope (one fp16/bf16 "
            "dtype, no dropout, no bias, causal/no-mask, q heads divisible "
            "by kv heads); mate receives sbhd transposed to bshd, plus the "
            "module's softmax_scale and deterministic flag. Every other "
            "eligible input runs TE's own UnfusedDotProductAttention backend "
            "(torch softmax fallback on MUSA) with its mask, GQA, bias and "
            "dropout-RNG contracts; non-padding mask types drop the mask "
            "tensor exactly like the flash path they replace. Segmented THD "
            "is sliced into per-sequence sbhd calls of the same dispatch, "
            "keeping padded offsets and zero-gradient edges for empty "
            "sequences. Padding masks are normalized to the shapes TE's "
            "get_full_mask consumes; unrecognized masks give up so the "
            "vendor error surfaces. CP, FP8 DPA, special softmax, windowed "
            "attention and max-logit remain upstream. "
            "MEGATRON_MUSA_PATCH_ATTN_BACKEND=auto|mudnn|mate|unfused "
            "selects the kernel order (auto by default; mudnn restores the "
            "forward-only selection, mate prefers the TileLang kernels, "
            "unfused forces the reference backend). The unfused route costs "
            "O(S^2) memory; mate's TileLang kernels JIT-compile per "
            "head-dim class on first use (minutes, cached in ~/.tilelang -- "
            "concurrent multi-rank compiles into the shared cache have "
            "crashed runs, pre-warm once); THD slicing adds one host sync "
            "of cu_seqlens metadata. Declines entirely when the TE "
            "hard-coded flash marker is gone (source probe)."
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
    HookPatch(
        id="transformer_engine.dot-product-attention.capability-dispatch",
        trigger="transformer_engine.musa.pytorch.attention",
        run=_install_native_dpa_dispatch,
        undo=_undo_native_dpa_dispatch,
        rationale=(
            "Direct TransformerEngine models (te.pytorch.TransformerLayer in "
            "the megatron-FSDP suite, 56 mfsdp cases) run attention through "
            "TE's own DotProductAttention, which the MUSA port pins to the "
            "flash backend. FP32-activation models die on 'FlashAttention "
            "only supports FP16 and BF16 data types, or Float8Tensors' where "
            "NVIDIA's backend selector would pick the unfused backend; the "
            "same flash backward window applies as on the Megatron wrapper. "
            "Megatron-scope patches never reach this class."
        ),
        strategy=(
            "Wrap DotProductAttention.forward in place -- constructor, musa "
            "hooks, parameters and every binding stay TE's (MT-TE's own "
            "__init__ resolves super() through its module global, so class "
            "replacement breaks construction) -- and reroute only eligible "
            "dense (sbhd/bshd, no cu_seqlens, plain tensors, no checkpoint "
            "core attention) calls whose inputs the flash kernel cannot serve "
            "(measured forward window and backward window, dropout) to TE's "
            "own UnfusedDotProductAttention backend, with the same padding-"
            "mask normalization and mask-tensor dropping as the Megatron "
            "wrapper dispatch. Everything else calls the original forward so "
            "its own errors surface. Declines when the TE hard-coded flash "
            "marker is gone (source probe)."
        ),
        upstream=("TransformerEngine pytorch/attention.py:DotProductAttention.forward"),
        remove_when=(
            "Remove together with megatron.te.attention.capability-dispatch "
            "when the MUSA TransformerEngine selects backends by input "
            "capability; re-run the megatron-FSDP te_transformer cases with "
            "this patch disabled before deleting."
        ),
    ),
)
