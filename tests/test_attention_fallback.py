"""Attention capability dispatch: flash gate, unfused routing, THD slicing."""

from types import SimpleNamespace

import pytest

from megatron_musa_patch.patches import _attention

torch = pytest.importorskip("torch")


def _musa_query(dtype, head_dim):
    return SimpleNamespace(
        device=SimpleNamespace(type="musa"), dtype=dtype, size=lambda dim: head_dim
    )


def _tedpa_stub(dtype, head_dim, dropout=0.0, training=False):
    """A stand-in exposing the attributes the dispatch paths read."""
    stub = SimpleNamespace()
    stub.training = training
    stub.attention_dropout = dropout
    stub.qkv_format = "sbhd"
    stub.window_size = None
    return stub


@pytest.fixture
def musa_live(monkeypatch):
    monkeypatch.setattr(_attention, "_musa_live", lambda: True)
    return True


@pytest.mark.parametrize(
    "training,dropout,expected",
    [
        (False, 0.0, False),
        (True, 0.0, False),
        (True, 0.2, True),
        (False, 0.2, False),
    ],
)
def test_flash_gate_accepts_bf16_supported_dims(musa_live, training, dropout, expected):
    """BF16/FP16 with head_dim 64..192 runs natively unless dropout applies."""
    stub = _tedpa_stub(torch.bfloat16, 128, dropout=dropout, training=training)
    assert _attention._flash_kernel_unsupported(stub, _musa_query(torch.bfloat16, 128)) is expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("training", [False, True])
def test_flash_gate_rejects_wider_dtypes(musa_live, dtype, training):
    """MuDNN flash rejects non-FP16/BF16 inputs ('Unsupport Type FLOAT')."""
    stub = _tedpa_stub(dtype, 128, training=training)
    assert _attention._flash_kernel_unsupported(stub, _musa_query(dtype, 128))
    assert _attention._flash_kernel_unsupported(
        stub,
        _musa_query(torch.bfloat16, 128),
        _musa_query(dtype, 128),
        _musa_query(torch.bfloat16, 128),
    )


@pytest.mark.parametrize("head_dim", [32, 56, 256])
@pytest.mark.parametrize("training", [False, True])
def test_flash_gate_rejects_head_dim_outside_window(musa_live, head_dim, training):
    """The flash interface asserts head_dim >= 64 and <= 192."""
    stub = _tedpa_stub(torch.bfloat16, head_dim, training=training)
    assert _attention._flash_kernel_unsupported(stub, _musa_query(torch.bfloat16, head_dim))


@pytest.mark.parametrize("head_dim", [144, 168, 176, 184, 192])
def test_flash_backward_window_rejects_broken_dims_in_training(musa_live, head_dim):
    """MuDNN flash backward fails for 144 and 168..192 although forward works.

    Training calls may run backward (directly or through checkpoint
    recompute), so those head dims must not take the native flash path.
    """
    stub = _tedpa_stub(torch.bfloat16, head_dim, training=True)
    assert _attention._flash_kernel_unsupported(stub, _musa_query(torch.bfloat16, head_dim))
    assert not _attention._flash_kernel_unsupported_fwd(stub, _musa_query(torch.bfloat16, head_dim))


@pytest.mark.parametrize("head_dim", [144, 192])
def test_flash_backward_window_keeps_inference_on_flash(musa_live, head_dim):
    """Inference-only calls keep the fast forward for the full window."""
    stub = _tedpa_stub(torch.bfloat16, head_dim, training=False)
    assert not _attention._flash_kernel_unsupported(stub, _musa_query(torch.bfloat16, head_dim))


def test_flash_backward_window_covers_grad_enabled_calls(musa_live):
    """Grad-enabled calls with grad-requiring inputs count even in eval mode."""
    stub = _tedpa_stub(torch.bfloat16, 192, training=False)
    plain = _musa_query(torch.bfloat16, 192)
    assert not _attention._flash_kernel_unsupported(stub, plain)
    real = SimpleNamespace(
        device=SimpleNamespace(type="musa"),
        dtype=torch.bfloat16,
        size=lambda dim, d=192: d,
        requires_grad=True,
    )
    assert torch.is_grad_enabled()
    assert _attention._flash_kernel_unsupported(stub, real)


def test_flash_backward_window_rejects_mixed_head_dims(musa_live):
    """Every measured mixed qk/v pair fails the MuDNN flash backward."""
    stub = _tedpa_stub(torch.bfloat16, 192, training=True)
    assert _attention._flash_kernel_unsupported(
        stub,
        _musa_query(torch.bfloat16, 192),
        _musa_query(torch.bfloat16, 192),
        _musa_query(torch.bfloat16, 128),
    )


def test_attn_backend_env_values(monkeypatch):
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_ATTN_BACKEND", raising=False)
    assert _attention._attn_backend() == "auto"
    for raw, expected in (
        ("auto", "auto"),
        ("MUDNN", "mudnn"),
        (" mate ", "mate"),
        ("unfused", "unfused"),
        ("nonsense", "auto"),
    ):
        monkeypatch.setenv("MEGATRON_MUSA_PATCH_ATTN_BACKEND", raw)
        assert _attention._attn_backend() == expected


def _mate_tensors(dqk, dv, dtype=torch.bfloat16, heads=8, kv_heads=None):
    def t(last, h):
        return SimpleNamespace(
            device=SimpleNamespace(type="musa"), dtype=dtype, shape=(4, 8, h, last)
        )

    return t(dqk, heads), t(dqk, kv_heads or heads), t(dv, kv_heads or heads)


@pytest.mark.parametrize(
    "dqk,dv,mask,bias,dropout,expected",
    [
        (192, 128, "causal", None, 0.0, True),  # the MLA shape
        (160, 128, "no_mask", None, 0.0, True),
        (144, 144, "causal", None, 0.0, True),
        (192, 192, "causal_bottom_right", None, 0.0, True),
        (128, 128, "causal", None, 0.0, False),  # MuDNN-safe: never routed to mate
        (128, 64, "causal", None, 0.0, False),  # mate has no config for it
        (256, 128, "causal", None, 0.0, False),
        (192, 128, "causal", object(), 0.0, False),  # bias unsupported
        (192, 128, "causal", None, 0.2, False),  # mate has no dropout
        (192, 128, "padding_causal", None, 0.0, False),  # mask tensors unsupported
        (192, 128, "arbitrary", None, 0.0, False),
    ],
)
def test_mate_eligibility(musa_live, dqk, dv, mask, bias, dropout, expected):
    stub = _tedpa_stub(torch.bfloat16, dqk, dropout=dropout, training=dropout > 0)
    q, k, v = _mate_tensors(dqk, dv)
    assert _attention._mate_eligible(stub, q, k, v, mask, bias, "sbhd") is expected


def test_mate_eligibility_requires_divisible_gqa(musa_live):
    stub = _tedpa_stub(torch.bfloat16, 192)
    q, k, v = _mate_tensors(192, 128, heads=7, kv_heads=3)
    assert not _attention._mate_eligible(stub, q, k, v, "causal", None, "sbhd")


def test_mate_dense_forward_transposes_sbhd_and_forwards_flags(musa_live, monkeypatch):
    calls = {}

    def fake_fn(q, k, v, causal, softmax_scale, deterministic):
        calls.update(
            shape=tuple(q.shape),
            causal=causal,
            scale=softmax_scale,
            deterministic=deterministic,
            kshape=tuple(k.shape),
            vshape=tuple(v.shape),
        )
        return v.expand_as(q[..., : v.shape[-1]]).clone()

    monkeypatch.setattr(_attention, "_mate_flash_fn", lambda: fake_fn)
    stub = _tedpa_stub(torch.bfloat16, 192, training=True)
    stub.softmax_scale = 0.137
    stub.deterministic = True
    # sbhd: [s, b, h, d] -> mate sees [b, s, h, d]
    q = torch.randn(4, 2, 8, 192)
    k = torch.randn(4, 2, 8, 192)
    v = torch.randn(4, 2, 8, 128)
    out = _attention._mate_dense_forward(stub, q, k, v, "causal", "sbhd")
    assert calls["shape"] == (2, 4, 8, 192)
    assert calls["kshape"] == (2, 4, 8, 192)
    assert calls["vshape"] == (2, 4, 8, 128)
    assert calls["causal"] is True
    assert calls["scale"] == 0.137
    assert calls["deterministic"] is True
    assert tuple(out.shape) == (4, 2, 1024), "sbhd output flattened like TE's backends"


def test_mate_dense_forward_keeps_bshd(musa_live, monkeypatch):
    def fake_fn(q, k, v, causal, softmax_scale, deterministic):
        return v

    monkeypatch.setattr(_attention, "_mate_flash_fn", lambda: fake_fn)
    stub = _tedpa_stub(torch.bfloat16, 192, training=True)
    q = torch.randn(2, 4, 8, 192)
    v = torch.randn(2, 4, 8, 128)
    out = _attention._mate_dense_forward(stub, q, q, v, "no_mask", "bshd")
    assert tuple(out.shape) == (2, 4, 1024), "bshd output flattened like TE's backends"
    assert torch.equal(out, v.reshape(2, 4, -1))


def test_mate_dense_forward_declines_without_mate(musa_live, monkeypatch):
    monkeypatch.setattr(_attention, "_mate_flash_fn", lambda: None)
    stub = _tedpa_stub(torch.bfloat16, 192, training=True)
    q = torch.randn(4, 2, 8, 192)
    assert _attention._mate_dense_forward(stub, q, q, q, "causal", "sbhd") is None


def test_te_padding_mask_normalization():
    """Megatron mask forms map onto the shapes TE's get_full_mask consumes."""
    q_pad = torch.tensor([[False, False, True], [False] * 3])
    k_pad = torch.tensor([[False, True, True], [False] * 3])
    # Distinct Q/K masks cannot be collapsed into a self-attention vector.
    assert _attention._te_padding_mask((q_pad, k_pad), 3, 3) is None
    same = _attention._te_padding_mask((q_pad, q_pad), 3, 3)
    assert torch.equal(same[:, 0, 0], q_pad)
    # Equal sequence lengths do not turn cross-attention into self-attention.
    cross = _attention._te_padding_mask((q_pad, k_pad), 3, 3, "cross")
    assert torch.equal(cross[0][:, 0, 0], q_pad)
    assert torch.equal(cross[1][:, 0, 0], k_pad)
    k_pad5 = torch.tensor([[False] + [True] * 4, [False] * 5])
    cross = _attention._te_padding_mask((q_pad, k_pad5), 3, 5, "cross")
    assert isinstance(cross, tuple) and all(m.shape == (2, 1, 1, s) for m, s in zip(cross, (3, 5)))
    assert _attention._te_padding_mask(torch.zeros(2, 1, 3, 3, dtype=torch.bool), 3, 3) is None
    assert _attention._te_padding_mask((), 3, 3) is None
    assert _attention._te_padding_mask(k_pad, 3, 5) is None
    # Single masks: 2D and [b, 1, sk] become [b, 1, 1, sk].
    assert _attention._te_padding_mask(k_pad, 3, 3).shape == (2, 1, 1, 3)
    assert _attention._te_padding_mask(k_pad[:, None, :], 3, 3).shape == (2, 1, 1, 3)
    assert _attention._te_padding_mask(None, 3, 3) is None
    # Full [b, sq, sk] matrices and non-boolean masks are not claimed.
    assert _attention._te_padding_mask(torch.zeros(2, 3, 3, dtype=torch.bool), 3, 3) is None
    assert _attention._te_padding_mask(k_pad.float(), 3, 3) is None


def test_attention_patch_registered():
    patch = next(
        p for p in _attention.PATCHES if p.id == "megatron.te.attention.capability-dispatch"
    )
    assert patch.target.endswith("TEDotProductAttention.forward")
    assert "UnfusedDotProductAttention" in patch.strategy
    assert "use_flash_attention" in patch.rationale


def test_packed_lengths_reject_invalid_metadata():
    with pytest.raises(ValueError):
        _attention._packed_spans(torch.tensor([0, 3, 2]), None, 2)
    with pytest.raises(ValueError):
        _attention._packed_spans(torch.tensor([0, 3]), torch.tensor([0, 2]), 2)


@pytest.mark.parametrize(
    "attribute,value",
    [("cp_group", SimpleNamespace(size=lambda: 2)), ("window_size", (8, 0)), ("num_splits", 2)],
)
def test_unsupported_contract_delegates(monkeypatch, attribute, value):
    stub = _tedpa_stub(torch.float32, 4)
    setattr(stub, attribute, value)
    monkeypatch.setattr(_attention, "_musa_live", lambda: True)
    q = torch.randn(2, 1, 1, 4)
    calls = []
    wrapped = _attention._tedpa_forward(lambda *a, **kw: calls.append(kw) or "native")
    assert wrapped(stub, q, q, q, None, "causal") == "native"
    assert len(calls) == 1
    assert not _attention._dispatch_contract(stub, q, q, q, None, None)


def _device_required():
    pytest.importorskip("torch_musa")
    if not torch.musa.is_available():
        pytest.skip("no MUSA device")


def _te_attention(
    dtype, head_dim, heads, kv_heads, dropout=0.0, qkv_format="sbhd", mask_type="causal"
):
    import transformer_engine.pytorch as te

    module = te.DotProductAttention(
        num_attention_heads=heads,
        kv_channels=head_dim,
        num_gqa_groups=kv_heads,
        attention_dropout=dropout,
        qkv_format=qkv_format,
        attn_mask_type=mask_type,
        sequence_parallel=False,
    )
    return module.to(device="musa", dtype=dtype)


def _fp64_reference(q, k, v, causal=True):
    """Per-batch attention oracle; q [s, b, h, d], k/v [s, b, hk, d]."""

    def prep(x):
        return x.detach().double().cpu() if x.device.type == "musa" else x.double()

    q64 = prep(q).permute(1, 2, 0, 3)
    k64 = prep(k).permute(1, 2, 0, 3)
    v64 = prep(v).permute(1, 2, 0, 3)
    if q64.shape[1] != k64.shape[1]:
        rep = q64.shape[1] // k64.shape[1]
        k64 = k64.repeat_interleave(rep, 1)
        v64 = v64.repeat_interleave(rep, 1)
    s = q64.shape[2]
    scores = (q64 @ k64.transpose(-2, -1)) * q.shape[-1] ** -0.5
    if causal:
        scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), float("-inf"))
    return (torch.softmax(scores, -1) @ v64).permute(2, 0, 1, 3).reshape(s, q.shape[1], -1)


def test_musa_bf16_training_delegates_to_native_flash(musa_live):
    """BF16 training with a supported head dim runs the native flash path."""
    _device_required()
    torch.manual_seed(17)
    q, k, v = [
        torch.randn(4, 1, 2, 64, device="musa", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    calls = []
    wrapped = _attention._tedpa_forward(lambda *a, **kw: calls.append(a) or "native")
    stub = _tedpa_stub(q.dtype, 64, training=True)
    assert wrapped(stub, q, k, v, None, "causal") == "native"
    assert len(calls) == 1
    assert not _attention._flash_kernel_unsupported(stub, q, k, v)


def test_musa_mla_shape_routes_to_mate_and_matches_reference(musa_live, monkeypatch):
    """MLA's 192/128 training shape runs mate's flash and stays numerically sane.

    MuDNN's flash backward rejects the mixed head dims, so the dispatch must
    not call the native path; with mate installed the TileLang kernel serves
    the call, matching the fp64 oracle within bf16 tolerance and producing
    finite gradients. Skipped when mate is not importable.
    """
    _device_required()
    pytest.importorskip("mate")
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_ATTN_BACKEND", raising=False)
    torch.manual_seed(23)
    s, b, h = 64, 2, 8
    q = torch.randn(s, b, h, 192, device="musa", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(s, b, h, 192, device="musa", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(s, b, h, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    stub = SimpleNamespace(
        training=True,
        attention_dropout=0.0,
        softmax_scale=None,
        deterministic=True,
        window_size=None,
        qkv_format="sbhd",
        config=None,
        unfused_attention=None,
    )
    native_calls = []
    wrapped = _attention._tedpa_forward(lambda *a, **kw: native_calls.append(a) or "native")
    out = wrapped(stub, q, k, v, None, "causal")
    assert native_calls == [], "the mixed-dim training shape must not reach MuDNN flash"
    assert out.shape == (s, b, h * 128), "TE's dense contract flattens heads"
    assert out.dtype == torch.bfloat16
    ref = _fp64_reference(q, k, v)
    torch.testing.assert_close(out.float().cpu(), ref.float().cpu(), rtol=2e-2, atol=2e-2)
    grads = torch.autograd.grad(out.float().square().sum(), (q, k, v))
    for grad in grads:
        assert torch.isfinite(grad).all()
    del q, k, v, out, ref, grads
    torch.musa.empty_cache()


def test_musa_mla_shape_env_selects_backend_order(musa_live, monkeypatch):
    """ATTN_BACKEND selects the kernel order for the MuDNN-unsafe MLA shape.

    ``unfused`` must bypass mate entirely, ``mate`` must prefer the TileLang
    kernel over both alternatives, and ``auto`` lands on mate for the mixed
    192/128 training shape that MuDNN's backward rejects.
    """
    _device_required()
    pytest.importorskip("mate")
    torch.manual_seed(23)
    q = torch.randn(8, 1, 2, 192, device="musa", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(8, 1, 2, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)

    unfused_calls, mate_calls, native_calls = [], [], []

    def unfused_backend(*args, **kwargs):
        unfused_calls.append(kwargs.get("attn_mask_type"))
        return torch.zeros(8, 1, 256, device="musa", dtype=torch.bfloat16)

    def fake_mate(q_, k_, v_, **kwargs):
        mate_calls.append(kwargs.get("causal"))
        assert tuple(q_.shape) == (1, 8, 2, 192), "sbhd must reach mate as bshd"
        return torch.zeros(1, 8, 2, 128, device="musa", dtype=torch.bfloat16)

    stub = SimpleNamespace(
        training=True,
        attention_dropout=0.0,
        softmax_scale=None,
        deterministic=True,
        window_size=None,
        qkv_format="sbhd",
        config=None,
        unfused_attention=unfused_backend,
    )
    monkeypatch.setattr(_attention, "_mate_flash_fn", lambda: fake_mate)
    wrapped = _attention._tedpa_forward(lambda *a, **kw: native_calls.append(a) or "native")

    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ATTN_BACKEND", "unfused")
    out = wrapped(stub, q, q, v, None, "causal")
    assert (len(unfused_calls), len(mate_calls), len(native_calls)) == (1, 0, 0)
    assert out.shape == (8, 1, 256), "TE's dense contract flattens heads"

    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ATTN_BACKEND", "mate")
    wrapped(stub, q, q, v, None, "causal")
    assert (len(unfused_calls), len(mate_calls), len(native_calls)) == (1, 1, 0)

    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ATTN_BACKEND", "auto")
    wrapped(stub, q, q, v, None, "causal")
    assert (len(unfused_calls), len(mate_calls), len(native_calls)) == (1, 2, 0)

    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ATTN_BACKEND", "mudnn")
    wrapped(stub, q, q, v, None, "causal")
    assert (len(unfused_calls), len(mate_calls), len(native_calls)) == (1, 2, 1)
    del q, v, out
    torch.musa.empty_cache()


def test_musa_out_of_window_head_dim_routes_to_unfused(musa_live):
    """FP32/odd head dims run TE's own unfused backend, not the flash assert."""
    _device_required()
    torch.manual_seed(5)
    module = _te_attention(torch.bfloat16, 32, 4, 4)
    module.train()
    q = torch.randn(64, 1, 4, 32, device="musa", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(64, 1, 4, 32, device="musa", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(64, 1, 4, 32, device="musa", dtype=torch.bfloat16, requires_grad=True)
    calls = []
    wrapped = _attention._tedpa_forward(lambda *a, **kw: calls.append(a) or "native")
    out = wrapped(module, q, k, v, None, "causal")
    assert calls == []
    assert out.shape == (64, 1, 128)
    ref = _fp64_reference(q, k, v).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=2e-2, atol=2e-2)
    grads = torch.autograd.grad(out.float().square().sum(), (q, k, v))
    for grad in grads:
        assert torch.isfinite(grad).all()
    del module, q, k, v, out, ref
    torch.musa.empty_cache()


def test_musa_training_dropout_routes_to_unfused(musa_live):
    """Native flash traps on the device for dropout_p > 0; unfused handles it."""
    _device_required()
    torch.manual_seed(5)
    module = _te_attention(torch.bfloat16, 128, 8, 8, dropout=0.2)
    module.train()
    q = torch.randn(32, 1, 8, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(32, 1, 8, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(32, 1, 8, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    calls = []
    wrapped = _attention._tedpa_forward(lambda *a, **kw: calls.append(a) or "native")
    out = wrapped(module, q, k, v, None, "causal")
    assert calls == []
    (out.float().square().sum()).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    del module, q, k, v, out
    torch.musa.empty_cache()


def _megatron_like_original(module):
    """Stand in for Megatron's TEDotProductAttention.forward call shape.

    Production intercepts the Megatron extension forward, whose ``original``
    accepts ``attention_bias``/``packed_seq_params``/``num_splits``; emulate it
    over the raw TE module by pinning per-span sbhd layouts.
    """

    def original(
        self,
        q,
        k,
        v,
        attention_mask,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
        num_splits=None,
    ):
        assert self is module
        mask = getattr(attn_mask_type, "name", attn_mask_type)
        return module(q, k, v, attention_mask, qkv_format="sbhd", attn_mask_type=mask)

    return original


def test_musa_packed_thd_matches_per_sequence_reference(musa_live):
    """Padded THD runs per-sequence vendor calls with zero-valued padding rows."""
    _device_required()
    torch.manual_seed(5)
    lens, pads = (4, 0, 6), (4, 0, 10)
    heads, kv_heads, head_dim = 8, 8, 128
    total = sum(pads)
    module = _te_attention(
        torch.bfloat16, head_dim, heads, kv_heads, qkv_format="thd", mask_type="padding_causal"
    )
    module.train()
    q = torch.randn(total, heads, head_dim, device="musa", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(
        total, kv_heads, head_dim, device="musa", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        total, kv_heads, head_dim, device="musa", dtype=torch.bfloat16, requires_grad=True
    )
    packed = SimpleNamespace(
        cu_seqlens_q=torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="musa"
        ),
        cu_seqlens_kv=torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="musa"
        ),
        cu_seqlens_q_padded=torch.tensor(
            [0] + list(torch.tensor(pads).cumsum(0)), dtype=torch.int32, device="musa"
        ),
        cu_seqlens_kv_padded=torch.tensor(
            [0] + list(torch.tensor(pads).cumsum(0)), dtype=torch.int32, device="musa"
        ),
    )
    wrapped = _attention._tedpa_forward(_megatron_like_original(module))
    out = wrapped(module, q, k, v, None, "padding_causal", packed_seq_params=packed)
    assert out.shape == (total, heads * head_dim)
    out4 = out.reshape(total, heads, head_dim)
    q64, k64, v64 = (t.detach().double().cpu() for t in (q, k, v))
    start = 0
    for length, pad in zip(lens, pads):
        if length:
            ref = _fp64_reference(
                q64[start : start + length, None],
                k64[start : start + length, None],
                v64[start : start + length, None],
            )
            err = (
                (out4[start : start + length].double().cpu() - ref.reshape(length, heads, head_dim))
                .abs()
                .max()
                .item()
            )
            assert err < 2e-2, f"sequence error {err}"
        if pad > length:
            assert torch.count_nonzero(out4[start + length : start + pad]) == 0
        start += pad
    grads = torch.autograd.grad(out.float().square().sum(), (q, k, v))
    for grad in grads:
        assert torch.isfinite(grad).all()
    del module, q, k, v, out, out4
    torch.musa.empty_cache()


def test_musa_padding_tuple_mask_routes_to_unfused(musa_live):
    """Megatron-style padding tuples reach the unfused backend with TE shapes."""
    _device_required()
    torch.manual_seed(5)
    module = _te_attention(torch.bfloat16, 32, 4, 4, mask_type="padding_causal")
    module.train()
    s = 8
    q = torch.randn(s, 2, 4, 32, device="musa", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(s, 2, 4, 32, device="musa", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(s, 2, 4, 32, device="musa", dtype=torch.bfloat16, requires_grad=True)
    valid = torch.tensor([[True] * (s - 2) + [False] * 2, [True] * s], device="musa")
    calls = []
    wrapped = _attention._tedpa_forward(lambda *a, **kw: calls.append(a) or "native")
    out = wrapped(module, q, k, v, (~valid, ~valid), "padding_causal")
    assert calls == []
    out4 = out.reshape(s, 2, 4, 32)
    for b in range(2):
        length = int(valid[b].sum())
        ref = _fp64_reference(
            q[:length, b][:, None], k[:length, b][:, None], v[:length, b][:, None]
        )
        err = (out4[:length, b].double().cpu() - ref.reshape(length, 4, 32)).abs().max().item()
        assert err < 2e-2, f"batch {b} error {err}"
        assert torch.count_nonzero(out4[length:, b]) == 0
    del module, q, k, v, out, out4
    torch.musa.empty_cache()


@pytest.mark.parametrize("mask_name", ["causal_bottom_right", "padding_causal_bottom_right"])
def test_packed_preserves_bottom_right_causal_alignment(monkeypatch, mask_name):
    q = torch.zeros(2, 1, 1)
    k = torch.zeros(3, 1, 1)
    v = torch.tensor([1.0, 2.0, 6.0]).reshape(3, 1, 1)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2]), cu_seqlens_kv=torch.tensor([0, 3]))

    def dispatch(module, query, key, value, mask, enum, name, bias, layout, original):
        diagonal = key.shape[0] - query.shape[0] if name.endswith("bottom_right") else 0
        allowed = torch.ones(query.shape[0], key.shape[0], dtype=torch.bool).tril(diagonal)
        weights = allowed.float() / allowed.sum(-1, keepdim=True)
        return (weights @ value[:, 0, 0]).reshape(query.shape[0], 1, 1, 1)

    monkeypatch.setattr(_attention, "_dense_dispatch", dispatch)
    out = _attention._packed_forward(SimpleNamespace(), q, k, v, mask_name, packed, None)
    torch.testing.assert_close(out[:, 0], torch.tensor([1.5, 3.0]))


def test_capability_dispatch_declines_when_te_selects_by_capability(monkeypatch):
    """No hard-coded flash in TE -> the vendor path handles every case."""
    from megatron_musa_patch import _compat

    monkeypatch.setattr(_attention, "_musa_live", lambda: True)
    monkeypatch.setattr(_compat, "module_source_contains", lambda name, *m: False)
    original = object()
    # Decline = the factory returns None; the engine keeps the original.
    assert _attention._tedpa_forward(original) is None
