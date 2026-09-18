"""Selective activation must not rely on incidental registration order."""

from itertools import permutations
from types import SimpleNamespace

import pytest

from megatron_musa_patch.patches import _control_collectives, _layer_norm, _rope, _training


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_local_norm_flags_follow_class_in_any_registration_order(engine, stub_module, order):
    pytest.importorskip("torch")
    patches = [
        p
        for p in _layer_norm.PATCHES
        if p.id
        in {
            "megatron.fusions.fused-layer-norm.pure-torch",
            "megatron.fusions.fused-layer-norm.have-apex-flag",
            "megatron.fusions.persist-layer-norm.disable",
        }
    ]
    original = type("UpstreamNorm", (), {})
    module = stub_module(
        "megatron.core.fusions.fused_layer_norm",
        FusedLayerNorm=original,
        HAVE_FUSED_LAYER_NORM=False,
        HAVE_PERSIST_LAYER_NORM=True,
    )
    engine.register([patches[i] for i in order])
    engine.install()
    assert module.FusedLayerNorm._megatron_musa_patch_fallback
    assert module.HAVE_FUSED_LAYER_NORM is True
    assert module.HAVE_PERSIST_LAYER_NORM is False
    engine.unapply()
    assert module.FusedLayerNorm is original
    assert module.HAVE_FUSED_LAYER_NORM is False
    assert module.HAVE_PERSIST_LAYER_NORM is True


@pytest.mark.parametrize("switch", ["ONLY", "DISABLE"])
def test_flag_selection_does_not_enable_class_implicitly(engine, stub_module, monkeypatch, switch):
    original = type("UpstreamNorm", (), {})
    module = stub_module(
        "megatron.core.fusions.fused_layer_norm",
        FusedLayerNorm=original,
        HAVE_FUSED_LAYER_NORM=False,
        HAVE_PERSIST_LAYER_NORM=True,
    )
    patches = [p for p in _layer_norm.PATCHES if p.target.startswith(module.__name__ + ":")]
    selected = [p.id for p in patches if (p.attr_name == "FusedLayerNorm") == (switch == "DISABLE")]
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_" + switch, ",".join(selected))
    engine.register(patches)
    engine.install()
    assert module.FusedLayerNorm is original
    assert module.HAVE_FUSED_LAYER_NORM is False
    assert module.HAVE_PERSIST_LAYER_NORM is True
    for record in engine.report():
        assert record["status"] == "skipped"
        if record["requires"]:
            assert "requires applied companion" in record["detail"]


def test_block_norm_works_without_local_class_replacement(engine, stub_module):
    torch = pytest.importorskip("torch")
    upstream = type("BrokenApexNorm", (), {})
    local = stub_module("megatron.core.fusions.fused_layer_norm", FusedLayerNorm=upstream)
    block = stub_module("megatron.core.transformer.transformer_block", LayerNormImpl=object())
    engine.register([p for p in _layer_norm.PATCHES if p.attr_name == "LayerNormImpl"])
    engine.install()
    config = SimpleNamespace(
        normalization="RMSNorm", layernorm_zero_centered_gamma=False, sequence_parallel=False
    )
    norm = block.LayerNormImpl(config, 4)
    x = torch.randn(2, 4, requires_grad=True)
    expected = torch.nn.functional.rms_norm(x, (4,), norm.weight, norm.eps)
    torch.testing.assert_close(norm(x), expected)
    norm(x).sum().backward()
    assert x.grad is not None
    assert local.FusedLayerNorm is upstream


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_rope_dispatch_observes_live_kernels_without_mutating_config(
    engine, stub_module, monkeypatch, order
):
    kernel = lambda *args: None
    monkeypatch.setattr(_rope, "_apex_kernels", lambda: (kernel, kernel))
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=True)

    def dispatch(t, freqs, config, **kwargs):
        assert original_config.apply_rope_fusion is True  # including DURING dispatch
        return config.apply_rope_fusion

    original_config = config
    module = stub_module(
        _rope._ROPE_UTILS,
        fused_apply_rotary_pos_emb=None,
        fused_apply_rotary_pos_emb_thd=None,
        apply_rotary_pos_emb=dispatch,
    )
    engine.register([_rope.PATCHES[i] for i in order])
    engine.install()
    assert module.apply_rotary_pos_emb(None, None, config) is False
    assert module.apply_rotary_pos_emb(None, None, config, cu_seqlens=object()) is False
    # A third party can install native kernels after our dispatcher was built.
    module.fused_apply_rotary_pos_emb = kernel
    assert module.apply_rotary_pos_emb(None, None, config) is True


def test_native_rope_does_not_probe_apex(engine, stub_module, monkeypatch):
    def forbidden():
        pytest.fail("native kernels must not import/probe apex")

    monkeypatch.setattr(_rope, "_apex_kernels", forbidden)
    kernel = lambda *args: None
    stub_module(
        _rope._ROPE_UTILS,
        fused_apply_rotary_pos_emb=kernel,
        fused_apply_rotary_pos_emb_thd=kernel,
        apply_rotary_pos_emb=kernel,
    )
    engine.register(_rope.PATCHES)
    engine.install()


def test_checkpoint_context_alone_reports_missing_companion(engine, stub_module):
    original = lambda: "saved"
    module = stub_module("megatron.training.checkpointing", save_checkpoint=original)
    engine.register([p for p in _control_collectives.PATCHES if p.attr_name == "save_checkpoint"])
    engine.install()
    assert module.save_checkpoint is original
    assert engine.report()[0]["status"] == "skipped"
    assert "host-barrier-proxy" in engine.report()[0]["detail"]


@pytest.mark.parametrize("order", [(0,), (1,), (0, 1), (1, 0)])
def test_training_wrappers_are_selectable_and_commute(engine, stub_module, order):
    patches = [p for p in _training.PATCHES if p.attr_name == "validate_args"]
    module = stub_module("megatron.training.arguments", validate_args=lambda args: args)
    engine.register([patches[i] for i in order])
    engine.install()
    args = SimpleNamespace(
        profile=True, use_pytorch_profiler=False, overlap_grad_reduce=True, rank=1
    )
    assert module.validate_args(args) is args
    assert args.use_pytorch_profiler is (0 in order)
    assert args.overlap_grad_reduce is (1 not in order)
