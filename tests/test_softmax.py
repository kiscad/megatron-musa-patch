"""Fused-softmax availability: extension probe and torch fallback contract."""

import pytest

from megatron_musa_patch.patches import _softmax

torch = pytest.importorskip("torch")


def _patch():
    return next(p for p in _softmax.PATCHES if p.id == "megatron.softmax.kernel-availability.musa")


class _Softmax:
    attn_mask_type = "causal"
    scaled_masked_softmax_fusion = True
    input_in_float16 = True

    def get_batch_per_block(self, sq, sk, b, np):
        return 1


def test_kernel_available_returns_false_without_extension(monkeypatch):
    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "scaled_masked_softmax_cuda" else object(),
    )
    Mock_probe = lambda self, mask, b, np, sq, sk: pytest.fail(  # noqa: E731
        "must not reach the CUDA probe when the extension is absent"
    )
    wrapped = _patch().replace(Mock_probe)
    assert wrapped(_Softmax(), None, 2, 2, 64, 64) is False


def test_kernel_available_delegates_when_extension_exists(monkeypatch):
    import importlib.util

    calls = []

    def original(self, mask, b, np, sq, sk):
        calls.append((b, np, sq, sk))
        return True

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    wrapped = _patch().replace(original)
    assert wrapped(_Softmax(), None, 2, 4, 128, 128) is True
    assert calls == [(2, 4, 128, 128)]


def test_softmax_patch_target_is_the_class_method():
    patch = _patch()
    assert patch.attr_name == "FusedScaleMaskSoftmax.is_kernel_available"
    assert "scaled_masked_softmax_cuda" in patch.rationale
