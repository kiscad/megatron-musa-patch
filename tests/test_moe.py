"""Module-local MoE adapters: FP64 top-k reference and namespace forwarding."""

from types import SimpleNamespace

import pytest

from megatron_musa_patch.patches import _moe

torch = pytest.importorskip("torch")


def _proxy():
    return _moe._MoeTorchProxy(torch)


@pytest.mark.parametrize("dim", [1, -1, None])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("sorted_", [True, False])
def test_fp64_topk_matches_cpu_reference(dim, largest, sorted_):
    """The FP64 reference matches torch.topk semantics on CPU input."""
    generator = torch.Generator().manual_seed(7)
    scores = torch.rand(8, 6, dtype=torch.float64, generator=generator)
    values, indices = _moe._fp64_topk(torch, scores, 2, dim, largest, sorted_)
    expected = torch.topk(
        scores, k=2, dim=1 if dim is None else dim, largest=largest, sorted=sorted_
    )
    torch.testing.assert_close(values, expected.values)
    assert torch.equal(indices, expected.indices)
    assert values.dtype == torch.float64
    assert indices.dtype == torch.int64


def test_fp64_topk_keeps_gradients_on_the_original_tensor():
    scores = torch.rand(4, 8, dtype=torch.float64, requires_grad=True)
    values, indices = _moe._fp64_topk(torch, scores, 3, 1, True, True)
    assert values.requires_grad
    values.sum().backward()
    assert scores.grad is not None
    # Only the selected positions carry gradient.
    assert int(scores.grad.count_nonzero()) == 4 * 3


def test_proxy_adapts_only_fp64_and_forwards_the_rest(monkeypatch):
    proxy = _proxy()
    sentinel = object()
    monkeypatch.setattr(torch, "custom_kernel", lambda: sentinel, raising=False)
    # Non-float64 input uses the native kernel.
    out = proxy.topk(torch.rand(4, 8), 2, dim=1)
    expected = torch.topk(torch.rand(4, 8), 2, dim=1)  # shape contract only
    assert out.values.shape == expected.values.shape
    # Unknown attributes forward to the real namespace.
    assert proxy.custom_kernel() is sentinel
    assert proxy.zeros(2).shape == (2,)
    # out= is delegated verbatim.
    values = torch.empty(4, 2, dtype=torch.float32)
    indices = torch.empty(4, 2, dtype=torch.int64)
    proxy.topk(torch.rand(4, 8), 2, dim=1, out=(values, indices))
    assert values.abs().sum() > 0


def test_fp64_musa_input_uses_reference_path(monkeypatch):
    pytest.importorskip("torch_musa")
    if not torch.musa.is_available():
        pytest.skip("no visible MUSA device")
    scores = torch.rand(8, 6, dtype=torch.float64, device="musa")
    values, indices = _proxy().topk(scores, 2, dim=1)
    expected = torch.topk(scores.cpu(), k=2, dim=1)
    torch.testing.assert_close(values.cpu(), expected.values)
    assert values.device.type == "musa"
    assert indices.device == scores.device


def test_topk_patch_targets_moe_utils_namespace():
    patch = next(p for p in _moe.PATCHES if p.id == "megatron.moe.topk.fp64-reference")
    assert patch.target == "megatron.core.transformer.moe.moe_utils:torch"
    assert "TopkOut" in patch.rationale


# --- P05: fused permute/unpermute demotion -------------------------------


def _permute_env(stub_module, dtype_marker="fp32"):
    calls = []

    def original(
        tokens, routing_map, probs=None, num_out_tokens=None, fused=False, drop_and_pad=False
    ):
        calls.append(
            {
                "fused": fused,
                "probs": probs,
                "num_out_tokens": num_out_tokens,
                "drop_and_pad": drop_and_pad,
            }
        )
        return "permuted", None, "indices"

    reduce_op = SimpleNamespace(SUM="SUM")
    stub_module(
        "torch",
        float32="fp32",
        float64="fp64",
        float16="fp16",
        bfloat16="bf16",
        distributed=SimpleNamespace(ReduceOp=reduce_op),
        musa=SimpleNamespace(is_available=lambda: True),
    )
    stub_module("torch.distributed", ReduceOp=reduce_op)
    patch = next(p for p in _moe.PATCHES if p.id == "megatron.moe.permutation.unfused-musa")
    wrapped = patch.replace(original)

    class Tensor:
        device = SimpleNamespace(type="musa")
        shape = (4, 2)

        def __init__(self, dtype):
            self.dtype = dtype

    return wrapped, calls, Tensor


def test_permute_demotes_fused_for_broken_dtypes(stub_module):
    wrapped, calls, Tensor = _permute_env(stub_module)
    routing_map = "map"
    assert wrapped(Tensor("fp32"), routing_map, fused=True, num_out_tokens=8) == (
        "permuted",
        None,
        "indices",
    )
    assert calls[-1] == {"fused": False, "probs": None, "num_out_tokens": 8, "drop_and_pad": False}
    # float64 is broken as well
    wrapped(Tensor("fp64"), routing_map, fused=True)
    assert calls[-1]["fused"] is False
    # float16/bfloat16 keep the fused kernel
    wrapped(Tensor("fp16"), routing_map, fused=True)
    assert calls[-1]["fused"] is True
    wrapped(Tensor("bf16"), routing_map, fused=True)
    assert calls[-1]["fused"] is True
    # an explicit fused=False never gets upgraded
    wrapped(Tensor("fp32"), routing_map, fused=False)
    assert calls[-1]["fused"] is False


def test_unpermute_requires_its_permute_companion(stub_module):
    calls = []

    def original(
        tokens,
        sorted_indices,
        restore_shape,
        probs=None,
        routing_map=None,
        fused=False,
        drop_and_pad=False,
    ):
        calls.append(fused)
        return "restored"

    reduce_op = SimpleNamespace(SUM="SUM")
    stub_module(
        "torch",
        float32="fp32",
        float64="fp64",
        float16="fp16",
        bfloat16="bf16",
        distributed=SimpleNamespace(ReduceOp=reduce_op),
        musa=SimpleNamespace(is_available=lambda: True),
    )
    stub_module("torch.distributed", ReduceOp=reduce_op)
    patches = {p.id: p for p in _moe.PATCHES}
    unpermute = patches["megatron.moe.unpermutation.unfused-musa"]
    assert unpermute.requires == ("megatron.moe.permutation.unfused-musa",)

    tensor = SimpleNamespace(dtype="fp32", device=SimpleNamespace(type="musa"))
    wrapped = unpermute.replace(original)
    assert wrapped(tensor, "idx", torch.Size([4, 2]), fused=True) == "restored"
    assert calls[-1] is False


def test_moe_topk_declines_when_fp64_topk_works(monkeypatch):
    from megatron_musa_patch.patches import _moe

    monkeypatch.setattr(_moe, "_musa_live", lambda: True)
    monkeypatch.setattr(_moe, "_fp64_topk_works_on_musa", lambda: True)
    original = object()
    # Decline = replace() returns None; the engine then keeps the original.
    assert _moe._moe_torch_namespace(original) is None


def test_moe_topk_probe_failure_keeps_fallback(monkeypatch):
    from megatron_musa_patch.patches import _moe

    monkeypatch.setattr(_moe, "_musa_live", lambda: True)
    monkeypatch.setattr(_moe, "_fp64_topk_works_on_musa", lambda: False)
    original = object()
    assert isinstance(_moe._moe_torch_namespace(original), _moe._MoeTorchProxy)


@pytest.mark.parametrize("broken", [False, True])
def test_topk_capability_probe_preserves_rng(monkeypatch, broken):
    from megatron_musa_patch.patches import _moe

    # Route the probe's allocation to CPU; this checks probe side effects,
    # independent of whether the installed MUSA kernel supports float64.
    original_arange = torch.arange
    monkeypatch.setattr(
        torch,
        "arange",
        lambda *args, **kwargs: original_arange(*args, **dict(kwargs, device="cpu")),
    )
    if broken:

        def fail(*args, **kwargs):
            raise RuntimeError("unsupported")

        monkeypatch.setattr(torch, "topk", fail)
    state = torch.get_rng_state().clone()
    assert _moe._fp64_topk_works_on_musa() is (not broken)
    assert torch.equal(torch.get_rng_state(), state)


def test_fp64_topk_preserves_named_result():
    scores = torch.tensor([0.1, 0.3, 0.2], dtype=torch.float64)
    result = _moe._fp64_topk(torch, scores, 2, None, True, True)
    assert isinstance(result, torch.return_types.topk)
    torch.testing.assert_close(result.values, torch.tensor([0.3, 0.2], dtype=torch.float64))
    assert result.indices.tolist() == [1, 2]


def test_fp64_musa_out_is_delegated(monkeypatch):
    from unittest.mock import Mock

    native = Mock(return_value=object())
    namespace = SimpleNamespace(float64=torch.float64, topk=native)
    scores = SimpleNamespace(dtype=torch.float64, device=SimpleNamespace(type="musa"))
    outputs = (object(), object())
    monkeypatch.setattr(_moe, "_musa_live", lambda: True)
    result = _moe._MoeTorchProxy(namespace).topk(scores, 2, out=outputs)
    native.assert_called_once_with(scores, 2, dim=None, largest=True, sorted=True, out=outputs)
    assert result is native.return_value
