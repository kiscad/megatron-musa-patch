"""Boolean-key sorting contract: same permutation, nothing else touched."""

from __future__ import annotations

import pytest

from megatron_musa_patch.patches import _sort

torch = pytest.importorskip("torch")


def _argsort(original):
    return _sort._argsort_via_uint8(original)


def _sort_fn(original):
    return _sort._sort_via_uint8(original)


@pytest.fixture
def on_musa(monkeypatch):
    """Treat the CPU tensors under test as living on the MUSA device."""
    monkeypatch.setattr(_sort, "_on_musa", lambda tensor: True)


def test_bool_keys_are_sorted_as_uint8(on_musa):
    seen = {}

    def original(input, *args, **kwargs):
        seen.update(dtype=input.dtype, kwargs=kwargs)
        return torch.argsort(input, *args, **kwargs)

    wrapped = _argsort(original)
    keys = torch.tensor([[False, True, False, True]])

    indices = wrapped(keys, dim=-1, descending=True, stable=True)

    assert seen["dtype"] is torch.uint8, "the uint8 view keeps False<True ordering"
    assert seen["kwargs"] == {"dim": -1, "descending": True, "stable": True}
    assert indices.tolist() == [[1, 3, 0, 2]], "stable descending permutation preserved"


def test_the_permutation_matches_the_native_bool_sort(on_musa):
    torch.manual_seed(7)
    keys = torch.randint(0, 2, (4, 64)).bool()
    wrapped = _argsort(torch.argsort)

    for descending, stable in ((True, True), (False, True), (True, False)):
        expected = torch.argsort(keys, dim=-1, descending=descending, stable=stable)
        got = wrapped(keys, dim=-1, descending=descending, stable=stable)
        if stable:
            assert torch.equal(got, expected)
        else:
            assert torch.equal(keys.gather(-1, got), keys.gather(-1, expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.uint8])
def test_other_dtypes_reach_the_original_untouched(dtype, on_musa):
    def original(input, *args, **kwargs):
        assert input.dtype is dtype, "no conversion for dtypes the kernel serves"
        return "untouched"

    wrapped = _argsort(original)
    assert wrapped(torch.zeros(4, dtype=dtype)) == "untouched"


def test_bool_keys_off_the_musa_device_are_untouched(monkeypatch):
    monkeypatch.setattr(_sort, "_on_musa", lambda tensor: False)

    def original(input, *args, **kwargs):
        assert input.dtype is torch.bool, "only the MUSA kernel is missing"
        return "untouched"

    wrapped = _argsort(original)
    assert wrapped(torch.zeros(4, dtype=torch.bool)) == "untouched"


def test_non_tensor_inputs_are_passed_through():
    wrapped = _argsort(lambda input, *a, **k: ("sorted", input))
    assert wrapped([3, 1, 2]) == ("sorted", [3, 1, 2])


def test_sort_casts_its_values_back_to_bool(on_musa):
    wrapped = _sort_fn(torch.sort)
    keys = torch.tensor([False, True, True, False])

    values, indices = wrapped(keys, descending=True, stable=True)

    assert values.dtype is torch.bool
    assert values.tolist() == [True, True, False, False]
    assert indices.tolist() == [1, 2, 0, 3]


def test_the_hook_owns_all_four_spellings_and_undoes_them(monkeypatch):
    """torch is imported before activation, so this is a hook, not an AttrPatch."""
    import sys

    fake_torch = type(sys)("torch")
    fake_torch.musa = type("M", (), {"is_available": staticmethod(lambda: True)})()
    fake_torch.Tensor = type("Tensor", (), {})
    originals = {}
    for owner in (fake_torch, fake_torch.Tensor):
        for name in ("argsort", "sort"):
            originals[(id(owner), name)] = lambda *a, **k: "original"
            setattr(owner, name, originals[(id(owner), name)])
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(_sort, "_owned", {})

    assert _sort._install_bool_sort() is None
    assert len(_sort._owned) == 4, "both ops in both spellings"
    for owner in (fake_torch, fake_torch.Tensor):
        for name in ("argsort", "sort"):
            assert getattr(owner, name) is not originals[(id(owner), name)]

    _sort._uninstall_bool_sort()
    for owner in (fake_torch, fake_torch.Tensor):
        for name in ("argsort", "sort"):
            assert getattr(owner, name) is originals[(id(owner), name)]
    assert _sort._owned == {}


def test_the_hook_declines_without_a_live_musa_device(monkeypatch):
    import sys

    fake_torch = type(sys)("torch")
    fake_torch.musa = type("M", (), {"is_available": staticmethod(lambda: False)})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(_sort, "_owned", {})

    assert _sort._install_bool_sort() is False
    assert _sort._owned == {}


def test_undo_keeps_a_later_owners_binding(monkeypatch):
    import sys

    fake_torch = type(sys)("torch")
    fake_torch.musa = type("M", (), {"is_available": staticmethod(lambda: True)})()
    fake_torch.Tensor = type("Tensor", (), {})
    for owner in (fake_torch, fake_torch.Tensor):
        for name in ("argsort", "sort"):
            setattr(owner, name, lambda *a, **k: "original")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(_sort, "_owned", {})
    _sort._install_bool_sort()

    later = lambda *a, **k: "later"  # noqa: E731 - a third party replaced it afterwards
    fake_torch.argsort = later

    _sort._uninstall_bool_sort()

    assert fake_torch.argsort is later
    assert _sort._owned == {}


def test_the_ledger_is_complete():
    (patch,) = _sort.PATCHES
    assert patch.id == "torch.sort.bool-keys"
    assert patch.trigger == "megatron"
    assert patch.undo is not None
    assert patch.rationale and patch.strategy and patch.upstream and patch.remove_when


@pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="requires a live MUSA device",
)
def test_musa_bool_argsort_fails_natively_and_matches_cpu_once_adapted():
    """The failure this patch exists for, plus the permutation it must preserve."""
    keys = torch.randint(0, 2, (8, 512), device="musa").bool()

    with pytest.raises(RuntimeError, match="Sort"):
        torch.Tensor.argsort(keys, dim=-1, descending=True, stable=True)

    wrapped = _argsort(torch.Tensor.argsort)
    indices = wrapped(keys, dim=-1, descending=True, stable=True)

    assert torch.equal(indices.cpu(), keys.cpu().argsort(dim=-1, descending=True, stable=True))
