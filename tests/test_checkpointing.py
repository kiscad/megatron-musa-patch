"""DCP device-selector tests without importing the vendor Megatron/TE stack.

Hardware save/load and external async process safety require separate tests.
"""

from __future__ import annotations

import pytest

from megatron_musa_patch.patches import _checkpointing


@pytest.fixture
def dcp_selector(stub_module):
    from types import SimpleNamespace

    _checkpointing._uninstall_dcp_device()
    device = SimpleNamespace(type="musa")
    cuda = SimpleNamespace(current_stream=lambda: SimpleNamespace(device=device))
    stub_module("torch", musa=SimpleNamespace(is_available=lambda: True), cuda=cuda)
    original = lambda: "cuda"
    module = stub_module(
        "torch.distributed.checkpoint.filesystem", _get_available_device_type=original
    )
    yield module, original, device
    _checkpointing._uninstall_dcp_device()


def test_dcp_selector_lifecycle_and_real_cuda(dcp_selector):
    import inspect

    module, original, device = dcp_selector
    assert _checkpointing._install_dcp_device()
    replacement = module._get_available_device_type
    assert inspect.signature(replacement) == inspect.signature(original)
    assert replacement() == "musa"
    assert not _checkpointing._install_dcp_device()
    assert module._get_available_device_type is replacement
    device.type = "cuda"
    assert replacement() == "cuda"
    _checkpointing._uninstall_dcp_device()
    assert module._get_available_device_type is original
    assert _checkpointing._install_dcp_device()
    third_party = lambda: "other"
    module._get_available_device_type = third_party
    _checkpointing._uninstall_dcp_device()
    assert module._get_available_device_type is third_party


@pytest.mark.parametrize("selected", ["cpu", "musa", "xpu", None])
def test_dcp_selector_preserves_other_devices(dcp_selector, selected):
    module, _, _ = dcp_selector
    module._get_available_device_type = lambda: selected
    assert _checkpointing._install_dcp_device()
    assert module._get_available_device_type() == selected


def test_dcp_selector_without_musa(dcp_selector, monkeypatch):
    import sys

    module, original, _ = dcp_selector
    monkeypatch.setattr(sys.modules["torch"].musa, "is_available", lambda: False)
    assert not _checkpointing._install_dcp_device()
    assert module._get_available_device_type is original
