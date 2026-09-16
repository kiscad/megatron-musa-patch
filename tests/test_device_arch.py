"""Architecture policy and reversible import-triggered hooks, without a GPU."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron_musa_patch.patches import _device_arch, _distributed


@pytest.mark.parametrize(
    "raw,expected",
    [(None, (8, 3)), ("", (8, 3)), ("  ", (8, 3)), ("9", (9, 0)), (" 9.1 ", (9, 1)), ("10.0", (10, 0))],
)
def test_architecture_parse(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("MEGATRON_MUSA_PATCH_ARCH", raising=False)
    else:
        monkeypatch.setenv("MEGATRON_MUSA_PATCH_ARCH", raw)
    assert _device_arch.arch_tuple() == expected
    assert _device_arch.arch_major() == expected[0]


@pytest.mark.parametrize("raw", ["ampere", "8.3.1", "-1.0", "8.-3", "8.", ".3"])
def test_invalid_architecture_warns_and_falls_back(monkeypatch, raw, caplog):
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ARCH", raw)
    with caplog.at_level(logging.WARNING, logger="megatron_musa_patch"):
        assert _device_arch.arch_tuple() == (8, 3)
    assert "falling back to 8.3" in caplog.text


def test_arch_wrapper_snapshots_configuration(monkeypatch):
    original = Mock()
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ARCH", "9.1")
    replacement = _device_arch._replace_arch_version(original)
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ARCH", "8.3")
    assert replacement() == 9
    original.assert_not_called()


@pytest.fixture()
def capability_hook(stub_module, monkeypatch):
    monkeypatch.setattr(_device_arch, "_capability_override", None)
    original = Mock(return_value=(3, 1))
    cuda = SimpleNamespace(get_device_capability=original)
    musa = SimpleNamespace(get_device_capability=original)
    stub_module("torch", cuda=cuda, musa=musa)
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ARCH", "8.3")
    yield cuda, musa, original
    _device_arch._uninstall_torch_capability()


def test_capability_hook_is_reversible_and_idempotent(capability_hook):
    cuda, musa, original = capability_hook
    assert _device_arch._install_torch_capability() is None
    replacement = cuda.get_device_capability
    assert replacement() == (8, 3)
    assert replacement(device=2) == (8, 3)
    assert musa.get_device_capability is original
    assert _device_arch._install_torch_capability() is False
    assert cuda.get_device_capability is replacement
    _device_arch._uninstall_torch_capability()
    assert cuda.get_device_capability is original
    _device_arch._uninstall_torch_capability()
    assert cuda.get_device_capability is original


def test_capability_undo_does_not_overwrite_later_owner(capability_hook):
    cuda, _, _ = capability_hook
    _device_arch._install_torch_capability()
    later = Mock()
    cuda.get_device_capability = later
    _device_arch._uninstall_torch_capability()
    assert cuda.get_device_capability is later


def test_capability_can_be_reinstalled_with_new_configuration(capability_hook, monkeypatch):
    cuda, _, original = capability_hook
    _device_arch._install_torch_capability()
    _device_arch._uninstall_torch_capability()
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ARCH", "9.0")
    _device_arch._install_torch_capability()
    assert cuda.get_device_capability() == (9, 0)
    _device_arch._uninstall_torch_capability()
    assert cuda.get_device_capability is original


def test_capability_undo_restores_uncached_proxy_state(stub_module, monkeypatch):
    native_capability = Mock(return_value=(3, 1))

    class Proxy:
        def __getattr__(self, name):
            if name != "get_device_capability":
                raise AttributeError(name)
            self.get_device_capability = native_capability
            return native_capability

    proxy = Proxy()
    stub_module("torch", cuda=proxy)
    monkeypatch.setattr(_device_arch, "_capability_override", None)
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ARCH", "8.3")
    try:
        _device_arch._install_torch_capability()
        assert proxy.get_device_capability() == (8, 3)
        _device_arch._uninstall_torch_capability()
        assert "get_device_capability" not in vars(proxy)
        assert proxy.get_device_capability is native_capability
    finally:
        _device_arch._uninstall_torch_capability()


def test_capability_undo_does_not_resolve_deleted_proxy_attribute(stub_module, monkeypatch):
    lookups = []

    class Proxy:
        def __getattr__(self, name):
            lookups.append(name)
            raise AttributeError(name)

    proxy = Proxy()
    stub_module("torch", cuda=proxy)
    monkeypatch.setattr(_device_arch, "_capability_override", None)
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_ARCH", "8.3")
    try:
        _device_arch._install_torch_capability()
        del proxy.get_device_capability
        _device_arch._uninstall_torch_capability()
        assert not lookups
        assert "get_device_capability" not in vars(proxy)
    finally:
        _device_arch._uninstall_torch_capability()


@pytest.fixture()
def teardown_hook(stub_module, monkeypatch):
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_TEARDOWN", raising=False)
    monkeypatch.setattr(_distributed, "_teardown_callback", None)
    dist = stub_module(
        "torch.distributed",
        is_available=Mock(return_value=True),
        is_initialized=Mock(return_value=True),
        destroy_process_group=Mock(),
    )
    stub_module("torch", distributed=dist)
    atexit = stub_module("atexit", register=Mock(), unregister=Mock())
    yield dist, atexit
    _distributed._uninstall_clean_teardown()


def test_teardown_hook_registers_once_and_unregisters_without_destroying(teardown_hook):
    dist, atexit = teardown_hook
    assert _distributed._install_clean_teardown() is None
    callback = atexit.register.call_args.args[0]
    assert _distributed._install_clean_teardown() is False
    atexit.register.assert_called_once_with(callback)
    _distributed._uninstall_clean_teardown()
    atexit.unregister.assert_called_once_with(callback)
    dist.destroy_process_group.assert_not_called()
    assert _distributed._teardown_callback is None
    _distributed._uninstall_clean_teardown()
    atexit.unregister.assert_called_once()


def test_teardown_switch_records_skip(teardown_hook, monkeypatch):
    _, atexit = teardown_hook
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_TEARDOWN", "0")
    assert _distributed._install_clean_teardown() is False
    atexit.register.assert_not_called()


@pytest.mark.parametrize("available,initialized", [(True, True), (True, False), (False, True)])
def test_teardown_only_destroys_initialized_available_group(teardown_hook, available, initialized):
    dist, atexit = teardown_hook
    dist.is_available.return_value = available
    dist.is_initialized.return_value = initialized
    _distributed._install_clean_teardown()
    atexit.register.call_args.args[0]()
    assert dist.destroy_process_group.call_count == int(available and initialized)


def test_teardown_failure_is_best_effort(teardown_hook):
    dist, atexit = teardown_hook
    dist.destroy_process_group.side_effect = RuntimeError("already shutting down")
    _distributed._install_clean_teardown()
    atexit.register.call_args.args[0]()
    dist.destroy_process_group.assert_called_once()


def test_import_hooks_register_undo_handlers():
    capability = _device_arch.PATCHES[0]
    teardown = _distributed.PATCHES[0]
    assert capability.undo is _device_arch._uninstall_torch_capability
    assert teardown.undo is _distributed._uninstall_clean_teardown
