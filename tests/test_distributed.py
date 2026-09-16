"""The teardown hook owns only its callback, independently of device adaptation."""
import atexit
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron_musa_patch.patches import _distributed


@pytest.mark.parametrize("initialized", [False, True])
def test_teardown_is_independent_and_uninstall_does_not_destroy_groups(engine, stub_module, monkeypatch, initialized):
    callbacks = []
    dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: initialized,
                           destroy_process_group=Mock())
    stub_module("megatron")
    stub_module("torch", distributed=dist)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(atexit, "unregister", callbacks.remove)
    engine.register(_distributed.PATCHES)
    engine.install()
    engine.install()
    assert len(callbacks) == 1
    callbacks[0]()
    assert dist.destroy_process_group.call_count == int(initialized)
    engine.unapply()
    assert callbacks == []
    assert dist.destroy_process_group.call_count == int(initialized)
    assert _distributed._teardown_callback is None
