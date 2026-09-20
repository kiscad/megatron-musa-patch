"""Contract tests for the interpreter backports in ``patches/_python_compat``."""

from __future__ import annotations

import sys
import typing

import pytest

from megatron_musa_patch.patches import _python_compat


@pytest.fixture
def without_override(monkeypatch):
    """Present an interpreter that has no ``typing.override`` (CPython < 3.12)."""
    monkeypatch.delattr(typing, "override", raising=False)
    monkeypatch.setattr(_python_compat, "_owned", {})
    yield
    _python_compat._owned.clear()


def test_installs_override_when_absent(without_override):
    assert _python_compat._install_typing_override() is None
    assert callable(typing.override)


def test_installed_override_returns_the_method_and_marks_it(without_override):
    _python_compat._install_typing_override()

    class Base:
        def step(self):
            return "base"

    class Child(Base):
        @typing.override
        def step(self):
            return "child"

    assert Child().step() == "child"
    assert getattr(Child.step, "__override__", False) is True


def test_falls_back_when_typing_extensions_is_missing(without_override, monkeypatch):
    monkeypatch.setitem(sys.modules, "typing_extensions", None)

    assert _python_compat._install_typing_override() is None
    assert typing.override is _python_compat._fallback_override


def test_fallback_tolerates_objects_that_reject_attributes():
    class Slotted:
        __slots__ = ()

    method = Slotted.__init__
    assert _python_compat._fallback_override(method) is method


def test_declines_when_the_interpreter_already_provides_override(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(typing, "override", sentinel, raising=False)
    monkeypatch.setattr(_python_compat, "_owned", {})

    assert _python_compat._install_typing_override() is False
    assert typing.override is sentinel
    assert _python_compat._owned == {}


def test_undo_removes_only_what_the_patch_owns(without_override):
    _python_compat._install_typing_override()
    _python_compat._uninstall_typing_override()

    assert not hasattr(typing, "override")
    assert _python_compat._owned == {}


def test_undo_is_idempotent(without_override):
    _python_compat._install_typing_override()
    _python_compat._uninstall_typing_override()
    _python_compat._uninstall_typing_override()

    assert not hasattr(typing, "override")


def test_undo_keeps_a_later_owners_binding(without_override):
    _python_compat._install_typing_override()
    other = object()
    typing.override = other

    _python_compat._uninstall_typing_override()

    assert typing.override is other
    del typing.override


def test_undo_without_install_leaves_the_interpreter_alone(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(typing, "override", sentinel, raising=False)
    monkeypatch.setattr(_python_compat, "_owned", {})

    _python_compat._uninstall_typing_override()

    assert typing.override is sentinel


def test_hook_runs_before_the_trigger_module_executes(
    engine, fake_package, without_override, monkeypatch
):
    """The failure this patch exists for: ``from typing import override`` at import time."""
    # Another test (or the real stack) may have left ``megatron`` in sys.modules;
    # the watcher only sees a fresh import.
    monkeypatch.delitem(sys.modules, "megatron", raising=False)
    name = fake_package("megatron", "from typing import override\n\nvalue = 1\n")
    engine.register(_python_compat.PATCHES)
    engine.install()

    module = __import__(name)

    assert module.value == 1
