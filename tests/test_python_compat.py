"""Interpreter backports: install only when missing, undo only what we own."""

from __future__ import annotations

import sys
import typing

import pytest

from megatron_musa_patch.patches import _python_compat


@pytest.fixture
def without_override(monkeypatch):
    """Run as if this interpreter predates PEP 698 (Python < 3.12)."""
    monkeypatch.delattr(typing, "override", raising=False)
    _python_compat._owned.clear()
    yield
    typing.__dict__.pop("override", None)
    _python_compat._owned.clear()


def _patch():
    return next(p for p in _python_compat.PATCHES
                if p.id == "python.typing.override.backport")


def test_installs_typing_override_when_absent(without_override):
    assert _python_compat._install_typing_override() is None
    assert typing.override is sys.modules["typing_extensions"].override


def test_installed_decorator_keeps_pep698_runtime_behaviour(without_override):
    _python_compat._install_typing_override()

    class Base:
        def forward(self):
            return "base"

    class Child(Base):
        @typing.override
        def forward(self):
            return "child"

    assert Child().forward() == "child"
    assert Child.forward.__override__ is True


def test_falls_back_without_typing_extensions(without_override, monkeypatch):
    monkeypatch.setitem(sys.modules, "typing_extensions", None)
    assert _python_compat._install_typing_override() is None
    assert typing.override is _python_compat._fallback_override


def test_fallback_marks_and_returns_the_same_object():
    def method():
        return 1

    assert _python_compat._fallback_override(method) is method
    assert method.__override__ is True
    # Objects that reject attribute assignment are returned unchanged.
    assert _python_compat._fallback_override(len) is len


def test_declines_when_the_interpreter_already_has_override(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(typing, "override", sentinel, raising=False)
    _python_compat._owned.clear()
    assert _python_compat._install_typing_override() is False
    assert typing.override is sentinel
    # Nothing was owned, so undo must not remove the interpreter's own name.
    _python_compat._uninstall_typing_override()
    assert typing.override is sentinel


def test_undo_removes_only_our_binding_and_is_idempotent(without_override):
    _python_compat._install_typing_override()
    _python_compat._uninstall_typing_override()
    assert not hasattr(typing, "override")
    _python_compat._uninstall_typing_override()
    assert not hasattr(typing, "override")


def test_undo_does_not_overwrite_a_later_owner(without_override):
    _python_compat._install_typing_override()
    replacement = object()
    typing.override = replacement
    _python_compat._uninstall_typing_override()
    assert typing.override is replacement


def test_hook_runs_before_a_megatron_module_that_imports_override(
    engine, fake_package, monkeypatch, without_override
):
    """The blocker is an import-time `from typing import override` in Megatron."""
    monkeypatch.delitem(sys.modules, "megatron", raising=False)
    fake_package("megatron", "from typing import override\n\nvalue = 1\n")
    engine.register(_python_compat.PATCHES)
    engine.install()

    import megatron  # the real blocker shape: fails without the hook

    assert megatron.value == 1
    assert engine.report()[0]["status"] == "applied"


def test_ledger_names_the_upstream_files_and_the_interpreter_gap():
    patch = _patch()
    assert patch.trigger == "megatron"
    assert "typing.override" in patch.strategy or "override" in patch.strategy
    assert "gpt.py" in patch.upstream and "hybrid.py" in patch.upstream
    assert "3.12" in patch.rationale
