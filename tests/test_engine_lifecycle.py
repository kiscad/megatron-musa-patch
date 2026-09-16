"""Regression coverage for composition, ownership and failure diagnostics."""
from __future__ import annotations

import importlib
import sys
import types

import pytest

from megatron_musa_patch._engine import AttrPatch, Engine, HookPatch
from megatron_musa_patch._errors import MegatronMusaPatchError, PatchConflict, PatchTargetMissing


def _wrap(label):
    def replace(old):
        def wrapper():
            return old() + label
        return wrapper
    return replace


def test_same_target_chain_is_idempotent_and_reloadable(engine, fake_package):
    name = fake_package("chain_target", "def fn(): return 'base'\n")
    engine.register([
        AttrPatch("chain.one", f"{name}:fn", _wrap("1")),
        AttrPatch("chain.two", f"{name}:fn", _wrap("2")),
    ])
    engine.apply_now()
    module = sys.modules[name]
    assert module.fn() == "base12"
    engine.apply_now()
    assert module.fn() == "base12"
    importlib.reload(module)
    assert module.fn() == "base12"
    engine.unapply()
    assert module.fn() == "base"
    engine.apply_now()
    assert module.fn() == "base12"


def test_register_after_install_rebuilds_chain(engine, stub_module):
    module = stub_module("live_registration", fn=lambda: "base")
    engine.register([AttrPatch("one", "live_registration:fn", _wrap("1"))])
    engine.install()
    engine.register([AttrPatch("two", "live_registration:fn", _wrap("2"))])
    assert module.fn() == "base12"
    engine.unapply()
    assert module.fn() == "base"


def test_failed_registration_is_atomic(engine):
    patch = AttrPatch("duplicate", "json:loads", lambda old: old)
    with pytest.raises(ValueError):
        engine.register([patch, patch])
    assert engine.report() == []


def test_failure_is_recorded_and_target_chain_is_atomic(engine, stub_module):
    module = stub_module("atomic_chain", fn=lambda: "base")
    original = module.fn

    def fail(old):
        raise ValueError("broken factory")

    engine.register([
        AttrPatch("good", "atomic_chain:fn", _wrap("1")),
        AttrPatch("bad", "atomic_chain:fn", fail),
    ])
    with pytest.raises(MegatronMusaPatchError, match="broken factory"):
        engine.install()
    assert module.fn is original
    assert engine.report()[1]["status"] == "failed"
    assert engine.report()[0]["status"] == "pending"


def test_missing_attribute_is_recorded_failed(engine, stub_module):
    stub_module("missing_attr")
    engine.register([AttrPatch("missing", "missing_attr:nope", lambda old: 1)])
    with pytest.raises(PatchTargetMissing):
        engine.install()
    assert engine.report()[0]["status"] == "failed"


def test_failed_version_check_can_be_retried(engine, monkeypatch):
    from megatron_musa_patch import _engine

    def fail():
        raise ValueError("version rejected")

    monkeypatch.setattr(_engine, "check_version", fail)
    with pytest.raises(ValueError):
        engine.install()
    assert not engine._installed
    monkeypatch.setattr(_engine, "check_version", lambda: None)
    engine.install()
    assert engine._installed


def test_optional_target_and_broken_dependency_are_distinct(engine, fake_package):
    name = fake_package("broken_dependency", "import missing_dependency_xyz\nvalue = 1\n")
    engine.register([AttrPatch("broken", f"{name}:value", lambda old: 2)])
    with pytest.raises(ModuleNotFoundError, match="missing_dependency_xyz"):
        engine.apply_now()
    assert engine.report()[0]["status"] == "failed"


def test_internal_import_error_is_not_swallowed(engine, fake_package):
    name = fake_package("broken_import", "raise ImportError('ABI mismatch')\n")
    engine.register([AttrPatch("broken", f"{name}:value", lambda old: 2)])
    with pytest.raises(ImportError, match="ABI mismatch"):
        engine.apply_now()


def test_missing_parent_is_optional(engine):
    engine.register([AttrPatch("optional", "missing_parent_xyz.child:value", lambda old: 2)])
    engine.apply_now()
    assert engine.report()[0]["status"] == "skipped"


def test_unapply_clears_pending_and_filter_state(engine, stub_module, monkeypatch):
    module = stub_module("filter_cycle", value=1)
    engine.register([AttrPatch("filter", "filter_cycle:value", lambda old: old + 1)])
    engine.install()
    assert module.value == 2
    engine.unapply()
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_DISABLE", "filter")
    engine.apply_now()
    assert module.value == 1
    assert engine.pending_modules() == []
    engine.unapply()
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_DISABLE")
    engine.apply_now()
    assert module.value == 2


def test_apply_now_respects_master_switch_after_install(engine, fake_package, monkeypatch):
    name = fake_package("master_cycle", "value = 1\n")
    engine.register([AttrPatch("master", f"{name}:value", lambda old: 2)])
    engine.install()
    monkeypatch.setenv("MEGATRON_MUSA_PATCH", "0")
    engine.apply_now()
    assert name not in sys.modules


def test_reversible_and_irreversible_hooks(engine, stub_module):
    stub_module("hook_cycle")
    calls = []
    engine.register([
        HookPatch("reversible", "hook_cycle", lambda: calls.append("run"), undo=lambda: calls.append("undo")),
        HookPatch("irreversible", "hook_cycle", lambda: calls.append("once")),
        HookPatch("declined", "hook_cycle", lambda: False),
    ])
    engine.install()
    assert [r["status"] for r in engine.report()] == ["applied", "applied", "skipped"]
    engine.unapply()
    assert calls == ["run", "once", "undo"]
    assert engine.is_applied("irreversible")
    engine.install()
    assert calls == ["run", "once", "undo", "run"]


def test_failed_hook_keeps_following_hooks_pending_for_retry(engine, stub_module):
    stub_module("hook_retry")
    calls = []

    def flaky():
        calls.append("flaky")
        if calls.count("flaky") == 1:
            raise RuntimeError("try again")

    engine.register([
        HookPatch("first", "hook_retry", lambda: calls.append("first")),
        HookPatch("flaky", "hook_retry", flaky),
        HookPatch("last", "hook_retry", lambda: calls.append("last")),
    ])
    with pytest.raises(RuntimeError, match="try again"):
        engine.install()
    engine.apply_now()
    assert calls == ["first", "flaky", "flaky", "last"]


def test_failed_hook_undo_blocks_reinstall_until_unapply_succeeds(engine, stub_module):
    stub_module("hook_undo_retry")
    state = {"fail_undo": True}

    def undo():
        if state["fail_undo"]:
            raise RuntimeError("undo exploded")
        state["undone"] = True

    engine.register([HookPatch("undo-fail", "hook_undo_retry", lambda: None, undo=undo)])
    engine.install()
    assert engine.is_applied("undo-fail")

    try:
        with pytest.raises(MegatronMusaPatchError, match="undo exploded"):
            engine.unapply()
        assert engine._records["undo-fail"].status == "failed"
        with pytest.raises(PatchConflict, match="cleanup is incomplete"):
            engine.install()
    finally:
        # A failing undo must not poison the engine fixture's teardown either.
        state["fail_undo"] = False
        engine.unapply()  # retry succeeds

    assert state["undone"]
    assert not engine.is_applied("undo-fail")
    engine.install()  # and reinstall works again
    assert engine.is_applied("undo-fail")


def test_descriptors_and_inherited_attributes_restore_exactly(engine, stub_module):
    class Parent:
        @staticmethod
        def static(value):
            return value + 1

        @classmethod
        def class_method(cls, value):
            return cls.__name__, value

    class Child(Parent):
        pass

    original = vars(Parent)["class_method"]
    stub_module("descriptor_target", Parent=Parent, Child=Child)
    engine.register([
        AttrPatch("static", "descriptor_target:Child.static", lambda old: lambda v: old(v) + 1),
        AttrPatch("class", "descriptor_target:Parent.class_method", lambda old: lambda cls, v: old(cls, v + 1)),
    ])
    engine.install()
    assert Child().static(1) == 3
    assert Parent.class_method(1) == ("Parent", 2)
    assert Child.class_method(1) == ("Child", 2)
    engine.unapply()
    assert "static" not in vars(Child)
    assert vars(Parent)["class_method"] is original


def test_alias_repair_is_bounded_and_never_rebinds_flags(engine, stub_module):
    original = lambda: "base"
    target = stub_module("alias_target", fn=original, flag=True)
    inside = stub_module("megatron.alias_test", fn=original, flag=True)
    outside = stub_module("outside_alias", fn=original, flag=True)
    engine.register([
        AttrPatch("alias", "alias_target:fn", _wrap("1")),
        AttrPatch("flag", "alias_target:flag", lambda old: False),
    ])
    engine.install()
    assert inside.fn is target.fn
    assert outside.fn is original
    assert inside.flag and outside.flag
    engine.unapply()
    assert inside.fn is original


def test_unapply_preserves_external_writes(engine, stub_module):
    target = stub_module("ownership", fn=lambda: "base")
    engine.register([AttrPatch("owned", "ownership:fn", _wrap("1"))])
    engine.install()
    external = lambda: "external"
    target.fn = external
    with pytest.raises(PatchConflict):
        engine.apply_now()
    engine.unapply()
    assert target.fn is external


def test_multiple_engines_apply_all_hooks_and_attributes(engine, fake_package):
    name = fake_package("multiple_engines", "value = 1\n")
    calls = []
    other = Engine()
    try:
        engine.register([HookPatch("first", name, lambda: calls.append(1))])
        other.register([
            HookPatch("second", name, lambda: calls.append(2)),
            AttrPatch("value", f"{name}:value", lambda old: 42),
        ])
        engine.install()
        other.install()
        module = importlib.import_module(name)
        assert calls == [1, 2]
        assert module.value == 42
    finally:
        other.unapply()


@pytest.mark.parametrize("target", ["a:b:c", "a..b:c", "a:b.", "a:b..c"])
def test_malformed_targets_fail_early(target):
    with pytest.raises(ValueError, match="malformed"):
        AttrPatch("bad", target, lambda old: old)


@pytest.mark.parametrize("rebuild", [False, True])
def test_alias_commit_failure_restores_target_and_existing_aliases(engine, stub_module, monkeypatch, rebuild):
    from megatron_musa_patch import _engine

    original = lambda: "base"
    module = stub_module("atomic_alias", fn=original)
    alias = stub_module("megatron.atomic_alias", fn=original)
    engine.register([AttrPatch("first", "atomic_alias:fn", _wrap("1"))])
    if rebuild:
        engine.install()
    baseline = module.fn

    def fail(*args):
        raise RuntimeError("alias commit failed")

    with monkeypatch.context() as patcher:
        patcher.setattr(_engine, "_rebind_aliases", fail)
        with pytest.raises(MegatronMusaPatchError, match="alias commit failed"):
            if rebuild:
                engine.register([AttrPatch("second", "atomic_alias:fn", _wrap("2"))])
            else:
                engine.install()
    assert module.fn is baseline
    assert alias.fn is baseline
    engine.unapply()
    assert module.fn is original
    assert alias.fn is original


def test_failed_attribute_undo_is_retryable_and_does_not_block_other_cleanup(engine, stub_module):
    class Target:
        fail = False

        def __setattr__(self, name, value):
            if self.fail and name == "value" and value == 1:
                raise RuntimeError("attribute undo failed")
            object.__setattr__(self, name, value)

    target = Target()
    target.value = 1
    module = stub_module("retry_undo", target=target, other=1)
    engine.register([
        AttrPatch("other", "retry_undo:other", lambda old: 2),
        AttrPatch("target", "retry_undo:target.value", lambda old: 2),
    ])
    engine.install()
    target.fail = True
    try:
        with pytest.raises(MegatronMusaPatchError, match="attribute undo failed"):
            engine.unapply()
        assert module.other == 1
        assert target.value == 2
        with pytest.raises(PatchConflict, match="cleanup is incomplete"):
            engine.install()
    finally:
        target.fail = False
        engine.unapply()
    assert target.value == 1


def test_reload_retires_declined_patch_and_repairs_old_consumers(engine, fake_package, stub_module):
    state = {"needed": True}
    name = fake_package("conditional_reload", "def fn(): return 'upstream'\n")
    engine.register([AttrPatch("conditional", f"{name}:fn",
                               lambda old: _wrap("patched")(old) if state["needed"] else None)])
    engine.apply_now()
    module = sys.modules[name]
    consumer = stub_module("megatron.late_consumer", fn=module.fn)
    state["needed"] = False
    importlib.reload(module)
    assert consumer.fn is module.fn
    assert consumer.fn() == "upstream"
    assert engine.report()[0]["status"] == "skipped"
    assert not engine._bindings
    engine.unapply()
    assert consumer.fn is module.fn
