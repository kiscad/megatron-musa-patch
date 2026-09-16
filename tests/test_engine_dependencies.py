"""Companion requirements are explicit, ordered, and never auto-enabled."""
import pytest

from megatron_musa_patch._engine import AttrPatch


def test_missing_dependency_can_be_registered_later(engine, stub_module):
    module = stub_module("companions", provider=False, consumer=False)
    engine.register([AttrPatch("consumer", "companions:consumer", lambda old: True,
                               requires=("provider",))])
    engine.install()
    assert module.consumer is False
    assert engine.report()[0]["status"] == "skipped"
    engine.register([AttrPatch("provider", "companions:provider", lambda old: True)])
    assert module.provider and module.consumer


def test_declined_companion_skips_only_its_consumer(engine, stub_module):
    module = stub_module("companions", provider=False, consumer=False, unrelated=False)
    engine.register([
        AttrPatch("consumer", "companions:consumer", lambda old: True, requires=("provider",)),
        AttrPatch("provider", "companions:provider", lambda old: None),
        AttrPatch("unrelated", "companions:unrelated", lambda old: True),
    ])
    engine.install()
    assert module.consumer is False
    assert module.unrelated is True
    assert "requires applied companion" in engine.report()[0]["detail"]


@pytest.mark.parametrize("target", ["other:provider", "companions:consumer"])
def test_dependencies_cannot_cross_modules_or_depend_on_same_chain(engine, target):
    with pytest.raises(ValueError, match="another attribute in the same module"):
        engine.register([
            AttrPatch("consumer", "companions:consumer", lambda old: True, requires=("provider",)),
            AttrPatch("provider", target, lambda old: True),
        ])
    assert engine.report() == []


def test_dependency_cycle_is_rejected_atomically(engine):
    with pytest.raises(ValueError, match="cyclic patch requirements"):
        engine.register([
            AttrPatch("a", "companions:a", lambda old: True, requires=("b",)),
            AttrPatch("b", "companions:b", lambda old: True, requires=("a",)),
        ])
    assert engine.report() == []
