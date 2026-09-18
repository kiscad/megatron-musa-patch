"""Declarative gates must be deterministic and safe before target resolution."""

from types import SimpleNamespace

import pytest

from megatron_musa_patch import _compat
from megatron_musa_patch._engine import AttrPatch, HookPatch


@pytest.mark.parametrize("override", ["1", "true", "*", "Transformer.Engine"])
def test_override_all_and_normalized_names(monkeypatch, override):
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_IGNORE_VERSION_GATES", override)
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "3.0")
    assert not _compat.check_version_gate("transformer_engine <2.1")[0]


@pytest.mark.parametrize(
    "installed,bound,blocked",
    [
        ("2.0", "==2.0.0", False),
        ("2.0.0", "<=2.0", False),
        ("2.0.0.1", ">2.0.0", False),
        ("2.0.0+vendor", "==2.0", False),
        ("2.0rc1", ">=2.0", False),
        ("unknown", "<2.1", True),
        ("2..0", "<3", True),
        ("2.0.post1", "==2", False),
        ("2.0.dev1+vendor.1", "==2", False),
    ],
)
def test_release_comparison(monkeypatch, installed, bound, blocked):
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_IGNORE_VERSION_GATES", raising=False)
    monkeypatch.setattr(_compat, "distribution_version", lambda name: installed)
    assert _compat.check_version_gate("vendor " + bound)[0] is blocked


@pytest.mark.parametrize(
    "spec",
    [
        "vendor >=2junk",
        "vendor <2.*",
        "vendor ==2.0rc1",
        "vendor >=2..0",
        "vendor >=2,",
        "vendor ~=2.0",
    ],
)
def test_gate_bounds_are_strict_numeric_releases(spec):
    with pytest.raises(ValueError):
        _compat.parse_version_gate(spec)


@pytest.mark.parametrize("patch_type", [AttrPatch, HookPatch])
def test_gate_validation_shared(patch_type):
    args = (
        dict(target="example:value", replace=lambda x: x)
        if patch_type is AttrPatch
        else dict(trigger="example", run=lambda: None)
    )
    with pytest.raises(ValueError):
        patch_type("invalid", version_gates=["vendor >=2"], **args)


def test_blocked_gate_does_not_resolve_removed_target(engine, stub_module, monkeypatch):
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "3.0")
    module = stub_module("gate_removed")
    engine.register(
        [
            AttrPatch(
                "removed",
                "gate_removed:OldClass.method",
                lambda x: pytest.fail("must not run"),
                version_gates=("vendor <3",),
            )
        ]
    )
    engine.install()
    assert engine.report()[0]["status"] == "skipped"
    assert not hasattr(module, "OldClass")


def test_source_probe_handles_top_level_module(tmp_path, monkeypatch):
    source = tmp_path / "probe.py"
    source.write_text("MARKER = True\n")
    monkeypatch.setattr(
        _compat,
        "find_spec_without_watchers",
        lambda name: SimpleNamespace(origin=str(source), submodule_search_locations=None),
    )
    assert _compat.module_source_contains("probe", "MARKER") is True
    assert _compat.module_source_contains("probe.child", "MARKER") is False


def test_missing_metadata_keeps_target_policy(monkeypatch):
    monkeypatch.setattr(_compat, "distribution_version", lambda name: None)
    assert not _compat.check_version_gate("vendor >=2")[0]


def test_gated_chain_dependencies_and_reinstall(engine, stub_module, monkeypatch):
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "3.0")
    module = stub_module("gate_chain", value=1, consumer=1)
    engine.register(
        [
            AttrPatch("consumer", "gate_chain:consumer", lambda old: 9, requires=("old",)),
            AttrPatch(
                "old", "gate_chain:value", lambda old: old + 10, version_gates=("vendor <3",)
            ),
            AttrPatch("always", "gate_chain:value", lambda old: old * 2),
        ]
    )
    engine.install()
    assert (module.value, module.consumer) == (2, 1)
    assert "requires" in engine.report()[0]["detail"]
    engine.unapply()
    assert (module.value, module.consumer) == (1, 1)
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_IGNORE_VERSION_GATES", "1")
    engine.install()
    assert (module.value, module.consumer) == (22, 9)
    engine.unapply()
    assert (module.value, module.consumer) == (1, 1)


def test_hook_gate_rechecked_on_reinstall(engine, stub_module, monkeypatch):
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "3")
    stub_module("gate_hook")
    calls = []
    engine.register(
        [
            HookPatch(
                "hook",
                "gate_hook",
                lambda: calls.append("run"),
                undo=lambda: calls.append("undo"),
                version_gates=("vendor <3",),
            )
        ]
    )
    engine.install()
    assert calls == []
    engine.unapply()
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_IGNORE_VERSION_GATES", "1")
    engine.install()
    engine.unapply()
    assert calls == ["run", "undo"]


def test_metadata_not_cached(monkeypatch):
    versions = iter(["2", "3"])
    monkeypatch.setattr(_compat, "distribution_version", lambda name: next(versions))
    assert not _compat.check_version_gate("vendor <3")[0]
    assert _compat.check_version_gate("vendor <3")[0]


@pytest.mark.parametrize("value", ["0", "false", "off", ""])
def test_false_override_values_do_not_bypass(monkeypatch, value):
    from megatron_musa_patch import _env

    monkeypatch.setenv("MEGATRON_MUSA_PATCH_IGNORE_VERSION_GATES", value)
    assert not _env.version_gate_overrides()


def test_source_probe_does_not_import_parents(tmp_path, monkeypatch):
    import sys

    package = tmp_path / "gate_source_probe"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('must not import')\n")
    (package / "child.py").write_text("MARKER = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _compat.module_source_contains("gate_source_probe.child", "MARKER") is True
    assert "gate_source_probe" not in sys.modules
    assert _compat.module_source_contains("gate_source_probe.child", "ABSENT") is False
    assert _compat.module_source_contains("gate_source_probe.missing", "MARKER") is False


def test_hook_metadata_failure_is_reported(engine, stub_module, monkeypatch):
    def broken_metadata(name):
        raise OSError("metadata unreadable")

    monkeypatch.setattr(_compat, "distribution_version", broken_metadata)
    stub_module("gate_error")
    engine.register(
        [
            HookPatch(
                "broken",
                "gate_error",
                lambda: pytest.fail("must not run"),
                version_gates=("vendor <3",),
            )
        ]
    )
    with pytest.raises(OSError, match="metadata unreadable"):
        engine.install()
    assert engine.report()[0]["status"] == "failed"
    assert "metadata unreadable" in engine.report()[0]["detail"]
