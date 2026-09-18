"""Maintenance contract: every patch is identifiable and independently reviewable."""

from __future__ import annotations

import sys

import pytest

from megatron_musa_patch import _compat
from megatron_musa_patch._compat import SUPPORTED_VERSION_SPEC, megatron_importable, parse_version
from megatron_musa_patch._engine import AppliedPatch, AttrPatch, HookPatch
from megatron_musa_patch._errors import MegatronMissing
from megatron_musa_patch.patches import MODULES, PATCHES


def test_patch_ids_are_nonempty_and_unique():
    assert PATCHES
    ids = [patch.id for patch in PATCHES]
    assert all(ids)
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.id)
def test_every_patch_has_an_actionable_maintenance_record(patch):
    for field in ("rationale", "strategy", "upstream", "remove_when"):
        value = getattr(patch, field)
        assert value.strip(), f"{patch.id} has no {field}"
        if field != "upstream":
            assert len(value.split()) >= 3, f"{patch.id}: {field} must explain, not just label"
        assert AppliedPatch(patch).as_dict()[field] == value


#: AttrPatch scopes above Megatron, each a reviewed exception that names its
#: target explicitly. ``megatron.`` remains the default for everything else.
_NON_MEGATRON_SCOPES = {
    # mcore-bridge's GDN forward re-imports fla's chunk_gated_delta_rule into
    # its own namespace, so the MUSA kernel choice is unreachable through
    # Megatron's binding alone on the path ms-swift's
    # ``--bridge_backend mcore-bridge`` executes.
    "mcore_bridge.ssm.gated-delta-rule.tilelang": "mcore_bridge",
    # Direct TransformerEngine models (te.pytorch.TransformerLayer in the
    # megatron-FSDP suite) construct these modules inside TE's own namespace;
    # Megatron's wrapper patches cannot reach them.
    "transformer_engine.layer-norm-linear.native-unfused": "transformer_engine",
    "transformer_engine.layer-norm-mlp.native-unfused": "transformer_engine",
}


def test_targets_and_hooks_have_explicit_scope():
    for patch in PATCHES:
        if isinstance(patch, AttrPatch):
            scope = _NON_MEGATRON_SCOPES.get(patch.id, "megatron")
            assert patch.module_name.startswith(scope + ".")
            assert ":" in patch.target
            assert patch.rebind_prefixes == (scope,)
        else:
            # transformer_engine is the one sanctioned exception: MT-TE's
            # import-time factory shim breaks eager torch.jit.script before
            # any Megatron import (ms-swift's zigzag_ring_attn), so the guard
            # must activate at the TE boundary.
            expected_triggers = {
                # Direct-TE models (te.pytorch.TransformerLayer) may never
                # import Megatron at all, and MT-TE's own __init__ resolves
                # super() through its module global, so the capability
                # dispatch must be installed in place at the musa TE
                # boundary; the factory shim and safe-seed hooks must fire
                # before any Megatron import.
                "megatron.te.factory-shim.torchscript-compat": "transformer_engine",
                "megatron.te.utils-module.safe-seed": "transformer_engine",
                "transformer_engine.dot-product-attention.capability-dispatch": (
                    "transformer_engine.musa.pytorch.attention"
                ),
            }
            expected = expected_triggers.get(patch.id, "megatron")
            assert patch.trigger == expected
            assert callable(patch.undo), f"{patch.id} must clean up owned changes"


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_each_module_exports_patch_tuple(module):
    assert isinstance(module.PATCHES, tuple)
    assert module.PATCHES


def test_companion_requirements_are_explicit_and_resolvable():
    from megatron_musa_patch._engine import Engine

    Engine._validate_dependencies(list(PATCHES))
    ids = {patch.id for patch in PATCHES}
    for patch in PATCHES:
        if isinstance(patch, AttrPatch):
            assert set(patch.requires) <= ids
            assert AppliedPatch(patch).as_dict()["requires"] == list(patch.requires)


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_patch_modules_do_not_import_other_patch_modules(module):
    import ast
    from pathlib import Path

    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level != 1, f"{module.__name__} imports a sibling patch module"
            assert not (node.module or "").startswith("megatron_musa_patch.patches")
        elif isinstance(node, ast.Import):
            assert not any(
                alias.name.startswith("megatron_musa_patch.patches") for alias in node.names
            )


def test_megatron_importable_probe_is_side_effect_free():
    before = set(sys.modules)
    megatron_importable()
    assert not {"torch", "megatron"} & (set(sys.modules) - before)


def test_probe_does_not_wake_the_import_watcher(engine, monkeypatch):
    monkeypatch.delitem(sys.modules, "megatron", raising=False)
    monkeypatch.delitem(sys.modules, "megatron.core", raising=False)
    calls = []
    engine.register([HookPatch("t.probe", "megatron", lambda: calls.append(1))])
    engine.install()
    megatron_importable()
    assert calls == []


def test_missing_megatron_is_reported_separately(monkeypatch):
    monkeypatch.setattr(_compat, "megatron_importable", lambda: False)
    assert _compat.check_megatron_present(strict=False) is False
    with pytest.raises(MegatronMissing):
        _compat.check_megatron_present(strict=True)
    monkeypatch.setattr(_compat, "megatron_importable", lambda: True)
    assert _compat.check_megatron_present(strict=True) is True


def test_version_spec_is_parseable():
    assert parse_version(SUPPORTED_VERSION_SPEC.split(",")[0].lstrip(">="))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.16.1", (0, 16, 1)),
        ("0.16.1rc0", (0, 16, 1)),
        ("1.2", (1, 2)),
        ("2", (2,)),
        (None, ()),
        ("", ()),
    ],
)
def test_parse_version(text, expected):
    assert parse_version(text) == expected


def test_supported_version_spec_bounds_are_enforced():
    assert parse_version("0.14.0") == (0, 14, 0)
    assert parse_version("0.16.1rc0") == (0, 16, 1)
    lo, _, hi = SUPPORTED_VERSION_SPEC.partition(",")
    assert parse_version(lo.lstrip(">=")) >= (0, 14)
    assert parse_version(hi.lstrip("<")) == (0, 17)
    assert _compat._in_supported_range((0, 14)) and _compat._in_supported_range((0, 16, 1))
    assert not _compat._in_supported_range((0, 13, 9)) and not _compat._in_supported_range(
        (0, 17, 0)
    )


def test_version_drift_warns_or_raises_per_strict_switch(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(_compat, "megatron_version", lambda: "0.13.2")
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_STRICT", raising=False)
    with caplog.at_level(logging.WARNING, logger="megatron_musa_patch"):
        assert _compat.check_version() == "0.13.2"
    assert "0.13.2" in caplog.text
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_STRICT", "1")
    with pytest.raises(_compat.UnsupportedMegatronVersion):
        _compat.check_version()


def test_env_flag_parsing(monkeypatch):
    from megatron_musa_patch import _env

    for raw, expected in [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        (" no ", False),
    ]:
        monkeypatch.setenv("MEGATRON_MUSA_PATCH_PROBE_X", raw)
        assert _env.flag("PROBE_X", True) is expected, raw
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_PROBE_X", "garbage")
    assert _env.flag("PROBE_X", True) is True and _env.flag("PROBE_X", False) is False
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_PROBE_X")
    assert _env.flag("PROBE_X", False) is False
