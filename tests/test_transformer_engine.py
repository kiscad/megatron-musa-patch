"""Transformer Engine call-shape adapter tests."""

from __future__ import annotations

from unittest.mock import Mock

from megatron_musa_patch.patches import _transformer_engine


def test_te_version_predicate_is_not_overridden(engine, stub_module):
    """The fork's reported version decides; no blanket true result any more."""
    original = Mock(return_value=False, __name__="is_te_min_version")
    module = stub_module("megatron.core.utils", is_te_min_version=original)

    engine.register(_transformer_engine.PATCHES)
    engine.install()

    assert module.is_te_min_version is original
    assert module.is_te_min_version("999.0.0") is False
    original.assert_called_once_with("999.0.0")
    assert all(r["id"] != "megatron.core.utils.te-version-check.ignore"
               for r in engine.report())


def test_mem_monitor_shim_installs_and_undos():
    import sys

    for name in ("musa_patch", "musa_patch.mem_utils"):
        sys.modules.pop(name, None)
    try:
        assert _transformer_engine._install_mem_monitor_shim() is True
        from musa_patch.mem_utils import MemMonitor  # noqa: PLC0415
    finally:
        installed = _transformer_engine._mem_monitor_owned.copy()
        _transformer_engine._uninstall_mem_monitor_shim()
    assert MemMonitor.max_token_num == 0
    # The fork's accounting idiom keeps its behavior.
    MemMonitor.max_token_num = max(MemMonitor.max_token_num, 5)
    assert MemMonitor.max_token_num == 5
    assert not installed.keys() & sys.modules.keys()
    assert not _transformer_engine._mem_monitor_owned


def test_mem_monitor_shim_never_shadows_an_existing_musa_patch(monkeypatch):
    import sys
    import types

    existing = types.ModuleType("musa_patch")
    monkeypatch.setitem(sys.modules, "musa_patch", existing)
    assert _transformer_engine._install_mem_monitor_shim() is False
    assert sys.modules["musa_patch"] is existing
    assert not _transformer_engine._mem_monitor_owned


def test_mem_monitor_shim_requires_the_musa_fork(monkeypatch):
    import sys

    monkeypatch.setattr(_transformer_engine, "_te_fork_needs_mem_monitor", lambda: False)
    assert _transformer_engine._install_mem_monitor_shim() is False
    assert "musa_patch" not in sys.modules


def test_mem_monitor_shim_installs_through_engine(engine):
    import sys

    sys.modules.pop("musa_patch", None)
    engine.register(_transformer_engine.PATCHES)
    sys.modules["megatron"] = Mock()
    try:
        engine.install()
        statuses = {r["id"]: r["status"] for r in engine.report()}
        assert statuses["megatron.te.grouped-linear.mem-monitor-compat"] == "applied"
        assert "musa_patch.mem_utils" in sys.modules
    finally:
        engine.unapply()
        sys.modules.pop("megatron", None)
    assert "musa_patch" not in sys.modules
    assert not _transformer_engine._mem_monitor_owned


class TeFork:
    """Stands in for the MUSA TE fork's ``cpu_offload`` module."""

    def __init__(self, arity=5, variadic=False):
        self.calls = []
        if variadic:
            def target(*args, **kwargs):
                self.calls.append((args, kwargs))
                return "variadic"
        elif arity == 5:
            def target(enabled, num_layers, model_layers, offload_activations,
                       offload_weights):
                self.calls.append((enabled, num_layers, model_layers,
                                   offload_activations, offload_weights))
                return "five"
        else:
            def target(enabled, num_layers, model_layers, offload_activations,
                       offload_weights, double_buffering):
                self.calls.append((enabled, num_layers, model_layers, offload_activations,
                                   offload_weights, double_buffering))
                return "six"
        self.target = target


def _cpu_offload_patch():
    return next(p for p in _transformer_engine.PATCHES
                if p.id == "megatron.te.cpu-offload-context.signature-dispatch")


def _upstream_wrapper(target):
    """Megatron's own six-argument wrapper as it behaves for a TE >= 2.5 report.

    Upstream picks this branch from ``is_te_min_version("2.5.0")``; the adapter
    keeps the call correct when the fork's real signature lags its version.
    """
    def get_cpu_offload_context(enabled, num_layers, model_layers, activation_offloading,
                                weight_offloading, double_buffering):
        return target(enabled, num_layers, model_layers, activation_offloading,
                      weight_offloading, double_buffering)

    return get_cpu_offload_context


def test_cpu_offload_context_uses_the_fork_signature(engine, stub_module):
    fork = TeFork(arity=5)
    upstream = _upstream_wrapper(fork.target)
    module = stub_module("megatron.core.extensions.transformer_engine",
                         _get_cpu_offload_context=fork.target,
                         get_cpu_offload_context=upstream)
    engine.register([_cpu_offload_patch()])
    engine.install()

    assert engine.report()[0]["status"] == "applied"
    assert module.get_cpu_offload_context(True, 4, 4, True, False, True) == "five"
    assert fork.calls == [(True, 4, 4, True, False)]


def test_cpu_offload_context_leaves_a_six_argument_fork_alone(engine, stub_module):
    fork = TeFork(arity=6)
    upstream = _upstream_wrapper(fork.target)
    module = stub_module("megatron.core.extensions.transformer_engine",
                         _get_cpu_offload_context=fork.target,
                         get_cpu_offload_context=upstream)
    engine.register([_cpu_offload_patch()])
    engine.install()

    assert engine.report()[0]["status"] == "skipped"
    assert module.get_cpu_offload_context(True, 4, 4, True, False, True) == "six"


def test_cpu_offload_context_declines_on_an_unknown_signature(engine, stub_module):
    fork = TeFork(variadic=True)
    upstream = _upstream_wrapper(fork.target)
    module = stub_module("megatron.core.extensions.transformer_engine",
                         _get_cpu_offload_context=fork.target,
                         get_cpu_offload_context=upstream)
    engine.register([_cpu_offload_patch()])
    engine.install()

    assert engine.report()[0]["status"] == "skipped"
    assert module.get_cpu_offload_context is upstream


def test_cpu_offload_context_declines_without_transformer_engine(engine, stub_module):
    module = stub_module("megatron.core.extensions.transformer_engine",
                         get_cpu_offload_context=None)
    engine.register([_cpu_offload_patch()])
    engine.install()

    assert engine.report()[0]["status"] == "skipped"
    assert module.get_cpu_offload_context is None


def test_quantized_init_context_and_ownership(monkeypatch):
    import sys
    import types
    from contextlib import contextmanager
    import pytest

    module = types.ModuleType('transformer_engine.pytorch')
    recipe_module = types.ModuleType('transformer_engine.common.recipe')
    class DelayedScaling:
        pass
    recipe_module.DelayedScaling = DelayedScaling
    active = []

    @contextmanager
    def fp8_model_init(enabled=True, recipe=None, preserve_high_precision_init_val=False):
        active.append((enabled, recipe, preserve_high_precision_init_val))
        try:
            yield
        finally:
            active.pop()

    module.fp8_model_init = fp8_model_init
    import torch
    monkeypatch.setattr(torch, 'musa', types.SimpleNamespace(is_available=lambda: True), raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, recipe_module.__name__, recipe_module)
    monkeypatch.setattr(_transformer_engine, '_te_fork_needs_mem_monitor', lambda: True)
    try:
        assert _transformer_engine._install_quantized_model_init()
        assert not _transformer_engine._install_quantized_model_init()
        recipe = DelayedScaling()
        with module.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
            assert active == [(True, recipe, True)]
            with pytest.raises(RuntimeError, match='body failed'):
                with module.quantized_model_init(False):
                    assert len(active) == 2
                    raise RuntimeError('body failed')
            assert len(active) == 1
        assert active == []
        with pytest.raises(NotImplementedError):
            module.quantized_model_init(recipe=object())
        _transformer_engine._uninstall_quantized_model_init()
        assert not hasattr(module, 'quantized_model_init')
        assert _transformer_engine._install_quantized_model_init()
        foreign = object()
        module.quantized_model_init = foreign
        _transformer_engine._uninstall_quantized_model_init()
        assert module.quantized_model_init is foreign
        assert not _transformer_engine._install_quantized_model_init()
    finally:
        _transformer_engine._uninstall_quantized_model_init()
