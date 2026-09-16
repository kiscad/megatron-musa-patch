"""Transformer Engine version compatibility policy tests."""

from __future__ import annotations

from unittest.mock import Mock

from megatron_musa_patch.patches import _transformer_engine


def test_te_version_predicate_is_ignored():
    original = Mock(return_value=False)
    wrapped = _transformer_engine._ignore_te_min_version(original)

    assert wrapped("999.0.0", check_equality=False) is True
    assert wrapped.__wrapped__ is original
    original.assert_not_called()


def test_te_version_patch_applies_to_megatron_utils(engine, stub_module):
    original = Mock(return_value=False)
    module = stub_module("megatron.core.utils", is_te_min_version=original)

    engine.register(_transformer_engine.PATCHES)
    engine.install()

    assert module.is_te_min_version("0.0.1") is True
    assert module.is_te_min_version("999.0.0", check_equality=False) is True
    assert engine.report()[0]["status"] == "applied"
    original.assert_not_called()


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
    """Megatron's own six-argument wrapper as it behaves after the version patch.

    ``is_te_min_version("2.5.0")`` is patched to true, so upstream always takes
    its six-argument branch -- the call the fork cannot accept.
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
