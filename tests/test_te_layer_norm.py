"""Norm-linear fallback: gradients, parameter names and upstream opt-out."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import integration_env

from megatron_musa_patch.patches import _layer_norm

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_norm_linear_contract(stub_module, monkeypatch, zero_centered, dtype):
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_TE_FUSED_LAYERNORM", raising=False)

    class Linear(torch.nn.Module):
        def __init__(self, input_size, output_size, *, config, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(output_size, input_size, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.randn(output_size, dtype=dtype))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight), self.bias

    def sharded(self, *args, **kwargs):
        return self.state_dict()

    stub_module(
        "megatron.core.extensions.transformer_engine", HAVE_TE=True, TEColumnParallelLinear=Linear
    )
    fallback = _layer_norm._unfused_te_layer_norm_linear(
        SimpleNamespace(sharded_state_dict=sharded)
    )
    config = SimpleNamespace(
        normalization="LayerNorm",
        layernorm_epsilon=1e-5,
        layernorm_zero_centered_gamma=zero_centered,
        params_dtype=dtype,
        sequence_parallel=True,
    )
    module = fallback(16, 8, config=config)
    x = torch.randn(4, 16, dtype=dtype, requires_grad=True)
    ref_x = x.detach().double().requires_grad_()
    ref_gamma = module.layer_norm_weight.detach().double().requires_grad_()
    gamma = ref_gamma + 1 if zero_centered else ref_gamma
    ref_bias = module.layer_norm_bias.detach().double().requires_grad_()
    norm = torch.nn.functional.layer_norm(ref_x, (16,), gamma, ref_bias, 1e-5)
    expected = torch.nn.functional.linear(norm, module.weight.detach().double())
    actual, bias = module(x)
    assert bias is module.bias
    tol = dict(atol=0.15, rtol=0.03) if dtype == torch.bfloat16 else dict(atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual.double(), expected, **tol)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad.double())
    torch.testing.assert_close(x.grad.double(), ref_x.grad, **tol)
    torch.testing.assert_close(module.layer_norm_weight.grad.double(), ref_gamma.grad, **tol)
    keys = {"weight", "bias", "layer_norm_weight"}
    keys.add("layer_norm_bias")
    torch.testing.assert_close(module.layer_norm_bias.grad.double(), ref_bias.grad, **tol)
    assert set(module.sharded_state_dict()) == keys
    assert module.layer_norm_weight.sequence_parallel
    assert module.layer_norm_weight.allreduce
    copied = deepcopy(module)
    torch.testing.assert_close(copied(x)[0], actual)
    clone = fallback(16, 8, config=config)
    clone.load_state_dict(module.state_dict(), strict=True)
    torch.testing.assert_close(clone(x)[0], actual)


@pytest.mark.parametrize("zero_centered", [False, True])
def test_rmsnorm_constructs_original_fused_module(stub_module, monkeypatch, zero_centered):
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_TE_FUSED_LAYERNORM", raising=False)
    calls = []

    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise AssertionError("RMSNorm must not construct the unfused TE Linear")

    class Original(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            calls.append((args, kwargs))

        def forward(self, x):
            return x

        def sharded_state_dict(self):
            return {}

    stub_module(
        "megatron.core.extensions.transformer_engine", HAVE_TE=True, TEColumnParallelLinear=Linear
    )
    patched = _layer_norm._unfused_te_layer_norm_linear(Original)
    config = SimpleNamespace(normalization="RMSNorm", layernorm_zero_centered_gamma=zero_centered)
    options = dict(config=config, bias=False, skip_bias_add=True, tp_group=object(), stride=3)
    module = patched(16, 8, **options)
    assert type(module) is Original
    assert isinstance(module, patched)
    assert type(module).forward is Original.forward
    assert not hasattr(module, "_megatron_musa_patch_fallback")
    assert calls == [((16, 8), options)]
    x = torch.randn(2, 16)
    assert module(x) is x


def _te_norm_env(stub_module, monkeypatch, normalization="LayerNorm", fused_residual=False):
    """Stub TE and the extension module, then build the TENorm stand-in."""

    class Norm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-5, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
            self.eps = eps

    stub_module("transformer_engine", pytorch=SimpleNamespace(LayerNorm=Norm, RMSNorm=Norm))
    stub_module("megatron.core.extensions.transformer_engine", HAVE_TE=True)
    monkeypatch.setattr(_layer_norm, "_musa_live", lambda: True)

    upstream_calls = []

    def original(*args, **kwargs):
        upstream_calls.append((args, kwargs))
        return "upstream-norm"

    config = SimpleNamespace(
        normalization=normalization,
        sequence_parallel=False,
        layernorm_zero_centered_gamma=False,
        fused_residual_rmsnorm=fused_residual,
    )
    return _layer_norm._te_norm_unfused(original), config, upstream_calls


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_te_norm_accepts_has_residual(stub_module, monkeypatch, normalization):
    """core 0.19 builds TENorm with has_residual; the stand-in must take it.

    Without it every TransformerLayer construction died with
    ``TENormMusa.__new__() got an unexpected keyword argument 'has_residual'``.
    """
    patched, config, upstream_calls = _te_norm_env(stub_module, monkeypatch, normalization)

    built = patched(config, 8, 1e-5, has_residual=True)

    assert getattr(built, "_megatron_musa_patch_fallback", False) is True
    assert upstream_calls == []  # fused residual is off: the fallback owns this path
    assert patched(config, 8) is not None  # the 0.16-era call shape still works


def test_te_norm_delegates_fused_residual_to_upstream(stub_module, monkeypatch):
    """The fallback does not emulate TEFusedResidualRMSNorm, so upstream builds it."""
    patched, config, upstream_calls = _te_norm_env(
        stub_module, monkeypatch, "RMSNorm", fused_residual=True
    )

    assert patched(config, 8, 1e-5, has_residual=True) == "upstream-norm"
    assert upstream_calls == [((config, 8, 1e-5, True), {})]

    # Only the fused-residual combination delegates; a plain norm stays local.
    assert patched(config, 8, 1e-5, has_residual=False) != "upstream-norm"


def test_te_norm_linear_opt_out(monkeypatch):
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_TE_FUSED_LAYERNORM", "1")
    assert _layer_norm._unfused_te_layer_norm_linear(object()) is None


def test_te_unavailable_skips_norm_linear_patch(engine, stub_module):
    # Match upstream: deriving from a MagicMock produces another mock rather
    # than a real class, discarding methods from the class body.
    te = MagicMock()

    class Original(te.pytorch.LayerNormLinear):
        def sharded_state_dict(self):
            return {}

    assert not hasattr(Original, "sharded_state_dict")
    module = stub_module(
        "megatron.core.extensions.transformer_engine",
        HAVE_TE=False,
        TELayerNormColumnParallelLinear=Original,
        TEColumnParallelLinear=te.pytorch.Linear,
    )
    patch = next(p for p in _layer_norm.PATCHES if p.id == "megatron.te.layer-norm-linear.unfused")
    engine.register([patch])
    engine.install()

    assert module.TELayerNormColumnParallelLinear is Original
    assert module.HAVE_TE is False
    assert engine.report()[0]["status"] == "skipped"


def _norm_linear_env(stub_module):
    """Stub the TE extension module and return the patched class."""

    class Linear(torch.nn.Module):
        def __init__(self, input_size, output_size, *, config, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(output_size, input_size))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight), None

    def sharded(self, *args, **kwargs):
        return self.state_dict()

    stub_module(
        "megatron.core.extensions.transformer_engine", HAVE_TE=True, TEColumnParallelLinear=Linear
    )
    original = SimpleNamespace(sharded_state_dict=sharded)
    return _layer_norm._unfused_te_layer_norm_linear(original), original


def _norm_config(normalization):
    return SimpleNamespace(
        normalization=normalization,
        layernorm_epsilon=1e-5,
        layernorm_zero_centered_gamma=False,
        params_dtype=torch.float32,
        sequence_parallel=False,
        add_bias_linear=False,
    )


def test_subclass_constructs_itself_for_rmsnorm(stub_module):
    """heterogeneous Gathered subclasses must not be routed to the fused class.

    The fused signature needs positional input/output sizes; the Gathered
    replacement is built as (config, tp_comm_buffer_name) and used to raise
    ``missing 2 required positional arguments`` / ``unexpected layer_number``.
    """
    patched, original = _norm_linear_env(stub_module)
    calls = []

    class Gathered(patched):
        def __init__(self, config, tp_comm_buffer_name, *args, **kwargs):
            calls.append((config, tp_comm_buffer_name, args, kwargs))
            super().__init__(
                input_size=config.hidden_size,
                output_size=config.hidden_size,
                config=config,
                gather_output=False,
                bias=config.add_bias_linear,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )

    config = _norm_config("RMSNorm")
    config.hidden_size = 16
    # Exact call shapes observed from the upstream heterogeneous builds.
    module = Gathered(config=config, tp_comm_buffer_name="linear_attn")
    assert type(module) is Gathered
    assert isinstance(module, patched)
    module = Gathered(config=config, tp_comm_buffer_name="linear_attn", layer_number=2)
    assert type(module) is Gathered
    assert calls[1][3] == {"layer_number": 2}

    # RMSNorm matches the fused module's layout: norm weight, no norm bias.
    assert module.layer_norm_bias is None
    assert sum(p.numel() for p in module.parameters()) == 16 * 16 + 16

    x = torch.randn(2, 16)
    out, bias = module(x)
    reference = torch.nn.functional.rms_norm(x, (16,), module.layer_norm_weight, 1e-5)
    torch.testing.assert_close(out, torch.nn.functional.linear(reference, module.weight))
    assert bias is None


def test_subclass_with_layernorm_keeps_norm_bias(stub_module):
    patched, _ = _norm_linear_env(stub_module)

    class Gathered(patched):
        def __init__(self, config, tp_comm_buffer_name, **kwargs):
            super().__init__(
                input_size=config.hidden_size,
                output_size=config.hidden_size,
                config=config,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )

    config = _norm_config("LayerNorm")
    config.hidden_size = 16
    module = Gathered(config=config, tp_comm_buffer_name="linear_mlp")
    assert module.layer_norm_bias is not None
    assert sum(p.numel() for p in module.parameters()) == 16 * 16 + 16 + 16
    x = torch.randn(2, 16)
    out, _ = module(x)
    reference = torch.nn.functional.layer_norm(
        x, (16,), module.layer_norm_weight, module.layer_norm_bias, 1e-5
    )
    torch.testing.assert_close(out, torch.nn.functional.linear(reference, module.weight))


def test_base_class_rmsnorm_still_uses_fused_module(stub_module):
    patched, original = _norm_linear_env(stub_module)
    constructed = []

    class Fused(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            constructed.append((args, kwargs))

        def forward(self, x):
            return x

    original_sharded = original.sharded_state_dict

    class Original(Fused):
        def sharded_state_dict(self, *args, **kwargs):
            return original_sharded(self, *args, **kwargs)

    stub_module(
        "megatron.core.extensions.transformer_engine", HAVE_TE=True, TEColumnParallelLinear=Fused
    )
    patched = _layer_norm._unfused_te_layer_norm_linear(Original)
    config = _norm_config("RMSNorm")
    module = patched(16, 8, config=config)
    assert type(module) is Original
    assert constructed, "exact base with RMSNorm must keep the fused construction"


@pytest.mark.integration
@pytest.mark.parametrize("ranks", [1, 2])
def test_musa_fp8_norm_linear(ranks):
    """Real TE FP8 parameters, BF16/FP8 backward, TP/SP and checkpoint sharding."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    if os.environ.get("MEGATRON_MUSA_RUN_INTEGRATION") != "1":
        pytest.skip("set MEGATRON_MUSA_RUN_INTEGRATION=1 with a working Megatron/MUSA stack")
    env = integration_env({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={ranks}",
            str(Path(__file__).with_name("te_layer_norm_smoke.py")),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("TE_PASS") == 8 * ranks
