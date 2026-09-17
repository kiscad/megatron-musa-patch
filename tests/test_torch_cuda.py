"""CPU-safe lifecycle tests plus optional real MUSA/torchada contracts.

Unit tests use small synthetic modules: importing this test file never imports
an accelerator dependency. Only tests requesting the ``torch`` fixture require
a usable MUSA device; adapter availability alone is not enough to run them.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from megatron_musa_patch._errors import MusaUnavailable
from megatron_musa_patch.backends import torch_cuda


@pytest.fixture
def fake_backend(monkeypatch):
    """Model the public torch/torchada API without touching a real runtime."""
    state = {"available": True, "patched": False, "adapter_calls": 0}

    class Tensor:
        type_name = "torch.musa.FloatTensor"

        def type(self, *args, **kwargs):
            return self if args or kwargs else self.type_name

        def musa(self, *args, **kwargs):
            return "original", args, kwargs

        def to(self, *args, **kwargs):
            return "to", args, kwargs

    class Device:
        def __init__(self, device, index=None):
            if isinstance(device, Device):
                self.type, self.index = device.type, device.index
            else:
                kind, _, suffix = device.partition(":")
                self.type = kind
                self.index = int(suffix) if suffix else index
            if self.index is not None and self.index < 0:
                raise RuntimeError("Device index must not be negative")

        def __str__(self):
            return self.type if self.index is None else f"{self.type}:{self.index}"

    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.device = Device
    torch.preserve_format = object()
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: False
    cuda.CUDAGraph = object()
    musa = types.ModuleType("torch_musa")
    musa.is_available = lambda: state["available"]
    musa.MUSAGraph = object()
    graphs = types.ModuleType("torch.musa.graphs")
    musa.graphs = graphs
    torch.cuda, torch.musa = cuda, musa
    adapter = types.ModuleType("torchada")
    adapter.is_patched = lambda: state["patched"]
    adapter.is_musa_platform = lambda: True
    adapter.__version__ = "test"

    def apply_patches():
        state["patched"] = True
        state["adapter_calls"] += 1
        torch.external_adapter_marker = True

    adapter.apply_patches = apply_patches
    for name, module in {
        "torch": torch,
        "torch.cuda": cuda,
        "torch.cuda.graphs": graphs,
        "torch_musa": musa,
        "torchada": adapter,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    # Preserve any unrelated real-runtime backend state a caller already owns.
    monkeypatch.setattr(torch_cuda, "_APPLIED", False)
    monkeypatch.setattr(torch_cuda, "_OVERRIDES", [])
    yield types.SimpleNamespace(torch=torch, adapter=adapter, state=state, graphs=graphs)
    torch_cuda.unapply()


def test_backend_import_is_lazy():
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        MEGATRON_MUSA_PATCH="0",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        PYTHONPATH=str(root / "src"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from megatron_musa_patch.backends import torch_cuda; "
            "torch_cuda.torchada_version(); torch_cuda.unapply(); "
            "assert not torch_cuda.is_applied(); "
            "assert not {'torch', 'torch_musa', 'torchada'} & sys.modules.keys()",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_apply_unapply_reinstall_restores_exact_bindings(fake_backend):
    torch = fake_backend.torch
    originals = dict(vars(torch.cuda)), dict(vars(torch.Tensor)), dict(vars(fake_backend.graphs))
    torch_cuda.unapply()  # Safe before activation.
    torch_cuda.apply()
    first_type = torch.Tensor.type
    first_musa = torch.Tensor.musa
    assert torch_cuda.is_applied()
    assert torch.cuda.is_available is torch.musa.is_available
    assert torch.cuda.CUDAGraph is torch.musa.MUSAGraph
    assert fake_backend.graphs.CUDAGraph is torch.musa.MUSAGraph
    torch_cuda.apply()
    assert torch.Tensor.type is first_type
    assert torch.Tensor.musa is first_musa
    assert fake_backend.state["adapter_calls"] == 1

    torch_cuda.unapply()
    assert not torch_cuda.is_applied()
    assert not torch_cuda._OVERRIDES
    assert dict(vars(torch.cuda)) == originals[0]
    assert dict(vars(torch.Tensor)) == originals[1]
    assert dict(vars(fake_backend.graphs)) == originals[2]
    assert torch.external_adapter_marker  # External adapter is not reversible.
    assert fake_backend.adapter.is_patched()
    torch_cuda.unapply()

    torch_cuda.apply()
    assert torch.Tensor.type is not first_type
    assert torch.Tensor.type.__wrapped__ is originals[1]["type"]
    assert torch.Tensor.musa.__wrapped__ is originals[1]["musa"]
    assert fake_backend.state["adapter_calls"] == 1


def test_availability_is_live_musa_probe_not_constant(fake_backend):
    torch_cuda.apply()
    assert fake_backend.torch.cuda.is_available()
    fake_backend.state["available"] = False
    assert fake_backend.torch.cuda.is_available() is False


def test_no_musa_device_does_not_import_adapter(fake_backend, monkeypatch):
    fake_backend.state["available"] = False
    monkeypatch.setitem(sys.modules, "torchada", None)
    original_type = fake_backend.torch.Tensor.type
    with pytest.raises(MusaUnavailable, match="no MUSA device"):
        torch_cuda.apply()
    assert fake_backend.torch.Tensor.type is original_type
    assert fake_backend.state["adapter_calls"] == 0
    assert not torch_cuda.is_applied()
    assert not torch_cuda._OVERRIDES


@pytest.mark.parametrize("dependency", ["torch", "torch_musa", "torchada"])
def test_missing_dependencies_are_actionable(fake_backend, monkeypatch, dependency):
    monkeypatch.setitem(sys.modules, dependency, None)
    with pytest.raises(MusaUnavailable, match="not importable"):
        torch_cuda.apply()
    assert not torch_cuda.is_applied()
    assert not torch_cuda._OVERRIDES


def test_adapter_non_musa_noop_is_not_success(fake_backend):
    fake_backend.state["patched"] = True
    fake_backend.adapter.is_musa_platform = lambda: False
    with pytest.raises(MusaUnavailable, match="TORCHADA_PLATFORM"):
        torch_cuda.apply()
    assert not torch_cuda.is_applied()
    assert not torch_cuda._OVERRIDES


def test_adapter_failed_to_patch_is_not_success(fake_backend):
    fake_backend.adapter.apply_patches = lambda: None
    with pytest.raises(MusaUnavailable, match="did not finish"):
        torch_cuda.apply()
    assert not torch_cuda.is_applied()
    assert not torch_cuda._OVERRIDES


def test_external_adapter_failure_does_not_claim_rollback(fake_backend):
    original_type = fake_backend.torch.Tensor.type

    def broken_adapter():
        fake_backend.torch.external_adapter_marker = True
        raise RuntimeError("external adapter failed")

    fake_backend.adapter.apply_patches = broken_adapter
    with pytest.raises(RuntimeError, match="external adapter failed"):
        torch_cuda.apply()
    assert fake_backend.torch.external_adapter_marker
    assert fake_backend.torch.Tensor.type is original_type
    assert not torch_cuda.is_applied()
    assert not torch_cuda._OVERRIDES


def test_older_adapter_public_api_remains_supported(fake_backend):
    del fake_backend.adapter.is_musa_platform
    del fake_backend.adapter.__version__
    torch_cuda.apply()
    assert torch_cuda.is_applied()


def test_failed_override_rolls_back_and_can_retry(fake_backend, monkeypatch):
    torch = fake_backend.torch
    originals = dict(vars(torch.cuda)), dict(vars(torch.Tensor)), dict(vars(fake_backend.graphs))
    fix = torch_cuda._fix_tensor_musa_for_subclasses

    def fail_after_mutation(torch):
        fix(torch)
        raise RuntimeError("injected activation failure")

    monkeypatch.setattr(torch_cuda, "_fix_tensor_musa_for_subclasses", fail_after_mutation)
    with pytest.raises(RuntimeError, match="injected activation failure"):
        torch_cuda.apply()
    assert not torch_cuda.is_applied()
    assert not torch_cuda._OVERRIDES
    assert dict(vars(torch.cuda)) == originals[0]
    assert dict(vars(torch.Tensor)) == originals[1]
    assert dict(vars(fake_backend.graphs)) == originals[2]
    assert fake_backend.adapter.is_patched()
    monkeypatch.setattr(torch_cuda, "_fix_tensor_musa_for_subclasses", fix)
    torch_cuda.apply()
    assert torch.Tensor.type.__wrapped__ is originals[1]["type"]
    assert torch_cuda.is_applied()


def test_assignment_that_mutates_then_raises_is_rolled_back(fake_backend):
    class FailingCuda(types.ModuleType):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == "CUDAGraph":
                raise RuntimeError("assignment failed after mutation")

    cuda = FailingCuda("torch.cuda")
    cuda.is_available = lambda: False
    fake_backend.torch.cuda = cuda
    original = dict(vars(cuda))
    with pytest.raises(RuntimeError, match="assignment failed after mutation"):
        torch_cuda.apply()
    assert dict(vars(cuda)) == original
    assert not torch_cuda._OVERRIDES
    assert not torch_cuda.is_applied()


def test_failed_unapply_keeps_cleanup_retryable(fake_backend):
    original_available = lambda: False
    state = {"reject_undo": False}

    class FailingUndoCuda(types.ModuleType):
        def __setattr__(self, name, value):
            if name == "is_available" and value is original_available and state["reject_undo"]:
                raise RuntimeError("temporary cleanup failure")
            super().__setattr__(name, value)

    cuda = FailingUndoCuda("torch.cuda")
    cuda.is_available = original_available
    fake_backend.torch.cuda = cuda
    torch_cuda.apply()
    state["reject_undo"] = True
    with pytest.raises(RuntimeError, match="temporary cleanup failure"):
        torch_cuda.unapply()
    assert len(torch_cuda._OVERRIDES) == 1
    state["reject_undo"] = False
    torch_cuda.unapply()
    assert cuda.is_available is original_available
    assert not torch_cuda._OVERRIDES
    assert not torch_cuda.is_applied()


def test_unapply_uses_post_adapter_baseline(fake_backend):
    torch = fake_backend.torch
    original = torch.Tensor.type

    def adapter_type(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    apply_adapter = fake_backend.adapter.apply_patches

    def adapter_with_type_override():
        apply_adapter()
        torch.Tensor.type = adapter_type

    fake_backend.adapter.apply_patches = adapter_with_type_override
    torch_cuda.apply()
    assert torch.Tensor.type.__wrapped__ is adapter_type
    torch_cuda.unapply()
    assert torch.Tensor.type is adapter_type


def test_unapply_preserves_subsequent_third_party_replacement(fake_backend):
    torch = fake_backend.torch
    torch_cuda.apply()
    later_type = lambda self: "later"
    torch.Tensor.type = later_type
    del torch.cuda.CUDAGraph
    torch_cuda.unapply()
    assert torch.Tensor.type is later_type
    assert "CUDAGraph" not in vars(torch.cuda)
    torch_cuda.apply()
    assert torch.Tensor.type.__wrapped__ is later_type


def test_proxy_bookkeeping_does_not_resolve_or_cache_original(fake_backend):
    original_available = lambda: False

    class CudaProxy(types.ModuleType):
        def __getattr__(self, name):
            if name == "is_available":
                self.is_available = original_available
                return original_available
            raise AttributeError(name)

    proxy = CudaProxy("torch.cuda")
    fake_backend.torch.cuda = proxy
    torch_cuda.apply()
    assert proxy.is_available is fake_backend.torch.musa.is_available
    torch_cuda.unapply()
    assert "is_available" not in vars(proxy)
    assert "CUDAGraph" not in vars(proxy)
    assert proxy.is_available is original_available


def test_tensor_type_query_and_conversion_contract(fake_backend):
    torch_cuda.apply()
    tensor = fake_backend.torch.Tensor()
    assert tensor.type() == "torch.cuda.FloatTensor"
    tensor.type_name = "torch.FloatTensor"
    assert tensor.type() == "torch.FloatTensor"
    tensor.type_name = "torch.musa_like.FloatTensor"
    assert tensor.type() == "torch.musa_like.FloatTensor"
    assert tensor.type("torch.FloatTensor") is tensor
    assert tensor.type(dtype="torch.FloatTensor", non_blocking=True) is tensor


def test_plain_tensor_musa_delegates_unchanged(fake_backend):
    torch_cuda.apply()
    tensor = fake_backend.torch.Tensor()
    marker = object()
    assert tensor.musa(marker, True, option=marker) == (
        "original", (marker, True), {"option": marker}
    )


@pytest.mark.parametrize("device,index", [(None, None), (2, 2), ("musa", None), ("musa:3", 3)])
def test_subclass_musa_preserves_transfer_options(fake_backend, device, index):
    torch = fake_backend.torch

    class Subclass(torch.Tensor):
        pass

    torch_cuda.apply()
    memory_format = object()
    kind, args, kwargs = Subclass().musa(device, True, memory_format=memory_format)
    assert kind == "to"
    assert not args
    assert kwargs["device"].type == "musa"
    assert kwargs["device"].index == index
    assert kwargs["non_blocking"] is True
    assert kwargs["memory_format"] is memory_format
    _, _, defaults = Subclass().musa()
    assert defaults["non_blocking"] is False
    assert defaults["memory_format"] is torch.preserve_format


def test_subclass_musa_accepts_device_object_and_rejects_cpu(fake_backend):
    torch = fake_backend.torch

    class Subclass(torch.Tensor):
        pass

    torch_cuda.apply()
    _, _, kwargs = Subclass().musa(device=torch.device("musa", 1))
    assert kwargs["device"].index == 1
    with pytest.raises(RuntimeError, match="must be musa"):
        Subclass().musa("cpu")
    with pytest.raises(TypeError):
        Subclass().musa(0, device=1)
    with pytest.raises(TypeError):
        Subclass().musa(unexpected=True)


def test_graph_alias_without_graphs_module_import(fake_backend, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch.cuda.graphs")
    torch_cuda.apply()
    assert fake_backend.graphs.CUDAGraph is fake_backend.torch.musa.MUSAGraph
    torch_cuda.unapply()
    assert "CUDAGraph" not in vars(fake_backend.graphs)


def test_backend_without_graph_support_remains_supported(fake_backend):
    original_graph = fake_backend.torch.cuda.CUDAGraph
    del fake_backend.torch.musa.MUSAGraph
    torch_cuda.apply()
    assert fake_backend.torch.cuda.CUDAGraph is original_graph


def test_hook_metadata_and_engine_unapply(fake_backend, engine, monkeypatch):
    from megatron_musa_patch.patches._torch_backend import PATCHES

    patch = PATCHES[0]
    assert "torchada" in patch.strategy
    assert "undo only project-owned bindings" in patch.strategy
    assert patch.undo is torch_cuda.unapply
    monkeypatch.setenv("MEGATRON_MUSA_PATCH", "1")
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_ONLY", raising=False)
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_DISABLE", raising=False)
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    original_type = fake_backend.torch.Tensor.type
    engine.register(PATCHES)
    engine.install()
    assert torch_cuda.is_applied()
    engine.unapply()
    assert not torch_cuda.is_applied()
    assert fake_backend.torch.Tensor.type is original_type
    engine.install()
    assert torch_cuda.is_applied()
    assert fake_backend.torch.Tensor.type.__wrapped__ is original_type


def test_hook_does_not_acquire_another_callers_active_layer(fake_backend, engine, monkeypatch):
    from megatron_musa_patch.patches._torch_backend import PATCHES

    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    torch_cuda.apply()
    active_type = fake_backend.torch.Tensor.type
    engine.register(PATCHES)
    engine.install()
    assert engine.report()[0]["status"] == "skipped"
    engine.unapply()
    assert torch_cuda.is_applied()
    assert fake_backend.torch.Tensor.type is active_type


# ---------------------------------------------------------------------------
# Real hardware contracts: these alone skip on CPU-only environments.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def torch():
    runtime = pytest.importorskip("torch")
    pytest.importorskip("torch_musa")
    if not runtime.musa.is_available():
        pytest.skip("no visible MUSA device")
    pytest.importorskip("torchada")
    already_applied = torch_cuda.is_applied()
    torch_cuda.apply()
    yield runtime
    if not already_applied:
        torch_cuda.unapply()


def test_apply_is_idempotent(torch):
    before = torch.cuda.is_available
    before_type = torch.Tensor.type
    torch_cuda.apply()
    assert torch.cuda.is_available is before
    assert torch.Tensor.type is before_type
    assert torch_cuda.is_applied()


def test_torchada_is_the_alias_layer(torch):
    assert importlib.import_module("torchada").is_patched()
    assert torch_cuda.torchada_version() != "unknown"


def test_is_available_uses_real_musa(torch):
    assert torch.cuda.is_available is torch.musa.is_available
    assert torch.cuda.is_available() is torch.musa.is_available()


def test_tensor_type_reports_cuda_names(torch):
    """Megatron asserts ``param.grad.type() == 'torch.cuda.FloatTensor'``."""
    assert torch.zeros(3, device="musa").type() == "torch.cuda.FloatTensor"
    assert torch.zeros(3).type() == "torch.FloatTensor"


def test_cuda_graph_class_is_bound(torch):
    assert torch.cuda.CUDAGraph is torch.musa.MUSAGraph
    assert importlib.import_module("torch.cuda.graphs").CUDAGraph is torch.musa.MUSAGraph


def test_musa_move_works_for_te_float8_tensor(torch):
    """FP8 params survive Megatron's ``module.cuda(current_device())`` move."""
    te = pytest.importorskip("transformer_engine.pytorch")
    with te.fp8_model_init(enabled=True):
        lin = te.Linear(32, 32, params_dtype=torch.bfloat16)
    assert type(lin.weight).__name__ == "Float8Tensor"
    lin.cuda(torch.cuda.current_device())
    assert lin.weight.device.type == "musa"
    assert lin.weight.device.index == torch.cuda.current_device()


def test_real_subclass_transfer_preserves_memory_format(torch):
    class Subclass(torch.Tensor):
        pass

    tensor = torch.zeros(2, 3, 4, 5).as_subclass(Subclass)
    moved = tensor.musa(non_blocking=True, memory_format=torch.channels_last)
    assert isinstance(moved, Subclass)
    assert moved.device.type == "musa"
    assert moved.is_contiguous(memory_format=torch.channels_last)
    with pytest.raises(RuntimeError, match="must be musa"):
        tensor.musa("cpu")


def test_torchada_device_queries(torch):
    assert torch.cuda.device_count() == torch.musa.device_count()
    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0
    assert torch.cuda.get_device_properties(0).name == torch.musa.get_device_name(0)


def test_torchada_device_string_rewriting(torch):
    assert torch.zeros(3, device="cuda").is_cuda
    assert torch.zeros(3, device="cuda:0").device.type == "musa"
    assert torch.zeros(3, device=torch.device("cuda")).is_cuda
    assert torch.empty(3, device="cuda").device.type == "musa"
    assert torch.arange(3, device="cuda").device.type == "musa"
    assert torch.full((3,), 1.0, device="cuda").device.type == "musa"
    assert torch.ones(3, device="cuda").device.type == "musa"
    assert torch.tensor([1.0], device="cuda").device.type == "musa"


def test_torchada_tensor_transfer_helpers(torch):
    cpu = torch.zeros(3)
    assert cpu.to("cuda").is_cuda
    assert cpu.to(torch.device("cuda")).is_cuda
    assert cpu.cuda().is_cuda
    assert cpu.cuda(0).device == torch.device("musa", 0)
    assert cpu.is_cuda is False
    assert torch.zeros(3, device="musa").is_cuda is True

    module = torch.nn.Linear(4, 4)
    assert module.cuda() is module
    assert module.weight.is_cuda
    assert torch.nn.Linear(4, 4).to("cuda").weight.is_cuda


def test_torchada_streams_events_and_rng(torch):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda.current_stream()
    event = torch.cuda.Event()
    event.record()
    torch.cuda.synchronize()
    state = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    torch.cuda.manual_seed(0)


def test_torchada_nvtx_is_a_noop(torch):
    torch.cuda.nvtx.range_push("x")
    with torch.cuda.nvtx.range("y"):
        pass
    torch.cuda.nvtx.range_pop()


def test_torchada_memory_api(torch):
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() >= 0
    assert torch.cuda.memory_reserved() >= 0
    assert torch.cuda.max_memory_allocated() >= 0
    memory = importlib.import_module("torch.cuda.memory")
    assert memory is torch.musa.memory
    memory._record_memory_history(enabled=None)


def test_torchada_stream_capture_probe(torch):
    """Pin behavior, not identity: torch_musa may wrap this probe itself."""
    assert torch.cuda.is_current_stream_capturing() is False


def test_torchada_rewrites_nccl_to_mccl(torch, tmp_path):
    """``--distributed-backend nccl`` is the only GPU choice Megatron allows."""
    import torch.distributed as dist

    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{tmp_path}/pg_init",
        rank=0,
        world_size=1,
    )
    try:
        assert dist.get_backend() == "mccl"
    finally:
        dist.destroy_process_group()


def test_tensor_musa_inside_device_context(torch):
    """P10: Megatron pins a CPU tensor and calls ``.cuda()`` under CP.

    ``get_pos_emb_on_this_cp_rank`` runs ``torch.tensor(..., pin_memory=True)
    .cuda(non_blocking=True)`` while a ``torch.device`` context is active. The
    torch_musa C shim re-enters the Python-level ``Tensor.musa`` with shifted
    arguments, which used to raise ``device() received an invalid combination
    of arguments - got (Tensor)``.
    """
    x = torch.tensor([0, 7], device="cpu", pin_memory=True)
    with torch.device("cuda"):
        moved = x.cuda(non_blocking=True)
    assert moved.device.type == "musa"
    assert moved.tolist() == [0, 7]
    assert x.device.type == "cpu"


def test_tensor_musa_reentry_shifts_the_real_tensor_back(torch):
    """The C shim re-enters with (placeholder-self, tensor, ...kwargs).

    Route that call shape through the same transfer path as tensor subclasses
    instead of feeding the placeholder to ``torch.device``.
    """
    x = torch.randn(3)
    wrapper = torch.Tensor.musa
    moved = wrapper(object(), x, non_blocking=True)
    assert moved.device.type == "musa"
    torch.testing.assert_close(moved.cpu(), x.to(torch.device("musa")).cpu())
    # A real tensor in self keeps the existing dispatch.
    assert wrapper(x, 0).device == torch.device("musa", 0)


def test_tensor_musa_same_device_transfer_is_no_copy(torch):
    a = torch.randn(4, device="musa")
    assert a.musa().data_ptr() == a.data_ptr()


def test_tensor_musa_accepts_musa_device_forms(torch):
    x = torch.randn(3)
    for device in (0, "musa:0", torch.device("musa"), None):
        assert x.musa(device).device.type == "musa"


def test_tensor_musa_subclass_accepts_cuda_spelling(torch):
    class Subclass(torch.Tensor):
        pass

    tensor = torch.randn(3).as_subclass(Subclass)
    moved = tensor.musa("cuda:0")
    assert moved.device == torch.device("musa", 0)
    assert isinstance(moved, Subclass)
