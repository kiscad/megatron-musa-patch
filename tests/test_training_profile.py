"""Argument and startup compatibility policies (no GPU required)."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron_musa_patch.patches import _training


@pytest.mark.parametrize("rank", [0, 1])
def test_pytorch_is_enabled_with_rank_zero_warning(rank, caplog):
    args = SimpleNamespace(profile=True, use_pytorch_profiler=False, rank=rank,
                           profile_step_start=4, profile_step_end=6, profile_ranks=[0, 1])
    original = Mock(return_value=args)
    wrapped = _training._enable_pytorch_profile_validate_args(original)
    defaults = {"profile": True}
    with caplog.at_level(logging.WARNING, logger="megatron_musa_patch"):
        assert wrapped(args, defaults=defaults) is args
    original.assert_called_once_with(args, defaults=defaults)
    assert args.profile is True
    assert args.use_pytorch_profiler is True
    assert args.profile_ranks == [0, 1]
    assert args.profile_step_start == 4 and args.profile_step_end == 6
    assert ("--profile automatically enables --use-pytorch-profiler" in caplog.text) == (rank == 0)


@pytest.mark.parametrize("profile,use_pytorch", [(False, False), (False, True), (True, True)])
def test_explicit_pytorch_and_disabled_profiling_are_preserved(profile, use_pytorch, caplog):
    args = SimpleNamespace(profile=profile, use_pytorch_profiler=use_pytorch)
    wrapped = _training._enable_pytorch_profile_validate_args(lambda args: args)
    assert wrapped(args) is args
    assert args.profile is profile
    assert args.use_pytorch_profiler is use_pytorch
    assert not caplog.records


def test_defaults_and_replaced_namespace_are_normalized():
    supplied = SimpleNamespace()
    validated = SimpleNamespace(profile=True)
    wrapped = _training._enable_pytorch_profile_validate_args(lambda args: validated)
    assert wrapped(supplied) is validated
    assert validated.profile is True
    assert validated.use_pytorch_profiler is True
    assert not hasattr(supplied, "profile")


def test_missing_profile_is_untouched():
    args = SimpleNamespace()
    wrapped = _training._enable_pytorch_profile_validate_args(lambda args: args)
    assert wrapped(args) is args
    assert not hasattr(args, "profile")


def test_validation_errors_propagate():
    original = Mock(side_effect=ValueError("invalid training arguments"))
    wrapped = _training._enable_pytorch_profile_validate_args(original)
    with pytest.raises(ValueError, match="invalid training arguments"):
        wrapped(SimpleNamespace(profile=True))


@pytest.fixture()
def overlap_policy(monkeypatch):
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_DP_OVERLAP", raising=False)
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_TP_OVERLAP", raising=False)
    return _training._ignore_overlap_flags_validate_args


@pytest.mark.parametrize("rank", [0, 1])
def test_dp_overlap_policy_disables_dependencies_but_preserves_other_overlap(
    overlap_policy, rank, caplog
):
    args = SimpleNamespace(
        rank=rank,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
        tp_comm_overlap=True,
        overlap_p2p_comm=True,
    )
    defaults = {"overlap_grad_reduce": True}

    def original(args, defaults):
        assert args.overlap_grad_reduce is False
        assert args.overlap_param_gather is False
        assert args.overlap_param_gather_with_optimizer_step is False
        return args

    wrapped = overlap_policy(original)
    with caplog.at_level(logging.WARNING, logger="megatron_musa_patch"):
        assert wrapped(args, defaults=defaults) is args
    assert defaults == {"overlap_grad_reduce": True}
    assert args.tp_comm_overlap is True
    assert args.overlap_p2p_comm is True
    assert ("disabling DP overlap" in caplog.text) == (rank == 0)
    assert wrapped.__wrapped__ is original


def test_overlap_default_requests_are_warned_and_cannot_reenable(overlap_policy, caplog):
    args = SimpleNamespace(overlap_grad_reduce=None)
    defaults = {"overlap_grad_reduce": True, "arbitrary": 123}

    def original(args, defaults):
        for key, value in defaults.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
        return args

    with caplog.at_level(logging.WARNING, logger="megatron_musa_patch"):
        result = overlap_policy(original)(args, defaults)
    assert result.overlap_grad_reduce is False
    assert result.arbitrary == 123
    assert defaults["overlap_grad_reduce"] is True
    assert "disabling DP overlap" in caplog.text


def test_disabled_overlap_does_not_warn_or_change_call_shape(overlap_policy, caplog):
    args = SimpleNamespace(overlap_grad_reduce=False, overlap_param_gather=False)
    original = Mock(return_value=args)
    assert overlap_policy(original)(args=args) is args
    original.assert_called_once_with(args)
    assert not caplog.records


@pytest.mark.parametrize(
    "preferred,legacy,skip",
    [(None, "1", True), ("1", "0", True), ("0", "1", False), (None, "0", False)],
)
def test_dp_overlap_switch_overrides_legacy_alias(overlap_policy, monkeypatch, preferred, legacy, skip):
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_TP_OVERLAP", legacy)
    if preferred is not None:
        monkeypatch.setenv("MEGATRON_MUSA_PATCH_DP_OVERLAP", preferred)
    replacement = overlap_policy(Mock())
    assert (replacement is None) is skip


def test_overlap_validator_exception_propagates(overlap_policy):
    original = Mock(side_effect=ValueError("upstream validation failed"))
    with pytest.raises(ValueError, match="upstream validation failed"):
        overlap_policy(original)(SimpleNamespace())


@pytest.mark.parametrize("ckpt_format", ["torch_dist", "torch", "torch_dcp", "fsdp_dtensor"])
def test_live_argument_patch_chain_never_rewrites_checkpoint_format(
    overlap_policy, ckpt_format
):
    args = SimpleNamespace(ckpt_format=ckpt_format, async_save=True, profile=True)
    wrapped = lambda args: args
    for patch in _training.PATCHES:
        if patch.target == "megatron.training.arguments:validate_args":
            replacement = patch.replace(wrapped)
            if replacement is not None:
                wrapped = replacement
    assert wrapped(args) is args
    assert args.ckpt_format == ckpt_format
    assert args.async_save is True
    assert args.use_pytorch_profiler is True
    assert not any(p.id == "megatron.training.ckpt-format.no-torch-dist" for p in _training.PATCHES)


def test_legacy_loader_noop_does_not_call_original():
    original = Mock()
    args = SimpleNamespace(rank=0)
    wrapped = _training._noop_fused_kernels_load(original)
    assert wrapped(args) is None
    assert wrapped.__wrapped__ is original
    original.assert_not_called()


def test_jit_warmup_policy_is_optional(monkeypatch):
    original = Mock()
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_JIT_WARMUP", raising=False)
    assert _training._noop_set_jit_fusion_options(original)() is None
    original.assert_not_called()
    monkeypatch.setenv("MEGATRON_MUSA_PATCH_JIT_WARMUP", "1")
    assert _training._noop_set_jit_fusion_options(original) is None
