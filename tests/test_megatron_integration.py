"""End-to-end checks against a real Megatron-LM installation.

Run in a subprocess on purpose: the patch set is process-global, and mixing it
with the unit tests would make failures order-dependent.

Point the tests at a Megatron-LM source checkout with::

    MEGATRON_LM_PATH=/path/to/Megatron-LM pytest tests/test_megatron_integration.py

Without it, whatever ``megatron`` is importable is used (typically the
``megatron-core`` wheel, which exercises the ``megatron.core`` patches only).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from conftest import integration_env

SCRIPT = textwrap.dedent(
    """
    import json, sys

    import megatron_musa_patch as mmp

    mmp.apply()

    report = {record["id"]: record for record in mmp.report()}
    unresolved = {
        pid: record["status"]
        for pid, record in report.items()
        if record["status"] in {"pending", "failed"}
    }

    result = {
        "unresolved": unresolved,
        "statuses": {pid: record["status"] for pid, record in report.items()},
    }

    try:
        import megatron.training  # noqa: F401

        result["has_training"] = True
    except ImportError:
        result["has_training"] = False

    # --- targeted assertions ------------------------------------------------
    import torch
    from megatron.core.fusions import fused_layer_norm as fln

    result["layer_norm_is_patched"] = (
        fln.FusedLayerNorm.__module__ == mmp.patches._layer_norm.__name__
    )

    from megatron.core.transformer import transformer_block as tb

    result["block_layer_norm_is_local"] = tb.LayerNormImpl is fln.FusedLayerNorm

    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2)
    norm = fln.FusedLayerNorm(config, 8)
    x = torch.randn(2, 3, 8)
    y = norm(x)
    expected = torch.nn.functional.layer_norm(x, (8,), norm.weight, norm.bias, norm.eps)
    result["layer_norm_matches_reference"] = bool(torch.allclose(y, expected, atol=1e-6))
    result["layer_norm_sequence_parallel_marker"] = bool(
        getattr(norm.weight, "sequence_parallel", False) == config.sequence_parallel
    )

    # apply_rope_fusion is argparse's default (--no-rope-fusion is the opt-out),
    # so a stack without fused kernels cannot start a run at all.
    import megatron.core.models.common.embeddings.rope_utils as rope_utils

    result["fused_rope_kernels"] = [
        rope_utils.fused_apply_rotary_pos_emb is not None,
        rope_utils.fused_apply_rotary_pos_emb_thd is not None,
    ]
    rope_config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=2, apply_rope_fusion=True
    )
    result["rope_fusion_config_accepted"] = bool(rope_config.apply_rope_fusion)

    # TransformerBlock calls this unconditionally while building the model, so
    # the TE call shape has to match the installed fork even with offloading off.
    # Arguments mirror Megatron's defaults: offloading disabled, both target
    # flags left on (the MUSA fork rejects two false flags even when disabled).
    from megatron.core.extensions import transformer_engine as te_ext

    if te_ext.get_cpu_offload_context is not None:
        te_ext.get_cpu_offload_context(False, 1, 1, True, True, False)
        result["cpu_offload_context_ok"] = True

    if result["has_training"]:
        from megatron.training import utils as training_utils

        result["device_arch_version"] = training_utils.get_device_arch_version()

        from megatron.legacy import fused_kernels

        result["fused_kernels_load_is_noop"] = fused_kernels.load(None) is None

    print("@@RESULT@@" + json.dumps(result))
    """
)


@pytest.fixture(scope="module")
def result() -> dict:
    if os.environ.get("MEGATRON_MUSA_RUN_INTEGRATION") != "1":
        pytest.skip("set MEGATRON_MUSA_RUN_INTEGRATION=1 with a working Megatron/MUSA stack")
    env = integration_env()

    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if completed.returncode != 0:
        pytest.fail(
            "integration script failed\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )
    for line in completed.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@") :])
    pytest.fail(f"no result marker in output:\n{completed.stdout}\n{completed.stderr}")


@pytest.mark.integration
def test_every_patch_resolved(result):
    """No patch may be left pending or failed once ``apply()`` has returned."""
    assert result["unresolved"] == {}
    assert set(result["statuses"].values()) <= {"applied", "skipped"}


@pytest.mark.integration
def test_layer_norm_patches_applied(result):
    assert result["layer_norm_is_patched"]
    assert result["block_layer_norm_is_local"]
    assert result["layer_norm_matches_reference"]
    assert result["layer_norm_sequence_parallel_marker"]


@pytest.mark.integration
def test_rope_fusion_is_usable(result):
    assert all(result["fused_rope_kernels"]), result["fused_rope_kernels"]
    assert result["rope_fusion_config_accepted"]


@pytest.mark.integration
def test_cpu_offload_context_matches_the_installed_te(result):
    if "cpu_offload_context_ok" not in result:
        pytest.skip("Transformer Engine is not importable")
    assert result["cpu_offload_context_ok"]


@pytest.mark.integration
def test_training_patches(result):
    if not result["has_training"]:
        pytest.skip("megatron.training is not importable (megatron-core wheel only)")
    assert result["device_arch_version"] == 8
    assert result["fused_kernels_load_is_noop"]
