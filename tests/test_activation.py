"""Public activation and entry-point loading, isolated from unit registries."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def _run(script, **switches):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEGATRON_MUSA_PATCH")}
    env.update(switches)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_explicit_import_only_installs_watcher():
    _run("""
        import sys
        import megatron_musa_patch as m
        assert 'torch' not in sys.modules
        assert 'megatron' not in sys.modules
        assert m.report()
        assert {r['status'] for r in m.report()} == {'pending'}
        m.uninstall()
        assert not m.ENGINE._installed
    """)


def test_entrypoint_load_honours_autoload_switch():
    _run("""
        import sys, types
        from importlib.machinery import ModuleSpec
        torch = types.ModuleType('torch')
        torch.__spec__ = ModuleSpec('torch', None)
        torch.__spec__._initializing = True
        sys.modules['torch'] = torch
        import megatron_musa_patch as m
        assert not m.ENGINE._installed
        m.torch_backend_autoload()
        assert not m.ENGINE._installed
        torch.__spec__._initializing = False
        m.install()  # explicit activation still works after EntryPoint.load
        assert m.ENGINE._installed
    """, MEGATRON_MUSA_PATCH_AUTOLOAD="0")


def test_entrypoint_catches_install_failure():
    _run("""
        import sys, types
        from importlib.machinery import ModuleSpec
        torch = types.ModuleType('torch')
        torch.__spec__ = ModuleSpec('torch', None)
        torch.__spec__._initializing = True
        sys.modules['torch'] = torch
        import megatron_musa_patch as m
        from megatron_musa_patch import activation
        def fail(): raise RuntimeError('test failure')
        activation.install = fail
        m.torch_backend_autoload()  # must not break import torch
        assert not m.ENGINE._installed
    """)


def test_disabled_apply_never_probes_or_imports_megatron(monkeypatch):
    from megatron_musa_patch import activation

    monkeypatch.setenv("MEGATRON_MUSA_PATCH", "0")

    def forbidden():
        pytest.fail("disabled apply probed Megatron")

    monkeypatch.setattr(activation, "check_megatron_present", forbidden)
    activation.apply()
