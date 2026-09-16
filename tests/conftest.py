"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
import textwrap
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Unit collection must not install the process-wide watcher. Hardware tests
# activate explicitly in isolated subprocesses.
os.environ["MEGATRON_MUSA_PATCH"] = "0"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
from megatron_musa_patch._engine import Engine  # noqa: E402


def integration_env(extra: dict | None = None) -> dict:
    """Environment for subprocess tests: patch package on, src importable."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEGATRON_MUSA_PATCH")}
    env["MEGATRON_MUSA_PATCH"] = "1"
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    paths = [str(Path(__file__).resolve().parents[1] / "src")]
    if env.get("MEGATRON_LM_PATH"):
        paths.append(env["MEGATRON_LM_PATH"])
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if extra:
        env.update(extra)
    return env


@pytest.fixture
def engine(monkeypatch) -> Engine:
    """A private registry, so tests never touch the process-wide one."""
    for name in list(os.environ):
        if name.startswith("MEGATRON_MUSA_PATCH"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("MEGATRON_MUSA_PATCH", "1")
    fresh = Engine()
    yield fresh
    fresh.unapply()


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    """Create an importable throw-away package on ``sys.path``.

    Returns a factory ``make(name, source)`` that writes ``<tmp>/<name>.py``
    and returns the dotted module name.  Modules created this way are removed
    from ``sys.modules`` again at teardown.
    """
    root = tmp_path / "fakepkgs"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))
    created: list[str] = []

    def make(name: str, source: str) -> str:
        path = root / f"{name}.py"
        path.write_text(textwrap.dedent(source))
        created.append(name)
        return name

    yield make

    for name in created:
        sys.modules.pop(name, None)


@pytest.fixture
def stub_module(monkeypatch):
    """Install a synthetic module in ``sys.modules`` and clean it up after."""
    created: list[str] = []

    def make(name: str, **attributes) -> types.ModuleType:
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        created.append(name)
        return module

    yield make

    for name in created:
        sys.modules.pop(name, None)
