"""Activation channels.

There are exactly three ways this package can come alive, and all of them end
up in the same idempotent :meth:`~megatron_musa_patch._engine.Engine.install`:

1. **Automatic** -- the ``torch.backends`` entry point declared in
   ``pyproject.toml``.  PyTorch imports every entry point in that group at the
   very end of ``import torch``, which is before any Megatron module can be
   imported.  Kill switches: ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` (torch's own)
   or ``MEGATRON_MUSA_PATCH_AUTOLOAD=0`` (ours).
2. **Explicit** -- ``import megatron_musa_patch`` at the top of a script.
3. **Imperative** -- :func:`megatron_musa_patch.apply`, e.g. from a test
   fixture or a launcher.

The automatic channel deliberately does *not* touch anything until Megatron is
actually imported: installing a CUDA compatibility layer into every Python
process in the environment just because the package happens to be installed
would be rude, and the training script is the only thing that needs it.
"""

from __future__ import annotations

import logging

from . import _env
from ._compat import check_megatron_present
from ._engine import ENGINE
from .patches import PATCHES

__all__ = ["install", "apply", "uninstall", "is_applied", "report", "torch_backend_autoload"]

logger = logging.getLogger("megatron_musa_patch")

_registered = False


def _register() -> None:
    global _registered
    if not _registered:
        ENGINE.register(PATCHES)
        _registered = True


def install() -> None:
    """Register the patch set and start watching for Megatron imports.

    Cheap and safe to call from inside ``import torch``: it imports neither
    ``torch`` nor ``megatron`` and only adds one finder to ``sys.meta_path``.
    """
    if not _env.enabled():
        return
    _register()
    ENGINE.install()


def apply() -> None:
    """Install *and* apply every patch right now.

    Unlike :func:`install` this forces the device compatibility layer to be
    built immediately instead of waiting for Megatron to be imported.  Use it
    when you want the CUDA compatibility layer available to your own code, or
    in tests.

    This is also where a missing Megatron is reported: calling ``apply()`` is an
    explicit statement of intent, whereas the automatic channel runs inside
    ``import torch`` in every process in the environment and must stay quiet.
    """
    if not _env.enabled() or not check_megatron_present():
        return
    install()
    ENGINE.apply_now()


def uninstall() -> None:
    """Restore the original objects and stop watching imports."""
    ENGINE.unapply()


def is_applied() -> bool:
    """Have any patches been applied?"""
    return ENGINE.is_applied()


def report() -> list[dict]:
    """Machine-readable description of every registered patch."""
    return ENGINE.report()


def torch_backend_autoload() -> None:
    """Entry point for the ``torch.backends`` group -- must never raise.

    This runs inside ``import torch``.  An exception here would break *every*
    process in the environment, so failures are reported through the logging
    module instead.
    """
    if not _env.autoload_enabled():
        logger.debug(
            "%s_AUTOLOAD=0 -- megatron-musa-patch will not activate automatically",
            _env.ENV_PREFIX,
        )
        return
    try:
        install()
    except Exception:  # noqa: BLE001 - never break `import torch`
        logger.exception(
            "megatron-musa-patch failed to activate; the training run will most "
            "likely fail later. Set %s_DEBUG=1 for details.",
            _env.ENV_PREFIX,
        )
