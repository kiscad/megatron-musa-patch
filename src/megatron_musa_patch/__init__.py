"""megatron-musa-patch -- adapt upstream Megatron-LM to Moore Threads MUSA GPUs.

Quick start
-----------
Install the package and do nothing else::

    pip install --no-deps megatron-musa-patch  # runtime dependencies preinstalled
    torchrun --nproc_per_node=8 pretrain_gpt.py ...

The ``torch.backends`` entry point in ``pyproject.toml`` makes PyTorch call
:func:`megatron_musa_patch.activation.torch_backend_autoload` at the end of
``import torch``.  From there a ``sys.meta_path`` watcher waits for ``megatron``
to be imported and patches target modules after they execute. Activate before
Megatron uses CUDA APIs; late activation repairs only supported module aliases,
not existing instances, closures or class bases.

Prefer to be explicit?  ``import megatron_musa_patch`` at the top of your script
does the same thing.  Need it applied right now?  Call
:func:`megatron_musa_patch.apply`.

Inspect what happened::

    >>> import megatron_musa_patch
    >>> megatron_musa_patch.report()          # doctest: +SKIP
    [{'id': 'torch.cuda.compat-layer', 'kind': 'hook', 'status': 'applied', ...}, ...]

Switches (see :mod:`megatron_musa_patch._env` for the full list)::

    MEGATRON_MUSA_PATCH=0                  disable everything
    MEGATRON_MUSA_PATCH_AUTOLOAD=0         disable only the automatic channel
    MEGATRON_MUSA_PATCH_DISABLE=id1,id2    skip individual patches
    MEGATRON_MUSA_PATCH_ONLY=id1           run only these patches
    MEGATRON_MUSA_PATCH_STRICT=1           fail on unsupported upstream versions
    MEGATRON_MUSA_PATCH_DEBUG=1            log every applied patch

Extending
---------
Register your own patch so it is applied with the same ordering guarantees::

    from megatron_musa_patch import AttrPatch, ENGINE

    ENGINE.register([AttrPatch(
        id="my-project.foo",
        target="megatron.core.some.module:SomeClass",
        replace=lambda original: MyReplacement,
        rationale="why this is needed",
    )])
    ENGINE.apply_now()

``replace`` receives the object currently bound to the target, so a patch can
wrap the original (``functools.wraps``) instead of replacing it outright.
"""

from __future__ import annotations

from . import _env
from ._compat import SUPPORTED_VERSION_SPEC, logger, megatron_version
from ._engine import ENGINE, AppliedPatch, AttrPatch, HookPatch
from ._errors import (
    MegatronMissing,
    MegatronMusaPatchError,
    MusaUnavailable,
    PatchConflict,
    PatchTargetMissing,
    UnsupportedMegatronVersion,
)
from .activation import torch_backend_autoload  # also the "torch.backends" entry point target
from .activation import apply, install, is_applied, report, uninstall


def _distribution_version() -> str:
    """Read our own version from the installed metadata.

    Single source of truth: ``pyproject.toml``.  Falls back to a marker when
    running from a bare source checkout that was never installed.
    """
    import importlib.metadata as md

    try:
        return md.version("megatron-musa-patch")
    except md.PackageNotFoundError:  # pragma: no cover - uninstalled checkout
        return "0.0.0+unknown"


__version__ = _distribution_version()

__all__ = [
    "__version__",
    # activation
    "install",
    "apply",
    "uninstall",
    "is_applied",
    "report",
    "torch_backend_autoload",
    # engine / extension points
    "ENGINE",
    "AttrPatch",
    "HookPatch",
    "AppliedPatch",
    # errors
    "MegatronMusaPatchError",
    "PatchTargetMissing",
    "PatchConflict",
    "UnsupportedMegatronVersion",
    "MegatronMissing",
    "MusaUnavailable",
    # metadata
    "SUPPORTED_VERSION_SPEC",
    "megatron_version",
    "logger",
]


def _loaded_by_torch() -> bool:
    """EntryPoint.load imports this package before invoking its callback.

    Defer installation during torch initialization so AUTOLOAD=0 and the
    callback's error boundary are not bypassed by this module's import effect.
    This probes sys.modules only; it never imports torch.
    """
    import sys

    torch = sys.modules.get("torch")
    return bool(getattr(getattr(torch, "__spec__", None), "_initializing", False))


if _env.enabled() and not _loaded_by_torch():
    install()
