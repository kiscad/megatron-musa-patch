"""Compatibility helpers: upstream version detection and target resolution.

This module is intentionally **standard-library only** and must never import
``torch`` or ``megatron`` at module scope: it runs inside ``import torch`` via
the ``torch.backends`` entry point, and it also runs inside ``import
megatron_musa_patch`` before Megatron exists.
"""

from __future__ import annotations

import importlib.metadata as md
import logging
import re
import sys
from typing import Any

from . import _env
from ._errors import MegatronMissing, PatchTargetMissing, UnsupportedMegatronVersion

__all__ = [
    "SUPPORTED_VERSION_SPEC",
    "logger",
    "parse_version",
    "megatron_distribution",
    "megatron_version",
    "megatron_importable",
    "check_version",
    "check_megatron_present",
    "split_target",
    "require_attr",
]

logger = logging.getLogger("megatron_musa_patch")

#: Marks the engine's ``_ImportWatcher`` on ``sys.meta_path``.  Probing code
#: (``megatron_importable``) must skip watchers, because a plain
#: ``find_spec("megatron")`` would run the pending hook patches as a side
#: effect.  Duck-typed on purpose: the watcher sets the attribute, the probe
#: reads it, and neither needs to import the other's module.
META_PATH_WATCHER_MARKER = "__megatron_musa_patch_import_watcher__"

#: The upstream range this patch set has been written against.  Only the
#: symbols we actually touch matter, so a narrow-but-honest range is better
#: than pretending to support everything.
SUPPORTED_VERSION_SPEC = ">=0.14,<0.17"

#: Distribution names that may carry an upstream Megatron-LM / Megatron-Core.
_DISTRIBUTIONS = ("megatron-core", "megatron_core", "megatron-lm", "Megatron-LM")

_VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def parse_version(text: str | None) -> tuple[int, ...]:
    """Parse the leading numeric part of a version string.

    ``"0.16.1rc0"`` -> ``(0, 16, 1)``.  We deliberately do not depend on
    ``packaging``: a missing dependency inside the activation path would break
    every interpreter in the environment (see the README).
    """
    if not text:
        return ()
    match = _VERSION_RE.search(text)
    if not match:
        return ()
    return tuple(int(part) for part in match.groups() if part is not None)


def megatron_distribution() -> str | None:
    """Name of the installed distribution that provides ``megatron``."""
    for name in _DISTRIBUTIONS:
        try:
            md.version(name)
        except md.PackageNotFoundError:
            continue
        return name
    return None


def megatron_version() -> str | None:
    """Version of the installed Megatron-LM / Megatron-Core, or ``None``.

    Uses import metadata on purpose: importing ``megatron.core.package_info``
    would drag ``torch`` (and, on some builds, initialise the device) into
    processes that only wanted to inspect the environment.
    """
    dist = megatron_distribution()
    if dist is None:
        return None
    try:
        return md.version(dist)
    except md.PackageNotFoundError:  # pragma: no cover - race with pip
        return None


def _in_supported_range(version: tuple[int, ...]) -> bool:
    if not version:
        return True  # unknown -> do not block, just warn elsewhere
    return (0, 14) <= version < (0, 17)


def check_version(*, strict: bool | None = None) -> str | None:
    """Warn (or raise) when upstream is outside :data:`SUPPORTED_VERSION_SPEC`."""
    version = megatron_version()
    if version is None:
        logger.debug(
            "no Megatron distribution metadata found; skipping version check "
            "(running from a source checkout?)"
        )
        return None
    if _in_supported_range(parse_version(version)):
        return version

    message = (
        f"megatron-musa-patch was written against Megatron {SUPPORTED_VERSION_SPEC}, "
        f"but {version} is installed. Patches are applied defensively and each one "
        "fails loudly if its target moved, but behaviour is not validated. "
        f"Set {_env.ENV_PREFIX}_STRICT=1 to turn this into an error."
    )
    if strict is None:
        strict = _env.strict()
    if strict:
        raise UnsupportedMegatronVersion(message)
    logger.warning(message)
    return version


def megatron_importable() -> bool:
    """Is there a ``megatron`` package on ``sys.path``?

    Two properties matter here:

    * **No imports.**  ``import megatron`` would pull in torch and a partially
      initialised Megatron into a process that only asked a question.  And
      ``megatron`` is a namespace package anyway, so we only care whether the
      name resolves at all -- a found spec may well have ``loader is None``.
    * **No watcher wakes.**  The import watcher treats any ``find_spec`` for a
      watched module as "an import is starting", which is true for the import
      machinery but *not* for probes.  Once it sits on ``sys.meta_path`` (the
      automatic channel installs it at the end of ``import torch``), a plain
      ``importlib.util.find_spec("megatron")`` would run the pending hook
      patches -- building the whole torch.cuda compatibility layer in an
      unrelated process.  The walk below skips every finder marked with
      :data:`META_PATH_WATCHER_MARKER`, mirroring ``Engine._find_real_spec``.
    """
    if sys.modules.get("megatron") is not None:
        return True
    return find_spec_without_watchers("megatron") is not None


def find_spec_without_watchers(fullname: str, path=None, target=None):
    """Resolve through real finders without importing parents or firing hooks.

    A finder returning None means 'not mine'; an exception is a broken lookup
    and must propagate rather than being misreported as an absent package.
    """
    for finder in sys.meta_path:
        if getattr(finder, META_PATH_WATCHER_MARKER, False):
            continue
        find_spec = getattr(finder, "find_spec", None)
        if find_spec is not None:
            spec = find_spec(fullname, path, target)
            if spec is not None:
                return spec
    return None


def check_megatron_present(*, strict: bool | None = None) -> bool:
    """Warn (or raise) when nothing provides ``megatron``.

    Only called from the explicit entry points (:func:`apply`,
    ``import megatron_musa_patch``).  The automatic channel must stay silent
    here: it runs at the end of ``import torch`` in *every* process in the
    environment, most of which will never import Megatron.
    """
    if megatron_importable():
        return True

    message = (
        "megatron-musa-patch found no `megatron` package to patch, so nothing "
        "will take effect. Put a Megatron-LM checkout on PYTHONPATH "
        "(`git clone https://github.com/NVIDIA/Megatron-LM && git checkout "
        "core_v0.16.1`) or `pip install megatron-core`. Set "
        f"{_env.ENV_PREFIX}_STRICT=1 to turn this into an error."
    )
    if strict is None:
        strict = _env.strict()
    if strict:
        raise MegatronMissing(message)
    logger.warning(message)
    return False


def split_target(target: str) -> tuple[str, str]:
    """Split ``"pkg.module:attr"`` into ``("pkg.module", "attr")``.

    ``"pkg.module.attr"`` is also accepted; the last dot separates the
    attribute.  The colon form is preferred because it is unambiguous.
    """
    if ":" in target:
        module, _, attr = target.partition(":")
    else:
        module, _, attr = target.rpartition(".")
    if not module or not attr or not all(
        part.isidentifier() for part in (module + "." + attr).split(".")
    ):
        raise ValueError(f"malformed patch target: {target!r}")
    return module, attr


def require_attr(module: Any, dotted: str, *, patch_id: str = "<unknown>") -> Any:
    """Return ``module.a.b.c`` or raise :class:`PatchTargetMissing`."""
    obj: Any = module
    walked: list[str] = []
    for part in dotted.split("."):
        walked.append(part)
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise PatchTargetMissing(
                patch_id=patch_id,
                target=f"{getattr(module, '__name__', module)}.{'.'.join(walked)}",
                version=megatron_version(),
                detail="attribute lookup failed",
            ) from None
    return obj
