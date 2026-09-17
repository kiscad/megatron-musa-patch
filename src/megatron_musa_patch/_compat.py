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
from functools import lru_cache
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
    "distribution_version",
    "te_version",
    "torch_musa_version",
    "transformers_version",
    "module_source_contains",
    "parse_version_gate",
    "check_version_gate",
    "validate_version_gates",
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


def distribution_version(name: str) -> str | None:
    """Installed version of an external distribution, or ``None``.

    Used by patches whose applicability is tied to a specific broken range of
    a vendor package (torch_musa, transformer_engine, transformers, ...).
    Version strings are advisory here: the MUSA forks are known to report
    numbers that do not follow the upstream API timeline, so callers must
    pair these with a capability or source probe whenever one exists.
    """
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def te_version() -> str | None:
    """Version string of the installed Transformer Engine (MUSA fork)."""
    for name in ("transformer_engine", "transformer_engine_cu12", "transformer-engine"):
        version = distribution_version(name)
        if version is not None:
            return version
    return None


def torch_musa_version() -> str | None:
    """Version string of the installed torch_musa package."""
    return distribution_version("torch_musa")


def transformers_version() -> str | None:
    """Version string of the installed transformers package."""
    return distribution_version("transformers")


def module_source_contains(module_name: str, *markers: str) -> bool | None:
    """Check markers in a module's source **without executing the module**.

    Several MUSA-fork defects this package works around are identified by
    exact code patterns whose presence depends on the vendor build, and the
    vendor version strings do not follow the upstream API timeline.  Reading
    source markers provide a conservative build fingerprint, not proof of
    kernel correctness. The probe never runs the module body.

    Dotted names are resolved from the top-level package's spec by joining
    the remaining parts as file paths -- resolving a submodule through the
    import system would import (and execute) its parents.

    Returns ``True`` when every marker appears in the source, ``False`` when
    the source is readable and any marker is missing, and ``None`` when the
    probe cannot decide (unsupported layout or unreadable source).
    A missing package or submodule returns False.
    Callers choose their own policy for the unknown case.
    """
    if not markers:
        raise ValueError("module_source_contains requires at least one marker")
    parts = module_name.split(".")
    try:
        spec = find_spec_without_watchers(parts[0])
    except Exception:  # noqa: BLE001 - a broken lookup is an undecided probe
        return None
    if spec is None:
        return False  # the top-level package is genuinely absent
    origin = getattr(spec, "origin", None)
    if not origin or not origin.endswith(".py"):
        return None  # namespace/zip layout: undecidable without executing
    from pathlib import Path

    if len(parts) == 1:
        target = Path(origin)
    elif not getattr(spec, "submodule_search_locations", None):
        return False
    else:
        target = Path(origin).parent.joinpath(*parts[1:])
    if target.is_dir():
        target = target / "__init__.py"
    elif not target.exists():
        target = target.with_name(target.name + ".py") if not target.name.endswith(".py") else target
    if not target.exists():
        return False  # submodule file genuinely absent
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read()
    except OSError:
        return None
    return all(marker in source for marker in markers)


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


#: Comparison operators accepted in declarative version gates.
_GATE_OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@lru_cache(maxsize=256)
def parse_version_gate(spec: str) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    """Parse AND-ed numeric release bounds; this is not PEP 440 ordering.

    Bounds contain only dot-separated integers. Installed prerelease/local
    suffixes are ignored deliberately, like the existing Megatron version
    guard. Parsing is cached, but environment and installed metadata are not.
    """
    match = re.fullmatch(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(.+?)\s*", spec)
    if match is None:
        raise ValueError(f"malformed version gate: {spec!r}")
    name, remainder = match.groups()
    checks = []
    for part in remainder.split(","):
        comparison = re.fullmatch(r"\s*(>=|<=|==|!=|>|<)\s*([0-9]+(?:\.[0-9]+)*)\s*", part)
        if comparison is None:
            raise ValueError(f"malformed comparison {part!r} in gate: {spec!r}")
        op, release = comparison.groups()
        checks.append((op, tuple(map(int, release.split(".")))))
    return _env.normalize_distribution_name(name), tuple(checks)


def validate_version_gates(gates: tuple[str, ...]) -> None:
    """Shared declaration validation for AttrPatch and HookPatch."""
    if not isinstance(gates, tuple) or any(not isinstance(g, str) for g in gates):
        raise ValueError("version_gates must be a tuple of gate strings")
    for gate in gates:
        parse_version_gate(gate)


def check_version_gate(spec: str) -> tuple[bool, str]:
    """Return (blocked, detail) using installed metadata, without imports.

    Missing metadata preserves existing target/capability handling; malformed
    installed versions block rather than accidentally satisfying an upper bound.
    Ignore switches affect only version gates, never capability probes or ONLY.
    """
    name, checks = parse_version_gate(spec)
    overrides = _env.version_gate_overrides()
    if "*" in overrides or name in overrides:
        return False, f"gate overridden by {_env.ENV_PREFIX}_IGNORE_VERSION_GATES: {spec}"
    installed = distribution_version(name)
    if installed is None:
        return False, f"{name} metadata unavailable (gate {spec} not enforced)"
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)*)(?:(?:a|b|rc|[.-]?(?:dev|post))[0-9]*)*"
        r"(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?",
        installed,
    )
    if match is None:
        return True, f"{name} has an unrecognized version {installed!r} (gate {spec})"
    version = tuple(map(int, match.group(1).split(".")))
    for op, bound in checks:
        width = max(len(version), len(bound))
        actual = version + (0,) * (width - len(version))
        expected = bound + (0,) * (width - len(bound))
        if not _GATE_OPS[op](actual, expected):
            return True, f"{name} {installed} outside declared range ({spec})"
    return False, f"{name} {installed} within {spec}"
