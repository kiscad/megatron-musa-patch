"""Environment-variable switches.

All switches live in one place so the documented behaviour and the code cannot
drift apart.  Every switch is read *lazily* (at access time, never cached at
import time) because the interpreter that imports this package during
``import torch`` is usually not the one that decides the values -- test suites
and wrapper scripts set them just before importing Megatron.

=================================  ====================================  =========
Variable                           Meaning                               Default
=================================  ====================================  =========
``MEGATRON_MUSA_PATCH``            Master switch. ``0`` makes both the   ``1``
                                   auto channel *and* ``import
                                   megatron_musa_patch`` a no-op.
``..._AUTOLOAD``                   Only the automatic (entry point)      ``1``
                                   channel.
``..._STRICT``                     Version drift raises instead of       ``0``
                                   warning.
``..._DISABLE``                    Comma separated patch ids to skip.    *(empty)*
``..._ONLY``                       Comma separated patch ids to run      *(empty)*
                                   (whitelist; overrides ``_DISABLE``).
``..._DEBUG``                      Log every applied patch at INFO.      ``0``
``..._ARCH``                       Faked NVIDIA compute capability,      ``8.3``
                                   e.g. ``9.0``.
``..._BLOCK_LAYERNORM``            ``upstream`` keeps                     ``local``
                                   ``LayerNormImpl = TENorm``.
``..._TE_FUSED_LAYERNORM``         ``1`` restores TE fused norm-linear. ``0``
``..._TE_NORM``                    ``1`` keeps TE's standalone             ``0``
                                   LayerNorm/RMSNorm (TENorm) instead of
                                   the functional fallback.
``..._IGNORE_VERSION_GATES``       ``1`` disables every declarative       *(empty)*
                                   version gate; a comma-separated list
                                   of package names bypasses only those
                                   packages (e.g. ``transformer_engine``)
                                   for trialling an out-of-range build.
``..._ROPE_FUSION``                ``0`` declines the apex fused-RoPE     ``1``
                                   fallback, keeping upstream's
                                   "apply_rope_fusion is not available"
                                   verdict.
``..._JIT_WARMUP``                 ``1`` keeps upstream's JIT warm-up.   ``0``
``..._CKPT_FORK``                  ``1`` keeps upstream's forked         ``0``
                                   checkpoint writer.
``..._DP_OVERLAP``                 ``1`` honours the DP-overlap flags    ``0``
                                   again (legacy spelling
                                   ``..._TP_OVERLAP`` applies when
                                   ``..._DP_OVERLAP`` is unset).
``..._TEARDOWN``                   ``0`` skips the clean shutdown        ``1``
                                   handler.
=================================  ====================================  =========
"""

from __future__ import annotations

import os
import re

__all__ = [
    "ENV_PREFIX",
    "flag",
    "value",
    "enabled",
    "autoload_enabled",
    "strict",
    "debug",
    "disabled_ids",
    "only_ids",
    "patch_enabled",
]

ENV_PREFIX = "MEGATRON_MUSA_PATCH"

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f", ""}


def _env_name(suffix: str | None = None) -> str:
    return ENV_PREFIX if suffix is None else f"{ENV_PREFIX}_{suffix}"


def value(suffix: str | None = None, default: str | None = None) -> str | None:
    """Raw string value of a switch."""
    return os.environ.get(_env_name(suffix), default)


def flag(suffix: str | None = None, default: bool = True) -> bool:
    """Boolean value of a switch.  Unparseable values fall back to ``default``."""
    raw = os.environ.get(_env_name(suffix))
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    return default


def enabled() -> bool:
    """Master switch."""
    return flag(None, True)


def autoload_enabled() -> bool:
    """Is the automatic activation channel allowed to run?"""
    return enabled() and flag("AUTOLOAD", True)


def strict() -> bool:
    """Turn upstream-version drift into a hard error."""
    return flag("STRICT", False)


def debug() -> bool:
    """Verbose per-patch logging."""
    return flag("DEBUG", False)


def _id_set(suffix: str) -> frozenset[str]:
    raw = os.environ.get(_env_name(suffix), "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def disabled_ids() -> frozenset[str]:
    return _id_set("DISABLE")


def only_ids() -> frozenset[str]:
    return _id_set("ONLY")


def patch_enabled(patch_id: str) -> bool:
    """Apply the ``ONLY`` / ``DISABLE`` filters to a single patch id."""
    only = only_ids()
    if only:
        return patch_id in only
    return patch_id not in disabled_ids()


def normalize_distribution_name(name: str) -> str:
    """Use distribution-name equivalence for metadata and override lists."""
    return re.sub(r"[-_.]+", "-", name).lower()


def version_gate_overrides() -> frozenset[str]:
    """Packages whose declarative version gates are forcibly satisfied.

    ``..._IGNORE_VERSION_GATES=1`` (or ``true``) overrides every gate;
    a comma-separated list of package names overrides only those packages.
    Used to trial a workaround outside its declared applicability range.
    """
    raw = os.environ.get(_env_name("IGNORE_VERSION_GATES"))
    if not raw:
        return frozenset()
    raw = raw.strip().lower()
    if raw in _FALSY:
        return frozenset()
    if raw in _TRUTHY:
        return frozenset({"*"})
    return frozenset(
        normalize_distribution_name(part.strip()) for part in raw.split(",") if part.strip()
    )
