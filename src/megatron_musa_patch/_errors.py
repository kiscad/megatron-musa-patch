"""Error types raised by :mod:`megatron_musa_patch`.

Everything derives from :class:`MegatronMusaPatchError` so callers can catch a
single base class.  The errors are deliberately verbose: this package patches a
fast-moving third-party library, so when upstream drifts we want the failure to
name the exact symbol, the detected version and the patch that broke, instead
of a bare ``AttributeError`` deep inside a training step.
"""

from __future__ import annotations

__all__ = [
    "MegatronMusaPatchError",
    "PatchTargetMissing",
    "PatchConflict",
    "UnsupportedMegatronVersion",
    "MegatronMissing",
    "MusaUnavailable",
]


class MegatronMusaPatchError(RuntimeError):
    """Base class for every error raised by this package."""


class PatchTargetMissing(MegatronMusaPatchError):
    """A patch target no longer exists in the installed upstream library."""

    def __init__(self, patch_id: str, target: str, version: str | None, detail: str = ""):
        self.patch_id = patch_id
        self.target = target
        self.version = version
        message = (
            f"patch {patch_id!r} cannot be applied: {target!r} was not found in the "
            f"installed upstream package (detected version: {version or 'unknown'}). "
            "Upstream Megatron-LM probably renamed or removed this symbol."
        )
        if detail:
            message += f" [{detail}]"
        super().__init__(message)


class PatchConflict(MegatronMusaPatchError):
    """Something else already claimed the attribute we wanted to patch."""


class UnsupportedMegatronVersion(MegatronMusaPatchError):
    """The installed upstream version is outside the supported range."""


class MegatronMissing(MegatronMusaPatchError):
    """Nothing importable provides the ``megatron`` package."""


class MusaUnavailable(MegatronMusaPatchError):
    """torch_musa is missing, or no MUSA device is visible."""
