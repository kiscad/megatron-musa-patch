"""The patch ledger.

Every patch this package applies to Megatron lives in one of the modules below
as declarative data.  Adding a patch means adding a record here -- you should
never need to copy an upstream file into this repository.

Ledger convention (borrowed from ``vllm-ascend``, which learned it the hard
way): each record documents *what* it targets, the observed *root cause*, the
chosen *strategy*, *which upstream file* it adapts, and upgrade review/removal
conditions. ``remove_when`` is a maintenance contract: a patch whose condition
has been met should be removed, not kept "just in case". Compatibility policy
may change kernel selection, fusion, overlap or I/O parallelism; validate
correctness and performance on the actual upgraded stack.
"""

from __future__ import annotations

from . import (
    _torch_backend,
)

__all__ = ["PATCHES", "MODULES"]

#: Order matters for patches that touch the same symbol.  The device
#: compatibility layer goes first so that anything reading ``torch.cuda`` while
#: its own replacement is being built already sees a working namespace.
MODULES = (
    _torch_backend,
)

#: Same-target factories compose in registration order. Same-module companion
#: requirements are declared on AttrPatch.requires and ordered by the engine.
PATCHES = tuple(patch for module in MODULES for patch in module.PATCHES)
