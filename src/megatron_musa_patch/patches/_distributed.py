"""Best-effort process-group teardown for normal interpreter exit.

Runs that leave MCCL groups live during Python finalization have exhibited
watchdog errors and, on some stacks, aborts. An atexit callback improves the
normal-exit path; it cannot make teardown deterministic across ranks, handle
SIGKILL, or replace a launcher's explicit coordinated cleanup.
"""

from __future__ import annotations

from typing import Callable

from .. import _env
from .._engine import HookPatch

__all__ = ["PATCHES"]

_teardown_callback: Callable[[], None] | None = None


def _install_clean_teardown() -> bool | None:
    global _teardown_callback
    if not _env.flag("TEARDOWN", True) or _teardown_callback is not None:
        return False

    import atexit

    import torch.distributed as dist

    def _teardown() -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001 - best effort during interpreter exit
            pass

    atexit.register(_teardown)
    _teardown_callback = _teardown


def _uninstall_clean_teardown() -> None:
    global _teardown_callback
    if _teardown_callback is None:
        return
    import atexit

    atexit.unregister(_teardown_callback)
    _teardown_callback = None


PATCHES = (
    HookPatch(
        id="torch.distributed.clean-teardown",
        trigger="megatron",
        run=_install_clean_teardown,
        undo=_uninstall_clean_teardown,
        rationale=(
            "MUSA bring-up runs that left process groups alive at interpreter exit "
            "showed MCCL watchdog/finalization errors. The affected entry points "
            "did not explicitly destroy the default group."
        ),
        strategy=(
            "Register one best-effort atexit callback to destroy an initialized "
            "process group, without adding a barrier. TEARDOWN=0 skips registration; "
            "undo unregisters only this callback and does not destroy a live group."
        ),
        upstream="pytorch/pytorch torch/distributed/distributed_c10d.py; MCCL shutdown",
        remove_when=(
            "Review on torch_musa/MCCL and launcher/Megatron upgrades; remove when "
            "the entry point owns explicit cleanup or repeated multi-rank normal "
            "exit tests pass with TEARDOWN=0. Test failure/interrupt paths separately."
        ),
    ),
)
