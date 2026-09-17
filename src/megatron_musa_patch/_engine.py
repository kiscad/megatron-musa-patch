"""Small, lazy patch registry with per-target transactions and explicit undo.

Attribute factories compose in registration order. A target is committed only
when every factory succeeds; repeat application does not stack wrappers. Hooks
run before their trigger and own their cleanup. Activation must precede code
that *uses* CUDA APIs: existing instances, closures and class bases cannot be
repaired after the fact. No upstream source is rewritten.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import inspect
import logging
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Union

from . import _env
from ._compat import (
    META_PATH_WATCHER_MARKER,
    check_version,
    find_spec_without_watchers,
    megatron_version,
    require_attr,
    split_target,
    validate_version_gates,
    check_version_gate,
)
from ._errors import MegatronMusaPatchError, PatchConflict

__all__ = ["AttrPatch", "HookPatch", "AppliedPatch", "Engine", "Patch"]
logger = logging.getLogger("megatron_musa_patch")


@dataclass(frozen=True)
class AttrPatch:
    """Replace a ``module:attribute`` (including ``module:Class.method``).

    ``replace(current)`` returns a replacement, or None to decline. Built-in
    records must explain the root cause (rationale), implementation (strategy),
    upstream location and a testable removal condition. Same-name function/class
    aliases are repaired only in ``rebind_prefixes``; primitive flags are never
    globally rebound. Static/class method descriptors are preserved.

    ``requires`` names companion AttrPatches on other attributes of the same
    module. Companions run first regardless of registration order. Missing,
    disabled or declined companions leave the consumer skipped with a reason;
    filtering never implicitly enables another patch. Cycles are rejected.
    """

    id: str
    target: str
    replace: Callable[[Any], Any]
    rationale: str = ""
    upstream: str = ""
    remove_when: str = ""
    strategy: str = ""
    rebind_prefixes: tuple[str, ...] = ("megatron",)
    # Required companion patches on other targets in this same module. These
    # are ordered explicitly, never enabled implicitly by ONLY/DISABLE.
    requires: tuple[str, ...] = ()
    # Declarative applicability gates, e.g. ("transformer_engine >=2.0,<2.1",).
    # Evaluated by the engine at apply time against installed distributions;
    # a blocked gate skips the patch with the reason in the report. Pair a
    # version gate with a capability probe whenever the vendor's version
    # numbering does not track its API.
    version_gates: tuple[str, ...] = ()

    @property
    def module_name(self) -> str:
        return split_target(self.target)[0]

    @property
    def attr_name(self) -> str:
        return split_target(self.target)[1]

    @property
    def attr_leaf(self) -> str:
        return self.attr_name.rpartition(".")[2]

    def __post_init__(self) -> None:
        split_target(self.target)
        if not callable(self.replace):
            raise TypeError("AttrPatch.replace must be callable")
        if not isinstance(self.rebind_prefixes, tuple) or any(
            not prefix or not isinstance(prefix, str) for prefix in self.rebind_prefixes
        ):
            raise ValueError("rebind_prefixes must be a tuple of nonempty module prefixes")
        if not isinstance(self.requires, tuple) or any(
            not isinstance(item, str) or not item or item == self.id for item in self.requires
        ):
            raise ValueError("requires must contain nonempty companion patch ids, not self")
        validate_version_gates(self.version_gates)


@dataclass(frozen=True)
class HookPatch:
    """Run before ``trigger`` is imported; return False to decline.

    ``undo`` cleans up owned changes. A hook without undo is one-shot even across
    uninstall/reinstall, and remains reported as applied: irreversible external
    effects must not be described as restored. A failing hook must clean up its
    own partial mutations before raising.
    """

    id: str
    trigger: str
    run: Callable[[], Any]
    rationale: str = ""
    upstream: str = ""
    remove_when: str = ""
    strategy: str = ""
    undo: Callable[[], None] | None = None
    # Declarative applicability gates, evaluated by the engine before run().
    version_gates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.trigger or not all(part.isidentifier() for part in self.trigger.split(".")):
            raise ValueError("HookPatch.trigger must name a module")
        if not callable(self.run):
            raise TypeError("HookPatch.run must be callable")
        if self.undo is not None and not callable(self.undo):
            raise TypeError("HookPatch.undo must be callable")
        validate_version_gates(self.version_gates)


Patch = Union[AttrPatch, HookPatch]


@dataclass
class AppliedPatch:
    """Public diagnostic record; runtime bindings are kept separately."""

    patch: Patch
    status: str = "pending"  # pending | applied | skipped | failed
    detail: str = ""

    def as_dict(self) -> dict:
        patch = self.patch
        return {
            "id": patch.id,
            "kind": "attr" if isinstance(patch, AttrPatch) else "hook",
            "target": patch.target if isinstance(patch, AttrPatch) else f"{patch.trigger} (hook)",
            "version_gates": list(patch.version_gates),
            "status": self.status,
            "detail": self.detail,
            "rationale": patch.rationale,
            "strategy": patch.strategy,
            "upstream": patch.upstream,
            "remove_when": patch.remove_when,
            "requires": list(patch.requires) if isinstance(patch, AttrPatch) else [],
        }


@dataclass
class _Binding:
    owner: Any
    leaf: str
    original: Any  # raw descriptor, not its bound value
    replacement: Any
    owned: bool  # inherited attributes must be deleted on undo
    rebind_prefixes: tuple[str, ...] = ()
    aliases: list[tuple[types.ModuleType, str, Any]] = field(default_factory=list)

    def restore(self, targets: set) -> None:
        if inspect.getattr_static(self.owner, self.leaf, None) is self.replacement:
            if self.owned:
                setattr(self.owner, self.leaf, self.original)
            else:
                delattr(self.owner, self.leaf)
        else:
            logger.warning("not restoring %s: another writer replaced the patch", self.leaf)
        for module, name, original in reversed(self.aliases):
            if vars(module).get(name) is self.replacement:
                setattr(module, name, original)
        # Also repair same-name consumers imported after initial application.
        _rebind_aliases(self.owner, self.leaf, self.replacement, self.original,
                       self.rebind_prefixes, targets)


class _PostExecLoader(importlib.abc.Loader):
    def __init__(self, wrapped, fullname: str, engines: list[Engine]):
        self._wrapped = wrapped
        self._fullname = fullname
        self._engines = engines

    def create_module(self, spec):
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        try:
            self._wrapped.exec_module(module)
        except Exception as exc:
            for engine in self._engines:
                engine._mark_import_failed(self._fullname, exc)
            raise
        for engine in self._engines:
            if engine._installed:
                engine._apply_for_module(self._fullname, module, trigger="import")

    def __getattr__(self, item):
        return getattr(self._wrapped, item)


class _ImportWatcher(importlib.abc.MetaPathFinder):
    """Coordinate watchers so multiple engines neither recurse nor hide peers.

    Hooks fire from find_spec because namespace packages have no exec_module.
    Consequently an external speculative find_spec can also run hooks. Internal
    availability probes deliberately bypass all watchers.
    """

    def __init__(self, engine: Engine):
        self._engine = engine
        setattr(self, META_PATH_WATCHER_MARKER, True)

    def find_spec(self, fullname, path=None, target=None):
        if not self._engine._watches(fullname):
            return None
        engines = [
            finder._engine for finder in reversed(sys.meta_path)
            if isinstance(finder, _ImportWatcher) and finder._engine._watches(fullname)
        ]
        spec = find_spec_without_watchers(fullname, path, target)
        if spec is None:
            for engine in engines:
                engine._mark_absent(fullname)
            return None
        for engine in engines:
            engine._run_hooks(fullname)
        if spec.loader is not None and any(fullname in engine._attrs for engine in engines):
            loader = spec.loader
            if isinstance(loader, importlib.util.LazyLoader):
                loader = loader.loader
            spec.loader = _PostExecLoader(loader, fullname, engines)
        return spec


class Engine:
    """Registry + import watcher. Configure before imports, not during training.

    Each target chain is atomic; arbitrary hook effects or imports are not a
    process-wide transaction. On failure report() retains diagnostics and
    unapply() can undo successful, owned changes.
    """

    def __init__(self) -> None:
        self._records: dict[str, AppliedPatch] = {}
        self._attrs: dict[str, list[AttrPatch]] = {}
        self._hooks: dict[str, list[HookPatch]] = {}
        self._bindings: dict[tuple[str, str], _Binding] = {}
        self._undo_order: list[Union[tuple[str, str], HookPatch]] = []
        self._running_hooks: set[str] = set()
        self._watcher = _ImportWatcher(self)
        self._installed = False

    def register(self, patches: Iterable[Patch]) -> None:
        """Validate a whole batch, then register it (also works after install)."""
        patches = tuple(patches)
        ids = set(self._records)
        for patch in patches:
            if not isinstance(patch, (AttrPatch, HookPatch)):
                raise TypeError("patches must be AttrPatch or HookPatch instances")
            if not patch.id:
                raise ValueError("patch id must not be empty")
            if patch.id in ids:
                raise ValueError(f"duplicate patch id: {patch.id!r}")
            ids.add(patch.id)
        self._validate_dependencies([record.patch for record in self._records.values()] + list(patches))
        for patch in patches:
            self._records[patch.id] = AppliedPatch(patch)
            if self._installed:
                self._schedule(patch)
        if self._installed:
            self._apply_already_imported()

    @staticmethod
    def _validate_dependencies(patches: list[Patch]) -> None:
        """Validate target-level DAGs without importing dependency modules.

        Missing companions may be registered later or intentionally omitted;
        consumers report skipped until they are present and applied. Cross-module
        requirements are deliberately unsupported: imports own that ordering.
        """
        records = {patch.id: patch for patch in patches}
        edges = {}
        for patch in patches:
            if not isinstance(patch, AttrPatch):
                continue
            key = (patch.module_name, patch.attr_name)
            for pid in patch.requires:
                dependency = records.get(pid)
                if dependency is None:
                    continue
                if (not isinstance(dependency, AttrPatch)
                        or dependency.module_name != patch.module_name
                        or dependency.attr_name == patch.attr_name):
                    raise ValueError(f"{patch.id}: requires {pid} must target another attribute in the same module")
                edges.setdefault(key, set()).add((dependency.module_name, dependency.attr_name))
        visited, visiting = set(), set()

        def visit(key):
            if key in visiting:
                raise ValueError(f"cyclic patch requirements at {key}")
            if key in visited:
                return
            visiting.add(key)
            for dependency in edges.get(key, ()):
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in edges:
            visit(key)

    def _schedule(self, patch: Patch) -> None:
        record = self._records[patch.id]
        if record.status == "applied" and isinstance(patch, HookPatch):
            return  # irreversible hook retained across uninstall
        if not _env.patch_enabled(patch.id):
            record.status, record.detail = "skipped", "disabled by environment"
        elif isinstance(patch, AttrPatch):
            self._attrs.setdefault(patch.module_name, []).append(patch)
        else:
            self._hooks.setdefault(patch.trigger, []).append(patch)

    def install(self) -> None:
        """Watch future imports and patch fully imported targets immediately."""
        if not _env.enabled() or self._installed:
            return
        if self._bindings or any(
            isinstance(item, HookPatch) and self._records[item.id].status == "failed"
            for item in self._undo_order
        ):
            raise PatchConflict("patch cleanup is incomplete; retry unapply() before install()")
        check_version()  # a failed check must not latch _installed
        unknown = (_env.only_ids() | _env.disabled_ids()) - self._records.keys()
        if unknown:
            logger.warning("unknown patch id(s) in environment: %s", ", ".join(sorted(unknown)))
        self._installed = True
        for record in self._records.values():
            self._schedule(record.patch)
        sys.meta_path.insert(0, self._watcher)
        self._apply_already_imported()

    def apply_now(self) -> None:
        """Import all targets; skip missing targets, never broken dependencies."""
        if not _env.enabled():
            return
        self.install()
        for module_name in list(dict.fromkeys((*self._hooks, *self._attrs))):
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                # Missing target/parent is optional, a missing dependency inside
                # a present target is a real error and must reach the caller.
                if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
                    self._mark_absent(module_name)
                    continue
                self._mark_import_failed(module_name, exc)
                raise
            except Exception as exc:
                self._mark_import_failed(module_name, exc)
                raise
            self._run_hooks(module_name)
            self._apply_for_module(module_name, module, trigger="apply-now")

    def _apply_already_imported(self) -> None:
        for module_name in list(self._hooks):
            if sys.modules.get(module_name) is not None:
                self._run_hooks(module_name)
        for module_name in list(self._attrs):
            module = sys.modules.get(module_name)
            if module is not None and not getattr(getattr(module, "__spec__", None), "_initializing", False):
                self._apply_for_module(module_name, module, trigger="already-imported")

    def _watches(self, fullname: str) -> bool:
        return self._installed and (fullname in self._attrs or fullname in self._hooks)

    def _mark_absent(self, fullname: str) -> None:
        patches = self._attrs.pop(fullname, []) + self._hooks.pop(fullname, [])
        for patch in patches:
            record = self._records[patch.id]
            record.status, record.detail = "skipped", f"module {fullname!r} is not installed"

    def _mark_import_failed(self, fullname: str, exc: Exception) -> None:
        for patch in self._attrs.get(fullname, ()):
            record = self._records[patch.id]
            if record.status == "pending":
                record.status, record.detail = "failed", f"import failed: {type(exc).__name__}: {exc}"

    def _run_hooks(self, module_name: str) -> None:
        if module_name in self._running_hooks:
            return
        self._running_hooks.add(module_name)
        try:
            for patch in self._hooks.get(module_name, ()):
                record = self._records[patch.id]
                if record.status in {"applied", "skipped"}:
                    continue
                try:
                    gate_detail = self._version_gate_block(patch)
                    if gate_detail is not None:
                        record.status, record.detail = "skipped", f"version gate: {gate_detail}"
                        self._log(record)
                        continue
                    result = patch.run()
                except Exception as exc:
                    record.status, record.detail = "failed", f"{type(exc).__name__}: {exc}"
                    raise
                record.status = "skipped" if result is False else "applied"
                record.detail = "hook declined to run" if result is False else "hook completed"
                if result is not False:
                    self._undo_order.append(patch)
                self._log(record)
            self._hooks.pop(module_name, None)
        finally:
            self._running_hooks.remove(module_name)

    def _apply_for_module(self, module_name: str, module: types.ModuleType, *, trigger: str) -> None:
        groups: dict[str, list[AttrPatch]] = {}
        for patch in self._attrs.get(module_name, ()):
            groups.setdefault(patch.attr_name, []).append(patch)
        visited = set()

        def apply_target(attr):
            if attr in visited:
                return
            visited.add(attr)
            for patch in groups[attr]:
                for pid in patch.requires:
                    record = self._records.get(pid)
                    if record is not None and record.patch.attr_name in groups:
                        apply_target(record.patch.attr_name)
            self._apply_target(module, attr, groups[attr], trigger)

        for attr in groups:
            apply_target(attr)

    @staticmethod
    def _version_gate_block(patch: Patch) -> str | None:
        """Detail of the first blocked declarative version gate, or None."""
        for gate in patch.version_gates:
            blocked, detail = check_version_gate(gate)
            if blocked:
                return detail
        return None

    def _apply_target(self, module, attr: str, patches: list[AttrPatch], trigger: str) -> None:
        key = (module.__name__, attr)
        previous = self._bindings.get(key)
        active = patches[0]
        mutations = []
        try:
            gates = {p.id: self._version_gate_block(p) for p in patches}
            # A version excluded by every patch may have removed the symbol.
            # No binding to restore means there is nothing to resolve or mutate.
            if previous is None and all(detail is not None for detail in gates.values()):
                for patch in patches:
                    record = self._records[patch.id]
                    record.status, record.detail = "skipped", f"version gate: {gates[patch.id]}"
                    self._log(record)
                return
            owner_path, _, leaf = attr.rpartition(".")
            owner = require_attr(module, owner_path, patch_id=active.id) if owner_path else module
            require_attr(module, attr, patch_id=active.id)
            raw = inspect.getattr_static(owner, leaf)
            if previous is not None and previous.owner is owner and raw is previous.replacement:
                if all(self._records[p.id].status in {"applied", "skipped"} for p in patches) and not any(
                    self._records[p.id].detail.startswith("requires ")
                    and all(self.is_applied(pid) for pid in p.requires) for p in patches
                ):
                    return
                # Newly registered patch on an existing target: rebuild the
                # whole chain from the baseline, not from an already wrapped fn.
                current = previous.original
            else:
                if previous is not None and trigger != "import":
                    raise PatchConflict(f"patch target {key!r} changed outside the engine; uninstall first")
                current = raw
            baseline = current
            outcomes = []
            for active in patches:
                gate_detail = gates[active.id]
                if gate_detail is not None:
                    outcomes.append((active, "skipped", f"version gate: {gate_detail}"))
                    continue
                unavailable = [pid for pid in active.requires if not self.is_applied(pid)]
                if unavailable:
                    outcomes.append((active, "skipped", "requires applied companion(s): " + ", ".join(unavailable)))
                    continue
                # Preserve binding semantics for staticmethod/classmethod.
                value = current.__func__ if isinstance(current, (staticmethod, classmethod)) else current
                replacement = active.replace(value)
                if replacement is None:
                    outcomes.append((active, "skipped", "patch declined to run"))
                    continue
                if isinstance(current, (staticmethod, classmethod)) and not isinstance(replacement, (staticmethod, classmethod)):
                    replacement = type(current)(replacement)
                current = replacement
                outcomes.append((active, "applied", f"trigger={trigger}"))
            if any(status == "applied" for _, status, _ in outcomes):
                binding = _Binding(owner, leaf, baseline, current, leaf in vars(owner))
                mutations.append((owner, leaf, raw, leaf in vars(owner), current))
                setattr(owner, leaf, current)
                if previous is not None:
                    # Existing consumer aliases may still point to the previous
                    # generation after reload. Keep their original undo values.
                    for consumer, name, original in previous.aliases:
                        if vars(consumer).get(name) is previous.replacement:
                            mutations.append((consumer, name, previous.replacement, True, current))
                            setattr(consumer, name, current)
                            binding.aliases.append((consumer, name, baseline))
                    binding.owned = previous.owned if previous.owner is owner else binding.owned
                if not owner_path:
                    prefixes = tuple(dict.fromkeys(prefix for p in patches for prefix in p.rebind_prefixes))
                    # An explicitly registered target has its own lifecycle.
                    # Rebinding it as a consumer would invalidate its ownership
                    # (e.g. initialize/training both patch the same JIT helper).
                    binding.rebind_prefixes = prefixes
                    declared = set()
                    for record in self._records.values():
                        patch = record.patch
                        if isinstance(patch, AttrPatch):
                            declared.add((patch.module_name, patch.attr_name))
                    binding.aliases.extend(_rebind_aliases(module, leaf, raw, current, prefixes, declared))
                self._bindings[key] = binding
                if key not in self._undo_order:
                    self._undo_order.append(key)
            elif previous is not None:
                # Reload may make a conditional patch unnecessary. Consumers
                # holding the old replacement must follow the new upstream
                # baseline rather than keeping a retired wrapper indefinitely.
                for consumer, name, _ in previous.aliases:
                    if vars(consumer).get(name) is previous.replacement:
                        mutations.append((consumer, name, previous.replacement, True, baseline))
                        setattr(consumer, name, baseline)
                declared = {(p.module_name, p.attr_name) for r in self._records.values()
                            if isinstance(p := r.patch, AttrPatch)}
                _rebind_aliases(owner, leaf, previous.replacement, baseline,
                                previous.rebind_prefixes, declared)
                self._bindings.pop(key)
                self._undo_order.remove(key)
            for patch, status, detail in outcomes:
                record = self._records[patch.id]
                record.status, record.detail = status, detail
                self._log(record)
        except Exception as exc:
            for changed_owner, name, old, owned, replacement in reversed(mutations):
                if inspect.getattr_static(changed_owner, name, None) is replacement:
                    if owned:
                        setattr(changed_owner, name, old)
                    else:
                        delattr(changed_owner, name)
            record = self._records[active.id]
            record.status, record.detail = "failed", f"{type(exc).__name__}: {exc}"
            if isinstance(exc, MegatronMusaPatchError):
                raise
            raise MegatronMusaPatchError(
                f"patch {active.id!r} failed for {active.target!r} "
                f"(megatron={megatron_version() or 'unknown'}, trigger={trigger}): {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _log(record: AppliedPatch) -> None:
        level = logging.INFO if _env.debug() else logging.DEBUG
        logger.log(level, "%s: %s (%s)", record.patch.id, record.status, record.detail)

    def unapply(self) -> None:
        """Undo owned changes in reverse application order and clear watchers.

        Third-party writes are not overwritten. Attribute/hook cleanup failures are
        reported and raised after attempting the other cleanups.
        """
        if self._watcher in sys.meta_path:
            sys.meta_path.remove(self._watcher)
        self._installed = False
        self._attrs.clear()
        self._hooks.clear()
        errors = []
        retained = []
        targets = {(p.module_name, p.attr_name) for r in self._records.values()
                   if isinstance(p := r.patch, AttrPatch)}
        for item in reversed(self._undo_order):
            if isinstance(item, HookPatch):
                record = self._records[item.id]
                if item.undo is None:
                    record.detail = "hook has no undo; effects remain after uninstall"
                    retained.append(item)
                    continue
                try:
                    item.undo()
                except Exception as exc:
                    record.status, record.detail = "failed", f"undo failed: {exc}"
                    errors.append(exc)
                    retained.append(item)
            else:
                try:
                    self._bindings[item].restore(targets)
                except Exception as exc:
                    errors.append(exc)
                    retained.append(item)
                    for record in self._records.values():
                        patch = record.patch
                        if isinstance(patch, AttrPatch) and (patch.module_name, patch.attr_name) == item:
                            record.status, record.detail = "failed", f"undo failed: {exc}"
                else:
                    self._bindings.pop(item)
        self._undo_order = list(reversed(retained))
        retained_ids = {item.id for item in retained if isinstance(item, HookPatch)}
        retained_ids.update(record.patch.id for record in self._records.values()
                            if isinstance(record.patch, AttrPatch)
                            and (record.patch.module_name, record.patch.attr_name) in retained)
        for record in self._records.values():
            if record.patch.id not in retained_ids:
                record.status, record.detail = "pending", "uninstalled"
        if errors:
            raise MegatronMusaPatchError(f"patch cleanup failed: {errors[0]}; see report()") from errors[0]

    def is_applied(self, patch_id: str | None = None) -> bool:
        if patch_id is not None:
            record = self._records.get(patch_id)
            return bool(record and record.status == "applied")
        return any(record.status == "applied" for record in self._records.values())

    def report(self) -> list[dict]:
        return [record.as_dict() for record in self._records.values()]

    def pending_modules(self) -> list[str]:
        return sorted({
            name for watch in (self._attrs, self._hooks) for name, patches in watch.items()
            if any(self._records[p.id].status == "pending" for p in patches)
        })

    def __iter__(self) -> Iterator[AppliedPatch]:
        return iter(self._records.values())


def _rebind_aliases(module, attr: str, old: Any, new: Any, prefixes: tuple[str, ...], targets: set):
    """Bounded same-name alias repair; never scan attributes via __getattr__.

    Transactional: a failure in one consumer rolls back the aliases already
    rebound, so the caller's binding stays the single source of truth.
    """
    if not isinstance(old, (types.FunctionType, types.BuiltinFunctionType, type)):
        return []
    aliases = []
    for name, consumer in list(sys.modules.items()):
        if (name, attr) in targets or consumer is module or type(consumer) is not types.ModuleType:
            continue
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        if attr in vars(consumer) and vars(consumer)[attr] is old:
            try:
                setattr(consumer, attr, new)
            except Exception:
                for rollback_module, rollback_attr, rollback_old in reversed(aliases):
                    try:
                        setattr(rollback_module, rollback_attr, rollback_old)
                    except Exception:  # noqa: BLE001 - best effort rollback
                        pass
                raise
            aliases.append((consumer, attr, old))
    return aliases


ENGINE = Engine()
