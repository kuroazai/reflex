"""
Plug-in registry — the single place reflex building blocks are discovered.

Kinds:
    predicate  factory(**params) -> Predicate          (YAML: {predicate: <name>, ...})
    recipe     factory(*, name, **params) -> TriggerSpec (YAML: macro.kind)
    expect     factory(**params) -> Expectation         (YAML: expect: {kind: <name>, ...})
    sink       factory(**params) -> input sink          (reflex run --sink <name>)
    source     factory(**params) -> frame source        (reflex run --source <name>)
    context    factory(**params) -> ContextProvider     (YAML: context: {<name>: {...}})

Built-ins self-register on first use. Third-party code plugs in either with
a decorator in any imported module:

    from reflex.registry import register_predicate

    @register_predicate("hp_bar_below")
    def hp_bar_below(*, region, ratio: float = 0.3): ...

or, with zero imports from the host, via a package entry point:

    [project.entry-points."reflex.plugins"]
    mygame = "mygame.reflex_plugins:register"     # called with the Registry

Every factory's keyword parameters are introspected so the DSL can validate
YAML (unknown / missing params) and `reflex list` can document them.
"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

KINDS = ("predicate", "recipe", "expect", "sink", "source", "context")
ENTRY_POINT_GROUP = "reflex.plugins"

_EMPTY = inspect.Parameter.empty


@dataclass(frozen=True)
class ParamInfo:
    name: str
    required: bool
    default: Any
    annotation: str

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.annotation, "required": self.required}
        if not self.required:
            d["default"] = _short(self.default)
        return d


def _short(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return [_short(x) for x in v]
    return repr(v)


@dataclass(frozen=True)
class Plugin:
    kind: str
    name: str
    factory: Callable[..., Any]
    params: tuple[ParamInfo, ...]
    doc: str
    origin: str
    adapter: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None
    """Optional YAML-param rewriter: (params, builder) -> factory kwargs."""
    extra_params: tuple[str, ...] = ()
    """YAML-only param names the adapter consumes (e.g. template_paths)."""
    tags: tuple[str, ...] = field(default_factory=tuple)
    aliases: dict[str, str] = field(default_factory=dict)
    """YAML shorthand -> real param name (e.g. dialog -> dialog_present)."""

    def param(self, name: str) -> ParamInfo | None:
        for p in self.params:
            if p.name == name:
                return p
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "origin": self.origin,
            "doc": self.doc,
            "params": [p.describe() for p in self.params],
            "yaml_only_params": list(self.extra_params),
            "aliases": dict(self.aliases),
            "tags": list(self.tags),
        }


def introspect_params(factory: Callable[..., Any], *, skip: tuple[str, ...] = ()) -> tuple[ParamInfo, ...]:
    out: list[ParamInfo] = []
    for p in inspect.signature(factory).parameters.values():
        if p.name in skip or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = p.annotation
        ann_s = "" if ann is _EMPTY else (ann if isinstance(ann, str) else getattr(ann, "__name__", repr(ann)))
        out.append(ParamInfo(name=p.name, required=p.default is _EMPTY, default=None if p.default is _EMPTY else p.default, annotation=ann_s))
    return tuple(out)


class UnknownPlugin(KeyError):
    pass


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Plugin]] = {k: {} for k in KINDS}
        self._builtins_loaded = False
        self._entry_points_loaded = False
        self.load_errors: list[dict[str, str]] = []

    # ------------------------------------------------------------ registration

    def register(
        self,
        kind: str,
        name: str,
        factory: Callable[..., Any] | None = None,
        *,
        adapter: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None,
        extra_params: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        aliases: dict[str, str] | None = None,
        replace: bool = False,
        doc: str | None = None,
    ):
        """Register directly, or use as a decorator when `factory` is None."""
        if kind not in KINDS:
            raise ValueError(f"unknown plug-in kind {kind!r}; expected one of {KINDS}")

        def _do(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._items[kind] and not replace and self._items[kind][name].factory is not fn:
                raise ValueError(f"{kind} plug-in {name!r} already registered by {self._items[kind][name].origin}")
            skip = ("name",) if kind == "recipe" else ()
            text = doc if doc is not None else (inspect.getdoc(fn) or "")
            self._items[kind][name] = Plugin(
                kind=kind,
                name=name,
                factory=fn,
                params=introspect_params(fn, skip=skip),
                doc=text.strip().splitlines()[0] if text.strip() else "",
                origin=f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', '?'))}",
                adapter=adapter,
                extra_params=tuple(extra_params),
                tags=tuple(tags),
                aliases=dict(aliases or {}),
            )
            return fn

        if factory is None:
            return _do
        return _do(factory)

    def unregister(self, kind: str, name: str) -> None:
        self._items.get(kind, {}).pop(name, None)

    # ------------------------------------------------------------ lookup

    def _ensure_loaded(self) -> None:
        if not self._builtins_loaded:
            self._builtins_loaded = True
            from reflex import builtins as _b  # noqa: F401 — registers on import

            _b.register_all(self)
        if not self._entry_points_loaded:
            self._entry_points_loaded = True
            self.load_entry_points()

    def get(self, kind: str, name: str) -> Plugin:
        self._ensure_loaded()
        items = self._items.get(kind)
        if items is None:
            raise ValueError(f"unknown plug-in kind {kind!r}")
        if name not in items:
            close = difflib.get_close_matches(name, list(items), n=3)
            hint = f" — did you mean {', '.join(close)}?" if close else ""
            raise UnknownPlugin(f"no {kind} plug-in named {name!r}{hint} (known: {', '.join(sorted(items))})")
        return items[name]

    def names(self, kind: str) -> list[str]:
        self._ensure_loaded()
        return sorted(self._items[kind])

    def describe(self, kind: str | None = None) -> list[dict[str, Any]]:
        self._ensure_loaded()
        kinds = [kind] if kind else list(KINDS)
        out: list[dict[str, Any]] = []
        for k in kinds:
            for n in sorted(self._items[k]):
                out.append(self._items[k][n].describe())
        return out

    def build(self, kind: str, name: str, **kwargs: Any) -> Any:
        return self.get(kind, name).factory(**kwargs)

    # ------------------------------------------------------------ discovery

    def load_entry_points(self, group: str = ENTRY_POINT_GROUP) -> list[str]:
        """Import every `reflex.plugins` entry point and call it with this registry.

        A broken plug-in never breaks the engine: its error is recorded in
        `load_errors` (and shown by `reflex list`).
        """
        loaded: list[str] = []
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group=group)
        except Exception as exc:  # noqa: BLE001
            self.load_errors.append({"entry_point": group, "error": repr(exc)})
            return loaded
        for ep in eps:
            try:
                obj = ep.load()
                if callable(obj):
                    obj(self)
                loaded.append(ep.name)
            except Exception as exc:  # noqa: BLE001
                self.load_errors.append({"entry_point": f"{ep.name} = {ep.value}", "error": repr(exc)})
        return loaded


REGISTRY = Registry()


def register_predicate(name: str, **kw: Any):
    return REGISTRY.register("predicate", name, **kw)


def register_recipe(name: str, **kw: Any):
    return REGISTRY.register("recipe", name, **kw)


def register_expect(name: str, **kw: Any):
    return REGISTRY.register("expect", name, **kw)


def register_sink(name: str, **kw: Any):
    return REGISTRY.register("sink", name, **kw)


def register_source(name: str, **kw: Any):
    return REGISTRY.register("source", name, **kw)


def register_context(name: str, **kw: Any):
    return REGISTRY.register("context", name, **kw)
