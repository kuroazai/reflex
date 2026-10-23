"""
YAML DSL for declarative reflex macros — built entirely from the registry.

Any registered recipe / predicate / expectation is usable from YAML with no
code edits; parameters are validated against the factory's signature, so a
typo fails loudly with the macro id and field instead of silently doing
nothing.

File shape (either a single `macro:` or a `macros:` list; `guards:` and
`context:` optional):

    guards:                         # any hit suppresses ALL macros
      payment_screen:
        predicate: template_match
        region: [0, 0, 1920, 1080]
        template_paths: [templates/buy_button.png]
        threshold: 0.8

    context:                        # fill view.extras each tick
      scene: {kind: static, values: {scene: dialog}}

    macros:
      - id: dialog_advance
        kind: dialog_advance_loop   # a registered recipe
        priority: 60
        params:
          dialog: {predicate: color_at, xy: [120, 384], hsv_lo: [0,0,220], hsv_hi: [179,50,255]}
          advance_button: a
        gates:                      # all must match before the predicate runs
          - {predicate: extras_equals, key: scene, value: dialog}
        expect: {kind: gone, within_ms: 600}
        trip_after_unverified: 3

Predicate shorthands: {any_of: [...]}, {all_of: [...]}, {not: {...}}.
Actions (on_match `do:`): "a", {click: [x, y]}, {wait: 120}, {chord: [a, b]},
{down: right}, {up: right}, {key: a, hold_ms: 90}.
Host objects (trackers, providers) are injected as `$name` via bindings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from reflex import actions as _actions
from reflex.core import Predicate, Region, TriggerSpec
from reflex.expect import Expectation
from reflex.registry import REGISTRY, Plugin, Registry, UnknownPlugin


class DSLError(ValueError):
    pass


# Spec-level knobs a macro may set on top of whatever its recipe builds.
SPEC_KNOBS = (
    "priority",
    "timeout_s",
    "poll_hz",
    "max_fires",
    "cooldown_s",
    "inter_press_ms",
    "hold_ms",
    "reverify_on_fire",
    "require_consecutive_hits",
    "miss_tolerance",
    "foreground_only",
    "trip_after_unverified",
)
MACRO_KEYS = {"id", "name", "kind", "params", "gates", "expect", "description", "enabled", "tags", *SPEC_KNOBS}


@dataclass
class MacroSet:
    specs: list[TriggerSpec] = field(default_factory=list)
    guards: dict[str, Predicate] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def extend(self, other: MacroSet) -> None:
        names = {s.name for s in self.specs}
        for s in other.specs:
            if s.name in names:
                raise DSLError(f"duplicate macro id {s.name!r} (in {other.sources})")
            self.specs.append(s)
        for k, g in other.guards.items():
            if k in self.guards:
                raise DSLError(f"duplicate guard {k!r}")
            self.guards[k] = g
        self.context.update(other.context)
        self.sources.extend(other.sources)
        self.skipped.extend(other.skipped)


class Builder:
    """Turns YAML mappings into registry objects, with located errors."""

    def __init__(
        self,
        *,
        registry: Registry = REGISTRY,
        base_dir: Path | None = None,
        bindings: dict[str, Any] | None = None,
        where: str = "",
    ) -> None:
        self.registry = registry
        self.base_dir = base_dir
        self.bindings = dict(bindings or {})
        self.where = where

    # ------------------------------------------------------------ errors

    def err(self, path: str, msg: str) -> DSLError:
        loc = f"{self.where}: " if self.where else ""
        return DSLError(f"{loc}{path}: {msg}")

    # ------------------------------------------------------------ values

    def resolve_binding(self, value: Any, path: str) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            key = value[1:]
            if key not in self.bindings:
                raise self.err(path, f"unbound reference {value!r} (bindings: {sorted(self.bindings) or 'none'})")
            return self.bindings[key]
        return value

    def coerce(self, plugin: Plugin, pname: str, value: Any, path: str) -> Any:
        value = self.resolve_binding(value, path)
        info = plugin.param(pname)
        ann = info.annotation if info else ""
        if value is None:
            return None
        if "Expectation" in ann:
            if isinstance(value, (list, tuple)):
                return [self.expectation(v, f"{path}[{i}]") for i, v in enumerate(value)]
            return self.expectation(value, path)
        if "Predicate" in ann:
            if isinstance(value, (list, tuple)):
                return [self.predicate(v, f"{path}[{i}]") for i, v in enumerate(value)]
            return self.predicate(value, path)
        if "Region" in ann:
            return self.region(value, path)
        if not ann and isinstance(value, dict) and ({"predicate", "any_of", "all_of", "not"} & set(value)):
            # Untyped plug-in param that is clearly a predicate spec.
            return self.predicate(value, path)
        if (pname.endswith("_action") or "Callable" in ann) and isinstance(value, (list, tuple, str)):
            seq = [value] if isinstance(value, str) else value
            try:
                return _actions.press_keys([_actions.from_spec(a) for a in seq])
            except ValueError as exc:
                raise self.err(path, str(exc)) from None
        if isinstance(value, list) and ("uple" in ann or ann == ""):
            return tuple(tuple(v) if isinstance(v, list) else v for v in value)
        return value

    def region(self, v: Any, path: str) -> Region:
        if isinstance(v, Region):
            return v
        if isinstance(v, dict) and {"x", "y", "w", "h"} <= set(v):
            return Region(int(v["x"]), int(v["y"]), int(v["w"]), int(v["h"]))
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return Region(int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        raise self.err(path, f"expected region {{x,y,w,h}} or [x,y,w,h], got {v!r}")

    def kwargs_for(self, plugin: Plugin, params: dict[str, Any], path: str, *, skip: tuple = ()) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise self.err(path, f"params must be a mapping, got {type(params).__name__}")
        if plugin.aliases:
            params = dict(params)
            for short, real in plugin.aliases.items():
                if short in params:
                    if real in params:
                        raise self.err(path, f"both {short!r} and its alias target {real!r} given")
                    params[real] = params.pop(short)
        known = {p.name for p in plugin.params} | set(plugin.extra_params)
        unknown = sorted(set(params) - known - set(skip))
        if unknown:
            raise self.err(
                path,
                f"unknown param(s) {unknown} for {plugin.kind} {plugin.name!r}; accepts "
                f"{sorted(known - set(skip))}"
                + (f" (aliases: {plugin.aliases})" if plugin.aliases else ""),
            )
        raw = dict(params)
        if plugin.adapter is not None:
            try:
                raw = plugin.adapter(raw, self)
            except DSLError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise self.err(path, f"{plugin.name}: {exc}") from None
        out: dict[str, Any] = {}
        for k, v in raw.items():
            out[k] = self.coerce(plugin, k, v, f"{path}.{k}")
        missing = [p.name for p in plugin.params if p.required and p.name not in out and p.name not in skip]
        if missing:
            raise self.err(path, f"{plugin.kind} {plugin.name!r} missing required param(s) {missing}")
        return out

    def _plugin(self, kind: str, name: str, path: str) -> Plugin:
        try:
            return self.registry.get(kind, name)
        except UnknownPlugin as exc:
            raise self.err(path, str(exc).strip('"')) from None

    def _call(self, plugin: Plugin, kwargs: dict[str, Any], path: str) -> Any:
        try:
            return plugin.factory(**kwargs)
        except (TypeError, ValueError, KeyError) as exc:
            raise self.err(path, f"{plugin.kind} {plugin.name!r}: {exc}") from None

    # ------------------------------------------------------------ objects

    def predicate(self, spec: Any, path: str = "predicate") -> Predicate:
        spec = self.resolve_binding(spec, path)
        if callable(spec):
            return spec
        if not isinstance(spec, dict):
            raise self.err(path, f"predicate must be a mapping, got {spec!r}")
        if "any_of" in spec or "all_of" in spec:
            key = "any_of" if "any_of" in spec else "all_of"
            parts = spec[key]
            if not isinstance(parts, list) or not parts:
                raise self.err(path, f"{key} needs a non-empty list")
            plugin = self._plugin("predicate", key, path)
            return self._call(plugin, {"predicates": [self.predicate(p, f"{path}.{key}[{i}]") for i, p in enumerate(parts)]}, path)
        if "not" in spec:
            plugin = self._plugin("predicate", "not", path)
            return self._call(plugin, {"predicate": self.predicate(spec["not"], f"{path}.not")}, path)
        kind = spec.get("predicate")
        if not kind:
            raise self.err(path, f"missing 'predicate' key in {spec!r}")
        plugin = self._plugin("predicate", str(kind), path)
        params = {k: v for k, v in spec.items() if k != "predicate"}
        return self._call(plugin, self.kwargs_for(plugin, params, path), path)

    def expectation(self, spec: Any, path: str = "expect") -> Expectation:
        spec = self.resolve_binding(spec, path)
        if isinstance(spec, Expectation):
            return spec
        if isinstance(spec, str):
            spec = {"kind": spec}
        if not isinstance(spec, dict) or "kind" not in spec:
            raise self.err(path, f"expect must be a name or {{kind: ..., ...}}, got {spec!r}")
        plugin = self._plugin("expect", str(spec["kind"]), path)
        params = {k: v for k, v in spec.items() if k != "kind"}
        return self._call(plugin, self.kwargs_for(plugin, params, path), path)

    def context_provider(self, name: str, spec: Any, path: str):
        spec = self.resolve_binding(spec, path)
        if callable(spec):
            return spec
        if not isinstance(spec, dict) or "kind" not in spec:
            raise self.err(path, f"context entry must be {{kind: ..., ...}} or a $binding, got {spec!r}")
        plugin = self._plugin("context", str(spec["kind"]), path)
        params = {k: v for k, v in spec.items() if k != "kind"}
        return self._call(plugin, self.kwargs_for(plugin, params, path), path)

    def macro(self, m: Any, path: str = "macro") -> TriggerSpec | None:
        if not isinstance(m, dict):
            raise self.err(path, f"macro must be a mapping, got {type(m).__name__}")
        name = str(m.get("id") or m.get("name") or "")
        if not name:
            raise self.err(path, "macro requires 'id'")
        path = f"macro[{name}]"
        unknown = sorted(set(m) - MACRO_KEYS)
        if unknown:
            raise self.err(path, f"unknown key(s) {unknown}; allowed: {sorted(MACRO_KEYS)}")
        if m.get("enabled", True) is False:
            return None
        kind = str(m.get("kind") or "")
        if not kind:
            raise self.err(path, "macro requires 'kind' (a registered recipe)")
        plugin = self._plugin("recipe", kind, f"{path}.kind")
        params = dict(m.get("params") or {})
        kwargs = self.kwargs_for(plugin, params, f"{path}.params")
        if plugin.param("priority") is not None and "priority" in m and "priority" not in kwargs:
            kwargs["priority"] = int(m["priority"])
        kwargs["name"] = name
        spec: TriggerSpec = self._call(plugin, kwargs, path)
        if not isinstance(spec, TriggerSpec):
            raise self.err(path, f"recipe {kind!r} returned {type(spec).__name__}, not TriggerSpec")

        overrides: dict[str, Any] = {}
        for knob in SPEC_KNOBS:
            if knob in m:
                overrides[knob] = m[knob]
        if "gates" in m:
            gates = m["gates"]
            if not isinstance(gates, list):
                raise self.err(f"{path}.gates", "gates must be a list of predicates")
            overrides["gates"] = tuple(self.predicate(g, f"{path}.gates[{i}]") for i, g in enumerate(gates))
        if "expect" in m and m["expect"] is not None:
            overrides["expect"] = self.expectation(m["expect"], f"{path}.expect")
        extra = dict(spec.extra)
        extra.setdefault("kind", kind)
        if m.get("description"):
            extra["description"] = str(m["description"])
        if m.get("tags"):
            extra["tags"] = list(m["tags"])
        if self.where:
            extra["source"] = self.where
        overrides["extra"] = extra
        try:
            return spec.with_overrides(**overrides)
        except (TypeError, ValueError) as exc:
            raise self.err(path, str(exc)) from None


def build_macro_set(raw: Any, *, builder: Builder) -> MacroSet:
    out = MacroSet(sources=[builder.where] if builder.where else [])
    if raw is None:
        return out
    macros: list[Any]
    if isinstance(raw, list):
        macros = [m["macro"] if isinstance(m, dict) and "macro" in m else m for m in raw]
    elif isinstance(raw, dict):
        allowed = {"macro", "macros", "guards", "context", "bindings_required", "description"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise builder.err("file", f"unknown top-level key(s) {unknown}; allowed: {sorted(allowed)}")
        if "macros" in raw:
            macros = list(raw["macros"] or [])
        elif "macro" in raw:
            macros = [raw["macro"]]
        else:
            macros = []
        for gname, gspec in (raw.get("guards") or {}).items():
            out.guards[str(gname)] = builder.predicate(gspec, f"guards[{gname}]")
        for cname, cspec in (raw.get("context") or {}).items():
            out.context[str(cname)] = builder.context_provider(str(cname), cspec, f"context[{cname}]")
    else:
        raise builder.err("file", "unrecognised macro file shape")
    seen: set = set()
    for i, m in enumerate(macros):
        spec = builder.macro(m, f"macros[{i}]")
        if spec is None:
            out.skipped.append(str(m.get("id") if isinstance(m, dict) else i))
            continue
        if spec.name in seen:
            raise builder.err(f"macros[{i}]", f"duplicate macro id {spec.name!r}")
        seen.add(spec.name)
        out.specs.append(spec)
    return out


def load_macro_set(
    path: str | Path,
    *,
    bindings: dict[str, Any] | None = None,
    registry: Registry = REGISTRY,
) -> MacroSet:
    """Load one YAML file, or every *.yaml / *.yml in a directory."""
    p = Path(path)
    if p.is_dir():
        total = MacroSet()
        for f in sorted(list(p.glob("*.yaml")) + list(p.glob("*.yml"))):
            total.extend(load_macro_set(f, bindings=bindings, registry=registry))
        return total
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DSLError(f"{p}: invalid YAML: {exc}") from None
    builder = Builder(registry=registry, base_dir=p.parent, bindings=bindings, where=str(p))
    return build_macro_set(raw, builder=builder)


def load_macros_yaml(path: str | Path, *, bindings: dict[str, Any] | None = None) -> list[TriggerSpec]:
    """Back-compat: the specs of one file."""
    return load_macro_set(path, bindings=bindings).specs


def load_macros_dir(path: str | Path, *, bindings: dict[str, Any] | None = None) -> list[TriggerSpec]:
    """Back-compat: the specs of every *.yaml / *.yml in a directory."""
    p = Path(path)
    if not p.is_dir():
        return []
    return load_macro_set(p, bindings=bindings).specs


def build_predicate(spec: dict[str, Any], *, bindings: dict[str, Any] | None = None) -> Predicate:
    return Builder(bindings=bindings).predicate(spec)
