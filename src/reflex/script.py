"""Step-by-step macro scripts.

Watchers (reflex.dsl) react: "whenever X is on screen, do Y". Scripts run a
sequence: wait for this, click that, loop until done. A script can also carry
watchers, which keep running during every wait (close a popup, top up health)
while the steps get on with the job.

    name: daily login
    target:
      window: {title: "My Game"}
      input: postmessage          # background input; or sendinput
    vars: {runs: 0}
    watchers:
      - id: close_popup
        kind: on_match
        params: {when: {predicate: image, template: img/x_button.png}, do: [{click: [600, 80]}]}
    steps:
      - wait_for: {image: img/play.png}
        timeout_s: 20
        as: play
      - click: $play
      - repeat: {times: 5}
        steps:
          - press: space
          - wait: 400
          - set: {runs: "${runs + 1}"}
      - if: {text: {region: [0, 0, 300, 40], contains: "Reward"}}
        then: [{click: {image: img/claim.png}}]
        else: [{log: "no reward today"}]

`reflex schema` prints every step and its fields; `reflex validate` checks a
script (keys, images, expressions, predicates) without running it.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from reflex.core import FrameView, Match
from reflex.dsl import Builder, DSLError, MacroSet, build_macro_set
from reflex.expr import ExprError, evaluate, has_refs, resolve
from reflex.io import EventSink, NoopEventSink, check_frame, jsonable, sink_capabilities
from reflex.registry import REGISTRY, Registry, UnknownPlugin
from reflex.runtime import ReflexRuntime


class ScriptError(ValueError):
    """The script file is wrong. Raised at load time, with the step's location."""


class StepFailed(RuntimeError):
    def __init__(self, path: str, msg: str) -> None:
        super().__init__(f"{path}: {msg}")
        self.path = path


class _Stop(Exception):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


# --------------------------------------------------------------------------- step table

COMMON = {"name", "comment", "when"}
"""Every step accepts: name (label in logs), comment (ignored), when (a condition; skip the step if false)."""

VERBS: dict[str, dict[str, Any]] = {
    "press": {"fields": {"hold_ms", "times", "interval_ms"},
              "doc": "Press a key, or a list of keys in order. press: space | press: [ctrl, s]"},
    "key_down": {"fields": set(), "doc": "Hold a key down until key_up (or the run ends)."},
    "key_up": {"fields": set(), "doc": "Release a held key."},
    "chord": {"fields": {"hold_ms"}, "doc": "Press keys together: chord: [ctrl, shift, s]"},
    "type": {"fields": {"interval_ms"}, "doc": "Type text (any characters). type: \"gg ${name}\""},
    "click": {"fields": {"button", "offset", "times", "interval_ms", "optional"},
              "doc": "Click a point: [x, y], a stored match ($var), or something to find now ({image: f.png})."},
    "double_click": {"fields": {"button", "offset", "optional"}, "doc": "Two clicks in quick succession."},
    "right_click": {"fields": {"offset", "optional"}, "doc": "Right-click a point (same targets as click)."},
    "move": {"fields": {"offset", "optional"}, "doc": "Move the pointer to a point."},
    "drag": {"fields": {"to", "button", "steps", "hold_ms", "offset"},
             "doc": "Press at a point, move to `to`, release. drag: [x1, y1] with to: [x2, y2]."},
    "scroll": {"fields": {"at", "horizontal"}, "doc": "Scroll the wheel by N notches (negative = down) at `at`."},
    "wait": {"fields": {"jitter_ms"}, "doc": "Pause for N milliseconds. Watchers keep running."},
    "wait_for": {"fields": {"timeout_s", "as", "poll_ms", "on_timeout"},
                 "doc": "Wait until a condition holds; `as` stores the match. on_timeout: fail | continue | [steps]"},
    "wait_gone": {"fields": {"timeout_s", "poll_ms", "on_timeout"}, "doc": "Wait until a condition stops holding."},
    "find": {"fields": {"as"}, "doc": "Check a condition once, no waiting; `as` stores the match or null."},
    "read_text": {"fields": {"as", "number"},
                  "doc": "OCR a region [x, y, w, h] into a variable; number: true stores the first number."},
    "if": {"fields": {"then", "else"}, "doc": "Run `then` steps if a condition holds, else `else` steps."},
    "repeat": {"fields": {"steps"},
               "doc": "Loop `steps`: repeat: {times: N} | {while: cond} | {until: cond} | {forever: true}, "
                      "optional as: i (loop counter) and max: N (safety cap)."},
    "set": {"fields": set(), "doc": "Set variables: set: {runs: \"${runs + 1}\"}"},
    "log": {"fields": set(), "doc": "Write a message to the run log. log: \"run ${runs} done\""},
    "call": {"fields": {"with"}, "doc": "Run a function from `functions:`; `with` sets variables first."},
    "break": {"fields": set(), "doc": "Leave the innermost repeat. break: true"},
    "continue": {"fields": set(), "doc": "Skip to the next repeat iteration. continue: true"},
    "stop": {"fields": set(), "doc": "End the script successfully. stop: \"reason\""},
    "fail": {"fields": set(), "doc": "End the script as failed. fail: \"reason\""},
    "screenshot": {"fields": {"region"}, "doc": "Save the current frame (or a region) as PNG: screenshot: shots/a.png"},
    "focus": {"fields": set(), "doc": "Bring the target window to the front (needed by sendinput). focus: true"},
    "watcher": {"fields": set(), "doc": "Turn a watcher on or off: watcher: {enable: id} | {disable: id}"},
    "stick": {"fields": {"x", "y", "hold_ms"},
              "doc": "Gamepad stick (target gamepad: true): stick: left with x/y in [-1, 1]; recentres after hold_ms."},
}

TOP_KEYS = {"name", "description", "target", "vars", "timeout_s", "poll_ms", "hotkeys", "humanize", "watchers",
            "guards", "functions", "steps", "hold_ms", "input_delay_ms", "focus_wait_s"}

# Shorthand predicate forms: {image: "a.png"} == {predicate: image, template: "a.png"}.
SCALAR_PARAM = {"image": "template", "text": "contains", "color_search": "rgb", "stable": "for_ms",
                "template_match": "template_paths"}


@dataclass
class Step:
    verb: str
    value: Any
    opts: dict[str, Any]
    path: str
    children: dict[str, list[Step]] = field(default_factory=dict)
    cond: Any = None


@dataclass
class Script:
    name: str
    steps: list[Step]
    functions: dict[str, list[Step]] = field(default_factory=dict)
    vars: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    watchers: MacroSet = field(default_factory=MacroSet)
    timeout_s: float = 3600.0
    poll_ms: int = 50
    hold_ms: int = 50
    input_delay_ms: int = 30
    focus_wait_s: float = 30.0
    hotkeys: dict[str, str] = field(default_factory=dict)
    humanize: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    base_dir: Path | None = None
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- conditions


@dataclass
class Cond:
    kind: str  # pred | expr | not | any | all
    spec: Any = None
    parts: list[Cond] = field(default_factory=list)


class _Compiler:
    def __init__(self, *, base_dir: Path | None, registry: Registry, where: str, check_keys: bool) -> None:
        self.base_dir = base_dir
        self.registry = registry
        self.where = where
        self.check_keys = check_keys
        self.function_names: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    def err(self, path: str, msg: str) -> ScriptError:
        return ScriptError(f"{self.where}: {path}: {msg}" if self.where else f"{path}: {msg}")

    # ------------------------------------------------------------ predicates

    def expand_predicate(self, spec: Any, path: str) -> Any:
        if not isinstance(spec, dict):
            raise self.err(path, f"expected a condition mapping, got {spec!r}")
        if "predicate" in spec:
            return spec
        for comb in ("any_of", "all_of"):
            if comb in spec:
                return {comb: [self.expand_predicate(p, f"{path}.{comb}[{i}]") for i, p in enumerate(spec[comb])]}
        if "not" in spec:
            return {"not": self.expand_predicate(spec["not"], f"{path}.not")}
        if len(spec) == 1:
            (key, val), = spec.items()
            try:
                self.registry.get("predicate", key)
            except UnknownPlugin as exc:
                raise self.err(path, str(exc).strip('"')) from None
            if isinstance(val, dict):
                return {"predicate": key, **val}
            if key in SCALAR_PARAM:
                return {"predicate": key, SCALAR_PARAM[key]: val}
            raise self.err(path, f"{key}: give its parameters as a mapping")
        raise self.err(path, f"cannot read {spec!r} as a condition; use {{predicate: name, ...}} or {{name: {{...}}}}")

    def check_predicate(self, spec: Any, path: str) -> None:
        if has_refs(spec):
            return  # built at run time once variables exist
        try:
            Builder(registry=self.registry, base_dir=self.base_dir, where="").predicate(spec, path)
        except DSLError as exc:
            raise self.err(path, str(exc)) from None
        except FileNotFoundError as exc:
            raise self.err(path, str(exc)) from None

    def condition(self, spec: Any, path: str) -> Cond:
        if isinstance(spec, bool):
            return Cond("expr", "True" if spec else "False")
        if isinstance(spec, str):
            src = spec[2:-1] if spec.startswith("${") and spec.endswith("}") else spec
            self.check_expr(src, path)
            return Cond("expr", src)
        if not isinstance(spec, dict):
            raise self.err(path, f"a condition is an expression string or a mapping, got {spec!r}")
        if "expr" in spec and len(spec) == 1:
            return self.condition(str(spec["expr"]), path)
        if "not" in spec and len(spec) == 1:
            return Cond("not", parts=[self.condition(spec["not"], f"{path}.not")])
        for comb, kind in (("any_of", "any"), ("all_of", "all")):
            if comb in spec and len(spec) == 1:
                items = spec[comb]
                if not isinstance(items, list) or not items:
                    raise self.err(path, f"{comb} needs a non-empty list")
                return Cond(kind, parts=[self.condition(c, f"{path}.{comb}[{i}]") for i, c in enumerate(items)])
        pspec = self.expand_predicate(spec, path)
        self.check_predicate(pspec, path)
        return Cond("pred", pspec)

    def check_expr(self, src: str, path: str) -> None:
        import ast

        try:
            ast.parse(src.strip(), mode="eval")
        except SyntaxError as exc:
            raise self.err(path, f"bad expression {src!r}: {exc.msg}") from None

    def check_exprs_in(self, value: Any, path: str) -> None:
        if isinstance(value, str) and "${" in value:
            import re

            for m in re.finditer(r"\$\{([^}]*)\}", value):
                self.check_expr(m.group(1), path)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                self.check_exprs_in(v, f"{path}[{i}]")
        elif isinstance(value, dict):
            for k, v in value.items():
                self.check_exprs_in(v, f"{path}.{k}")

    def check_key(self, key: Any, path: str) -> None:
        if not self.check_keys or has_refs(key):
            return
        from reflex.backends.gamepad import is_pad

        if is_pad(str(key)):
            return
        from reflex.backends.windows.keys import UnknownKey, vk_for

        try:
            vk_for(str(key))
        except UnknownKey as exc:
            raise self.err(path, str(exc)) from None

    def target(self, value: Any, path: str) -> None:
        """A click/move target: [x, y], $var, "center", or a condition to find."""
        if isinstance(value, str):
            if value in ("center", "pointer") or has_refs(value):
                return
            raise self.err(path, f"target must be [x, y], $var, 'center' or a condition, got {value!r}")
        if isinstance(value, list):
            if len(value) != 2:
                raise self.err(path, f"a point is [x, y], got {value!r}")
            self.check_exprs_in(value, path)
            return
        if isinstance(value, dict):
            self.condition(value, path)
            return
        raise self.err(path, f"cannot click {value!r}")

    # ------------------------------------------------------------ steps

    def steps(self, raw: Any, path: str) -> list[Step]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise self.err(path, "steps must be a list")
        return [self.step(s, f"{path}[{i}]") for i, s in enumerate(raw)]

    def step(self, raw: Any, path: str) -> Step:
        if isinstance(raw, str) and raw in ("break", "continue"):
            raw = {raw: True}
        if not isinstance(raw, dict):
            raise self.err(path, f"a step is a mapping like {{press: a}}, got {raw!r}")
        verbs = [k for k in raw if k in VERBS]
        if len(verbs) != 1:
            unknown = sorted(set(raw) - set(VERBS) - COMMON)
            hint = ""
            if unknown:
                import difflib

                close = difflib.get_close_matches(unknown[0], list(VERBS), n=1)
                hint = f" (did you mean {close[0]!r}?)" if close else ""
            raise self.err(path, f"a step needs exactly one action, found {verbs or unknown}{hint}; "
                                 f"actions: {', '.join(VERBS)}")
        verb = verbs[0]
        label = raw.get("name")
        path = f"{path}<{label}>" if label else f"{path}.{verb}"
        allowed = VERBS[verb]["fields"] | COMMON | {verb}
        if verb == "repeat":
            allowed = allowed | {"as", "max"}
        extra = sorted(set(raw) - allowed)
        if extra:
            raise self.err(path, f"{verb} does not take {extra}; it takes {sorted(allowed - COMMON - {verb})}")
        value = raw[verb]
        opts = {k: v for k, v in raw.items() if k not in (verb, "comment")}
        st = Step(verb=verb, value=value, opts=opts, path=path)
        if "when" in raw:
            st.opts["when"] = self.condition(raw["when"], f"{path}.when")
        self._compile_verb(st, raw)
        # Expressions anywhere else in the step (wait: "${...}", timeout_s, hold_ms, ...).
        nested = {"then", "else", "steps", "on_timeout", "when"}
        self.check_exprs_in({k: v for k, v in raw.items() if k not in nested}, path)
        return st

    def _compile_verb(self, st: Step, raw: dict) -> None:
        v, path = st.value, st.path
        verb = st.verb
        if verb == "press":
            for i, k in enumerate(v if isinstance(v, list) else [v]):
                self.check_key(k, f"{path}[{i}]")
        elif verb in ("key_down", "key_up"):
            self.check_key(v, path)
        elif verb == "chord":
            if not isinstance(v, list) or len(v) < 2:
                raise self.err(path, "chord needs a list of two or more keys")
            for i, k in enumerate(v):
                self.check_key(k, f"{path}[{i}]")
        elif verb in ("click", "double_click", "right_click", "move"):
            self.target(v, path)
            if st.opts.get("button") not in (None, "left", "right", "middle"):
                raise self.err(path, "button must be left, right or middle")
            if isinstance(v, dict) and not has_refs(v):
                st.cond = self.condition(v, path)
        elif verb == "drag":
            self.target(v, path)
            if "to" not in raw:
                raise self.err(path, "drag needs to: [x, y]")
            self.target(raw["to"], f"{path}.to")
        elif verb == "scroll":
            if "at" not in raw:
                raise self.err(path, "scroll needs at: [x, y] (or 'center')")
            self.target(raw["at"], f"{path}.at")
        elif verb in ("wait_for", "wait_gone", "find"):
            st.cond = self.condition(v, path)
            if verb == "find" and "as" not in raw:
                raise self.err(path, "find needs as: <variable>")
            ot = raw.get("on_timeout", "fail")
            if isinstance(ot, list):
                st.children["on_timeout"] = self.steps(ot, f"{path}.on_timeout")
            elif ot not in ("fail", "continue"):
                raise self.err(path, "on_timeout is fail, continue, or a list of steps")
        elif verb == "read_text":
            if not (isinstance(v, list) and len(v) == 4) and not has_refs(v):
                raise self.err(path, "read_text needs a region [x, y, w, h]")
            if "as" not in raw:
                raise self.err(path, "read_text needs as: <variable>")
        elif verb == "if":
            st.cond = self.condition(v, path)
            st.children["then"] = self.steps(raw.get("then"), f"{path}.then")
            st.children["else"] = self.steps(raw.get("else"), f"{path}.else")
            if not st.children["then"] and not st.children["else"]:
                raise self.err(path, "if needs then: and/or else: steps")
        elif verb == "repeat":
            if not isinstance(v, dict):
                raise self.err(path, "repeat: {times: N} | {while: cond} | {until: cond} | {forever: true}")
            modes = [k for k in ("times", "while", "until", "forever") if k in v]
            if len(modes) != 1:
                raise self.err(path, "repeat needs exactly one of times, while, until, forever")
            mode = modes[0]
            if mode in ("while", "until"):
                st.cond = self.condition(v[mode], f"{path}.{mode}")
            elif mode == "times":
                self.check_exprs_in(v["times"], f"{path}.times")
            for k in ("as", "max"):
                if k in v:
                    st.opts[k] = v[k]
            st.children["steps"] = self.steps(raw.get("steps"), f"{path}.steps")
            if not st.children["steps"]:
                raise self.err(path, "repeat needs steps:")
        elif verb == "set":
            if not isinstance(v, dict) or not v:
                raise self.err(path, "set needs a mapping of variable: value")
            self.check_exprs_in(v, path)
        elif verb == "call":
            self.calls.append((str(v), path))
        elif verb == "watcher":
            if not (isinstance(v, dict) and len(v) == 1 and next(iter(v)) in ("enable", "disable")):
                raise self.err(path, "watcher: {enable: id} or {disable: id}")
        elif verb in ("log", "stop", "fail", "type"):
            self.check_exprs_in(v, path)
        elif verb == "stick":
            if v not in ("left", "right") and not has_refs(v):
                raise self.err(path, "stick is left or right")
            if "x" not in raw and "y" not in raw:
                raise self.err(path, "stick needs x and/or y in [-1, 1]")


def _check_keys_default() -> bool:
    import sys

    return sys.platform == "win32"


def build_script(
    raw: Any,
    *,
    base_dir: Path | None = None,
    where: str = "",
    registry: Registry = REGISTRY,
    check_keys: bool | None = None,
) -> Script:
    comp = _Compiler(base_dir=base_dir, registry=registry, where=where,
                     check_keys=_check_keys_default() if check_keys is None else check_keys)
    if not isinstance(raw, dict):
        raise comp.err("file", "a script is a mapping with at least steps:")
    unknown = sorted(set(raw) - TOP_KEYS)
    if unknown:
        raise comp.err("file", f"unknown top-level key(s) {unknown}; allowed: {sorted(TOP_KEYS)}")
    if "steps" not in raw:
        raise comp.err("file", "a script needs steps:")
    functions_raw = raw.get("functions") or {}
    if not isinstance(functions_raw, dict):
        raise comp.err("functions", "functions is a mapping of name: [steps]")
    comp.function_names = set(functions_raw)
    functions = {str(n): comp.steps(s, f"functions.{n}") for n, s in functions_raw.items()}
    steps = comp.steps(raw["steps"], "steps")
    for name, path in comp.calls:
        if name not in comp.function_names:
            raise comp.err(path, f"no function named {name!r} (defined: {sorted(comp.function_names) or 'none'})")

    watchers = MacroSet()
    if raw.get("watchers") or raw.get("guards"):
        builder = Builder(registry=registry, base_dir=base_dir, where=where)
        try:
            watchers = build_macro_set({"macros": raw.get("watchers") or [], "guards": raw.get("guards") or {}},
                                       builder=builder)
        except DSLError as exc:
            raise ScriptError(str(exc)) from None

    target = raw.get("target") or {}
    if not isinstance(target, dict):
        raise comp.err("target", "target is a mapping, e.g. {window: {title: ...}, input: postmessage}")
    t_unknown = sorted(set(target) - {"window", "screen", "capture", "input", "all_windows", "require_focus",
                                      "gamepad"})
    if t_unknown:
        raise comp.err("target", f"unknown key(s) {t_unknown}")
    if target.get("input") not in (None, "postmessage", "sendinput"):
        raise comp.err("target.input", "input is postmessage (background) or sendinput (real input)")
    hotkeys = dict(raw.get("hotkeys") or {})
    for k in hotkeys:
        if k not in ("stop", "pause"):
            raise comp.err("hotkeys", f"hotkeys are stop and pause, not {k!r}")
    timeout_s = float(raw.get("timeout_s", 3600))
    if timeout_s <= 0:
        raise comp.err("timeout_s", "timeout_s must be > 0; scripts never run unbounded")
    vars_ = raw.get("vars") or {}
    if not isinstance(vars_, dict):
        raise comp.err("vars", "vars is a mapping")
    return Script(
        name=str(raw.get("name") or (Path(where).stem if where else "script")),
        description=str(raw.get("description") or ""),
        steps=steps,
        functions=functions,
        vars=dict(vars_),
        target=target,
        watchers=watchers,
        timeout_s=timeout_s,
        poll_ms=int(raw.get("poll_ms", 50)),
        hold_ms=int(raw.get("hold_ms", 50)),
        input_delay_ms=int(raw.get("input_delay_ms", 30)),
        focus_wait_s=float(raw.get("focus_wait_s", 30.0)),
        hotkeys=hotkeys,
        humanize=dict(raw.get("humanize") or {}),
        base_dir=base_dir,
        source_path=where,
        raw=raw,
    )


def load_script(path: str | Path, *, registry: Registry = REGISTRY, check_keys: bool | None = None) -> Script:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScriptError(f"{p}: invalid YAML: {exc}") from None
    return build_script(raw, base_dir=p.parent, where=str(p), registry=registry, check_keys=check_keys)


def is_script(raw: Any) -> bool:
    return isinstance(raw, dict) and "steps" in raw


# --------------------------------------------------------------------------- running


class Control:
    """Stop and pause shared by every runner in one session (and the hotkeys)."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.stop_reason = ""

    def stop(self, reason: str = "stopped") -> None:
        self.stop_reason = self.stop_reason or reason
        self.stop_event.set()

    def toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
        else:
            self.pause_event.set()


@dataclass
class ScriptResult:
    status: str  # completed | stopped | failed | timed_out
    instance: str
    elapsed_s: float
    steps_run: int
    vars: dict[str, Any]
    error: str | None = None
    failed_at: str | None = None
    log: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("completed", "stopped")

    def to_dict(self) -> dict[str, Any]:
        return jsonable({
            "status": self.status, "instance": self.instance, "elapsed_s": self.elapsed_s,
            "steps_run": self.steps_run, "vars": self.vars, "error": self.error, "failed_at": self.failed_at,
            "log": self.log[-50:],
        })


def match_to_var(m: Match) -> dict[str, Any]:
    meta = jsonable(dict(m.meta))
    c = meta.get("centroid_xy") or meta.get("xy")
    if isinstance(c, (list, tuple)) and len(c) == 2:
        meta["x"], meta["y"] = int(c[0]), int(c[1])
    meta["found"] = True
    return meta


class ScriptRunner:
    """Runs one script against one target (frame source + input sink)."""

    def __init__(
        self,
        script: Script,
        *,
        source: Any,
        sink: Any,
        events: EventSink | None = None,
        control: Control | None = None,
        instance: str = "main",
        extra_vars: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        ts_ns: Callable[[], int] = time.time_ns,
        registry: Registry = REGISTRY,
        focus: Callable[[], bool] | None = None,
        seed: int | None = None,
    ) -> None:
        self.script = script
        self.source = source
        self.sink = sink
        self.events = events or NoopEventSink()
        self.control = control or Control()
        self.instance = instance
        self.clock = clock
        self.sleep = sleep
        self.ts_ns = ts_ns
        self.registry = registry
        self._focus = focus
        self.vars: dict[str, Any] = {**script.vars, **(extra_vars or {})}
        self.caps = sink_capabilities(sink)
        self.steps_run = 0
        self.log: list[str] = []
        self.rng = random.Random(seed)
        self._deadline = 0.0
        self._last_view: FrameView | None = None
        self._rt: ReflexRuntime | None = None
        if script.watchers.specs or script.watchers.guards:
            target_ok = getattr(sink, "target_ok", None)
            self._rt = ReflexRuntime(source=source, sink=sink, events=self.events, clock=clock, sleep=sleep,
                                     ts_ns=ts_ns, guards=script.watchers.guards,
                                     context_providers=script.watchers.context,
                                     foreground_ok=target_ok if callable(target_ok) else None)
            for spec in script.watchers.specs:
                # Watchers live as long as the script; their own timeout would retire them early.
                self._rt.register(spec.with_overrides(timeout_s=max(spec.timeout_s, script.timeout_s + 60)))

    # ------------------------------------------------------------ plumbing

    def _emit(self, kind: str, **payload: Any) -> None:
        self.events.emit(kind, {"script": self.script.name, "instance": self.instance, **payload})

    def _say(self, msg: str) -> None:
        line = f"[{self.instance}] {msg}"
        self.log.append(line)
        self._emit("script.log", message=msg)

    def _check(self, path: str = "") -> None:
        if self.control.stop_event.is_set():
            raise _Stop("stopped", self.control.stop_reason or "stopped")
        if self.clock() >= self._deadline:
            raise _Stop("timed_out", f"script timeout of {self.script.timeout_s:g}s reached at {path or 'start'}")
        if self.control.pause_event.is_set():
            self._emit("script.paused", at=path)
            if hasattr(self.sink, "release_all"):
                self.sink.release_all()
            while self.control.pause_event.is_set():
                if self.control.stop_event.is_set():
                    raise _Stop("stopped", self.control.stop_reason or "stopped")
                self.sleep(0.05)
            self._emit("script.resumed", at=path)

    def _frame(self) -> FrameView:
        frame = check_frame(self.source.grab())
        view = FrameView(frame_bgr=frame, ts_ns=self.ts_ns())
        if self._rt is not None:
            self._rt.tick(frame_bgr=frame)
        self._last_view = view
        return view

    def _pause_ms(self, ms: float, path: str) -> None:
        """Sleep while still polling watchers and the stop/pause controls."""
        end = self.clock() + max(0.0, ms) / 1000.0
        step = max(0.005, self.script.poll_ms / 1000.0)
        while True:
            self._check(path)
            left = end - self.clock()
            if left <= 0:
                return
            if self._rt is not None:
                self._frame()
            self.sleep(min(step, left))

    def _r(self, value: Any, path: str) -> Any:
        try:
            return resolve(value, self.vars)
        except ExprError as exc:
            raise StepFailed(path, str(exc)) from None

    # ------------------------------------------------------------ conditions

    def _build(self, spec: Any, path: str):
        spec = self._r(spec, path) if has_refs(spec) else spec
        try:
            return Builder(registry=self.registry, base_dir=self.script.base_dir).predicate(spec, path)
        except DSLError as exc:
            raise StepFailed(path, str(exc)) from None

    def _prepare(self, cond: Cond, path: str) -> Any:
        """Build predicates once per use, so stateful ones (stable) start fresh."""
        if cond.kind == "pred":
            return ("pred", self._build(cond.spec, path))
        if cond.kind == "expr":
            return ("expr", cond.spec)
        return (cond.kind, [self._prepare(p, path) for p in cond.parts])

    def _eval(self, prepared: Any, view_fn: Callable[[], FrameView], path: str) -> Match | None:
        kind, body = prepared
        if kind == "expr":
            try:
                val = evaluate(body, self.vars)
            except ExprError as exc:
                raise StepFailed(path, str(exc)) from None
            return Match(meta={"predicate": "expr", "value": jsonable(val)}) if val else None
        if kind == "pred":
            return body(view_fn())
        if kind == "not":
            return None if self._eval(body[0], view_fn, path) is not None else Match(meta={"predicate": "not"})
        if kind == "any":
            for p in body:
                m = self._eval(p, view_fn, path)
                if m is not None:
                    return m
            return None
        last = None
        for p in body:
            last = self._eval(p, view_fn, path)
            if last is None:
                return None
        return last

    def _lazy_view(self) -> Callable[[], FrameView]:
        """One frame per evaluation, grabbed only if a predicate needs it."""
        cache: list[FrameView] = []

        def view() -> FrameView:
            if not cache:
                cache.append(self._frame())
            return cache[0]

        return view

    def _test(self, cond: Cond, path: str) -> Match | None:
        return self._eval(self._prepare(cond, path), self._lazy_view(), path)

    def _wait(self, cond: Cond, *, gone: bool, timeout_s: float, poll_ms: int, path: str) -> Match | None | bool:
        prepared = self._prepare(cond, path)
        end = self.clock() + timeout_s
        poll = max(1, poll_ms) / 1000.0
        while True:
            self._check(path)
            m = self._eval(prepared, self._lazy_view(), path)
            if gone and m is None:
                return True
            if not gone and m is not None:
                return m
            if self.clock() >= end:
                return False
            self.sleep(min(poll, max(0.0, end - self.clock())))

    # ------------------------------------------------------------ targets and input

    def _point(self, value: Any, path: str, *, optional: bool = False, cond: Cond | None = None,
               offset: Any = None) -> tuple[int, int] | None:
        if cond is not None or (isinstance(value, dict) and "x" not in value and "centroid_xy" not in value
                                and not value.get("found")):
            c = cond if cond is not None else _Compiler(base_dir=self.script.base_dir, registry=self.registry,
                                                        where="", check_keys=False).condition(self._r(value, path), path)
            m = self._test(c, path)
            if m is None:
                if optional:
                    return None
                raise StepFailed(path, "click target not found on screen")
            value = match_to_var(m)
        else:
            value = self._r(value, path)
        if value == "center":
            v = self._last_view or self._frame()
            x, y = v.width // 2, v.height // 2
        elif isinstance(value, dict):
            if not value.get("found", True) and optional:
                return None
            xy = value.get("centroid_xy") or value.get("xy")
            if xy is None and "x" in value and "y" in value:
                xy = (value["x"], value["y"])
            if xy is None:
                raise StepFailed(path, f"{value!r} has no position")
            x, y = int(xy[0]), int(xy[1])
        elif value is None:
            if optional:
                return None
            raise StepFailed(path, "target is null (a find that found nothing?)")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            x, y = int(value[0]), int(value[1])
        else:
            raise StepFailed(path, f"cannot click {value!r}")
        if offset is not None:
            dx, dy = self._r(offset, path)
            x, y = x + int(dx), y + int(dy)
        jitter = int(self.script.humanize.get("click_jitter_px", 0))
        if jitter:
            x += self.rng.randint(-jitter, jitter)
            y += self.rng.randint(-jitter, jitter)
        return (x, y)

    def _need(self, cap: str, path: str) -> None:
        if cap not in self.caps:
            raise StepFailed(path, f"the input sink {type(self.sink).__name__} cannot {cap}")

    def _ensure_target(self, path: str) -> None:
        target_ok = getattr(self.sink, "target_ok", None)
        if not callable(target_ok) or target_ok():
            return
        self._emit("script.waiting_for_focus", at=path)
        self._say(f"{path}: target window lost focus; waiting up to {self.script.focus_wait_s:g}s")
        end = self.clock() + self.script.focus_wait_s
        while not target_ok():
            self._check(path)
            if self.clock() >= end:
                raise StepFailed(path, "target window does not have focus, so input was not sent "
                                       "(use input: postmessage for background input, or a focus: step)")
            self.sleep(0.1)

    def _input(self, fn: Callable[[], bool], path: str, what: str) -> None:
        self._ensure_target(path)
        ok = fn()
        if not ok:
            raise StepFailed(path, f"{what} was refused by {type(self.sink).__name__}")
        delay = self.script.input_delay_ms + self._jitter()
        if delay > 0:
            self.sleep(delay / 1000.0)

    def _jitter(self) -> int:
        j = int(self.script.humanize.get("jitter_ms", 0))
        return self.rng.randint(0, j) if j > 0 else 0

    # ------------------------------------------------------------ execution

    def run(self) -> ScriptResult:
        started = self.clock()
        self._deadline = started + self.script.timeout_s
        self._emit("script.start", steps=len(self.script.steps), target=self.script.target)
        status, error, failed_at = "completed", None, None
        try:
            self._block(self.script.steps)
        except _Stop as stop:
            status = stop.status if stop.status != "finished" else "completed"
            if stop.status == "finished":
                self._say(f"stopped: {stop.reason}")
            else:
                error = stop.reason
        except StepFailed as exc:
            status, error, failed_at = "failed", str(exc), exc.path
        except (_Break, _Continue):
            status, error = "failed", "break/continue outside a repeat"
        except Exception as exc:  # noqa: BLE001 — report, never crash the host
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            if hasattr(self.sink, "release_all"):
                try:
                    self.sink.release_all()
                except Exception:  # noqa: BLE001
                    pass
        elapsed = round(self.clock() - started, 3)
        result = ScriptResult(status=status, instance=self.instance, elapsed_s=elapsed, steps_run=self.steps_run,
                              vars=jsonable(self.vars), error=error, failed_at=failed_at, log=list(self.log))
        self._emit("script.end", status=status, error=error, failed_at=failed_at, elapsed_s=elapsed,
                   steps_run=self.steps_run)
        return result

    def _block(self, steps: list[Step]) -> None:
        for st in steps:
            self._step(st)

    def _step(self, st: Step) -> None:
        self._check(st.path)
        when = st.opts.get("when")
        if isinstance(when, Cond) and self._test(when, f"{st.path}.when") is None:
            return
        self.steps_run += 1
        self._emit("script.step", at=st.path, verb=st.verb)
        getattr(self, f"_do_{st.verb}")(st)

    # keys -----------------------------------------------------------------

    def _do_press(self, st: Step) -> None:
        keys = self._r(st.value, st.path)
        keys = keys if isinstance(keys, list) else [keys]
        hold = int(self._r(st.opts.get("hold_ms", self.script.hold_ms), st.path))
        times = int(self._r(st.opts.get("times", 1), st.path))
        gap = int(self._r(st.opts.get("interval_ms", 0), st.path))
        self._need("key", st.path)
        for t in range(times):
            for k in keys:
                self._input(lambda k=k: self.sink.press(str(k), hold_ms=hold + self._jitter()), st.path, f"press {k}")
            if gap and t < times - 1:
                self._pause_ms(gap + self._jitter(), st.path)

    def _do_key_down(self, st: Step) -> None:
        self._need("key_down", st.path)
        k = str(self._r(st.value, st.path))
        self._input(lambda: self.sink.key_down(k), st.path, f"key_down {k}")

    def _do_key_up(self, st: Step) -> None:
        self._need("key_up", st.path)
        k = str(self._r(st.value, st.path))
        self._input(lambda: self.sink.key_up(k), st.path, f"key_up {k}")

    def _do_chord(self, st: Step) -> None:
        self._need("chord", st.path)
        keys = [str(k) for k in self._r(st.value, st.path)]
        hold = int(self._r(st.opts.get("hold_ms", self.script.hold_ms), st.path))

        def chord() -> bool:
            ok = all([bool(self.sink.key_down(k)) for k in keys])
            self.sleep(hold / 1000.0)
            return all([bool(self.sink.key_up(k)) for k in reversed(keys)]) and ok

        self._input(chord, st.path, f"chord {'+'.join(keys)}")

    def _do_type(self, st: Step) -> None:
        self._need("type", st.path)
        text = str(self._r(st.value, st.path))
        gap = int(self._r(st.opts.get("interval_ms", 0), st.path))
        self._input(lambda: self.sink.type_text(text, interval_ms=gap), st.path, "type")

    # mouse ------------------------------------------------------------------

    def _click(self, st: Step, button: str, count: int) -> None:
        self._need("click", st.path)
        pt = self._point(st.value, st.path, optional=bool(st.opts.get("optional")), cond=st.cond,
                         offset=st.opts.get("offset"))
        if pt is None:
            self._say(f"{st.path}: target not found, skipped (optional)")
            return
        gap = int(self._r(st.opts.get("interval_ms", 60), st.path))
        for i in range(count):
            self._input(lambda: self.sink.click(pt[0], pt[1], button=button), st.path, f"click {pt}")
            if i < count - 1:
                self.sleep(gap / 1000.0)

    def _do_click(self, st: Step) -> None:
        self._click(st, str(st.opts.get("button", "left")), int(self._r(st.opts.get("times", 1), st.path)))

    def _do_double_click(self, st: Step) -> None:
        st.opts.setdefault("interval_ms", 60)
        self._click(st, str(st.opts.get("button", "left")), 2)

    def _do_right_click(self, st: Step) -> None:
        self._click(st, "right", 1)

    def _do_move(self, st: Step) -> None:
        self._need("move", st.path)
        pt = self._point(st.value, st.path, optional=bool(st.opts.get("optional")), cond=st.cond,
                         offset=st.opts.get("offset"))
        if pt is not None:
            self._input(lambda: self.sink.move(pt[0], pt[1]), st.path, f"move {pt}")

    def _do_drag(self, st: Step) -> None:
        for cap in ("mouse_down", "move", "mouse_up"):
            self._need(cap, st.path)
        a = self._point(st.value, st.path, offset=st.opts.get("offset"))
        b = self._point(st.opts["to"], f"{st.path}.to")
        assert a is not None and b is not None
        button = str(st.opts.get("button", "left"))
        n = max(1, int(self._r(st.opts.get("steps", 12), st.path)))
        hold = int(self._r(st.opts.get("hold_ms", 300), st.path))
        self._input(lambda: self.sink.mouse_down(a[0], a[1], button=button), st.path, "drag press")
        for i in range(1, n + 1):
            x = round(a[0] + (b[0] - a[0]) * i / n)
            y = round(a[1] + (b[1] - a[1]) * i / n)
            self.sink.move(x, y)
            self.sleep(hold / 1000.0 / n)
        self._input(lambda: self.sink.mouse_up(b[0], b[1], button=button), st.path, "drag release")

    def _do_scroll(self, st: Step) -> None:
        self._need("scroll", st.path)
        clicks = int(self._r(st.value, st.path))
        pt = self._point(st.opts["at"], f"{st.path}.at")
        assert pt is not None
        horizontal = bool(st.opts.get("horizontal", False))
        self._input(lambda: self.sink.scroll(pt[0], pt[1], clicks=clicks, horizontal=horizontal), st.path, "scroll")

    # waiting and seeing -------------------------------------------------------

    def _do_wait(self, st: Step) -> None:
        ms = float(self._r(st.value, st.path))
        j = int(self._r(st.opts.get("jitter_ms", 0), st.path))
        self._pause_ms(ms + (self.rng.randint(0, j) if j else 0) + self._jitter(), st.path)

    def _timed_out(self, st: Step, what: str) -> None:
        ot = st.opts.get("on_timeout", "fail")
        self._emit("script.timeout", at=st.path, what=what)
        if "on_timeout" in st.children:
            self._say(f"{st.path}: {what}; running on_timeout steps")
            self._block(st.children["on_timeout"])
        elif ot == "continue":
            self._say(f"{st.path}: {what}; continuing")
        else:
            raise StepFailed(st.path, what)

    def _do_wait_for(self, st: Step) -> None:
        timeout = float(self._r(st.opts.get("timeout_s", 30), st.path))
        poll = int(self._r(st.opts.get("poll_ms", self.script.poll_ms), st.path))
        res = self._wait(st.cond, gone=False, timeout_s=timeout, poll_ms=poll, path=st.path)
        if isinstance(res, Match):
            if st.opts.get("as"):
                self.vars[str(st.opts["as"])] = match_to_var(res)
            self._emit("script.matched", at=st.path, match=res.meta)
            return
        if st.opts.get("as"):
            self.vars[str(st.opts["as"])] = None
        self._timed_out(st, f"not seen within {timeout:g}s")

    def _do_wait_gone(self, st: Step) -> None:
        timeout = float(self._r(st.opts.get("timeout_s", 30), st.path))
        poll = int(self._r(st.opts.get("poll_ms", self.script.poll_ms), st.path))
        if self._wait(st.cond, gone=True, timeout_s=timeout, poll_ms=poll, path=st.path) is True:
            return
        self._timed_out(st, f"still there after {timeout:g}s")

    def _do_find(self, st: Step) -> None:
        m = self._test(st.cond, st.path)
        self.vars[str(st.opts["as"])] = match_to_var(m) if m is not None else None

    def _do_read_text(self, st: Step) -> None:
        from reflex.ocr import numbers_in, read_text

        region = [int(v) for v in self._r(st.value, st.path)]
        res = read_text(self._frame().frame_bgr, tuple(region))  # type: ignore[arg-type]
        if st.opts.get("number"):
            nums = numbers_in(" ".join(res.lines))
            self.vars[str(st.opts["as"])] = nums[0] if nums else None
        else:
            self.vars[str(st.opts["as"])] = res.text

    def _do_screenshot(self, st: Step) -> None:
        import cv2

        path = Path(str(self._r(st.value, st.path)))
        if not path.is_absolute() and self.script.base_dir is not None:
            path = self.script.base_dir / path
        frame = self._frame().frame_bgr
        if "region" in st.opts:
            x, y, w, h = (int(v) for v in self._r(st.opts["region"], st.path))
            frame = frame[y : y + h, x : x + w]
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        self._say(f"saved {path}")

    # control flow -----------------------------------------------------------

    def _do_if(self, st: Step) -> None:
        m = self._test(st.cond, st.path)
        self._block(st.children["then"] if m is not None else st.children["else"])

    def _do_repeat(self, st: Step) -> None:
        spec = st.value
        counter = st.opts.get("as")
        cap = int(self._r(st.opts.get("max", 1_000_000), st.path))
        if "times" in spec:
            limit = min(int(self._r(spec["times"], st.path)), cap)
        else:
            limit = cap
        i = 0
        while i < limit:
            self._check(st.path)
            if "while" in spec and self._test(st.cond, f"{st.path}.while") is None:
                return
            if counter:
                self.vars[str(counter)] = i
            try:
                self._block(st.children["steps"])
            except _Break:
                return
            except _Continue:
                pass
            i += 1
            if "until" in spec and self._test(st.cond, f"{st.path}.until") is not None:
                return
        if "times" not in spec:
            self._say(f"{st.path}: stopped after max {cap} iterations")

    def _do_set(self, st: Step) -> None:
        for k, v in st.value.items():
            self.vars[str(k)] = self._r(v, f"{st.path}.{k}")

    def _do_log(self, st: Step) -> None:
        self._say(str(self._r(st.value, st.path)))

    def _do_call(self, st: Step) -> None:
        for k, v in (st.opts.get("with") or {}).items():
            self.vars[str(k)] = self._r(v, f"{st.path}.with.{k}")
        self._block(self.script.functions[str(st.value)])

    def _do_break(self, st: Step) -> None:
        raise _Break()

    def _do_continue(self, st: Step) -> None:
        raise _Continue()

    def _do_stop(self, st: Step) -> None:
        raise _Stop("finished", str(self._r(st.value, st.path)) if st.value not in (True, None) else "stop step")

    def _do_fail(self, st: Step) -> None:
        raise StepFailed(st.path, str(self._r(st.value, st.path)))

    def _do_focus(self, st: Step) -> None:
        if self._focus is None:
            raise StepFailed(st.path, "focus needs a window target")
        self._focus()
        self.sleep(0.15)

    def _do_stick(self, st: Step) -> None:
        self._need("stick", st.path)
        side = str(self._r(st.value, st.path))
        x = float(self._r(st.opts.get("x", 0), st.path))
        y = float(self._r(st.opts.get("y", 0), st.path))
        hold = int(self._r(st.opts.get("hold_ms", 200), st.path))
        self._input(lambda: self.sink.stick(side, x, y), st.path, f"stick {side}")
        if hold > 0:
            self._pause_ms(hold, st.path)
            self.sink.stick(side, 0, 0)

    def _do_watcher(self, st: Step) -> None:
        if self._rt is None:
            raise StepFailed(st.path, "this script has no watchers")
        (action, name), = st.value.items()
        if not self._rt.has_spec(str(name)):
            raise StepFailed(st.path, f"no watcher {name!r} (have {self._rt.names()})")
        if action == "enable":
            self._rt.enable(str(name))
        else:
            self._rt.disable(str(name), reason="script")


# --------------------------------------------------------------------------- targets and sessions


@dataclass
class Target:
    instance: str
    source: Any
    sink: Any
    vars: dict[str, Any]
    focus: Callable[[], bool] | None = None


def open_targets(
    script: Script,
    *,
    overrides: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    source: Any = None,
) -> list[Target]:
    """Build source/sink pairs from script.target (+ CLI overrides).

    One per matching window when all_windows is true, so a script can drive
    several game clients at once.
    """

    t = {**script.target, **{k: v for k, v in (overrides or {}).items() if v is not None}}
    targets = _open(t, script, dry_run=dry_run, source=source)
    if t.get("gamepad") and not dry_run:
        from reflex.backends.gamepad import GamepadSink

        for tg in targets:
            tg.sink = GamepadSink(tg.sink)
    return targets


def _open(t: dict[str, Any], script: Script, *, dry_run: bool, source: Any) -> list[Target]:
    from reflex.io import RecordingActionSink

    if source is not None:
        return [Target("main", source, RecordingActionSink() if dry_run else _sink_for(t, None), {}, None)]
    if "screen" in t:
        from reflex.backends.windows import ScreenSource, SendInputSink

        region = t["screen"]
        src = ScreenSource(None if region in (None, "all", True) else tuple(region))
        sink = RecordingActionSink() if dry_run else SendInputSink(origin=src.origin, require_focus=False)
        return [Target("screen", src, sink, {"screen": list(src.region)})]
    win_q = t.get("window")
    if not win_q:
        raise ScriptError("the script has no target: add target: {window: {title: ...}} or pass --title")
    from reflex.backends.windows import WindowSource, find_window, focus, list_windows

    if isinstance(win_q, str):
        win_q = {"title": win_q}
    if t.get("all_windows"):
        query = {k: v for k, v in win_q.items() if k in ("title", "title_regex", "class_name", "process")}
        windows = list_windows(**query)
        if not windows:
            find_window(**query)  # raises with the helpful message
    else:
        windows = [find_window(**win_q)]
    out = []
    for i, w in enumerate(windows):
        src = WindowSource(w, method=t.get("capture", "auto"))
        sink = RecordingActionSink() if dry_run else _sink_for(t, w)
        name = f"{i}:{w.title[:24]}" if len(windows) > 1 else "main"
        out.append(Target(name, src, sink, {"window": w.describe()},
                          focus=(lambda h=w.hwnd: focus(h))))
    return out


def _sink_for(t: Mapping[str, Any], window: Any) -> Any:
    from reflex.backends.windows import PostMessageSink, SendInputSink

    if t.get("input", "postmessage") == "sendinput":
        return SendInputSink(window, require_focus=bool(t.get("require_focus", True)))
    if window is None:
        raise ScriptError("postmessage input needs a window target")
    return PostMessageSink(window)


def run_script(
    script: Script,
    *,
    targets: list[Target] | None = None,
    events: EventSink | None = None,
    control: Control | None = None,
    hotkeys: bool = True,
    **open_kw: Any,
) -> list[ScriptResult]:
    """Run a script on every target (threads when there are several) and wait for all."""
    from reflex.hotkeys import DEFAULT_PAUSE, DEFAULT_STOP, HotkeyWatcher

    control = control or Control()
    targets = targets if targets is not None else open_targets(script, **open_kw)
    keys = None
    if hotkeys:
        stop_combo = script.hotkeys.get("stop", DEFAULT_STOP)
        pause_combo = script.hotkeys.get("pause", DEFAULT_PAUSE)
        keys = HotkeyWatcher({stop_combo: lambda: control.stop(f"hotkey {stop_combo}"),
                              pause_combo: control.toggle_pause}).start()
    try:
        runners = [ScriptRunner(script, source=t.source, sink=t.sink, events=events, control=control,
                                instance=t.instance, extra_vars=t.vars, focus=t.focus) for t in targets]
        if len(runners) == 1:
            return [runners[0].run()]
        results: list[ScriptResult | None] = [None] * len(runners)

        def go(i: int) -> None:
            results[i] = runners[i].run()

        threads = [threading.Thread(target=go, args=(i,), name=f"reflex-{r.instance}", daemon=True)
                   for i, r in enumerate(runners)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        return [r for r in results if r is not None]
    finally:
        if keys is not None:
            keys.stop()


def schedule_times(*, every_s: float | None, at: list[str] | None, now: float | None = None) -> float:
    """Seconds until the next scheduled start."""
    import datetime as dt

    now_dt = dt.datetime.fromtimestamp(now if now is not None else time.time())
    waits: list[float] = []
    if every_s:
        waits.append(float(every_s))
    for hhmm in at or []:
        h, m = (int(p) for p in str(hhmm).split(":"))
        nxt = now_dt.replace(hour=h, minute=m, second=0, microsecond=0)
        if nxt <= now_dt:
            nxt += dt.timedelta(days=1)
        waits.append((nxt - now_dt).total_seconds())
    if not waits:
        raise ValueError("schedule needs every_s or at")
    return min(waits)


def describe_steps() -> list[dict[str, Any]]:
    """The step table, for `reflex schema` and for agents writing scripts."""
    return [{"step": k, "fields": sorted(v["fields"]), "doc": v["doc"]} for k, v in VERBS.items()]


def frame_from_image(path: str | Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return img
