"""
Pluggable I/O: frame sources, input sinks, event sinks.

Input sinks are duck-typed. The minimum is `press(button, *, hold_ms)`.
A sink gains capabilities by implementing more methods:

    press(button, *, hold_ms) -> bool     capability "key"
    key_down(button) -> bool              capability "key_down"
    key_up(button) -> bool                capability "key_up"   (both => "chord")
    click(x, y, *, button="left") -> bool capability "click"

A sink may also expose an explicit `capabilities` set to narrow that.
`ActionExecutor` is the single place typed actions are turned into sink
calls; it refuses (returns ActionRefused) anything the sink can't do.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

import numpy as np

from reflex.actions import (
    Action,
    Chord,
    Click,
    Key,
    KeyDown,
    KeyUp,
    MouseDown,
    MouseUp,
    Move,
    Scroll,
    TypeText,
    Wait,
)


@runtime_checkable
class FrameSource(Protocol):
    def grab(self) -> np.ndarray: ...


@runtime_checkable
class InputSink(Protocol):
    def press(self, button: str, *, hold_ms: int) -> bool: ...


class EventSink(Protocol):
    def emit(self, kind: str, payload: dict[str, Any]) -> None: ...


LogCallback = Callable[..., None]


def noop_log(**kwargs: Any) -> None:
    return None


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


def check_frame(frame: Any, *, who: str = "FrameSource") -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise TypeError(f"{who} must return a (H, W, 3) ndarray, got shape={getattr(frame, 'shape', None)}")
    return frame


class CallableFrameSource:
    def __init__(self, fn: Callable[[], np.ndarray]) -> None:
        self._fn = fn

    def grab(self) -> np.ndarray:
        return check_frame(self._fn())


class StaticFrameSource:
    """Always returns the same frame (tests, dry runs)."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = check_frame(frame)
        self.grabs = 0

    def grab(self) -> np.ndarray:
        self.grabs += 1
        return self.frame


class TimedFrameSequenceSource:
    """Replays recorded frames against a clock.

    `frames` is a list of (t_s, frame). grab() returns the latest frame whose
    timestamp is <= clock(), so a virtual clock reproduces a live session's
    timing exactly (the frame on screen when the reflex looked).
    """

    def __init__(self, frames: Sequence[tuple[float, np.ndarray]], clock: Callable[[], float]) -> None:
        if not frames:
            raise ValueError("TimedFrameSequenceSource needs at least one frame")
        self._frames = sorted(((float(t), check_frame(f, who="replay frame")) for t, f in frames), key=lambda p: p[0])
        self._clock = clock
        self.last_index = 0

    def index_at(self, t: float) -> int:
        idx = 0
        for i, (ts, _f) in enumerate(self._frames):
            if ts <= t:
                idx = i
            else:
                break
        return idx

    def grab(self) -> np.ndarray:
        self.last_index = self.index_at(self._clock())
        return self._frames[self.last_index][1]

    @property
    def end_t(self) -> float:
        return self._frames[-1][0]

    def __len__(self) -> int:
        return len(self._frames)


def coerce_frame_source(source: Any) -> FrameSource:
    if hasattr(source, "grab"):
        return source
    if callable(source):
        return CallableFrameSource(source)
    raise TypeError(f"frame_source must be FrameSource or callable, got {type(source)!r}")


# ---------------------------------------------------------------------------
# Input sinks
# ---------------------------------------------------------------------------


class CallableInputSink:
    def __init__(self, fn: Callable[[str, int], bool]) -> None:
        self._fn = fn

    def press(self, button: str, *, hold_ms: int) -> bool:
        return bool(self._fn(button, hold_ms))


class RecordingInputSink:
    """Press-only recorder (legacy shape: calls == [(button, hold_ms)])."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def press(self, button: str, *, hold_ms: int) -> bool:
        self.calls.append((button, int(hold_ms)))
        return True


class RecordingActionSink:
    """Full-capability recorder used by dry runs and the replay harness."""

    capabilities = frozenset({"key", "key_down", "key_up", "chord", "click", "wait", "move", "mouse_down",
                              "mouse_up", "scroll", "type"})

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self.log: list[dict[str, Any]] = []

    def _rec(self, kind: str, **kw: Any) -> bool:
        self.log.append({"t": round(self._clock(), 6), "kind": kind, **kw})
        return True

    def press(self, button: str, *, hold_ms: int) -> bool:
        return self._rec("key", button=button, hold_ms=int(hold_ms))

    def key_down(self, button: str) -> bool:
        return self._rec("key_down", button=button)

    def key_up(self, button: str) -> bool:
        return self._rec("key_up", button=button)

    def click(self, x: int, y: int, *, button: str = "left") -> bool:
        return self._rec("click", x=int(x), y=int(y), button=button)

    def move(self, x: int, y: int) -> bool:
        return self._rec("move", x=int(x), y=int(y))

    def mouse_down(self, x: int, y: int, *, button: str = "left") -> bool:
        return self._rec("mouse_down", x=int(x), y=int(y), button=button)

    def mouse_up(self, x: int, y: int, *, button: str = "left") -> bool:
        return self._rec("mouse_up", x=int(x), y=int(y), button=button)

    def scroll(self, x: int, y: int, *, clicks: int = -1, horizontal: bool = False) -> bool:
        return self._rec("scroll", x=int(x), y=int(y), clicks=int(clicks), horizontal=bool(horizontal))

    def type_text(self, text: str, *, interval_ms: int = 0) -> bool:
        return self._rec("type", text=str(text))

    def release_all(self) -> None:
        self._rec("release_all")

    @property
    def calls(self) -> list[tuple[str, int]]:
        return [(e["button"], e["hold_ms"]) for e in self.log if e["kind"] == "key"]


def coerce_input_sink(sink: Any) -> Any:
    if hasattr(sink, "press") or hasattr(sink, "click"):
        return sink
    if callable(sink):
        return CallableInputSink(sink)
    raise TypeError(f"input_sink must be InputSink or callable, got {type(sink)!r}")


def sink_capabilities(sink: Any) -> frozenset[str]:
    declared = getattr(sink, "capabilities", None)
    if declared is not None:
        return frozenset(declared) | {"wait"}
    caps = {"wait"}
    if callable(getattr(sink, "press", None)):
        caps.add("key")
    if callable(getattr(sink, "key_down", None)):
        caps.add("key_down")
    if callable(getattr(sink, "key_up", None)):
        caps.add("key_up")
    if {"key_down", "key_up"} <= caps:
        caps.add("chord")
    for method, cap in (("click", "click"), ("move", "move"), ("mouse_down", "mouse_down"),
                        ("mouse_up", "mouse_up"), ("scroll", "scroll"), ("type_text", "type")):
        if callable(getattr(sink, method, None)):
            caps.add(cap)
    return frozenset(caps)


class ActionRefused(Exception):
    """The sink lacks the capability an action needs."""


class ActionExecutor:
    """Turns typed actions into sink calls. The only place input is sent."""

    def __init__(self, sink: Any, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self.sink = coerce_input_sink(sink)
        self.capabilities = sink_capabilities(self.sink)
        self._sleep = sleep

    def supports(self, action: Action) -> bool:
        return action.capability in self.capabilities

    def execute(self, action: Action, *, default_hold_ms: int) -> bool:
        if not self.supports(action):
            raise ActionRefused(f"sink {type(self.sink).__name__} cannot perform {action.capability!r} ({action.label()})")
        s = self.sink
        if isinstance(action, Wait):
            self._sleep(max(0, int(action.ms)) / 1000.0)
            return True
        if isinstance(action, Key):
            hold = default_hold_ms if action.hold_ms is None else action.hold_ms
            return bool(s.press(action.name, hold_ms=int(hold)))
        if isinstance(action, KeyDown):
            return bool(s.key_down(action.name))
        if isinstance(action, KeyUp):
            return bool(s.key_up(action.name))
        if isinstance(action, Chord):
            hold = default_hold_ms if action.hold_ms is None else action.hold_ms
            ok = all([bool(s.key_down(k)) for k in action.keys])
            self._sleep(max(0, int(hold)) / 1000.0)
            ok = all([bool(s.key_up(k)) for k in reversed(action.keys)]) and ok
            return ok
        if isinstance(action, Click):
            return bool(s.click(int(action.x), int(action.y), button=action.button))
        if isinstance(action, Move):
            return bool(s.move(int(action.x), int(action.y)))
        if isinstance(action, MouseDown):
            return bool(s.mouse_down(int(action.x), int(action.y), button=action.button))
        if isinstance(action, MouseUp):
            return bool(s.mouse_up(int(action.x), int(action.y), button=action.button))
        if isinstance(action, Scroll):
            return bool(s.scroll(int(action.x), int(action.y), clicks=int(action.clicks), horizontal=action.horizontal))
        if isinstance(action, TypeText):
            return bool(s.type_text(action.text, interval_ms=int(action.interval_ms)))
        raise ActionRefused(f"unknown action type {type(action).__name__}")


# ---------------------------------------------------------------------------
# Event sinks
# ---------------------------------------------------------------------------


class NoopEventSink:
    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        return None


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append((kind, dict(payload)))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.events]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [p for k, p in self.events if k == kind]


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={tuple(obj.shape)}>"
    return repr(obj)


class JsonlEventSink:
    """Appends one JSON line per event: {ts_ns, kind, session, ...payload}.

    This file is the monitoring contract: `reflex stats` and the dashboard
    read it; the replay harness can compare against it.
    """

    def __init__(self, path: str | Path, *, session: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session = session or time.strftime("%Y%m%dT%H%M%S")
        self._lock = threading.Lock()

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        body = jsonable(payload)
        # Reserved envelope keys win; colliding payload keys are renamed, never dropped.
        for key in ("kind", "session", "ts_ns"):
            if key in body:
                body[f"spec_{key}" if key == "kind" else f"payload_{key}"] = body.pop(key)
        rec = {"ts_ns": time.time_ns(), "kind": kind, "session": self.session}
        rec.update(body)
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)


class FanoutEventSink:
    def __init__(self, *sinks: EventSink) -> None:
        self.sinks = [s for s in sinks if s is not None]

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        for s in self.sinks:
            try:
                s.emit(kind, payload)
            except Exception:  # noqa: BLE001 — a broken monitor must not stop reflexes
                pass


class LogCallbackEventSink:
    """Adapts the event stream onto a `log(**kwargs)` callback (MacroEngine API)."""

    def __init__(self, log: LogCallback) -> None:
        self._log = log

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        short = kind.split(".", 1)[1] if kind.startswith("reflex.") else kind
        body = dict(payload)
        spec = body.pop("name", None)
        if "kind" in body:
            body["spec_kind"] = body.pop("kind")
        if "spec" in body:
            body["spec_meta"] = body.pop("spec")
        self._log(kind=short, spec=spec, **body)
