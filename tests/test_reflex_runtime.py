"""ReflexRuntime: ticking, priorities, deadman, cooldowns, lockouts."""

from __future__ import annotations

import numpy as np

from reflex.core import Match, TriggerSpec
from reflex.runtime import ReflexRuntime


class _StubFrameSource:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.grabs = 0

    def grab(self) -> np.ndarray:
        self.grabs += 1
        return self.frame


class _RecordingInputSink:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def press(self, button: str, *, hold_ms: int) -> bool:
        self.calls.append((button, hold_ms))
        return True


class _RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def emit(self, kind: str, payload):
        self.events.append((kind, dict(payload)))


def _always_match_spec(name: str, *, action=("a",), priority: int = 50, max_fires: int = 1, cooldown_s: float = 0.0, poll_hz: float = 1000.0):
    def pred(_v):
        return Match(meta={"name": name})

    def af(_m):
        return action

    return TriggerSpec(
        name=name,
        predicate=pred,
        action_factory=af,
        timeout_s=600.0,
        poll_hz=poll_hz,
        max_fires=max_fires,
        cooldown_s=cooldown_s,
        inter_press_ms=0,
        hold_ms=1,
        priority=priority,
    )


def _never_match_spec(name: str):
    def pred(_v):
        return None

    def af(_m):
        return ("z",)

    return TriggerSpec(name=name, predicate=pred, action_factory=af, poll_hz=1000.0, hold_ms=1, inter_press_ms=0)


def _fixed_clock():
    state = {"t": 0.0}

    def get():
        return state["t"]

    def advance(dt):
        state["t"] += dt

    return get, advance


def _fixed_ts():
    state = {"t": 1_000_000_000}

    def get():
        return state["t"]

    return get


def test_runtime_fires_when_predicate_matches():
    src = _StubFrameSource(np.zeros((10, 10, 3), dtype=np.uint8))
    sink = _RecordingInputSink()
    rt = ReflexRuntime(source=src, sink=sink, clock=_fixed_clock()[0], ts_ns=_fixed_ts())
    rt.register(_always_match_spec("interact", action=("right", "a")))
    report = rt.tick()
    assert report.fired == "interact"
    assert sink.calls == [("right", 1), ("a", 1)]
    assert report.candidates == 1


def test_runtime_does_not_fire_when_no_predicate_matches():
    src = _StubFrameSource(np.zeros((10, 10, 3), dtype=np.uint8))
    sink = _RecordingInputSink()
    rt = ReflexRuntime(source=src, sink=sink, clock=_fixed_clock()[0], ts_ns=_fixed_ts())
    rt.register(_never_match_spec("nothing"))
    report = rt.tick()
    assert report.fired is None
    assert sink.calls == []


def test_runtime_deadman_suppresses_firing():
    src = _StubFrameSource(np.zeros((10, 10, 3), dtype=np.uint8))
    sink = _RecordingInputSink()
    held = {"v": True}
    rt = ReflexRuntime(source=src, sink=sink, deadman=lambda: held["v"], clock=_fixed_clock()[0], ts_ns=_fixed_ts())
    rt.register(_always_match_spec("x"))
    report = rt.tick()
    assert report.suppressed_reason == "deadman"
    assert sink.calls == []


def test_runtime_priority_chooses_higher():
    src = _StubFrameSource(np.zeros((10, 10, 3), dtype=np.uint8))
    sink = _RecordingInputSink()
    rt = ReflexRuntime(source=src, sink=sink, clock=_fixed_clock()[0], ts_ns=_fixed_ts())
    rt.register(_always_match_spec("low", action=("a",), priority=10))
    rt.register(_always_match_spec("high", action=("b",), priority=99))
    report = rt.tick()
    assert report.fired == "high"
    assert sink.calls == [("b", 1)]


def test_runtime_cooldown_blocks_immediate_refire():
    clock_get, advance = _fixed_clock()
    src = _StubFrameSource(np.zeros((10, 10, 3), dtype=np.uint8))
    sink = _RecordingInputSink()
    rt = ReflexRuntime(source=src, sink=sink, clock=clock_get, ts_ns=_fixed_ts())
    rt.register(_always_match_spec("x", max_fires=10, cooldown_s=1.0))

    advance(0.001)
    assert rt.tick().fired == "x"
    advance(0.5)
    assert rt.tick().fired is None  # still in cooldown
    advance(0.6)
    assert rt.tick().fired == "x"


def test_runtime_max_fires_disables_after_limit():
    clock_get, advance = _fixed_clock()
    src = _StubFrameSource(np.zeros((10, 10, 3), dtype=np.uint8))
    sink = _RecordingInputSink()
    rt = ReflexRuntime(source=src, sink=sink, clock=clock_get, ts_ns=_fixed_ts())
    rt.register(_always_match_spec("x", max_fires=2))

    advance(0.001)
    assert rt.tick().fired == "x"
    advance(0.001)
    assert rt.tick().fired == "x"
    advance(0.001)
    assert rt.tick().fired is None
    advance(0.001)
    assert rt.tick().fired is None


def test_runtime_poll_hz_skips_too_soon():
    clock_get, advance = _fixed_clock()
    src = _StubFrameSource(np.zeros((10, 10, 3), dtype=np.uint8))
    sink = _RecordingInputSink()
    rt = ReflexRuntime(source=src, sink=sink, clock=clock_get, ts_ns=_fixed_ts())

    def pred(_v):
        return Match(meta={})

    spec = TriggerSpec(name="x", predicate=pred, action_factory=lambda _m: ("a",),
                       poll_hz=10.0, hold_ms=1, inter_press_ms=0, max_fires=10, cooldown_s=0.0)
    rt.register(spec)

    advance(0.01)
    rt.tick()  # records next_poll = +0.1
    advance(0.05)
    report = rt.tick()
    assert report.candidates == 0  # too soon


def test_runtime_lockout_blocks_all_fires():
    src = _StubFrameSource(np.zeros((10, 10, 3), dtype=np.uint8))
    sink = _RecordingInputSink()
    clock_get, advance = _fixed_clock()
    rt = ReflexRuntime(source=src, sink=sink, clock=clock_get, ts_ns=_fixed_ts())
    rt.register(_always_match_spec("x"))
    advance(0.001)
    rt.lockout(2.0)
    r = rt.tick()
    assert r.suppressed_reason and "lockout" in r.suppressed_reason
    advance(2.5)
    assert rt.tick().fired == "x"


def test_runtime_emits_events_on_register_and_fire():
    src = _StubFrameSource(np.zeros((10, 10, 3), dtype=np.uint8))
    sink = _RecordingInputSink()
    events = _RecordingEventSink()
    rt = ReflexRuntime(source=src, sink=sink, events=events, clock=_fixed_clock()[0], ts_ns=_fixed_ts())
    rt.register(_always_match_spec("x"))
    rt.tick()
    kinds = [k for k, _ in events.events]
    assert "reflex.registered" in kinds
    assert "reflex.fired" in kinds


def test_runtime_predicate_error_does_not_crash():
    src = _StubFrameSource(np.zeros((10, 10, 3), dtype=np.uint8))
    sink = _RecordingInputSink()
    events = _RecordingEventSink()

    def pred_explode(_v):
        raise RuntimeError("boom")

    spec = TriggerSpec(name="x", predicate=pred_explode, action_factory=lambda _m: ("a",),
                       poll_hz=1000.0, hold_ms=1, inter_press_ms=0, max_fires=1, cooldown_s=0.0)
    rt = ReflexRuntime(source=src, sink=sink, events=events, clock=_fixed_clock()[0], ts_ns=_fixed_ts())
    rt.register(spec)
    report = rt.tick()
    assert report.fired is None
    assert any(k == "reflex.predicate_error" for k, _ in events.events)
