from __future__ import annotations

import numpy as np
import pytest

from reflex import MacroEngine, Match, TriggerSpec
from reflex.io import RecordingInputSink


class _StepClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += float(dt)


class _FrameQueue:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self.calls = 0

    def grab(self) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("frame queue exhausted")
        self.calls += 1
        return self._frames.pop(0)


def _black(h: int = 16, w: int = 16) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _white(h: int = 16, w: int = 16) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_engine_fires_when_predicate_matches_and_reverifies():
    clock = _StepClock()
    sink = RecordingInputSink()
    frames = _FrameQueue([_black(), _black(), _white(), _white(), _white()])

    def predicate(view):
        if int(view.frame_bgr.mean()) > 200:
            return Match(meta={"why": "white"})
        return None

    spec = TriggerSpec(
        name="white_screen",
        predicate=predicate,
        action_factory=lambda m: ("a", "start"),
        timeout_s=10.0,
        poll_hz=10.0,
        inter_press_ms=20,
    )
    engine = MacroEngine(frames, sink, clock=clock.now, sleep=clock.sleep)
    outcome = engine.run(spec)

    assert outcome.status == "fired"
    assert outcome.fires == 1
    assert outcome.last_action_seq == ("a", "start")
    assert sink.calls == [("a", 60), ("start", 60)]


def test_engine_times_out_when_predicate_never_matches():
    clock = _StepClock()
    sink = RecordingInputSink()
    frames = _FrameQueue([_black()] * 1000)

    spec = TriggerSpec(
        name="never",
        predicate=lambda view: None,
        action_factory=lambda m: ("a",),
        timeout_s=0.5,
        poll_hz=20.0,
    )
    engine = MacroEngine(frames, sink, clock=clock.now, sleep=clock.sleep)
    outcome = engine.run(spec)

    assert outcome.status == "timed_out"
    assert outcome.fires == 0
    assert sink.calls == []


def test_engine_reverify_blocks_flicker_match():
    clock = _StepClock()
    sink = RecordingInputSink()
    sequence = [_black(), _white()] + [_black()] * 100
    frames = _FrameQueue(sequence)

    def predicate(view):
        if int(view.frame_bgr.mean()) > 200:
            return Match(meta={})
        return None

    spec = TriggerSpec(
        name="flicker",
        predicate=predicate,
        action_factory=lambda m: ("a",),
        timeout_s=0.5,
        poll_hz=20.0,
    )
    engine = MacroEngine(frames, sink, clock=clock.now, sleep=clock.sleep)
    outcome = engine.run(spec)

    assert outcome.status == "timed_out"
    assert sink.calls == []


def test_engine_validates_spec():
    with pytest.raises(ValueError):
        TriggerSpec(
            name="bad",
            predicate=lambda v: None,
            action_factory=lambda m: ("a",),
            timeout_s=0.0,
        )
