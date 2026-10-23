"""Watchdog macros: anti_stuck latch + recovery sequence."""

from __future__ import annotations

import numpy as np

from reflex.core import FrameView
from reflex.watchdog import anti_stuck, deadman_release


def _view(ts_ms):
    return FrameView(frame_bgr=np.zeros((10, 10, 3), dtype=np.uint8), ts_ns=int(ts_ms * 1_000_000))


def test_anti_stuck_does_not_fire_below_threshold():
    spec = anti_stuck(stuck_signal=lambda v: 0.4, threshold=0.85, stable_for_ms=200)
    assert spec.predicate(_view(0)) is None
    assert spec.predicate(_view(1000)) is None


def test_anti_stuck_requires_stable_window_above_threshold():
    spec = anti_stuck(stuck_signal=lambda v: 0.95, threshold=0.85, stable_for_ms=300)
    # First sample arms; second is too soon; third clears the window
    assert spec.predicate(_view(0)) is None
    assert spec.predicate(_view(100)) is None
    assert spec.predicate(_view(400)) is not None


def test_anti_stuck_resets_on_dip():
    score = {"v": 0.95}
    spec = anti_stuck(stuck_signal=lambda v: score["v"], threshold=0.85, stable_for_ms=300)
    assert spec.predicate(_view(0)) is None
    score["v"] = 0.20
    assert spec.predicate(_view(100)) is None
    score["v"] = 0.95
    # latch was reset, must wait full window again from new arm point
    assert spec.predicate(_view(150)) is None
    assert spec.predicate(_view(500)) is not None


def test_anti_stuck_action_returns_recovery_buttons():
    spec = anti_stuck(stuck_signal=lambda v: 1.0, threshold=0.5, stable_for_ms=0,
                      recovery_buttons=("down", "left", "down"))
    m = spec.predicate(_view(0))
    assert m is not None
    assert spec.action_factory(m) == ("down", "left", "down")


def test_anti_stuck_default_priority_higher_than_interaction():
    spec = anti_stuck(stuck_signal=lambda v: 0.0)
    assert spec.priority >= 90


def test_deadman_release_fires_only_when_held():
    held = {"v": False}
    spec = deadman_release(is_held=lambda: held["v"])
    assert spec.predicate(_view(0)) is None
    held["v"] = True
    m = spec.predicate(_view(1))
    assert m is not None
    assert spec.action_factory(m) == ("up", "down", "left", "right")
    assert spec.foreground_only is False
    assert spec.priority >= 99
