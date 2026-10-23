"""Recorder: raw input -> script, and the hooks' lifecycle."""

from __future__ import annotations

import sys
import time

import pytest
import yaml

from reflex.recorder import RawEvent, compile_events
from reflex.script import build_script


def K(t, kind, key, char=""):
    return RawEvent(t, kind, key=key, char=char)


def M(t, kind, x, y, button="left", delta=0):
    return RawEvent(t, kind, button=button, x=x, y=y, delta=delta)


def test_typing_becomes_one_type_step_and_pauses_become_waits():
    evs = []
    t = 0.0
    for ch in "gg":
        evs += [K(t, "key_down", ch, ch), K(t + 0.03, "key_up", ch)]
        t += 0.08
    evs += [K(1.0, "key_down", "enter"), K(1.05, "key_up", "enter")]
    script = compile_events(evs)
    assert script["steps"] == [{"type": "gg"}, {"wait": 890}, {"press": "enter"}]  # from release to press


def test_shift_and_layout_chars_and_chords():
    evs = [
        K(0.0, "key_down", "shift"), K(0.01, "key_down", "h", "H"), K(0.05, "key_up", "h"), K(0.06, "key_up", "shift"),
        K(0.07, "key_down", "i", "i"), K(0.09, "key_up", "i"),
        K(0.5, "key_down", "ctrl"), K(0.51, "key_down", "s"), K(0.55, "key_up", "s"), K(0.56, "key_up", "ctrl"),
        K(1.0, "key_down", "shift"), K(1.1, "key_up", "shift"),
    ]
    steps = compile_events(evs)["steps"]
    assert steps[0] == {"type": "Hi"}
    assert {"chord": ["ctrl", "s"]} in steps
    assert steps[-1] == {"press": "shift"}


def test_long_hold_and_autorepeat():
    evs = [K(0.0, "key_down", "w"), K(0.1, "key_down", "w"), K(0.2, "key_down", "w"), K(0.9, "key_up", "w")]
    assert compile_events(evs)["steps"] == [{"press": "w", "hold_ms": 900}]


def test_clicks_double_clicks_drags_and_scroll():
    evs = [
        M(0.0, "mouse_down", 10, 20), M(0.05, "mouse_up", 10, 20),
        M(0.15, "mouse_down", 10, 20), M(0.2, "mouse_up", 11, 20),
        M(1.0, "mouse_down", 100, 100, "right"), M(1.05, "mouse_up", 100, 100, "right"),
        M(2.0, "mouse_down", 50, 50), M(2.4, "mouse_up", 150, 80),
        M(3.0, "wheel", 70, 70, delta=-1), M(3.02, "wheel", 70, 70, delta=-1),
    ]
    steps = compile_events(evs)["steps"]
    assert steps[0] == {"double_click": [10, 20]}
    assert {"right_click": [100, 100]} in steps
    drag = next(s for s in steps if "drag" in s)
    assert drag["drag"] == [50, 50] and drag["to"] == [150, 80] and drag["hold_ms"] == 400
    assert steps[-1] == {"scroll": -2, "at": [70, 70]}


def test_clicks_can_use_recorded_images():
    evs = [M(0.0, "mouse_down", 30, 40), M(0.05, "mouse_up", 30, 40)]
    steps = compile_events(evs, images={0: "rec_images/click_001.png"})["steps"]
    assert steps == [{"click": {"image": "rec_images/click_001.png"}, "comment": "recorded at 30,40"}]


def test_keys_still_held_are_released_and_output_is_a_valid_script():
    evs = [K(0.0, "key_down", "d"), M(0.5, "mouse_down", 1, 1), M(0.6, "mouse_up", 1, 1)]
    script = compile_events(evs, target={"window": {"title": "x"}})
    assert script["steps"][-1]["key_up"] == "d"
    build_script(yaml.safe_load(yaml.safe_dump(script)), check_keys=False)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows hooks")
def test_hooks_see_real_keys_and_ignore_injected_ones_by_default():
    from reflex.backends.windows import SendInputSink, screen_capture_available
    from reflex.recorder import Recorder

    if not screen_capture_available():
        pytest.skip("desktop is locked: Windows refuses synthetic input until it is unlocked")
    sink = SendInputSink(require_focus=False)
    plain = Recorder(None).start()
    sink.press("f24", hold_ms=5)  # F24: no app binds it
    time.sleep(0.15)
    # Only F24 is looked at: the hooks also see whatever the person at the PC is doing.
    assert [e for e in plain.stop() if e.key == "f24"] == []  # reflex's own (injected) input is ignored

    rec = Recorder(None, record_injected=True).start()
    sink.press("f24", hold_ms=5)
    time.sleep(0.15)
    events = [e for e in rec.stop() if e.key == "f24"]
    assert [e.kind for e in events] == ["key_down", "key_up"]
    assert rec.stopped.is_set()
