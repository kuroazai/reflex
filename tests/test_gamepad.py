"""Gamepad sink with a stand-in controller (no driver needed)."""

from __future__ import annotations

import yaml

from reflex.backends.gamepad import GamepadSink, is_pad
from reflex.io import RecordingActionSink, sink_capabilities
from reflex.script import ScriptRunner, build_script


class FakePad:
    def __init__(self):
        self.calls = []

    def press_button(self, button):
        self.calls.append(("press", button))

    def release_button(self, button):
        self.calls.append(("release", button))

    def left_trigger(self, value):
        self.calls.append(("lt", value))

    def right_trigger(self, value):
        self.calls.append(("rt", value))

    def left_joystick_float(self, x_value_float, y_value_float):
        self.calls.append(("ls", x_value_float, y_value_float))

    def right_joystick_float(self, x_value_float, y_value_float):
        self.calls.append(("rs", x_value_float, y_value_float))

    def update(self):
        self.calls.append(("update",))


def test_pad_names():
    assert is_pad("pad_a") and is_pad("PAD_RT") and not is_pad("a")


def test_buttons_triggers_and_routing_to_keyboard():
    pad = FakePad()
    keys = RecordingActionSink()
    sink = GamepadSink(keys, pad=pad, sleep=lambda s: None)
    assert {"stick", "key", "click"} <= sink_capabilities(sink)
    sink.press("pad_a", hold_ms=10)
    sink.press("pad_rt", hold_ms=10)
    sink.press("space", hold_ms=10)
    sink.click(5, 6)
    actions = [c for c in pad.calls if c[0] != "update"]
    assert actions == [("press", "XUSB_GAMEPAD_A"), ("release", "XUSB_GAMEPAD_A"), ("rt", 255), ("rt", 0)]
    assert [e["kind"] for e in keys.log] == ["key", "click"]


def test_stick_clamps_and_release_all_recentres():
    pad = FakePad()
    sink = GamepadSink(pad=pad, sleep=lambda s: None)
    sink.key_down("pad_lb")
    sink.stick("left", 2.0, -0.5)
    sink.release_all()
    calls = [c for c in pad.calls if c[0] != "update"]
    assert ("ls", 1.0, -0.5) in calls
    assert ("release", "XUSB_GAMEPAD_LEFT_SHOULDER") in calls
    assert calls[-2:] == [("ls", 0.0, 0.0), ("rs", 0.0, 0.0)]


def test_stick_step_in_a_script():
    pad = FakePad()
    sink = GamepadSink(pad=pad, sleep=lambda s: None)
    script = build_script(yaml.safe_load("""
steps:
  - press: pad_start
  - stick: left
    x: 0
    y: 1
    hold_ms: 50
"""))
    t = [0.0]
    res = ScriptRunner(script, source=lambda: None, sink=sink, clock=lambda: t[0],
                       sleep=lambda d: t.__setitem__(0, t[0] + d)).run()
    assert res.status == "completed", res.error
    sticks = [c for c in pad.calls if c[0] == "ls"]
    assert sticks[:2] == [("ls", 0.0, 1.0), ("ls", 0.0, 0.0)]
