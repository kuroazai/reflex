"""A virtual Xbox 360 controller, for games that only listen to a gamepad.

Uses vgamepad, which drives the ViGEmBus driver. Install both yourself:

    pip install vgamepad      # its installer offers ViGEmBus; accept it (needs admin)

Button names (use them anywhere a key name goes, e.g. press: pad_a):

    pad_a pad_b pad_x pad_y pad_lb pad_rb pad_back pad_start pad_guide
    pad_ls pad_rs (stick clicks)  pad_up pad_down pad_left pad_right (d-pad)
    pad_lt pad_rt (triggers, fully pressed)

Sticks: stick(side, x, y) with x, y in [-1, 1]; the script step is
`stick: {side: left, x: 0, y: 1}` with hold_ms (then it re-centres).

`GamepadSink(other_sink)` sends pad_* names to the controller and everything
else (keys, mouse) to the other sink, so one script can use both.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

PAD_BUTTONS = {
    "pad_a": "XUSB_GAMEPAD_A", "pad_b": "XUSB_GAMEPAD_B", "pad_x": "XUSB_GAMEPAD_X", "pad_y": "XUSB_GAMEPAD_Y",
    "pad_lb": "XUSB_GAMEPAD_LEFT_SHOULDER", "pad_rb": "XUSB_GAMEPAD_RIGHT_SHOULDER",
    "pad_back": "XUSB_GAMEPAD_BACK", "pad_start": "XUSB_GAMEPAD_START", "pad_guide": "XUSB_GAMEPAD_GUIDE",
    "pad_ls": "XUSB_GAMEPAD_LEFT_THUMB", "pad_rs": "XUSB_GAMEPAD_RIGHT_THUMB",
    "pad_up": "XUSB_GAMEPAD_DPAD_UP", "pad_down": "XUSB_GAMEPAD_DPAD_DOWN",
    "pad_left": "XUSB_GAMEPAD_DPAD_LEFT", "pad_right": "XUSB_GAMEPAD_DPAD_RIGHT",
}
PAD_TRIGGERS = {"pad_lt", "pad_rt"}


class GamepadUnavailable(RuntimeError):
    pass


def is_pad(name: str) -> bool:
    n = str(name).lower()
    return n in PAD_BUTTONS or n in PAD_TRIGGERS


class GamepadSink:
    def __init__(self, other: Any = None, *, sleep: Callable[[float], None] = time.sleep, pad: Any = None) -> None:
        if pad is None:
            try:
                import vgamepad  # noqa: PLC0415
            except ImportError as exc:
                raise GamepadUnavailable(
                    "gamepad output needs vgamepad and the ViGEmBus driver: pip install vgamepad "
                    "(accept the ViGEmBus installer it offers; it needs admin)"
                ) from exc
            try:
                pad = vgamepad.VX360Gamepad()
            except Exception as exc:  # noqa: BLE001
                raise GamepadUnavailable(f"could not create a virtual controller (is ViGEmBus installed?): {exc}") \
                    from exc
            self._vg = vgamepad
        else:
            import types

            self._vg = types.SimpleNamespace(XUSB_BUTTON=getattr(pad, "XUSB_BUTTON", None))
        self.pad = pad
        self.other = other
        self._sleep = sleep
        self._held: set[str] = set()
        caps = {"key", "key_down", "key_up", "chord", "wait", "stick"}
        if other is not None:
            from reflex.io import sink_capabilities

            caps |= set(sink_capabilities(other))
        self.capabilities = frozenset(caps)

    # ------------------------------------------------------------ plumbing

    def _button(self, name: str) -> Any:
        enum = getattr(self._vg, "XUSB_BUTTON", None)
        return getattr(enum, PAD_BUTTONS[name]) if enum is not None else PAD_BUTTONS[name]

    def _set(self, name: str, down: bool) -> bool:
        n = name.lower()
        if n in PAD_TRIGGERS:
            value = 255 if down else 0
            (self.pad.left_trigger if n == "pad_lt" else self.pad.right_trigger)(value=value)
        elif down:
            self.pad.press_button(button=self._button(n))
        else:
            self.pad.release_button(button=self._button(n))
        self.pad.update()
        if down:
            self._held.add(n)
        else:
            self._held.discard(n)
        return True

    def __getattr__(self, item: str) -> Any:
        # Mouse and text go to the other sink.
        other = self.__dict__.get("other")
        if other is not None and item in ("click", "move", "mouse_down", "mouse_up", "scroll", "type_text"):
            return getattr(other, item)
        raise AttributeError(item)

    def target_ok(self) -> bool:
        fn = getattr(self.other, "target_ok", None)
        return bool(fn()) if callable(fn) else True

    # ------------------------------------------------------------ keys and buttons

    def key_down(self, button: str) -> bool:
        if is_pad(button):
            return self._set(button, True)
        if self.other is None:
            raise GamepadUnavailable(f"{button!r} is not a pad button and there is no keyboard sink")
        return bool(self.other.key_down(button))

    def key_up(self, button: str) -> bool:
        if is_pad(button):
            return self._set(button, False)
        return bool(self.other.key_up(button)) if self.other is not None else False

    def press(self, button: str, *, hold_ms: int) -> bool:
        if not is_pad(button):
            if self.other is None:
                raise GamepadUnavailable(f"{button!r} is not a pad button and there is no keyboard sink")
            return bool(self.other.press(button, hold_ms=hold_ms))
        self._set(button, True)
        self._sleep(max(0, int(hold_ms)) / 1000.0)
        return self._set(button, False)

    def stick(self, side: str, x: float, y: float) -> bool:
        x = max(-1.0, min(1.0, float(x)))
        y = max(-1.0, min(1.0, float(y)))
        if side == "left":
            self.pad.left_joystick_float(x_value_float=x, y_value_float=y)
        elif side == "right":
            self.pad.right_joystick_float(x_value_float=x, y_value_float=y)
        else:
            raise ValueError("stick side is left or right")
        self.pad.update()
        return True

    def release_all(self) -> None:
        for name in list(self._held):
            self._set(name, False)
        self.stick("left", 0, 0)
        self.stick("right", 0, 0)
        if self.other is not None and hasattr(self.other, "release_all"):
            self.other.release_all()
