"""Input sinks: real input (SendInput) and background input (PostMessage).

Both take coordinates in the target window's client area, the same space
`WindowSource` frames are in. Both remember which keys and buttons they are
holding, and `release_all()` lets them go: the kill switch and run shutdown
call it so a stopped macro never leaves a key stuck down.

SendInputSink
    Synthesises real keyboard and mouse input, so it goes to whatever window
    has focus and moves the real cursor. Keys are sent as scan codes, which
    games reading DirectInput or raw input need. Use it when the target
    ignores posted messages.

PostMessageSink
    Posts window messages straight to the target window. It works while the
    window is in the background or covered, never moves your cursor, and never
    steals focus, so you can keep using the PC. Emulators, launchers and most
    desktop apps accept it; many games read raw input instead and ignore it.
    Try it first, and fall back to SendInputSink if nothing happens.

Neither sink hides itself from anything. reflex sends ordinary Windows input
and makes no attempt to get past anti-cheat systems.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable

from reflex.backends.windows import _win32 as w
from reflex.backends.windows.keys import KeyCode, key_lparam, resolve
from reflex.backends.windows.windows import Window, find_window, is_foreground

ALL_CAPS = frozenset({
    "key", "key_down", "key_up", "chord", "click", "wait", "move", "mouse_down", "mouse_up", "scroll", "type",
})

_BUTTON_FLAGS = {
    "left": (w.MOUSEEVENTF_LEFTDOWN, w.MOUSEEVENTF_LEFTUP),
    "right": (w.MOUSEEVENTF_RIGHTDOWN, w.MOUSEEVENTF_RIGHTUP),
    "middle": (w.MOUSEEVENTF_MIDDLEDOWN, w.MOUSEEVENTF_MIDDLEUP),
}
_BUTTON_MSGS = {
    "left": (w.WM_LBUTTONDOWN, w.WM_LBUTTONUP, w.MK_LBUTTON),
    "right": (w.WM_RBUTTONDOWN, w.WM_RBUTTONUP, w.MK_RBUTTON),
    "middle": (w.WM_MBUTTONDOWN, w.WM_MBUTTONUP, w.MK_MBUTTON),
}
_ALT_VKS = {0x12, 0xA4, 0xA5}


def _button(name: str) -> str:
    b = str(name).lower()
    if b not in _BUTTON_FLAGS:
        raise ValueError(f"mouse button must be left, right or middle, got {name!r}")
    return b


def _resolve_window(window: Window | int | None, find: dict) -> Window | None:
    if isinstance(window, Window):
        return window
    if window is not None:
        return find_window(hwnd=int(window))
    if find:
        return find_window(**find)
    return None


class _HeldState:
    def __init__(self) -> None:
        self.keys: dict[str, KeyCode] = {}
        self.buttons: set[str] = set()


class SendInputSink:
    """Real input via SendInput. Goes to the focused window.

    window   — the target. Coordinates are its client area, converted to the
               screen at send time so a moved window still gets hit. If
               `require_focus` is true (default) input is refused (returns
               False) while another window has focus, instead of typing into
               the wrong program.
    origin   — with no window: the screen point that frame (0, 0) maps to,
               e.g. `ScreenSource.origin`.
    """

    capabilities = ALL_CAPS

    def __init__(
        self,
        window: Window | int | None = None,
        *,
        origin: tuple[int, int] = (0, 0),
        require_focus: bool = True,
        use_scancodes: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        **find: object,
    ) -> None:
        w.ensure_dpi_aware()
        self.window = _resolve_window(window, find)
        self.origin = (int(origin[0]), int(origin[1]))
        self.require_focus = bool(require_focus)
        self.use_scancodes = bool(use_scancodes)
        self._sleep = sleep
        self._held = _HeldState()
        self.sent = 0

    # ------------------------------------------------------------ plumbing

    def target_ok(self) -> bool:
        """True when input would reach the target (always true with no window)."""
        if self.window is None or not self.require_focus:
            return True
        return is_foreground(self.window.hwnd)

    def _to_screen(self, x: int, y: int) -> tuple[int, int]:
        if self.window is not None:
            pt = w.POINT(int(x), int(y))
            w.user32.ClientToScreen(self.window.hwnd, ctypes.byref(pt))
            return (int(pt.x), int(pt.y))
        return (self.origin[0] + int(x), self.origin[1] + int(y))

    def _send(self, *inputs: w.INPUT) -> bool:
        arr = (w.INPUT * len(inputs))(*inputs)
        n = w.user32.SendInput(len(inputs), arr, ctypes.sizeof(w.INPUT))
        self.sent += int(n)
        return n == len(inputs)

    def _key_input(self, kc: KeyCode, *, up: bool) -> w.INPUT:
        inp = w.INPUT(type=w.INPUT_KEYBOARD)
        flags = w.KEYEVENTF_KEYUP if up else 0
        if kc.extended:
            flags |= w.KEYEVENTF_EXTENDEDKEY
        if self.use_scancodes and kc.scan:
            inp.ki.wVk = 0
            inp.ki.wScan = kc.scan
            flags |= w.KEYEVENTF_SCANCODE
        else:
            inp.ki.wVk = kc.vk
        inp.ki.dwFlags = flags
        return inp

    def _mouse_input(self, flags: int, data: int = 0) -> w.INPUT:
        inp = w.INPUT(type=w.INPUT_MOUSE)
        inp.mi.dwFlags = flags
        inp.mi.mouseData = ctypes.c_uint32(int(data)).value
        return inp

    # ------------------------------------------------------------ keys

    def key_down(self, button: str) -> bool:
        if not self.target_ok():
            return False
        kc = resolve(button)
        ok = self._send(self._key_input(kc, up=False))
        if ok:
            self._held.keys[button] = kc
        return ok

    def key_up(self, button: str) -> bool:
        kc = self._held.keys.pop(button, None) or resolve(button)
        return self._send(self._key_input(kc, up=True))

    def press(self, button: str, *, hold_ms: int) -> bool:
        if not self.key_down(button):
            return False
        self._sleep(max(0, int(hold_ms)) / 1000.0)
        return self.key_up(button)

    def type_text(self, text: str, *, interval_ms: int = 0) -> bool:
        if not self.target_ok():
            return False
        ok = True
        for ch in str(text):
            for unit in _utf16_units(ch):
                down = w.INPUT(type=w.INPUT_KEYBOARD)
                down.ki.wScan = unit
                down.ki.dwFlags = w.KEYEVENTF_UNICODE
                up = w.INPUT(type=w.INPUT_KEYBOARD)
                up.ki.wScan = unit
                up.ki.dwFlags = w.KEYEVENTF_UNICODE | w.KEYEVENTF_KEYUP
                ok = self._send(down, up) and ok
            if interval_ms:
                self._sleep(interval_ms / 1000.0)
        return ok

    # ------------------------------------------------------------ mouse

    def move(self, x: int, y: int) -> bool:
        if not self.target_ok():
            return False
        sx, sy = self._to_screen(x, y)
        return bool(w.user32.SetCursorPos(sx, sy))

    def mouse_down(self, x: int, y: int, *, button: str = "left") -> bool:
        b = _button(button)
        if not self.move(x, y):
            return False
        ok = self._send(self._mouse_input(_BUTTON_FLAGS[b][0]))
        if ok:
            self._held.buttons.add(b)
        return ok

    def mouse_up(self, x: int, y: int, *, button: str = "left") -> bool:
        b = _button(button)
        self.move(x, y)
        self._held.buttons.discard(b)
        return self._send(self._mouse_input(_BUTTON_FLAGS[b][1]))

    def click(self, x: int, y: int, *, button: str = "left") -> bool:
        b = _button(button)
        if not self.move(x, y):
            return False
        down, up = _BUTTON_FLAGS[b]
        return self._send(self._mouse_input(down), self._mouse_input(up))

    def scroll(self, x: int, y: int, *, clicks: int = 1, horizontal: bool = False) -> bool:
        if not self.move(x, y):
            return False
        flag = w.MOUSEEVENTF_HWHEEL if horizontal else w.MOUSEEVENTF_WHEEL
        return self._send(self._mouse_input(flag, int(clicks) * w.WHEEL_DELTA))

    # ------------------------------------------------------------ safety

    def release_all(self) -> None:
        for name in list(self._held.keys):
            self.key_up(name)
        for b in list(self._held.buttons):
            self._held.buttons.discard(b)
            self._send(self._mouse_input(_BUTTON_FLAGS[b][1]))


class PostMessageSink:
    """Background input: window messages posted straight to the target.

    Clicks go to the deepest child window under the point, with coordinates
    converted into that child's client area, which is what most UI toolkits
    expect. Keys go to `key_hwnd` if given, otherwise the top-level window.
    """

    capabilities = ALL_CAPS

    def __init__(
        self,
        window: Window | int | None = None,
        *,
        key_hwnd: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        **find: object,
    ) -> None:
        w.ensure_dpi_aware()
        resolved = _resolve_window(window, find)
        if resolved is None:
            raise ValueError("PostMessageSink needs a window (Window, hwnd, or title=/process= ...)")
        self.window = resolved
        self.hwnd = resolved.hwnd
        self.key_hwnd = int(key_hwnd) if key_hwnd else self.hwnd
        self._sleep = sleep
        self._held = _HeldState()
        self.posted = 0

    def target_ok(self) -> bool:
        return bool(w.user32.IsWindow(self.hwnd))

    def _post(self, hwnd: int, msg: int, wparam: int, lparam: int) -> bool:
        ok = bool(w.user32.PostMessageW(hwnd, msg, wparam, ctypes.c_ssize_t(lparam).value))
        self.posted += int(ok)
        return ok

    # ------------------------------------------------------------ keys

    def key_down(self, button: str) -> bool:
        kc = resolve(button)
        alt = kc.vk in _ALT_VKS
        msg = w.WM_SYSKEYDOWN if alt else w.WM_KEYDOWN
        ok = self._post(self.key_hwnd, msg, kc.vk, key_lparam(kc.scan, extended=kc.extended, up=False, alt=alt))
        if ok:
            self._held.keys[button] = kc
        return ok

    def key_up(self, button: str) -> bool:
        kc = self._held.keys.pop(button, None) or resolve(button)
        alt = kc.vk in _ALT_VKS
        msg = w.WM_SYSKEYUP if alt else w.WM_KEYUP
        return self._post(self.key_hwnd, msg, kc.vk, key_lparam(kc.scan, extended=kc.extended, up=True, alt=alt))

    def press(self, button: str, *, hold_ms: int) -> bool:
        if not self.key_down(button):
            return False
        self._sleep(max(0, int(hold_ms)) / 1000.0)
        return self.key_up(button)

    def type_text(self, text: str, *, interval_ms: int = 0) -> bool:
        ok = True
        for ch in str(text):
            for unit in _utf16_units(ch):
                ok = self._post(self.key_hwnd, w.WM_CHAR, unit, 1) and ok
            if interval_ms:
                self._sleep(interval_ms / 1000.0)
        return ok

    # ------------------------------------------------------------ mouse

    def target_at(self, x: int, y: int) -> tuple[int, int, int]:
        """(hwnd, x, y): the deepest visible child under a client point, in its own coordinates."""
        hwnd, px, py = self.hwnd, int(x), int(y)
        for _ in range(16):
            child = w.user32.ChildWindowFromPointEx(
                hwnd, w.POINT(px, py), w.CWP_SKIPINVISIBLE | w.CWP_SKIPDISABLED | w.CWP_SKIPTRANSPARENT
            )
            child = int(child or 0)
            if not child or child == hwnd:
                break
            pt = w.POINT(px, py)
            w.user32.MapWindowPoints(hwnd, child, ctypes.byref(pt), 1)
            hwnd, px, py = child, int(pt.x), int(pt.y)
        return hwnd, px, py

    def move(self, x: int, y: int) -> bool:
        hwnd, px, py = self.target_at(x, y)
        mask = 0
        for b in self._held.buttons:
            mask |= _BUTTON_MSGS[b][2]
        return self._post(hwnd, w.WM_MOUSEMOVE, mask, w.make_lparam(px, py))

    def mouse_down(self, x: int, y: int, *, button: str = "left") -> bool:
        b = _button(button)
        hwnd, px, py = self.target_at(x, y)
        down, _up, mk = _BUTTON_MSGS[b]
        self._post(hwnd, w.WM_MOUSEMOVE, 0, w.make_lparam(px, py))
        ok = self._post(hwnd, down, mk, w.make_lparam(px, py))
        if ok:
            self._held.buttons.add(b)
        return ok

    def mouse_up(self, x: int, y: int, *, button: str = "left") -> bool:
        b = _button(button)
        hwnd, px, py = self.target_at(x, y)
        self._held.buttons.discard(b)
        return self._post(hwnd, _BUTTON_MSGS[b][1], 0, w.make_lparam(px, py))

    def click(self, x: int, y: int, *, button: str = "left") -> bool:
        b = _button(button)
        hwnd, px, py = self.target_at(x, y)
        down, up, mk = _BUTTON_MSGS[b]
        lp = w.make_lparam(px, py)
        ok = self._post(hwnd, w.WM_MOUSEMOVE, 0, lp)
        ok = self._post(hwnd, down, mk, lp) and ok
        return self._post(hwnd, up, 0, lp) and ok

    def scroll(self, x: int, y: int, *, clicks: int = 1, horizontal: bool = False) -> bool:
        hwnd, _px, _py = self.target_at(x, y)
        pt = w.POINT(int(x), int(y))
        w.user32.ClientToScreen(self.hwnd, ctypes.byref(pt))  # wheel messages carry screen coordinates
        delta = (int(clicks) * w.WHEEL_DELTA) & 0xFFFF
        msg = w.WM_MOUSEHWHEEL if horizontal else w.WM_MOUSEWHEEL
        return self._post(hwnd, msg, delta << 16, w.make_lparam(pt.x, pt.y))

    # ------------------------------------------------------------ safety

    def release_all(self) -> None:
        for name in list(self._held.keys):
            self.key_up(name)
        for b in list(self._held.buttons):
            self._held.buttons.discard(b)
            self._post(self.hwnd, _BUTTON_MSGS[b][1], 0, 0)


def _utf16_units(ch: str) -> list[int]:
    data = ch.encode("utf-16-le")
    return [int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)]
