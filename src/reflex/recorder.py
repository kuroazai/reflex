"""Record your own keyboard and mouse into a script.

    reflex record --title "My Game" -o farm.yaml --images

Low-level Windows hooks see every key and click. Only input aimed at the
target window is kept: clicks inside its client area, keys while it has
focus. Input that reflex itself sends is flagged by Windows as injected and
ignored, so a recording never captures its own playback. The stop hotkey
(ctrl+alt+end) ends the recording and is left out of it.

The result is an ordinary script: presses, typing (runs of letters become one
`type:` step), clicks, drags, scrolls and the pauses between them. With
images on, each click also saves a small crop of what was under the pointer
and clicks that image instead of a fixed pixel, so the script still works
after the window moves or the layout shifts a little.
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable, Sequence
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
LLKHF_INJECTED = 0x10
LLMHF_INJECTED = 0x01
WM_QUIT = 0x0012

_MOUSE_KIND = {
    0x0201: ("down", "left"), 0x0202: ("up", "left"),
    0x0204: ("down", "right"), 0x0205: ("up", "right"),
    0x0207: ("down", "middle"), 0x0208: ("up", "middle"),
}
_WHEEL = 0x020A
_KEY_DOWN = (0x0100, 0x0104)
_KEY_UP = (0x0101, 0x0105)
_MODIFIERS = {"shift", "lshift", "rshift", "ctrl", "lctrl", "rctrl", "alt", "lalt", "ralt", "lwin", "rwin"}
# US layout shifted punctuation, so shift+1 records as "!" inside typed text.
_SHIFTED = dict(zip("1234567890-=[];'\\,./`", "!@#$%^&*()_+{}:\"|<>?~", strict=True))


@dataclass(frozen=True)
class RawEvent:
    t: float
    """Seconds since recording started."""
    kind: str
    """key_down | key_up | mouse_down | mouse_up | wheel"""
    key: str = ""
    button: str = ""
    x: int = 0
    y: int = 0
    """Client-area coordinates of the target window (mouse events)."""
    delta: int = 0
    """Wheel notches, positive = up."""
    char: str = ""
    """key_down: the character this key typed under the real keyboard layout and shift state."""


# --------------------------------------------------------------------------- compile


def _key_name(vk: int) -> str:
    from reflex.backends.windows.keys import _VK

    preferred = {0x0D: "enter", 0x1B: "esc", 0x10: "shift", 0x11: "ctrl", 0x12: "alt", 0x2E: "delete",
                 0x5B: "lwin", 0x5D: "apps"}
    if vk in preferred:
        return preferred[vk]
    for name, code in _VK.items():
        if code == vk and not name.startswith("numpad"):
            return name
    return f"vk:{vk}"


def _printable(key: str) -> str | None:
    if len(key) == 1 and (key.isalnum() or key in ";=,-./`[]\\'"):
        return key
    if key == "space":
        return " "
    return None


def compile_events(
    events: Sequence[RawEvent],
    *,
    min_wait_ms: int = 60,
    drag_px: int = 8,
    images: dict[int, str] | None = None,
    name: str = "recorded",
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn raw events into a script mapping (ready for yaml.safe_dump).

    images maps the index of a mouse_down event to a saved crop; that click
    becomes `click: {image: ...}` with the recorded point kept as a comment.
    """
    steps: list[dict[str, Any]] = []
    held: dict[str, RawEvent] = {}
    mods: set[str] = set()
    last_t = 0.0
    typing: list[str] = []
    pending_down: dict[str, tuple[int, RawEvent]] = {}
    used_mods: set[str] = set()
    last_click_t = -1.0

    def flush_typing() -> None:
        if typing:
            steps.append({"type": "".join(typing)})
            typing.clear()

    def wait_until(t: float) -> None:
        nonlocal last_t
        gap_ms = int(round((t - last_t) * 1000))
        if gap_ms >= min_wait_ms:
            flush_typing()
            steps.append({"wait": gap_ms})
        last_t = t

    for i, ev in enumerate(events):
        if ev.kind == "key_down":
            if ev.key in held:
                continue  # auto-repeat
            wait_until(ev.t)
            held[ev.key] = ev
            if ev.key in _MODIFIERS:
                mods.add(ev.key)
        elif ev.kind == "key_up":
            down = held.pop(ev.key, None)
            if down is None:
                continue
            if ev.key in _MODIFIERS:
                mods.discard(ev.key)
                if ev.key not in used_mods:  # pressed and released on its own
                    flush_typing()
                    steps.append({"press": ev.key})
                    last_t = ev.t
                used_mods.discard(ev.key)
                continue
            hold_ms = int(round((ev.t - down.t) * 1000))
            active = sorted(m for m in mods if m in held)
            ch = _printable(ev.key)
            if down.char and set(active) <= {"shift"} and hold_ms < 250:
                used_mods.update(active)
                typing.append(down.char)
            elif active == ["shift"] and ch is not None and hold_ms < 250:
                used_mods.add("shift")
                typing.append(_SHIFTED.get(ch, ch.upper()))
            elif active:
                flush_typing()
                used_mods.update(active)
                steps.append({"chord": [*active, ev.key]})
            elif ch is not None and hold_ms < 250:
                typing.append(ch)
            elif hold_ms >= 250:
                flush_typing()
                steps.append({"press": ev.key, "hold_ms": hold_ms})
            else:
                flush_typing()
                steps.append({"press": ev.key})
            last_t = ev.t
        elif ev.kind == "mouse_down":
            wait_until(ev.t)
            flush_typing()
            pending_down[ev.button] = (i, ev)
        elif ev.kind == "mouse_up":
            got = pending_down.pop(ev.button, None)
            if got is None:
                continue
            idx, down = got
            moved = abs(ev.x - down.x) > drag_px or abs(ev.y - down.y) > drag_px
            if moved:
                steps.append({"drag": [down.x, down.y], "to": [ev.x, ev.y], "button": down.button,
                              "hold_ms": int(round((ev.t - down.t) * 1000))})
            else:
                verb = "right_click" if down.button == "right" else "click"
                if images and idx in images:
                    st: dict[str, Any] = {verb: {"image": images[idx]}, "comment": f"recorded at {down.x},{down.y}"}
                else:
                    st = {verb: [down.x, down.y]}
                if down.button == "middle":
                    st["button"] = "middle"
                # Two quick clicks on the same spot are a double click.
                prev = steps[-1] if steps else None
                gap_wait = None
                if prev is not None and "wait" in prev and len(steps) >= 2:
                    gap_wait, prev = prev["wait"], steps[-2]
                if (verb == "click" and prev is not None and prev.get("click") == st["click"]
                        and (gap_wait is None or gap_wait < 300) and ev.t - last_click_t < 0.5):
                    if gap_wait is not None:
                        steps.pop()
                    steps[-1] = {"double_click": st["click"], **({"comment": st["comment"]} if "comment" in st else {})}
                    last_click_t = -1.0
                else:
                    steps.append(st)
                    last_click_t = ev.t
            last_t = ev.t
        elif ev.kind == "wheel":
            wait_until(ev.t)
            flush_typing()
            if steps and "scroll" in steps[-1] and steps[-1]["at"] == [ev.x, ev.y]:
                steps[-1]["scroll"] += ev.delta
            else:
                steps.append({"scroll": ev.delta, "at": [ev.x, ev.y]})
    flush_typing()
    for key in sorted(held):
        steps.append({"key_up": key, "comment": "still held when recording stopped"})
    script: dict[str, Any] = {"name": name}
    if target:
        script["target"] = target
    script["steps"] = steps or [{"log": "nothing was recorded"}]
    return script


# --------------------------------------------------------------------------- hooks


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", wintypes.POINT), ("mouseData", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class Recorder:
    """Collects RawEvents aimed at one window until stop() (or the stop hotkey).

    on_click(index, x, y) is called on each mouse-down inside the window,
    before the click lands, so the caller can grab a frame of what was there.
    """

    def __init__(
        self,
        window: Any,
        *,
        stop_combo: Sequence[str] = ("ctrl", "alt", "end"),
        record_injected: bool = False,
        on_click: Callable[[int, int, int], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        from reflex.backends.windows import _win32 as w

        self._w = w
        self.window = window
        self.hwnd = int(window.hwnd) if window is not None else 0
        self.record_injected = record_injected
        self.on_click = on_click
        self.on_stop = on_stop
        self.events: list[RawEvent] = []
        self._stop_keys = set(stop_combo)
        self._down: set[str] = set()
        self._t0 = 0.0
        self._thread: threading.Thread | None = None
        self._tid = 0
        self._ready = threading.Event()
        self.stopped = threading.Event()
        self._lock = threading.Lock()
        self.error: BaseException | None = None

    # ------------------------------------------------------------ filters

    def _in_target(self) -> bool:
        if not self.hwnd:
            return True
        from reflex.backends.windows import is_foreground

        return is_foreground(self.hwnd)

    def _client(self, sx: int, sy: int) -> tuple[int, int, bool]:
        if not self.hwnd:
            return sx, sy, True
        from reflex.backends.windows.windows import client_rect_on_screen

        x, y, cw, ch = client_rect_on_screen(self.hwnd)
        cx, cy = sx - x, sy - y
        return cx, cy, (0 <= cx < cw and 0 <= cy < ch)

    def _char(self, vk: int, scan: int) -> str:
        """The printable character for this key under the active layout, or ''."""
        user32 = self._w.user32
        state = (ctypes.c_ubyte * 256)()
        if self._down & {"shift"}:
            state[0x10] = 0x80
        if user32.GetKeyState(0x14) & 1:  # caps lock on
            state[0x14] = 0x01
        if self._down & {"ctrl", "alt"}:
            return ""
        buf = ctypes.create_unicode_buffer(8)
        layout = user32.GetKeyboardLayout(user32.GetWindowThreadProcessId(self.hwnd, None) if self.hwnd else 0)
        # flag 0x4: do not change the keyboard state (keeps dead keys working for the user)
        n = user32.ToUnicodeEx(vk, scan, state, buf, 8, 0x4, layout)
        text = buf.value[:n] if n > 0 else ""
        return text if len(text) == 1 and text.isprintable() else ""

    def _add(self, ev: RawEvent) -> None:
        with self._lock:
            self.events.append(ev)

    # ------------------------------------------------------------ hook procs

    def _kb(self, n_code: int, w_param: int, l_param: int) -> int:
        try:
            if n_code == 0:
                info = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                injected = bool(info.flags & LLKHF_INJECTED)
                key = _key_name(int(info.vkCode))
                for alias, canon in (("lctrl", "ctrl"), ("rctrl", "ctrl"), ("lalt", "alt"), ("ralt", "alt"),
                                     ("lshift", "shift"), ("rshift", "shift")):
                    if key == alias:
                        key = canon
                down = w_param in _KEY_DOWN
                if not injected:
                    if down:
                        self._down.add(key)
                    else:
                        self._down.discard(key)
                if not injected and down and self._stop_keys <= self._down:
                    self._request_stop()
                elif (self.record_injected or not injected) and self._in_target() and key not in self._stop_keys:
                    char = self._char(int(info.vkCode), int(info.scanCode)) if down else ""
                    self._add(RawEvent(time.perf_counter() - self._t0, "key_down" if down else "key_up", key=key,
                                       char=char))
        except BaseException as exc:  # noqa: BLE001 — never raise into the OS
            self.error = exc
        return self._w.user32.CallNextHookEx(None, n_code, w_param, l_param)

    def _mouse(self, n_code: int, w_param: int, l_param: int) -> int:
        try:
            if n_code == 0 and (w_param in _MOUSE_KIND or w_param == _WHEEL):
                info = ctypes.cast(l_param, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                if self.record_injected or not (info.flags & LLMHF_INJECTED):
                    x, y, inside = self._client(int(info.pt.x), int(info.pt.y))
                    if inside:
                        t = time.perf_counter() - self._t0
                        if w_param == _WHEEL:
                            delta = ctypes.c_short((info.mouseData >> 16) & 0xFFFF).value // 120
                            self._add(RawEvent(t, "wheel", x=x, y=y, delta=delta))
                        else:
                            kind, button = _MOUSE_KIND[w_param]
                            if kind == "down" and self.on_click is not None:
                                self.on_click(len(self.events), x, y)
                            self._add(RawEvent(t, f"mouse_{kind}", button=button, x=x, y=y))
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
        return self._w.user32.CallNextHookEx(None, n_code, w_param, l_param)

    # ------------------------------------------------------------ lifecycle

    def _run(self) -> None:
        w = self._w
        user32, kernel32 = w.user32, w.kernel32
        proto = ctypes.WINFUNCTYPE(w.LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        w._sig(user32.SetWindowsHookExW, wintypes.HHOOK, ctypes.c_int, proto, wintypes.HINSTANCE, wintypes.DWORD)
        w._sig(user32.CallNextHookEx, w.LRESULT, wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        w._sig(user32.UnhookWindowsHookEx, wintypes.BOOL, wintypes.HHOOK)
        w._sig(user32.ToUnicodeEx, ctypes.c_int, wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_ubyte),
               wintypes.LPWSTR, ctypes.c_int, wintypes.UINT, wintypes.HKL)
        w._sig(user32.GetKeyboardLayout, wintypes.HKL, wintypes.DWORD)
        w._sig(user32.GetKeyState, ctypes.c_short, ctypes.c_int)
        w._sig(user32.PostThreadMessageW, wintypes.BOOL, wintypes.DWORD, wintypes.UINT, wintypes.WPARAM,
               wintypes.LPARAM)
        w._sig(kernel32.GetModuleHandleW, wintypes.HMODULE, wintypes.LPCWSTR)
        w._sig(kernel32.GetCurrentThreadId, wintypes.DWORD)
        w._sig(user32.GetMessageW, wintypes.BOOL, ctypes.c_void_p, w.HWND, wintypes.UINT, wintypes.UINT)
        self._kb_proc = proto(self._kb)  # keep references alive for the hook's lifetime
        self._ms_proc = proto(self._mouse)
        hinst = kernel32.GetModuleHandleW(None)
        self._tid = int(kernel32.GetCurrentThreadId())
        kb = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._kb_proc, hinst, 0)
        ms = user32.SetWindowsHookExW(WH_MOUSE_LL, self._ms_proc, hinst, 0)
        if not kb or not ms:
            self.error = OSError(ctypes.get_last_error(), "SetWindowsHookExW failed")
            self._ready.set()
            return
        self._t0 = time.perf_counter()
        self._ready.set()
        msg = (ctypes.c_byte * 64)()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                pass
        finally:
            user32.UnhookWindowsHookEx(kb)
            user32.UnhookWindowsHookEx(ms)
            self.stopped.set()

    def start(self) -> Recorder:
        self._thread = threading.Thread(target=self._run, name="reflex-recorder", daemon=True)
        self._thread.start()
        self._ready.wait(5.0)
        if self.error is not None:
            raise RuntimeError(f"recorder failed to start: {self.error!r}")
        return self

    def _request_stop(self) -> None:
        if self.on_stop is not None:
            self.on_stop()
        self._w.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)

    def stop(self) -> list[RawEvent]:
        if self._tid and not self.stopped.is_set():
            self._w.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(3.0)
        with self._lock:
            return list(self.events)

    def wait(self, timeout_s: float | None = None) -> bool:
        return self.stopped.wait(timeout_s)


def crop_around(frame: np.ndarray, x: int, y: int, size: int = 48) -> np.ndarray:
    h, w = frame.shape[:2]
    half = size // 2
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(w, x + half), min(h, y + half)
    return frame[y0:y1, x0:x1].copy()


def record_to_file(
    window: Any,
    out_path: str | Path,
    *,
    images: bool = False,
    max_s: float = 3600.0,
    crop_px: int = 48,
    min_wait_ms: int = 60,
    announce: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Record until the stop hotkey (or max_s), then write the script YAML."""
    import cv2
    import yaml

    from reflex.backends.windows import WindowSource

    out = Path(out_path)
    img_dir = out.with_suffix("")
    img_dir = img_dir.parent / f"{img_dir.name}_images"
    saved: dict[int, str] = {}
    source = WindowSource(window) if images else None
    lock = threading.Lock()

    def on_click(index: int, x: int, y: int) -> None:
        if source is None:
            return
        try:
            frame = source.grab()
        except Exception:  # noqa: BLE001 — a missed crop just means a plain coordinate click
            return
        crop = crop_around(frame, x, y, crop_px)
        if crop.size == 0 or float(crop.std()) < 4.0:
            return  # flat area: a template would match anywhere, so keep the coordinate
        img_dir.mkdir(parents=True, exist_ok=True)
        p = img_dir / f"click_{len(saved) + 1:03d}.png"
        cv2.imwrite(str(p), crop)
        with lock:
            saved[index] = p.relative_to(out.parent).as_posix()

    rec = Recorder(window, on_click=on_click).start()
    announce(f"recording input to {window.title!r}. Press ctrl+alt+end to stop.")
    rec.wait(max_s)
    events = rec.stop()
    target = {"window": {"title": window.title}, "input": "postmessage"}
    script = compile_events(events, images=saved, name=out.stem, target=target, min_wait_ms=min_wait_ms)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Recorded by reflex on {time.strftime('%Y-%m-%d %H:%M')}: {len(events)} input events.\n"
              "# Edit freely; check it with: reflex validate " + out.name + "\n")
    out.write_text(header + yaml.safe_dump(script, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return script
