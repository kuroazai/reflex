"""Global hotkeys: the kill switch and pause toggle every run gets.

Polls key state on a background thread, so it works whichever window has
focus, including a full-screen game, and needs no message loop. A combo
fires once per press (on the press edge, not repeatedly while held).

    keys = HotkeyWatcher({"ctrl+alt+end": stop.set, "ctrl+alt+p": toggle_pause})
    keys.start()
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable

DEFAULT_STOP = "ctrl+alt+end"
DEFAULT_PAUSE = "ctrl+alt+home"

_MODS = {"ctrl": 0x11, "control": 0x11, "shift": 0x10, "alt": 0x12, "win": 0x5B}


def parse_combo(combo: str) -> tuple[int, ...]:
    """'ctrl+shift+q' -> virtual-key codes, modifiers first."""
    from reflex.backends.windows.keys import vk_for

    parts = [p.strip().lower() for p in str(combo).split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty hotkey {combo!r}")
    return tuple(_MODS[p] if p in _MODS else vk_for(p) for p in parts)


class HotkeyWatcher:
    def __init__(self, bindings: dict[str, Callable[[], None]], *, poll_hz: float = 60.0) -> None:
        self._bindings = {combo: (parse_combo(combo), fn) for combo, fn in bindings.items()} \
            if sys.platform == "win32" else {}
        self._dt = 1.0 / float(poll_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fired: list[str] = []

    @property
    def active(self) -> bool:
        return bool(self._bindings)

    def _down(self, vk: int) -> bool:
        from reflex.backends.windows._win32 import user32

        return bool(user32.GetAsyncKeyState(vk) & 0x8000)

    def _loop(self) -> None:
        held = {combo: True for combo in self._bindings}  # ignore combos already down at start
        while not self._stop.wait(self._dt):
            for combo, (vks, fn) in self._bindings.items():
                down = all(self._down(vk) for vk in vks)
                if down and not held[combo]:
                    self.fired.append(combo)
                    try:
                        fn()
                    except Exception:  # noqa: BLE001 — a bad callback must not kill the kill switch
                        pass
                held[combo] = down

    def start(self) -> HotkeyWatcher:
        if self._bindings and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="reflex-hotkeys", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
            self._thread = None

    def __enter__(self) -> HotkeyWatcher:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
