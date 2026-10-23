"""Find and describe top-level windows."""

from __future__ import annotations

import ctypes
import re
from ctypes import wintypes
from dataclasses import asdict, dataclass
from typing import Any

from reflex.backends.windows._win32 import (
    GA_ROOT,
    POINT,
    PROCESS_QUERY_LIMITED_INFORMATION,
    RECT,
    WNDENUMPROC,
    ensure_dpi_aware,
    kernel32,
    user32,
)


@dataclass(frozen=True)
class Window:
    hwnd: int
    title: str
    class_name: str
    pid: int
    process: str
    visible: bool
    minimized: bool
    rect: tuple[int, int, int, int]
    """Outer window rect on screen: (x, y, w, h)."""
    client: tuple[int, int, int, int]
    """Client area on screen: (x, y, w, h). Frame and click coordinates are relative to this."""

    def describe(self) -> dict[str, Any]:
        return asdict(self)


class WindowNotFound(LookupError):
    pass


def _text(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _pid(hwnd: int) -> int:
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _process_name(pid: int) -> str:
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value.replace("/", "\\").rsplit("\\", 1)[-1]
        return ""
    finally:
        kernel32.CloseHandle(h)


def client_rect_on_screen(hwnd: int) -> tuple[int, int, int, int]:
    r = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(r)):
        raise WindowNotFound(f"window {hwnd:#x} is gone")
    pt = POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return (int(pt.x), int(pt.y), int(r.right - r.left), int(r.bottom - r.top))


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        raise WindowNotFound(f"window {hwnd:#x} is gone")
    return (int(r.left), int(r.top), int(r.right - r.left), int(r.bottom - r.top))


def describe(hwnd: int) -> Window:
    ensure_dpi_aware()
    pid = _pid(hwnd)
    return Window(
        hwnd=int(hwnd),
        title=_text(hwnd),
        class_name=_class(hwnd),
        pid=pid,
        process=_process_name(pid),
        visible=bool(user32.IsWindowVisible(hwnd)),
        minimized=bool(user32.IsIconic(hwnd)),
        rect=window_rect(hwnd),
        client=client_rect_on_screen(hwnd),
    )


def _handles() -> list[int]:
    out: list[int] = []

    @WNDENUMPROC
    def cb(hwnd, _lparam):
        out.append(int(hwnd))
        return True

    user32.EnumWindows(cb, 0)
    return out


def list_windows(
    *,
    title: str | None = None,
    title_regex: str | None = None,
    class_name: str | None = None,
    process: str | None = None,
    include_hidden: bool = False,
) -> list[Window]:
    """Top-level windows, filtered. `title` and `process` match case-insensitive substrings."""
    ensure_dpi_aware()
    rx = re.compile(title_regex, re.I) if title_regex else None
    found: list[Window] = []
    for hwnd in _handles():
        if not include_hidden and not user32.IsWindowVisible(hwnd):
            continue
        t = _text(hwnd)
        if not include_hidden and not t:
            continue
        if title is not None and title.lower() not in t.lower():
            continue
        if rx is not None and not rx.search(t):
            continue
        if class_name is not None and _class(hwnd) != class_name:
            continue
        if process is not None and process.lower() not in _process_name(_pid(hwnd)).lower():
            continue
        try:
            found.append(describe(hwnd))
        except WindowNotFound:
            continue  # closed while we were looking
    return found


def find_window(
    *,
    title: str | None = None,
    title_regex: str | None = None,
    class_name: str | None = None,
    process: str | None = None,
    hwnd: int | None = None,
) -> Window:
    """Exactly one window, or a WindowNotFound naming what was searched for and what exists."""
    if hwnd is not None:
        if not user32.IsWindow(int(hwnd)):
            raise WindowNotFound(f"no window with hwnd {int(hwnd):#x}")
        return describe(int(hwnd))
    if not any((title, title_regex, class_name, process)):
        raise ValueError("find_window needs title, title_regex, class_name, process or hwnd")
    hits = list_windows(title=title, title_regex=title_regex, class_name=class_name, process=process)
    if not hits:
        query = {k: v for k, v in dict(title=title, title_regex=title_regex, class_name=class_name,
                                         process=process).items() if v}
        some = ", ".join(repr(w.title) for w in list_windows()[:15])
        raise WindowNotFound(f"no visible window matches {query}. Visible windows include: {some}")
    # Prefer the foreground window, then the largest, so "the game" beats its launcher.
    fg = int(user32.GetForegroundWindow() or 0)
    hits.sort(key=lambda w: (w.hwnd != fg, -(w.client[2] * w.client[3])))
    return hits[0]


def foreground() -> int:
    return int(user32.GetForegroundWindow() or 0)


def is_foreground(hwnd: int) -> bool:
    fg = foreground()
    if not fg:
        return False
    return fg == int(hwnd) or int(user32.GetAncestor(fg, GA_ROOT) or 0) == int(hwnd)


def focus(hwnd: int) -> bool:
    return bool(user32.SetForegroundWindow(int(hwnd)))
