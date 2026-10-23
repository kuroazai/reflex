"""A real Win32 window that paints known colours and records the input it gets.

Used by the test suite and by `reflex doctor` to prove, on the user's own
machine, that capture and both input paths work before pointing them at a game.

    with TestWindow(title="reflex test") as tw:
        frame = WindowSource(tw.hwnd).grab()     # left half red, right half (a child window) blue
        PostMessageSink(tw.hwnd).press("a", hold_ms=10)
        tw.wait_for(lambda ev: ev[0] == "keydown")
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from typing import Any

from reflex.backends.windows import _win32 as w

user32 = w.user32
gdi32 = w.gdi32
kernel32 = w.kernel32

WNDPROC = ctypes.WINFUNCTYPE(w.LRESULT, w.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

WS_OVERLAPPEDWINDOW = 0x00CF0000
WS_VISIBLE = 0x10000000
WS_CHILD = 0x40000000
WS_BORDER = 0x00800000
ES_MULTILINE = 0x0004
WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
CW_USEDEFAULT = -0x80000000
WM_DESTROY = 0x0002
WM_PAINT = 0x000F
WM_CLOSE = 0x0010
WM_ERASEBKGND = 0x0014
SW_SHOWNOACTIVATE = 4
SWP_NOACTIVATE = 0x0010
SWP_NOZORDER = 0x0004

RED_BGR = (40, 40, 220)
BLUE_BGR = (220, 90, 30)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", wintypes.HDC),
        ("fErase", wintypes.BOOL),
        ("rcPaint", w.RECT),
        ("fRestore", wintypes.BOOL),
        ("fIncUpdate", wintypes.BOOL),
        ("rgbReserved", ctypes.c_byte * 32),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", w.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", w.POINT),
        ("lPrivate", wintypes.DWORD),
    ]


w._sig(user32.RegisterClassExW, wintypes.ATOM, ctypes.POINTER(WNDCLASSEXW))
w._sig(user32.UnregisterClassW, wintypes.BOOL, wintypes.LPCWSTR, wintypes.HINSTANCE)
w._sig(user32.CreateWindowExW, w.HWND, wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HWND, wintypes.HMENU, wintypes.HINSTANCE,
       wintypes.LPVOID)
w._sig(user32.DefWindowProcW, w.LRESULT, w.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
w._sig(user32.DestroyWindow, wintypes.BOOL, w.HWND)
w._sig(user32.GetMessageW, wintypes.BOOL, ctypes.POINTER(MSG), w.HWND, wintypes.UINT, wintypes.UINT)
w._sig(user32.TranslateMessage, wintypes.BOOL, ctypes.POINTER(MSG))
w._sig(user32.DispatchMessageW, w.LRESULT, ctypes.POINTER(MSG))
w._sig(user32.PostQuitMessage, None, ctypes.c_int)
w._sig(user32.BeginPaint, wintypes.HDC, w.HWND, ctypes.POINTER(PAINTSTRUCT))
w._sig(user32.EndPaint, wintypes.BOOL, w.HWND, ctypes.POINTER(PAINTSTRUCT))
w._sig(user32.FillRect, ctypes.c_int, wintypes.HDC, ctypes.POINTER(w.RECT), wintypes.HBRUSH)
w._sig(user32.ShowWindow, wintypes.BOOL, w.HWND, ctypes.c_int)
w._sig(user32.SetWindowPos, wintypes.BOOL, w.HWND, w.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
       wintypes.UINT)
w._sig(user32.InvalidateRect, wintypes.BOOL, w.HWND, ctypes.POINTER(w.RECT), wintypes.BOOL)
w._sig(gdi32.CreateSolidBrush, wintypes.HBRUSH, wintypes.COLORREF)
w._sig(user32.SetFocus, w.HWND, w.HWND)
w._sig(user32.SendMessageW, w.LRESULT, w.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
w._sig(kernel32.GetModuleHandleW, wintypes.HMODULE, wintypes.LPCWSTR)

_RECORDED = {
    w.WM_KEYDOWN: "keydown", w.WM_KEYUP: "keyup", w.WM_SYSKEYDOWN: "syskeydown", w.WM_SYSKEYUP: "syskeyup",
    w.WM_CHAR: "char", w.WM_LBUTTONDOWN: "lbuttondown", w.WM_LBUTTONUP: "lbuttonup",
    w.WM_RBUTTONDOWN: "rbuttondown", w.WM_RBUTTONUP: "rbuttonup", w.WM_MBUTTONDOWN: "mbuttondown",
    w.WM_MBUTTONUP: "mbuttonup", w.WM_MOUSEWHEEL: "wheel", w.WM_MOUSEHWHEEL: "hwheel",
}
_MOUSE = {w.WM_LBUTTONDOWN, w.WM_LBUTTONUP, w.WM_RBUTTONDOWN, w.WM_RBUTTONUP, w.WM_MBUTTONDOWN, w.WM_MBUTTONUP}


def _colorref(bgr: tuple[int, int, int]) -> int:
    b, g, r = bgr
    return (b << 16) | (g << 8) | r


class TestWindow:
    """A top-level window (left half red) with a child window (right half blue).

    `colors` overrides the two colours (BGR), e.g. to tell a covering window apart.
    `edit=True` puts a standard Windows text box (EDIT control) in the right half
    instead, focused within its own thread, to test typing into a real control.

    `events` is a list of (kind, which, wparam, x, y): which is "parent" or
    "child", x/y are the receiving window's client coordinates for mouse
    messages. Nothing is ever activated, so the window never steals focus.
    """

    __test__ = False  # not a pytest class

    def __init__(self, *, title: str = "reflex test window", size: tuple[int, int] = (400, 300),
                 pos: tuple[int, int] = (120, 120),
                 colors: tuple[tuple[int, int, int], tuple[int, int, int]] = (RED_BGR, BLUE_BGR),
                 edit: bool = False) -> None:
        self.title = title
        self.colors = colors
        self.with_edit = edit
        self.edit = 0
        self.size = size
        self.pos = pos
        self.events: list[tuple[str, str, int, int, int]] = []
        self.hwnd = 0
        self.child = 0
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._class = f"reflex_test_{id(self):x}"

    # ------------------------------------------------------------ window procedure

    def _wndproc(self, hwnd, msg, wparam, lparam):
        try:
            if msg in _RECORDED:
                which = "child" if int(hwnd or 0) == self.child else "parent"
                x = y = 0
                if msg in _MOUSE:
                    x = ctypes.c_short(lparam & 0xFFFF).value
                    y = ctypes.c_short((lparam >> 16) & 0xFFFF).value
                with self._lock:
                    self.events.append((_RECORDED[msg], which, int(wparam), int(x), int(y)))
                return 0
            if msg == WM_ERASEBKGND:
                return 1
            if msg == WM_PAINT:
                ps = PAINTSTRUCT()
                hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
                r = w.RECT()
                user32.GetClientRect(hwnd, ctypes.byref(r))
                color = self.colors[1] if int(hwnd or 0) == self.child else self.colors[0]
                brush = gdi32.CreateSolidBrush(_colorref(color))
                user32.FillRect(hdc, ctypes.byref(r), brush)
                gdi32.DeleteObject(brush)
                user32.EndPaint(hwnd, ctypes.byref(ps))
                return 0
            if msg == WM_DESTROY and int(hwnd or 0) == self.hwnd:
                user32.PostQuitMessage(0)
                return 0
        except BaseException as exc:  # noqa: BLE001 — never let an exception cross into Win32
            self._error = exc
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    # ------------------------------------------------------------ lifecycle

    def _run(self) -> None:
        try:
            w.ensure_dpi_aware()
            hinst = kernel32.GetModuleHandleW(None)
            self._proc = WNDPROC(self._wndproc)  # keep a reference: Win32 holds only the raw pointer
            wc = WNDCLASSEXW()
            wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
            wc.lpfnWndProc = self._proc
            wc.hInstance = hinst
            wc.lpszClassName = self._class
            if not user32.RegisterClassExW(ctypes.byref(wc)):
                raise OSError(ctypes.get_last_error(), "RegisterClassExW failed")
            x, y = self.pos
            ww, hh = self.size
            hwnd = user32.CreateWindowExW(0, self._class, self.title, WS_OVERLAPPEDWINDOW, x, y, ww, hh,
                                          None, None, hinst, None)
            if not hwnd:
                raise OSError(ctypes.get_last_error(), "CreateWindowExW failed")
            self.hwnd = int(hwnd)
            r = w.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(r))
            cw, ch = r.right - r.left, r.bottom - r.top
            if self.with_edit:
                edit = user32.CreateWindowExW(0, "EDIT", "", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_MULTILINE,
                                              cw // 2, 0, cw - cw // 2, ch, hwnd, None, hinst, None)
                self.edit = int(edit or 0)
                user32.SetFocus(edit)
            else:
                child = user32.CreateWindowExW(0, self._class, "child", WS_CHILD | WS_VISIBLE, cw // 2, 0,
                                               cw - cw // 2, ch, hwnd, None, hinst, None)
                self.child = int(child or 0)
            user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
            user32.InvalidateRect(hwnd, None, True)
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        msg = MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        user32.UnregisterClassW(self._class, kernel32.GetModuleHandleW(None))

    def start(self) -> TestWindow:
        self._thread = threading.Thread(target=self._run, name="reflex-test-window", daemon=True)
        self._thread.start()
        if not self._ready.wait(5.0):
            raise TimeoutError("test window did not start")
        if self._error is not None:
            raise RuntimeError(f"test window failed: {self._error!r}")
        time.sleep(0.15)  # first WM_PAINT
        return self

    def close(self) -> None:
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)
        if self._thread is not None:
            self._thread.join(3.0)
        self.hwnd = 0

    def __enter__(self) -> TestWindow:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------ inspection

    def move(self, x: int, y: int) -> None:
        user32.SetWindowPos(self.hwnd, None, int(x), int(y), 0, 0, SWP_NOACTIVATE | SWP_NOZORDER | 0x0001)

    def edit_text(self) -> str:
        """The EDIT control's current text (edit=True windows)."""
        if not self.edit:
            raise RuntimeError("this test window has no edit control")
        n = int(user32.SendMessageW(self.edit, WM_GETTEXTLENGTH, 0, 0))
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.SendMessageW(self.edit, WM_GETTEXT, n + 1, ctypes.cast(buf, ctypes.c_void_p).value)
        return buf.value

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    def snapshot(self) -> list[tuple[str, str, int, int, int]]:
        with self._lock:
            return list(self.events)

    def wait_for(self, cond: Callable[[tuple[str, str, int, int, int]], bool], *, timeout_s: float = 2.0,
                 count: int = 1) -> list[tuple[str, str, int, int, int]]:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            hits = [e for e in self.snapshot() if cond(e)]
            if len(hits) >= count:
                return hits
            time.sleep(0.01)
        return [e for e in self.snapshot() if cond(e)]
