"""Frame sources: the whole screen, a screen region, or one window.

`WindowSource` captures a window even while other windows cover it, by asking
the window to render itself (PrintWindow with PW_RENDERFULLCONTENT). Frame
coordinates are client-area pixels, which is also what the input sinks take,
so a match location can be clicked without conversion.

Limits worth knowing:
  - A minimized window has nothing to render. WindowSource raises rather than
    returning a black frame that every predicate would quietly fail on.
  - A few exclusive-fullscreen or protected games render black to PrintWindow.
    method="auto" detects an all-black frame and falls back to copying the
    window's area from the screen, which needs the window visible.
"""

from __future__ import annotations

import ctypes
import time
from typing import Literal

import numpy as np

from reflex.backends.windows._win32 import (
    BI_RGB,
    BITMAPINFO,
    CAPTUREBLT,
    DIB_RGB_COLORS,
    PW_RENDERFULLCONTENT,
    SM_CXVIRTUALSCREEN,
    SM_CYVIRTUALSCREEN,
    SM_XVIRTUALSCREEN,
    SM_YVIRTUALSCREEN,
    SRCCOPY,
    ensure_dpi_aware,
    gdi32,
    user32,
)
from reflex.backends.windows.windows import (
    Window,
    WindowNotFound,
    client_rect_on_screen,
    find_window,
    window_rect,
)


class CaptureError(RuntimeError):
    pass


class _DibBuffer:
    """A reusable memory DC + 32-bit top-down DIB section, resized on demand."""

    def __init__(self) -> None:
        self.w = self.h = 0
        self.mem_dc = None
        self.bitmap = None
        self.old = None
        self.bits = ctypes.c_void_p()

    def ensure(self, ref_dc, w: int, h: int) -> None:
        if (w, h) == (self.w, self.h) and self.mem_dc:
            return
        self.release()
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(bmi.bmiHeader)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # negative: top-down rows
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        self.mem_dc = gdi32.CreateCompatibleDC(ref_dc)
        if not self.mem_dc:
            raise CaptureError("CreateCompatibleDC failed")
        self.bitmap = gdi32.CreateDIBSection(ref_dc, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(self.bits), None, 0)
        if not self.bitmap:
            gdi32.DeleteDC(self.mem_dc)
            self.mem_dc = None
            raise CaptureError(f"CreateDIBSection failed for {w}x{h}")
        self.old = gdi32.SelectObject(self.mem_dc, self.bitmap)
        self.w, self.h = w, h

    def to_bgr(self) -> np.ndarray:
        gdi32.GdiFlush()
        buf = (ctypes.c_ubyte * (self.w * self.h * 4)).from_address(self.bits.value)
        bgra = np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 4)
        return bgra[:, :, :3].copy()

    def release(self) -> None:
        if self.mem_dc:
            if self.old:
                gdi32.SelectObject(self.mem_dc, self.old)
            if self.bitmap:
                gdi32.DeleteObject(self.bitmap)
            gdi32.DeleteDC(self.mem_dc)
        self.mem_dc = self.bitmap = self.old = None
        self.w = self.h = 0

    def __del__(self) -> None:  # pragma: no cover - best effort at interpreter exit
        try:
            self.release()
        except Exception:  # noqa: BLE001
            pass


def _blit_from_screen(buf: _DibBuffer, x: int, y: int, w: int, h: int) -> np.ndarray:
    if w <= 0 or h <= 0:
        raise CaptureError(f"empty capture area {w}x{h}")
    screen_dc = user32.GetDC(None)
    try:
        buf.ensure(screen_dc, w, h)
        if not gdi32.BitBlt(buf.mem_dc, 0, 0, w, h, screen_dc, x, y, SRCCOPY | CAPTUREBLT):
            if not screen_capture_available():
                raise CaptureError(
                    "the screen is locked or showing a secure desktop (lock screen, UAC prompt), so it "
                    "cannot be copied. WindowSource with method='printwindow' keeps working while locked."
                )
            raise CaptureError(f"BitBlt from screen failed at {(x, y, w, h)}")
        return buf.to_bgr()
    finally:
        user32.ReleaseDC(None, screen_dc)


def screen_capture_available() -> bool:
    """False while the workstation is locked or a secure desktop is up."""
    desk = user32.OpenInputDesktop(0, False, 0x0100)  # DESKTOP_READOBJECTS
    if not desk:
        return False
    user32.CloseDesktop(desk)
    return True


def virtual_screen() -> tuple[int, int, int, int]:
    """Bounding box of all monitors: (x, y, w, h). x/y can be negative."""
    ensure_dpi_aware()
    return (
        int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN)),
        int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN)),
        int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)),
        int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)),
    )


class ScreenSource:
    """Capture a fixed screen rectangle (default: every monitor).

    Frame pixel (0, 0) is screen point `origin`. Pair it with
    `SendInputSink(origin=source.origin)` so clicks land where matches were found.
    """

    def __init__(self, region: tuple[int, int, int, int] | None = None) -> None:
        ensure_dpi_aware()
        self.region = tuple(int(v) for v in region) if region is not None else virtual_screen()
        self._buf = _DibBuffer()
        self.grabs = 0

    @property
    def origin(self) -> tuple[int, int]:
        return (self.region[0], self.region[1])

    def grab(self) -> np.ndarray:
        x, y, w, h = self.region
        self.grabs += 1
        return _blit_from_screen(self._buf, x, y, w, h)

    def close(self) -> None:
        self._buf.release()


CaptureMethod = Literal["auto", "printwindow", "screen"]


class WindowSource:
    """Capture one window's client area, including while it is covered.

    Pass a `Window`, an `hwnd`, or the same filters as `find_window`
    (`title=`, `process=`, ...). The window is resolved once; if it closes,
    `grab()` raises WindowNotFound instead of returning stale frames.
    """

    def __init__(
        self,
        window: Window | int | None = None,
        *,
        method: CaptureMethod = "auto",
        **find: object,
    ) -> None:
        ensure_dpi_aware()
        if isinstance(window, Window):
            self.window = window
        elif window is not None:
            self.window = find_window(hwnd=int(window))
        else:
            self.window = find_window(**find)  # type: ignore[arg-type]
        if method not in ("auto", "printwindow", "screen"):
            raise ValueError(f"method must be auto, printwindow or screen, got {method!r}")
        self.hwnd = self.window.hwnd
        self.method: CaptureMethod = method
        self.active_method: str = "screen" if method == "screen" else "printwindow"
        self._buf = _DibBuffer()
        self.grabs = 0
        self.fallbacks = 0

    def client_origin(self) -> tuple[int, int]:
        x, y, _w, _h = client_rect_on_screen(self.hwnd)
        return (x, y)

    def _check_alive(self) -> None:
        if not user32.IsWindow(self.hwnd):
            raise WindowNotFound(f"window {self.window.title!r} ({self.hwnd:#x}) has closed")
        if user32.IsIconic(self.hwnd):
            raise CaptureError(
                f"window {self.window.title!r} is minimized; restore it (it can stay behind other windows)"
            )

    def _printwindow(self) -> np.ndarray:
        wx, wy, ww, wh = window_rect(self.hwnd)
        cx, cy, cw, ch = client_rect_on_screen(self.hwnd)
        if cw <= 0 or ch <= 0:
            raise CaptureError(f"window {self.window.title!r} has an empty client area")
        win_dc = user32.GetWindowDC(self.hwnd)
        try:
            self._buf.ensure(win_dc, ww, wh)
            if not user32.PrintWindow(self.hwnd, self._buf.mem_dc, PW_RENDERFULLCONTENT):
                raise CaptureError("PrintWindow failed")
            frame = self._buf.to_bgr()
        finally:
            user32.ReleaseDC(self.hwnd, win_dc)
        ox, oy = cx - wx, cy - wy
        return frame[oy : oy + ch, ox : ox + cw].copy()

    def _screen(self) -> np.ndarray:
        cx, cy, cw, ch = client_rect_on_screen(self.hwnd)
        return _blit_from_screen(self._buf, cx, cy, cw, ch)

    def grab(self) -> np.ndarray:
        self._check_alive()
        self.grabs += 1
        if self.active_method == "screen":
            return self._screen()
        frame = self._printwindow()
        if self.method == "auto" and int(frame.max(initial=0)) <= 3:
            # Rendered black: this window does not support PrintWindow.
            self.active_method = "screen"
            self.fallbacks += 1
            return self._screen()
        return frame

    def close(self) -> None:
        self._buf.release()


def benchmark(source: object, *, seconds: float = 2.0) -> dict[str, float]:
    """Frames per second and per-grab latency for any source with grab()."""
    times: list[float] = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        source.grab()  # type: ignore[attr-defined]
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    n = len(times)
    return {
        "grabs": n,
        "fps": round(n / seconds, 1),
        "ms_p50": round(times[n // 2], 2) if n else 0.0,
        "ms_p95": round(times[min(n - 1, int(n * 0.95))], 2) if n else 0.0,
    }
