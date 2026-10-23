"""ctypes bindings for the slice of Win32 that capture and input need.

Every handle-returning function has an explicit restype: the ctypes default is
a C int, which silently truncates 64-bit handles.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

if sys.platform != "win32":  # pragma: no cover - imported only on Windows
    raise ImportError("reflex.backends.windows needs Windows")

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
ULONG_PTR = ctypes.c_size_t
HANDLE = wintypes.HANDLE
HWND = wintypes.HWND
HDC = wintypes.HDC
HBITMAP = wintypes.HBITMAP
HGDIOBJ = wintypes.HGDIOBJ

# --------------------------------------------------------------------------- constants

SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0
BI_RGB = 0
PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
WHEEL_DELTA = 120

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_MBUTTONUP = 0x0208
WM_MOUSEWHEEL = 0x020A
WM_MOUSEHWHEEL = 0x020E
MK_LBUTTON = 0x0001
MK_RBUTTON = 0x0002
MK_MBUTTON = 0x0010

MAPVK_VK_TO_VSC = 0
CWP_SKIPINVISIBLE = 0x0001
CWP_SKIPDISABLED = 0x0002
CWP_SKIPTRANSPARENT = 0x0004

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# --------------------------------------------------------------------------- structures


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, HWND, wintypes.LPARAM)

# --------------------------------------------------------------------------- signatures


def _sig(fn, restype, *argtypes):
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


_sig(user32.EnumWindows, wintypes.BOOL, WNDENUMPROC, wintypes.LPARAM)
_sig(user32.GetWindowTextLengthW, ctypes.c_int, HWND)
_sig(user32.GetWindowTextW, ctypes.c_int, HWND, wintypes.LPWSTR, ctypes.c_int)
_sig(user32.GetClassNameW, ctypes.c_int, HWND, wintypes.LPWSTR, ctypes.c_int)
_sig(user32.IsWindow, wintypes.BOOL, HWND)
_sig(user32.IsWindowVisible, wintypes.BOOL, HWND)
_sig(user32.IsIconic, wintypes.BOOL, HWND)
_sig(user32.GetWindowRect, wintypes.BOOL, HWND, ctypes.POINTER(RECT))
_sig(user32.GetClientRect, wintypes.BOOL, HWND, ctypes.POINTER(RECT))
_sig(user32.ClientToScreen, wintypes.BOOL, HWND, ctypes.POINTER(POINT))
_sig(user32.ScreenToClient, wintypes.BOOL, HWND, ctypes.POINTER(POINT))
_sig(user32.MapWindowPoints, ctypes.c_int, HWND, HWND, ctypes.POINTER(POINT), wintypes.UINT)
_sig(user32.ChildWindowFromPointEx, HWND, HWND, POINT, wintypes.UINT)
_sig(user32.GetWindowThreadProcessId, wintypes.DWORD, HWND, ctypes.POINTER(wintypes.DWORD))
_sig(user32.GetForegroundWindow, HWND)
_sig(user32.SetForegroundWindow, wintypes.BOOL, HWND)
_sig(user32.GetAncestor, HWND, HWND, wintypes.UINT)
_sig(user32.GetDC, HDC, HWND)
_sig(user32.GetWindowDC, HDC, HWND)
_sig(user32.ReleaseDC, ctypes.c_int, HWND, HDC)
_sig(user32.PrintWindow, wintypes.BOOL, HWND, HDC, wintypes.UINT)
_sig(user32.SendInput, wintypes.UINT, wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
_sig(user32.PostMessageW, wintypes.BOOL, HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
_sig(user32.MapVirtualKeyW, wintypes.UINT, wintypes.UINT, wintypes.UINT)
_sig(user32.VkKeyScanW, ctypes.c_short, wintypes.WCHAR)
_sig(user32.GetAsyncKeyState, ctypes.c_short, ctypes.c_int)
_sig(user32.GetSystemMetrics, ctypes.c_int, ctypes.c_int)
_sig(user32.SetCursorPos, wintypes.BOOL, ctypes.c_int, ctypes.c_int)
_sig(user32.GetCursorPos, wintypes.BOOL, ctypes.POINTER(POINT))

_sig(gdi32.CreateCompatibleDC, HDC, HDC)
_sig(gdi32.CreateDIBSection, HBITMAP, HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
     ctypes.POINTER(ctypes.c_void_p), HANDLE, wintypes.DWORD)
_sig(gdi32.SelectObject, HGDIOBJ, HDC, HGDIOBJ)
_sig(gdi32.BitBlt, wintypes.BOOL, HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
     HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD)
_sig(gdi32.DeleteObject, wintypes.BOOL, HGDIOBJ)
_sig(gdi32.DeleteDC, wintypes.BOOL, HDC)
_sig(gdi32.GdiFlush, wintypes.BOOL)

_sig(kernel32.OpenProcess, HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_sig(kernel32.CloseHandle, wintypes.BOOL, HANDLE)
_sig(kernel32.QueryFullProcessImageNameW, wintypes.BOOL, HANDLE, wintypes.DWORD, wintypes.LPWSTR,
     ctypes.POINTER(wintypes.DWORD))

GA_ROOT = 2


def make_lparam(x: int, y: int) -> int:
    """Pack client coordinates the way WM_*BUTTON* messages expect."""
    return ((int(y) & 0xFFFF) << 16) | (int(x) & 0xFFFF)


_dpi_done = False


def ensure_dpi_aware() -> None:
    """Make window rects and capture sizes physical pixels, not scaled ones.

    Without this, a 150%-scaled display reports a 1920-wide window as 1280 and
    every coordinate drifts. Called once by every source and sink.
    """
    global _dpi_done
    if _dpi_done:
        return
    _dpi_done = True
    try:
        # PER_MONITOR_AWARE_V2
        user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        user32.SetProcessDpiAwarenessContext.argtypes = [HANDLE]
        if user32.SetProcessDpiAwarenessContext(HANDLE(-4)):
            return
    except AttributeError:
        pass
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        user32.SetProcessDPIAware()
