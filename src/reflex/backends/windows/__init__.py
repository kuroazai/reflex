"""Windows capture and input, in plain ctypes (no pywin32).

    from reflex.backends.windows import WindowSource, PostMessageSink, find_window

    win = find_window(title="Notepad")
    source = WindowSource(win)          # frames of its client area, even when covered
    sink = PostMessageSink(win)         # input without stealing focus
"""

from reflex.backends.windows.capture import (
    CaptureError,
    ScreenSource,
    WindowSource,
    benchmark,
    screen_capture_available,
    virtual_screen,
)
from reflex.backends.windows.input import PostMessageSink, SendInputSink
from reflex.backends.windows.keys import UnknownKey, known_keys
from reflex.backends.windows.windows import (
    Window,
    WindowNotFound,
    describe,
    find_window,
    focus,
    foreground,
    is_foreground,
    list_windows,
)

__all__ = [
    "CaptureError",
    "PostMessageSink",
    "ScreenSource",
    "SendInputSink",
    "UnknownKey",
    "Window",
    "WindowNotFound",
    "WindowSource",
    "benchmark",
    "describe",
    "find_window",
    "focus",
    "foreground",
    "is_foreground",
    "known_keys",
    "list_windows",
    "screen_capture_available",
    "virtual_screen",
]
