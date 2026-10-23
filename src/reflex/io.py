from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    def grab(self) -> np.ndarray:
        ...


@runtime_checkable
class InputSink(Protocol):
    def press(self, button: str, *, hold_ms: int) -> bool:
        ...


LogCallback = Callable[..., None]


def noop_log(**kwargs: Any) -> None:
    return None


class CallableFrameSource:
    def __init__(self, fn: Callable[[], np.ndarray]) -> None:
        self._fn = fn

    def grab(self) -> np.ndarray:
        frame = self._fn()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise TypeError(
                f"FrameSource must return a (H, W, 3) ndarray, got shape={getattr(frame, 'shape', None)}"
            )
        return frame


class CallableInputSink:
    def __init__(self, fn: Callable[[str, int], bool]) -> None:
        self._fn = fn

    def press(self, button: str, *, hold_ms: int) -> bool:
        return bool(self._fn(button, hold_ms))


class RecordingInputSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def press(self, button: str, *, hold_ms: int) -> bool:
        self.calls.append((button, int(hold_ms)))
        return True
