from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class FrameView:
    frame_bgr: np.ndarray
    ts_ns: int

    @property
    def height(self) -> int:
        return int(self.frame_bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.frame_bgr.shape[1])


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    w: int
    h: int

    def as_xywh(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def clip_to(self, frame: np.ndarray) -> Region:
        H, W = int(frame.shape[0]), int(frame.shape[1])
        x = max(0, min(self.x, W))
        y = max(0, min(self.y, H))
        w = max(0, min(self.w, W - x))
        h = max(0, min(self.h, H - y))
        return Region(x, y, w, h)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.clip_to(frame).as_xywh()
        return frame[y : y + h, x : x + w]

    @classmethod
    def whole(cls, frame: np.ndarray) -> Region:
        return cls(0, 0, int(frame.shape[1]), int(frame.shape[0]))

    @classmethod
    def around(cls, cx: int, cy: int, *, w: int, h: int) -> Region:
        return cls(int(cx - w // 2), int(cy - h // 2), int(w), int(h))


@dataclass(frozen=True)
class Match:
    meta: dict[str, Any] = field(default_factory=dict)


Predicate = Callable[[FrameView], Match | None]
ActionFactory = Callable[[Match], Sequence[str]]


@dataclass(frozen=True)
class TriggerSpec:
    name: str
    predicate: Predicate
    action_factory: ActionFactory
    timeout_s: float
    poll_hz: float = 30.0
    max_fires: int = 1
    cooldown_s: float = 0.0
    inter_press_ms: int = 50
    hold_ms: int = 60
    reverify_on_fire: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0; reflex never runs unbounded")
        if self.poll_hz <= 0 or self.poll_hz > 240:
            raise ValueError("poll_hz must be in (0, 240]")
        if self.max_fires < 1:
            raise ValueError("max_fires must be >= 1")
        if self.inter_press_ms < 0:
            raise ValueError("inter_press_ms must be >= 0")
        if self.hold_ms < 0:
            raise ValueError("hold_ms must be >= 0")


OutcomeStatus = Literal["fired", "timed_out", "killed", "error"]


@dataclass(frozen=True)
class TriggerOutcome:
    spec_name: str
    status: OutcomeStatus
    fires: int
    last_match_meta: dict[str, Any] | None
    last_action_seq: tuple[str, ...]
    elapsed_s: float
    error_message: str | None = None
    fire_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)
