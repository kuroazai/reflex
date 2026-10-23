from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from reflex.actions import Action
    from reflex.expect import Expectation


@dataclass(frozen=True)
class FrameView:
    """A single observed frame plus its capture timestamp.

    Predicates are pure: they receive a FrameView and return a Match or None.
    The same view is shared across predicates within one tick so that
    expensive ops (HSV convert, edge detect) can be cached on demand.

    `extras` carries per-tick context filled by the runtime: the previous
    frame (`prev_frame_bgr`) and whatever the registered context providers
    return (e.g. `scene` from a game-state reader). Gates read it.
    """

    frame_bgr: np.ndarray
    ts_ns: int
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.frame_bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.frame_bgr.shape[1])


@dataclass(frozen=True)
class Match:
    """The output of a predicate when it fires."""

    meta: dict[str, Any] = field(default_factory=dict)


Predicate = Callable[[FrameView], Match | None]


@dataclass(frozen=True)
class Region:
    """Axis-aligned screen-space rectangle in pixels."""

    x: int
    y: int
    w: int
    h: int

    def as_xywh(self) -> tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.w), int(self.h))

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h))

    def crop(self, frame_bgr: np.ndarray) -> np.ndarray:
        h_frame, w_frame = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        x0 = max(0, int(self.x))
        y0 = max(0, int(self.y))
        x1 = min(w_frame, int(self.x + self.w))
        y1 = min(h_frame, int(self.y + self.h))
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=frame_bgr.dtype)
        return frame_bgr[y0:y1, x0:x1]

    def clip_to(self, frame_bgr: np.ndarray) -> Region:
        h_frame, w_frame = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        x0 = max(0, int(self.x))
        y0 = max(0, int(self.y))
        x1 = max(x0, min(w_frame, int(self.x + self.w)))
        y1 = max(y0, min(h_frame, int(self.y + self.h)))
        return Region(x=x0, y=y0, w=x1 - x0, h=y1 - y0)

    def contains_xy(self, x: int, y: int) -> bool:
        return self.x <= int(x) < self.x + self.w and self.y <= int(y) < self.y + self.h

    @classmethod
    def whole(cls, frame_bgr: np.ndarray) -> Region:
        return cls(0, 0, int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))

    @classmethod
    def around(cls, cx: int, cy: int, *, w: int, h: int) -> Region:
        return cls(int(cx - w // 2), int(cy - h // 2), int(w), int(h))


ActionItem = Union[str, "Action"]
"""A plain string is shorthand for `Key(name)`; see reflex.actions."""

ActionFactory = Callable[[Match], Sequence[ActionItem]]


@dataclass(frozen=True)
class TriggerSpec:
    """A reflex declaration.

    Wires a predicate that decides "is this scene fireable?" to an
    action_factory that produces the input sequence to dispatch on a hit.
    Timing parameters control polling rate, hold/inter-press pacing,
    single-shot vs repeat behaviour, cooldowns, and reverify semantics.

    Validation knobs:
      gates    — context predicates that must ALL match before the trigger
                 predicate is even evaluated (e.g. scene == battle).
      expect   — postcondition checked on frames after each fire; the
                 runtime reports verified / unverified.
      trip_after_unverified — N consecutive unverified fires disable the
                 spec (circuit breaker against blind button spam). 0 = never.

    timeout_s must be > 0: a reflex never runs unbounded (watchdogs use a day).
    """

    name: str
    predicate: Predicate
    action_factory: ActionFactory
    timeout_s: float = 30.0
    poll_hz: float = 30.0
    max_fires: int = 1
    cooldown_s: float = 0.0
    inter_press_ms: int = 50
    hold_ms: int = 60
    reverify_on_fire: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
    priority: int = 50
    require_consecutive_hits: int = 1
    miss_tolerance: int = 0
    foreground_only: bool = True
    gates: tuple[Predicate, ...] = ()
    expect: Expectation | None = None
    trip_after_unverified: int = 3

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TriggerSpec.name is required")
        if self.timeout_s <= 0:
            raise ValueError(f"{self.name}: timeout_s must be > 0; reflex never runs unbounded")
        if self.poll_hz <= 0 or self.poll_hz > 1000:
            raise ValueError(f"{self.name}: poll_hz must be in (0, 1000]")
        if self.max_fires < 1:
            raise ValueError(f"{self.name}: max_fires must be >= 1")
        if self.inter_press_ms < 0 or self.hold_ms < 0:
            raise ValueError(f"{self.name}: inter_press_ms / hold_ms must be >= 0")
        if self.cooldown_s < 0:
            raise ValueError(f"{self.name}: cooldown_s must be >= 0")
        if self.require_consecutive_hits < 1:
            raise ValueError(f"{self.name}: require_consecutive_hits must be >= 1")
        if self.trip_after_unverified < 0:
            raise ValueError(f"{self.name}: trip_after_unverified must be >= 0")

    def with_priority(self, priority: int) -> TriggerSpec:
        return replace(self, priority=int(priority))

    def with_overrides(self, **changes: Any) -> TriggerSpec:
        return replace(self, **changes)


OutcomeStatus = Literal["fired", "timed_out", "killed", "error", "tripped", "goal_reached", "guarded"]


@dataclass(frozen=True)
class TriggerOutcome:
    """Structured result handed back to the slow tier (LLM / planner)."""

    spec_name: str
    status: OutcomeStatus
    fires: int
    last_match_meta: dict[str, Any] | None
    last_action_seq: tuple[str, ...]
    elapsed_s: float
    error_message: str | None = None
    fire_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    verified: int = 0
    unverified: int = 0
