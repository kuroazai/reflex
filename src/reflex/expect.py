"""
Expectations — postconditions that prove a fire had its intended effect.

"Pressed A" is not evidence. "Pressed A and the dialog box closed within
600 ms" is. A spec declares `expect=`; after each fire the runtime checks
the expectation on subsequent frames and emits:

    reflex.verified    — satisfied within the window (with evidence)
    reflex.unverified  — window elapsed without it (the press did nothing
                         visible: collision, lost focus, wrong screen …)

While a verification is pending the spec does not fire again, so a reflex
can never queue presses against states it has not seen (no button spam).
N consecutive unverified fires trip the spec (see TriggerSpec).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from reflex.core import FrameView, Match, Predicate, Region


@dataclass(frozen=True)
class ExpectContext:
    pre_view: FrameView
    view: FrameView
    fire_match: Match
    spec_predicate: Predicate
    elapsed_ms: float
    scratch: dict[str, Any] = field(default_factory=dict)
    """Per-fire scratch space (lives as long as one pending verification)."""


Check = Callable[[ExpectContext], dict[str, Any] | None]


@dataclass(frozen=True)
class Expectation:
    """A named postcondition.

    check(ctx) returns an evidence dict when satisfied, else None.
    Checked on frames from `after_ms` until `within_ms` after the fire.
    """

    name: str
    check: Check
    within_ms: int = 800
    after_ms: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.within_ms <= 0:
            raise ValueError("expectation within_ms must be > 0")
        if not (0 <= self.after_ms < self.within_ms):
            raise ValueError("expectation after_ms must be in [0, within_ms)")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "within_ms": self.within_ms, "after_ms": self.after_ms, **self.params}


def gone(*, predicate: Predicate | None = None, within_ms: int = 800, after_ms: int = 0) -> Expectation:
    """The trigger's own predicate (or `predicate`) stops matching.

    Typical: dialog advance → box closes / text changes; click a button →
    button disappears.
    """

    def check(ctx: ExpectContext) -> dict[str, Any] | None:
        p = predicate or ctx.spec_predicate
        if p(ctx.view) is None:
            return {"gone_after_ms": round(ctx.elapsed_ms, 1)}
        return None

    return Expectation("gone", check, within_ms=within_ms, after_ms=after_ms,
                       params={"predicate": "custom" if predicate else "self"})


def appears(*, predicate: Predicate, within_ms: int = 800, after_ms: int = 0) -> Expectation:
    """`predicate` matches (e.g. the menu panel opened, scene == battle)."""

    def check(ctx: ExpectContext) -> dict[str, Any] | None:
        m = predicate(ctx.view)
        if m is not None:
            return {"appeared_after_ms": round(ctx.elapsed_ms, 1), "match": m.meta}
        return None

    return Expectation("appears", check, within_ms=within_ms, after_ms=after_ms)


def changed(
    *,
    region: Region | None = None,
    diff_threshold: float = 6.0,
    min_changed_fraction: float = 0.01,
    within_ms: int = 800,
    after_ms: int = 0,
) -> Expectation:
    """The region differs from the frame at fire time (camera scrolled,
    cursor moved, text advanced). The cheapest generic "something happened".
    """

    def check(ctx: ExpectContext) -> dict[str, Any] | None:
        a = ctx.view.frame_bgr
        b = ctx.pre_view.frame_bgr
        if a.shape != b.shape:
            return {"changed_fraction": 1.0, "reason": "frame_shape_changed"}
        r = (region or Region.whole(a)).clip_to(a)
        if r.w == 0 or r.h == 0:
            return None
        da = a[r.y : r.y + r.h, r.x : r.x + r.w]
        db = b[r.y : r.y + r.h, r.x : r.x + r.w]
        diff = cv2.absdiff(da, db).mean(axis=2)
        frac = float((diff > diff_threshold).mean())
        if frac >= min_changed_fraction:
            return {"changed_fraction": round(frac, 4), "changed_after_ms": round(ctx.elapsed_ms, 1)}
        return None

    params = {"region": list(region.as_xywh()) if region else None, "min_changed_fraction": min_changed_fraction}
    return Expectation("changed", check, within_ms=within_ms, after_ms=after_ms, params=params)


def all_of(expectations: Sequence[Expectation], *, within_ms: int | None = None) -> Expectation:
    parts = list(expectations)
    if not parts:
        raise ValueError("expect all_of needs at least one expectation")
    def check(ctx: ExpectContext) -> dict[str, Any] | None:
        # Each part only needs to hold once inside the window.
        seen: dict[int, dict[str, Any]] = ctx.scratch.setdefault(f"all_of:{id(parts)}", {})
        for i, e in enumerate(parts):
            if i in seen:
                continue
            ev = e.check(ctx)
            if ev is not None:
                seen[i] = ev
        if len(seen) == len(parts):
            return {"parts": [seen[i] for i in range(len(parts))]}
        return None

    w = within_ms if within_ms is not None else max(e.within_ms for e in parts)
    return Expectation("all_of", check, within_ms=w, params={"parts": [e.name for e in parts]})


def any_of(expectations: Sequence[Expectation], *, within_ms: int | None = None) -> Expectation:
    parts = list(expectations)
    if not parts:
        raise ValueError("expect any_of needs at least one expectation")

    def check(ctx: ExpectContext) -> dict[str, Any] | None:
        for e in parts:
            ev = e.check(ctx)
            if ev is not None:
                return {"by": e.name, **ev}
        return None

    w = within_ms if within_ms is not None else max(e.within_ms for e in parts)
    return Expectation("any_of", check, within_ms=w, params={"parts": [e.name for e in parts]})


def frame_diff_fraction(a: np.ndarray, b: np.ndarray, *, threshold: float = 6.0) -> float:
    if a.shape != b.shape:
        return 1.0
    return float((cv2.absdiff(a, b).mean(axis=2) > threshold).mean())
