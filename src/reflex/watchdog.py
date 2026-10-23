"""
Built-in watchdog reflexes — the recovery layer.

Per the deep research report, recovery should be a *first-class* macro
family, not planner improvisation. Watchdogs are fired by state, not by
pixel predicates: an external StuckSignal is wired in and the watchdog
runs a deterministic recovery ladder (release inputs → reverse → sidestep
→ rearm). This is the boring, load-bearing piece that prevents agents
from getting visibly stuck on stream.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from reflex.core import FrameView, Match, TriggerSpec

StuckSignal = Callable[[FrameView], float]
"""Returns a [0, 1] score indicating how stuck the agent appears to be."""


@dataclass
class _StuckLatch:
    threshold: float
    stable_for_ms: int
    armed_since_ts_ns: int | None = None
    triggered: bool = False

    def update(self, score: float, ts_ns: int) -> bool:
        if score < self.threshold:
            self.armed_since_ts_ns = None
            self.triggered = False
            return False
        if self.armed_since_ts_ns is None:
            self.armed_since_ts_ns = ts_ns
        elapsed_ms = (ts_ns - self.armed_since_ts_ns) // 1_000_000
        return elapsed_ms >= self.stable_for_ms


def anti_stuck(
    *,
    name: str = "anti_stuck",
    stuck_signal: StuckSignal,
    threshold: float = 0.85,
    stable_for_ms: int = 300,
    recovery_buttons: tuple[str, ...] = ("down", "left"),
    cooldown_s: float = 1.5,
    poll_hz: float = 8.0,
    priority: int = 95,
    hold_ms: int = 80,
    inter_press_ms: int = 60,
) -> TriggerSpec:
    """Fire a deterministic recovery sequence when the agent is stuck.

    `stuck_signal(view) -> [0, 1]` is computed in the caller's pipeline
    (typical sources: progress.py stall counter, optical flow magnitude,
    repeated identical episodic frames). Once the signal stays above
    `threshold` for `stable_for_ms`, the watchdog fires
    `recovery_buttons` in order — by default a small "back off and step
    sideways" sequence — then enters cooldown.

    `priority` defaults to 95 so the watchdog preempts ordinary
    interaction reflexes once the stuckness signal latches.
    """
    latch = _StuckLatch(threshold=float(threshold), stable_for_ms=int(stable_for_ms))

    def predicate(view: FrameView) -> Match | None:
        score = float(stuck_signal(view))
        if not latch.update(score, view.ts_ns):
            return None
        return Match(
            meta={
                "predicate": "anti_stuck",
                "score": round(score, 3),
                "stable_for_ms": int(stable_for_ms),
            }
        )

    def action_factory(_match: Match) -> tuple[str, ...]:
        return tuple(recovery_buttons)

    return TriggerSpec(
        name=name,
        predicate=predicate,
        action_factory=action_factory,
        timeout_s=86_400.0,
        poll_hz=poll_hz,
        max_fires=10_000,
        cooldown_s=cooldown_s,
        inter_press_ms=inter_press_ms,
        hold_ms=hold_ms,
        reverify_on_fire=False,
        priority=priority,
        extra={
            "kind": "anti_stuck",
            "threshold": float(threshold),
            "stable_for_ms": int(stable_for_ms),
            "recovery_buttons": list(recovery_buttons),
        },
    )


def deadman_release(
    *,
    name: str = "deadman_release",
    is_held: Callable[[], bool],
    release_buttons: tuple[str, ...] = ("up", "down", "left", "right"),
    poll_hz: float = 240.0,
    priority: int = 100,
) -> TriggerSpec:
    """Highest-priority release of held inputs when the deadman is engaged.

    Note: the runtime's own deadman gate already suppresses *new* fires.
    This is for the case where you want to actively flush held inputs
    on engage — e.g. release all four arrows so the avatar stops moving
    even if the OS has stuck-key state.
    """

    def predicate(view: FrameView) -> Match | None:
        if not is_held():
            return None
        return Match(meta={"predicate": "deadman_release"})

    def action_factory(_match: Match) -> tuple[str, ...]:
        return tuple(release_buttons)

    return TriggerSpec(
        name=name,
        predicate=predicate,
        action_factory=action_factory,
        timeout_s=86_400.0,
        poll_hz=poll_hz,
        max_fires=10_000,
        cooldown_s=0.10,
        inter_press_ms=0,
        hold_ms=5,
        reverify_on_fire=False,
        priority=priority,
        foreground_only=False,
        extra={"kind": "deadman_release", "release_buttons": list(release_buttons)},
    )
