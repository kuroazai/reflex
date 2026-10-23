from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from reflex.core import FrameView, TriggerOutcome, TriggerSpec
from reflex.io import FrameSource, InputSink, LogCallback, noop_log


def _coerce_frame_source(source: Any) -> FrameSource:
    if hasattr(source, "grab"):
        return source
    if callable(source):
        from reflex.io import CallableFrameSource

        return CallableFrameSource(source)
    raise TypeError(f"frame_source must be FrameSource or callable, got {type(source)!r}")


def _coerce_input_sink(sink: Any) -> InputSink:
    if hasattr(sink, "press"):
        return sink
    if callable(sink):
        from reflex.io import CallableInputSink

        return CallableInputSink(sink)
    raise TypeError(f"input_sink must be InputSink or callable, got {type(sink)!r}")


class MacroEngine:
    def __init__(
        self,
        frame_source: Any,
        input_sink: Any,
        *,
        log: LogCallback = noop_log,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._frame_source = _coerce_frame_source(frame_source)
        self._input_sink = _coerce_input_sink(input_sink)
        self._log = log
        self._clock = clock
        self._sleep = sleep

    def run(self, spec: TriggerSpec) -> TriggerOutcome:
        started_at = self._clock()
        deadline = started_at + spec.timeout_s
        poll_dt = 1.0 / spec.poll_hz

        fires = 0
        last_fire_at = -math.inf
        last_match_meta: dict[str, Any] | None = None
        last_action_seq: tuple[str, ...] = ()
        history: list[dict[str, Any]] = []

        self._log(
            kind="armed",
            spec=spec.name,
            timeout_s=spec.timeout_s,
            poll_hz=spec.poll_hz,
            max_fires=spec.max_fires,
        )

        try:
            while self._clock() < deadline:
                view = self._capture()
                match = spec.predicate(view)

                if match is None:
                    self._sleep(poll_dt)
                    continue

                if (self._clock() - last_fire_at) < spec.cooldown_s:
                    self._sleep(poll_dt)
                    continue

                if spec.reverify_on_fire:
                    view2 = self._capture()
                    match2 = spec.predicate(view2)
                    if match2 is None:
                        self._log(kind="reverify_failed", spec=spec.name, candidate=match.meta)
                        self._sleep(poll_dt)
                        continue
                    match_to_fire = match2
                else:
                    match_to_fire = match

                action_seq = tuple(spec.action_factory(match_to_fire))

                self._log(
                    kind="firing",
                    spec=spec.name,
                    fire_index=fires + 1,
                    action_seq=action_seq,
                    match=match_to_fire.meta,
                )

                for i, button in enumerate(action_seq):
                    pressed = self._input_sink.press(button, hold_ms=spec.hold_ms)
                    self._log(
                        kind="pressed",
                        spec=spec.name,
                        button=button,
                        seq_index=i,
                        ok=pressed,
                    )
                    if i < len(action_seq) - 1 and spec.inter_press_ms > 0:
                        self._sleep(spec.inter_press_ms / 1000.0)

                fires += 1
                last_fire_at = self._clock()
                last_match_meta = dict(match_to_fire.meta)
                last_action_seq = action_seq
                history.append(
                    {
                        "fire_index": fires,
                        "ts_s_since_armed": last_fire_at - started_at,
                        "action_seq": action_seq,
                        "match_meta": last_match_meta,
                    }
                )

                self._log(kind="fired", spec=spec.name, fire_index=fires)

                if fires >= spec.max_fires:
                    return TriggerOutcome(
                        spec_name=spec.name,
                        status="fired",
                        fires=fires,
                        last_match_meta=last_match_meta,
                        last_action_seq=last_action_seq,
                        elapsed_s=self._clock() - started_at,
                        fire_history=tuple(history),
                    )

                self._sleep(poll_dt)
        except Exception as exc:
            self._log(kind="error", spec=spec.name, error=repr(exc))
            return TriggerOutcome(
                spec_name=spec.name,
                status="error",
                fires=fires,
                last_match_meta=last_match_meta,
                last_action_seq=last_action_seq,
                elapsed_s=self._clock() - started_at,
                error_message=repr(exc),
                fire_history=tuple(history),
            )

        self._log(kind="timed_out", spec=spec.name, fires=fires)
        return TriggerOutcome(
            spec_name=spec.name,
            status="timed_out",
            fires=fires,
            last_match_meta=last_match_meta,
            last_action_seq=last_action_seq,
            elapsed_s=self._clock() - started_at,
            fire_history=tuple(history),
        )

    def _capture(self) -> FrameView:
        frame = self._frame_source.grab()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise TypeError(
                f"FrameSource.grab returned invalid frame: shape={getattr(frame, 'shape', None)}"
            )
        return FrameView(frame_bgr=frame, ts_ns=time.time_ns())
