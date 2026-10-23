"""
MacroEngine — run ONE trigger to completion and hand back a TriggerOutcome.

This is the "LLM hands off a reactive task, then resumes" API:

    outcome = MacroEngine(frame_source, input_sink, log=cb).run(spec)
    # outcome.status in {fired, timed_out, tripped, error}

It is a thin wrapper over ReflexRuntime so single-shot and scheduled use
share one implementation (verification, gates, capability checks, events).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from reflex.core import Predicate, TriggerOutcome, TriggerSpec
from reflex.io import (
    EventSink,
    FanoutEventSink,
    LogCallback,
    LogCallbackEventSink,
    coerce_frame_source,
    coerce_input_sink,
    noop_log,
)
from reflex.runtime import ReflexRuntime


class MacroEngine:
    def __init__(
        self,
        frame_source: Any,
        input_sink: Any,
        *,
        log: LogCallback = noop_log,
        events: EventSink | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        guards: dict[str, Predicate] | None = None,
        context_providers: dict[str, Callable] | None = None,
        foreground_ok: Callable[[], bool] | None = None,
    ) -> None:
        self._source = coerce_frame_source(frame_source)
        self._sink = coerce_input_sink(input_sink)
        self._log = log
        self._events = events
        self._clock = clock
        self._sleep = sleep
        self._guards = guards
        self._context = context_providers
        self._foreground_ok = foreground_ok

    def run(self, spec: TriggerSpec) -> TriggerOutcome:
        started = self._clock()
        sink = FanoutEventSink(LogCallbackEventSink(self._log), self._events) if self._events else LogCallbackEventSink(self._log)
        rt = ReflexRuntime(
            source=self._source,
            sink=self._sink,
            events=sink,
            clock=self._clock,
            sleep=self._sleep,
            guards=self._guards,
            context_providers=self._context,
            foreground_ok=self._foreground_ok,
        )
        self._log(kind="armed", spec=spec.name, timeout_s=spec.timeout_s, poll_hz=spec.poll_hz, max_fires=spec.max_fires)
        timeout = spec.timeout_s
        if spec.expect is not None:
            timeout += spec.expect.within_ms / 1000.0
        try:
            rt.register(spec)
            rt.run(timeout_s=timeout)
        except Exception as exc:  # noqa: BLE001
            self._log(kind="error", spec=spec.name, error=repr(exc))
            return self._outcome(rt, spec, "error", started, error=repr(exc))

        st = rt.stats()[spec.name]
        if st["tripped"]:
            status = "tripped"
        elif st["fires"] >= spec.max_fires:
            status = "fired"
        elif st["disabled"]:
            status = "error"
        else:
            status = "timed_out"
            self._log(kind="timed_out", spec=spec.name, fires=st["fires"])
        err = st["disabled_reason"] if status == "error" else None
        return self._outcome(rt, spec, status, started, error=err)

    def _outcome(self, rt: ReflexRuntime, spec: TriggerSpec, status: str, started: float, *, error: str | None = None) -> TriggerOutcome:
        st = rt.stats().get(spec.name, {})
        hist = rt.history(spec.name)
        last = hist[-1] if hist else None
        return TriggerOutcome(
            spec_name=spec.name,
            status=status,  # type: ignore[arg-type]
            fires=int(st.get("fires", 0)),
            last_match_meta=dict(last["match_meta"]) if last else None,
            last_action_seq=tuple(last["action_seq"]) if last else (),
            elapsed_s=self._clock() - started,
            error_message=error,
            fire_history=tuple(hist),
            verified=int(st.get("verified", 0)),
            unverified=int(st.get("unverified", 0)),
        )
