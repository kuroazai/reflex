"""
ReflexRuntime — the fast-tier scheduler.

Each tick:

  1. deadman / lockout gates
  2. grab a frame; fill view.extras (previous frame + context providers)
  3. settle pending verifications (expect) → verified / unverified / tripped
  4. global guards — any hit suppresses ALL firing (fail closed on error)
  5. evaluate eligible specs: gates → predicate → debounce → cooldown
  6. fire the highest-priority candidate (reverified on a fresh frame),
     after checking the sink can perform every action in the sequence
  7. arm its expectation; emit events; accumulate timing stats

Design notes:

- Single-threaded and deterministic given (source, clock, sleep). The
  replay harness drives it with a virtual clock + recorded frames.
- The runtime never calls the LLM. The slow tier registers specs and reads
  outcomes (`run()` → RunResult, or the event stream).
- Every event of consequence goes to the EventSink. Kinds:
    reflex.registered / unregistered / fired / verified / unverified /
    tripped / guarded / guard_cleared / timeout / unsupported_action /
    reverify_failed / predicate_error / gate_error / action_factory_error /
    input_error / context_error / deadman / tick_stats / run_end
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reflex.actions import Action, labels, normalize_seq
from reflex.core import FrameView, Match, Predicate, TriggerSpec
from reflex.expect import ExpectContext
from reflex.io import (  # noqa: F401 — re-exported for older imports
    ActionExecutor,
    ActionRefused,
    EventSink,
    FrameSource,
    InputSink,
    NoopEventSink,
)

ContextProvider = Callable[[FrameView], dict[str, Any] | None]

# Back-compat alias (older code imported the private name).
_NoopEventSink = NoopEventSink


@dataclass
class _Pending:
    fire_index: int
    fired_at: float
    pre_view: FrameView
    match: Match
    scratch: dict[str, Any] = field(default_factory=dict)
    next_check: float = 0.0


@dataclass
class _SpecState:
    spec: TriggerSpec
    fires: int = 0
    consecutive_hits: int = 0
    consecutive_misses: int = 0
    last_fire_monotonic: float = 0.0
    last_match: Match | None = None
    armed_at_monotonic: float = 0.0
    next_poll_monotonic: float = 0.0
    disabled: bool = False
    disabled_reason: str | None = None
    timed_out: bool = False
    tripped: bool = False
    pending: _Pending | None = None
    verified: int = 0
    unverified: int = 0
    consecutive_unverified: int = 0
    gate_blocked: int = 0
    evaluations: int = 0
    matches: int = 0
    last_fire_actions: tuple[str, ...] = ()
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.disabled or self.timed_out or self.tripped or self.fires >= self.spec.max_fires

    def status(self) -> str:
        if self.tripped:
            return "tripped"
        if self.disabled:
            return "disabled"
        if self.timed_out:
            return "timed_out"
        if self.fires >= self.spec.max_fires:
            return "exhausted"
        if self.pending is not None:
            return "verifying"
        return "armed"


@dataclass
class TickReport:
    """Per-tick result from ReflexRuntime.tick(). Inspectable for tests."""

    ts_ns: int
    candidates: int
    fired: str | None
    fired_action: tuple[str, ...] = ()
    suppressed_reason: str | None = None
    matches: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    tick_ms: float = 0.0
    """Whole tick incl. input dispatch; tick_stats windows use tick_ms - action_ms."""
    eval_ms: float = 0.0
    action_ms: float = 0.0
    verified: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    status: str  # goal_reached | all_done | timed_out | stopped | deadman
    elapsed_s: float
    ticks: int
    fires: dict[str, int]
    stats: dict[str, dict[str, Any]]
    goal_match: dict[str, Any] | None = None


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


class ReflexRuntime:
    def __init__(
        self,
        *,
        source: FrameSource,
        sink: Any,
        events: EventSink | None = None,
        deadman: Callable[[], bool] | None = None,
        foreground_ok: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        ts_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], None] = time.sleep,
        perf: Callable[[], float] = time.perf_counter,
        guards: dict[str, Predicate] | None = None,
        context_providers: dict[str, ContextProvider] | None = None,
        tick_stats_every_s: float = 5.0,
    ) -> None:
        self._source = source
        self._executor = ActionExecutor(sink, sleep=sleep)
        self._sink = self._executor.sink
        self._events: EventSink = events or NoopEventSink()
        self._deadman = deadman or (lambda: False)
        self._foreground_ok = foreground_ok or (lambda: True)
        self._clock = clock
        self._ts_ns = ts_ns
        self._sleep = sleep
        self._perf = perf  # compute-cost timing, independent of the (possibly virtual) scheduling clock
        self._tick_p0 = 0.0
        self._specs: dict[str, _SpecState] = {}
        self._global_lockout_until: float = 0.0
        self._guards: dict[str, Predicate] = dict(guards or {})
        self._context: dict[str, ContextProvider] = dict(context_providers or {})
        self._prev_frame: np.ndarray | None = None
        self._last_view: FrameView | None = None
        self._guarded_by: str | None = None
        self._guard_hits: dict[str, int] = {}
        self._tick_stats_every_s = float(tick_stats_every_s)
        self._win_started = clock()
        self._win_tick_ms: list[float] = []
        self._win_eval_ms: list[float] = []
        self._win_suppressed: dict[str, int] = {}
        self.ticks = 0

    # ------------------------------------------------------------------ setup

    @property
    def capabilities(self) -> frozenset:
        return self._executor.capabilities

    def register(self, spec: TriggerSpec, *, replace: bool = False) -> None:
        if spec.name in self._specs and not replace:
            raise KeyError(f"reflex spec '{spec.name}' already registered")
        now = self._clock()
        self._specs[spec.name] = _SpecState(spec=spec, armed_at_monotonic=now, next_poll_monotonic=now)
        self._events.emit(
            "reflex.registered",
            {
                "name": spec.name,
                "priority": spec.priority,
                "kind": spec.extra.get("kind"),
                "max_fires": spec.max_fires,
                "timeout_s": spec.timeout_s,
                "poll_hz": spec.poll_hz,
                "gates": len(spec.gates),
                "expect": spec.expect.describe() if spec.expect else None,
            },
        )

    def unregister(self, name: str) -> None:
        if name in self._specs:
            del self._specs[name]
            self._events.emit("reflex.unregistered", {"name": name})

    def disable(self, name: str, reason: str = "manual") -> None:
        if name in self._specs:
            st = self._specs[name]
            st.disabled = True
            st.disabled_reason = reason

    def enable(self, name: str) -> None:
        """Re-arm a disabled or tripped spec (clears the trip counter)."""
        if name in self._specs:
            st = self._specs[name]
            st.disabled = False
            st.disabled_reason = None
            st.tripped = False
            st.consecutive_unverified = 0
            st.consecutive_hits = 0
            st.consecutive_misses = 0

    def add_guard(self, name: str, predicate: Predicate) -> None:
        self._guards[name] = predicate

    def add_context_provider(self, name: str, provider: ContextProvider) -> None:
        self._context[name] = provider

    def lockout(self, duration_s: float) -> None:
        self._global_lockout_until = max(self._global_lockout_until, self._clock() + duration_s)

    def has_spec(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> list[str]:
        return list(self._specs.keys())

    # ------------------------------------------------------------------ tick

    def _suppress(self, reason: str, now: float | None = None) -> None:
        key = reason.split("=", 1)[0]
        self._win_suppressed[key] = self._win_suppressed.get(key, 0) + 1
        if now is not None:
            # Treat due specs as polled so a suppressed runtime idles at the
            # specs' own poll rate instead of spinning.
            for st in self._specs.values():
                if not st.done and st.next_poll_monotonic <= now:
                    st.next_poll_monotonic = now + 1.0 / st.spec.poll_hz

    def _build_view(self, frame_bgr: np.ndarray, ts_ns: int) -> FrameView:
        extras: dict[str, Any] = {}
        if self._prev_frame is not None:
            extras["prev_frame_bgr"] = self._prev_frame
        view = FrameView(frame_bgr=frame_bgr, ts_ns=ts_ns, extras=extras)
        for name, provider in self._context.items():
            try:
                got = provider(view)
            except Exception as exc:  # noqa: BLE001
                self._events.emit("reflex.context_error", {"provider": name, "error": repr(exc)})
                continue
            if got:
                extras.update(got)
        return view

    def _settle_verifications(self, view: FrameView, now: float, report: TickReport) -> None:
        for state in self._specs.values():
            pend = state.pending
            spec = state.spec
            if pend is None or spec.expect is None:
                continue
            exp = spec.expect
            elapsed_ms = (now - pend.fired_at) * 1000.0
            if now < pend.next_check and elapsed_ms < exp.within_ms:
                continue
            pend.next_check = now + 1.0 / spec.poll_hz
            evidence = None
            if elapsed_ms >= exp.after_ms:
                ctx = ExpectContext(
                    pre_view=pend.pre_view,
                    view=view,
                    fire_match=pend.match,
                    spec_predicate=spec.predicate,
                    elapsed_ms=elapsed_ms,
                    scratch=pend.scratch,
                )
                try:
                    evidence = exp.check(ctx)
                except Exception as exc:  # noqa: BLE001
                    self._events.emit("reflex.expect_error", {"name": spec.name, "error": repr(exc)})
                    evidence = None
            if evidence is not None:
                state.pending = None
                state.verified += 1
                state.consecutive_unverified = 0
                report.verified.append(spec.name)
                self._mark_history(state, pend.fire_index, verdict="verified")
                self._events.emit(
                    "reflex.verified",
                    {"name": spec.name, "fire_index": pend.fire_index, "expect": exp.name,
                     "latency_ms": round(elapsed_ms, 1), "evidence": evidence},
                )
                continue
            if elapsed_ms >= exp.within_ms:
                state.pending = None
                state.unverified += 1
                state.consecutive_unverified += 1
                report.unverified.append(spec.name)
                self._mark_history(state, pend.fire_index, verdict="unverified")
                self._events.emit(
                    "reflex.unverified",
                    {"name": spec.name, "fire_index": pend.fire_index, "expect": exp.name,
                     "within_ms": exp.within_ms, "consecutive": state.consecutive_unverified},
                )
                if spec.trip_after_unverified and state.consecutive_unverified >= spec.trip_after_unverified:
                    state.tripped = True
                    self._events.emit(
                        "reflex.tripped",
                        {"name": spec.name, "consecutive_unverified": state.consecutive_unverified,
                         "fires": state.fires},
                    )

    @staticmethod
    def _mark_history(state: _SpecState, fire_index: int, *, verdict: str) -> None:
        for h in reversed(state.history):
            if h.get("fire_index") == fire_index:
                h["verdict"] = verdict
                return

    def _check_guards(self, view: FrameView) -> tuple[str, dict[str, Any]] | None:
        for name, guard in self._guards.items():
            try:
                m = guard(view)
            except Exception as exc:  # noqa: BLE001 — fail closed
                self._events.emit("reflex.guard_error", {"guard": name, "error": repr(exc)})
                return name, {"error": repr(exc)}
            if m is not None:
                return name, m.meta
        return None

    def tick(self, *, frame_bgr: np.ndarray | None = None) -> TickReport:
        t0 = self._clock()
        self._tick_p0 = self._perf()
        ts_ns_now = self._ts_ns()
        self.ticks += 1

        if self._deadman():
            self._events.emit("reflex.deadman", {})
            self._suppress("deadman", t0)
            return TickReport(ts_ns=ts_ns_now, candidates=0, fired=None, suppressed_reason="deadman")

        if self._global_lockout_until > t0:
            self._suppress("lockout", t0)
            return TickReport(
                ts_ns=ts_ns_now,
                candidates=0,
                fired=None,
                suppressed_reason=f"lockout_for={(self._global_lockout_until - t0):.2f}s",
            )

        grabbed_here = frame_bgr is None
        if frame_bgr is None:
            frame_bgr = self._source.grab()
        view = self._build_view(frame_bgr, ts_ns_now)
        self._last_view = view
        report = TickReport(ts_ns=ts_ns_now, candidates=0, fired=None)

        try:
            self._settle_verifications(view, t0, report)

            guard = self._check_guards(view) if self._guards else None
            if guard is not None:
                gname, gmeta = guard
                self._guard_hits[gname] = self._guard_hits.get(gname, 0) + 1
                if self._guarded_by != gname:
                    self._guarded_by = gname
                    self._events.emit("reflex.guarded", {"guard": gname, "match_meta": gmeta})
                self._suppress("guard", t0)
                report.suppressed_reason = f"guard={gname}"
                return report
            if self._guarded_by is not None:
                self._events.emit("reflex.guard_cleared", {"guard": self._guarded_by})
                self._guarded_by = None

            self._evaluate_and_fire(view, t0, grabbed_here, report)
            return report
        finally:
            self._prev_frame = frame_bgr
            report.tick_ms = round((self._perf() - self._tick_p0) * 1000.0, 3)
            self._record_timing(report)

    def _evaluate_and_fire(self, view: FrameView, t0: float, grabbed_here: bool, report: TickReport) -> None:
        candidates: list[tuple[_SpecState, Match]] = []
        fg_ok: bool | None = None

        for state in self._specs.values():
            spec = state.spec
            if state.disabled or state.timed_out or state.tripped or state.pending is not None:
                continue
            if state.fires >= spec.max_fires:
                continue
            if t0 - state.armed_at_monotonic >= spec.timeout_s:
                state.timed_out = True
                self._events.emit("reflex.timeout", {"name": spec.name, "fires": state.fires})
                continue
            if t0 < state.next_poll_monotonic:
                continue
            state.next_poll_monotonic = t0 + (1.0 / spec.poll_hz)

            if spec.foreground_only:
                if fg_ok is None:
                    fg_ok = bool(self._foreground_ok())
                if not fg_ok:
                    state.consecutive_hits = 0
                    continue

            if spec.gates and not self._gates_pass(state, view):
                continue

            state.evaluations += 1
            try:
                match = spec.predicate(view)
            except Exception as exc:  # noqa: BLE001
                self._events.emit("reflex.predicate_error", {"name": spec.name, "error": repr(exc)})
                continue

            if match is None:
                if state.consecutive_hits > 0 and state.consecutive_misses < spec.miss_tolerance:
                    state.consecutive_misses += 1
                else:
                    state.consecutive_hits = 0
                    state.consecutive_misses = 0
                continue

            state.matches += 1
            state.consecutive_hits += 1
            state.consecutive_misses = 0
            state.last_match = match
            report.matches.append((spec.name, match.meta))

            if state.consecutive_hits < spec.require_consecutive_hits:
                continue
            if state.fires > 0 and (t0 - state.last_fire_monotonic) < spec.cooldown_s:
                continue
            candidates.append((state, match))

        report.candidates = len(candidates)
        report.eval_ms = round((self._perf() - self._tick_p0) * 1000.0, 3)
        if not candidates:
            return

        candidates.sort(key=lambda t: (-t[0].spec.priority, t[0].armed_at_monotonic))
        chosen_state, chosen_match = candidates[0]
        spec = chosen_state.spec
        fire_view = view

        if spec.reverify_on_fire:
            if grabbed_here:
                fresh = self._source.grab()
                fire_view = FrameView(frame_bgr=fresh, ts_ns=self._ts_ns(), extras=view.extras)
            try:
                refreshed = spec.predicate(fire_view)
            except Exception as exc:  # noqa: BLE001
                self._events.emit("reflex.predicate_error", {"name": spec.name, "error": repr(exc), "phase": "reverify"})
                refreshed = None
            if refreshed is None:
                chosen_state.consecutive_hits = 0
                self._events.emit("reflex.reverify_failed", {"name": spec.name, "candidate": chosen_match.meta})
                return
            chosen_match = refreshed

        try:
            actions: tuple[Action, ...] = normalize_seq(spec.action_factory(chosen_match))
        except Exception as exc:  # noqa: BLE001
            self._events.emit("reflex.action_factory_error", {"name": spec.name, "error": repr(exc)})
            chosen_state.consecutive_hits = 0
            return

        refused = [a for a in actions if not self._executor.supports(a)]
        if refused:
            # Never half-send a sequence: refuse all of it, and disable the spec —
            # it cannot succeed with this sink.
            self.disable(spec.name, reason="unsupported_action")
            self._events.emit(
                "reflex.unsupported_action",
                {"name": spec.name, "actions": list(labels(refused)),
                 "needs": sorted({a.capability for a in refused}),
                 "sink_capabilities": sorted(self._executor.capabilities)},
            )
            return

        a0 = self._clock()
        ok_seq: list[bool] = []
        for i, action in enumerate(actions):
            if i > 0 and spec.inter_press_ms > 0:
                self._sleep(spec.inter_press_ms / 1000.0)
            ok = False
            try:
                ok = self._executor.execute(action, default_hold_ms=int(spec.hold_ms))
            except Exception as exc:  # noqa: BLE001
                self._events.emit("reflex.input_error", {"name": spec.name, "error": repr(exc), "action": action.label()})
            ok_seq.append(bool(ok))
        report.action_ms = round((self._clock() - a0) * 1000.0, 3)

        fire_t = self._clock()
        chosen_state.fires += 1
        chosen_state.last_fire_monotonic = t0
        chosen_state.consecutive_hits = 0
        chosen_state.last_fire_actions = labels(actions)
        fire_index = chosen_state.fires
        chosen_state.history.append(
            {
                "fire_index": fire_index,
                "t_since_armed_s": round(fire_t - chosen_state.armed_at_monotonic, 4),
                "action_seq": list(labels(actions)),
                "ok_seq": ok_seq,
                "match_meta": chosen_match.meta,
                "verdict": "pending" if spec.expect else "unchecked",
            }
        )
        if spec.expect is not None:
            chosen_state.pending = _Pending(fire_index=fire_index, fired_at=fire_t, pre_view=fire_view, match=chosen_match)

        report.fired = spec.name
        report.fired_action = labels(actions)
        self._events.emit(
            "reflex.fired",
            {
                "name": spec.name,
                "priority": spec.priority,
                "action": list(labels(actions)),
                "ok_seq": ok_seq,
                "all_ok": all(ok_seq) if ok_seq else True,
                "fires_total": fire_index,
                "match_meta": chosen_match.meta,
                "action_ms": report.action_ms,
                "expect": spec.expect.name if spec.expect else None,
            },
        )

    def _gates_pass(self, state: _SpecState, view: FrameView) -> bool:
        for i, gate in enumerate(state.spec.gates):
            try:
                ok = gate(view) is not None
            except Exception as exc:  # noqa: BLE001
                self._events.emit("reflex.gate_error", {"name": state.spec.name, "gate": i, "error": repr(exc)})
                ok = False
            if not ok:
                state.gate_blocked += 1
                state.consecutive_hits = 0
                return False
        return True

    # ------------------------------------------------------------------ timing

    def _record_timing(self, report: TickReport) -> None:
        if report.suppressed_reason is None or report.suppressed_reason.startswith("guard"):
            # Compute cost only: input dispatch (holds, waits) is reported per fire as action_ms.
            self._win_tick_ms.append(max(0.0, report.tick_ms - report.action_ms))
            self._win_eval_ms.append(report.eval_ms)
        now = self._clock()
        if self._tick_stats_every_s > 0 and now - self._win_started >= self._tick_stats_every_s:
            self.flush_tick_stats()

    def flush_tick_stats(self) -> dict[str, Any] | None:
        now = self._clock()
        window = now - self._win_started
        ticks = sorted(self._win_tick_ms)
        evals = sorted(self._win_eval_ms)
        payload: dict[str, Any] | None = None
        if ticks or self._win_suppressed:
            payload = {
                "window_s": round(window, 3),
                "ticks": len(ticks),
                "tick_hz": round(len(ticks) / window, 2) if window > 0 else None,
                "tick_ms_p50": round(_percentile(ticks, 0.50), 3),
                "tick_ms_p95": round(_percentile(ticks, 0.95), 3),
                "tick_ms_max": round(ticks[-1], 3) if ticks else 0.0,
                "eval_ms_p50": round(_percentile(evals, 0.50), 3),
                "eval_ms_p95": round(_percentile(evals, 0.95), 3),
                "suppressed": dict(self._win_suppressed),
                "specs_armed": sum(1 for s in self._specs.values() if not s.done),
            }
            self._events.emit("reflex.tick_stats", payload)
        self._win_started = now
        self._win_tick_ms = []
        self._win_eval_ms = []
        self._win_suppressed = {}
        return payload

    # ------------------------------------------------------------------ run loop

    def _next_due_in(self) -> float:
        now = self._clock()
        waits: list[float] = []
        for s in self._specs.values():
            if s.pending is not None and s.spec.expect is not None:
                deadline = s.pending.fired_at + s.spec.expect.within_ms / 1000.0
                waits.append(max(0.0, min(s.pending.next_check, deadline) - now))
            elif not s.done:
                waits.append(max(0.0, s.next_poll_monotonic - now))
        return min(waits) if waits else 0.01

    def run(
        self,
        *,
        timeout_s: float,
        until: Predicate | None = None,
        stop_when_all_done: bool = True,
        should_stop: Callable[[], bool] | None = None,
        max_idle_sleep_s: float = 0.05,
        min_idle_sleep_s: float = 0.001,
    ) -> RunResult:
        """Tick until `until` matches, every spec is done, or timeout.

        Pending verifications are always allowed to settle before an
        "all_done" exit so the result carries verdicts for every fire.
        """
        if timeout_s <= 0:
            raise ValueError("run() needs timeout_s > 0; reflexes never run unbounded")
        started = self._clock()
        deadline = started + float(timeout_s)
        status = "timed_out"
        goal_meta: dict[str, Any] | None = None
        n = 0
        while self._clock() < deadline:
            if should_stop is not None and should_stop():
                status = "stopped"
                break
            due_in = self._next_due_in()
            if due_in > 0 and n > 0:
                # Nothing to poll or verify yet: sleep (capped so stop checks stay live).
                self._sleep(min(max_idle_sleep_s, due_in))
                continue
            report = self.tick()
            n += 1
            if until is not None and report.suppressed_reason is None:
                try:
                    gm = until(self._last_view) if self._last_view is not None else None
                except Exception as exc:  # noqa: BLE001
                    self._events.emit("reflex.goal_error", {"error": repr(exc)})
                    gm = None
                if gm is not None:
                    status = "goal_reached"
                    goal_meta = gm.meta
                    break
            if stop_when_all_done and self._specs and all(
                s.done and s.pending is None for s in self._specs.values()
            ):
                status = "all_done"
                break
            self._sleep(min(max_idle_sleep_s, max(min_idle_sleep_s, self._next_due_in())))
        self.flush_tick_stats()
        result = RunResult(
            status=status,
            elapsed_s=round(self._clock() - started, 4),
            ticks=n,
            fires={k: s.fires for k, s in self._specs.items()},
            stats=self.stats(),
            goal_match=goal_meta,
        )
        self._events.emit(
            "reflex.run_end",
            {"status": status, "elapsed_s": result.elapsed_s, "ticks": n, "fires": result.fires,
             "goal_match": goal_meta},
        )
        return result

    # ------------------------------------------------------------------ inspection

    def stats(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, state in self._specs.items():
            checked = state.verified + state.unverified
            out[name] = {
                "status": state.status(),
                "priority": state.spec.priority,
                "kind": state.spec.extra.get("kind"),
                "fires": state.fires,
                "max_fires": state.spec.max_fires,
                "disabled": state.disabled,
                "disabled_reason": state.disabled_reason,
                "timed_out": state.timed_out,
                "tripped": state.tripped,
                "evaluations": state.evaluations,
                "matches": state.matches,
                "gate_blocked": state.gate_blocked,
                "verified": state.verified,
                "unverified": state.unverified,
                "verify_rate": round(state.verified / checked, 4) if checked else None,
                "pending": state.pending is not None,
                "last_actions": list(state.last_fire_actions),
                "last_match_meta": state.last_match.meta if state.last_match else None,
            }
        return out

    def history(self, name: str) -> list[dict[str, Any]]:
        st = self._specs.get(name)
        return list(st.history) if st else []

    def guard_hits(self) -> dict[str, int]:
        return dict(self._guard_hits)
