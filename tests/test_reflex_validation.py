"""Reflex v2: typed actions, capability refusal, expectations + trips,
gates, guards, context providers, run() loop, MacroEngine outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from reflex import expect
from reflex.actions import Chord, Click, Key, KeyDown, KeyUp, Wait, from_spec
from reflex.conditions import extras_equals, motion_present
from reflex.core import Match, Region, TriggerSpec
from reflex.engine import MacroEngine
from reflex.io import RecordingActionSink, RecordingEventSink, RecordingInputSink
from reflex.recipes import click_target, on_match
from reflex.runtime import ReflexRuntime


class VClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += float(dt)


class MutableSource:
    """Frame source whose frame the test (or a sink side-effect) changes."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def grab(self) -> np.ndarray:
        return self.frame


def _frame(v: int = 0) -> np.ndarray:
    return np.full((20, 20, 3), v, dtype=np.uint8)


def _bright(view):
    return Match(meta={"centroid_xy": (5, 6)}) if view.frame_bgr.mean() > 100 else None


def _rt(src, sink, clock, events=None, **kw):
    return ReflexRuntime(source=src, sink=sink, events=events or RecordingEventSink(),
                         clock=clock.now, sleep=clock.sleep, ts_ns=lambda: int(clock.t * 1e9), **kw)


# --------------------------------------------------------------------------- actions


def test_action_from_spec_forms():
    assert from_spec("a") == Key("a")
    assert from_spec({"key": "a", "hold_ms": 80}) == Key("a", hold_ms=80)
    assert from_spec({"down": "right"}) == KeyDown("right")
    assert from_spec({"up": "right"}) == KeyUp("right")
    assert from_spec({"chord": ["ctrl", "s"]}) == Chord(("ctrl", "s"))
    assert from_spec({"click": [10, 20], "button": "right"}) == Click(10, 20, "right")
    assert from_spec({"wait": 120}) == Wait(120)
    with pytest.raises(ValueError):
        from_spec({"teleport": 1})


def test_typed_actions_reach_full_sink_in_order():
    clock = VClock()
    sink = RecordingActionSink(clock=clock.now)
    rt = _rt(MutableSource(_frame(200)), sink, clock)
    rt.register(on_match(name="combo", when=_bright,
                         do=["a", {"click": [3, 4]}, {"wait": 100}, {"chord": ["x", "z"]}],
                         inter_press_ms=0, hold_ms=10))
    rep = rt.tick()
    assert rep.fired == "combo"
    kinds = [(e["kind"], e.get("button")) for e in sink.log]
    assert kinds == [("key", "a"), ("click", "left"), ("key_down", "x"), ("key_down", "z"),
                     ("key_up", "z"), ("key_up", "x")]
    assert rep.fired_action == ("a", "click:left@3,4", "wait:100ms", "x+z")


def test_press_only_sink_refuses_click_and_disables_spec():
    clock = VClock()
    sink = RecordingInputSink()  # press only
    ev = RecordingEventSink()
    rt = _rt(MutableSource(_frame(200)), sink, clock, events=ev)
    rt.register(on_match(name="clicker", when=_bright, do=["a", {"click": [1, 1]}]))
    rep = rt.tick()
    assert rep.fired is None
    assert sink.calls == []  # nothing half-sent
    refused = ev.of("reflex.unsupported_action")
    assert refused and refused[0]["needs"] == ["click"]
    assert rt.stats()["clicker"]["disabled_reason"] == "unsupported_action"


def test_click_target_clicks_match_centroid():
    clock = VClock()
    sink = RecordingActionSink(clock=clock.now)
    rt = _rt(MutableSource(_frame(200)), sink, clock)
    rt.register(click_target(name="btn", locator=_bright, offset_xy=(1, -1)))
    rt.tick()
    assert sink.log[0] == {"t": 0.0, "kind": "click", "x": 6, "y": 5, "button": "left"}


# --------------------------------------------------------------------------- expectations


def _dialog_world(clock, *, closes: bool):
    """A 'dialog' (bright frame) that closes when A is pressed — or doesn't."""
    src = MutableSource(_frame(200))

    class Sink(RecordingActionSink):
        def press(self, button, *, hold_ms):
            if closes:
                src.frame = _frame(0)
            return super().press(button, hold_ms=hold_ms)

    return src, Sink(clock=clock.now)


def test_expect_gone_verified_when_press_has_effect():
    clock = VClock()
    src, sink = _dialog_world(clock, closes=True)
    ev = RecordingEventSink()
    rt = _rt(src, sink, clock, events=ev)
    spec = on_match(name="advance", when=_bright, do=["a"], max_fires=5, poll_hz=20)
    rt.register(spec.with_overrides(expect=expect.gone(within_ms=500)))
    res = rt.run(timeout_s=2.0)
    st = rt.stats()["advance"]
    assert st["fires"] == 1 and st["verified"] == 1 and st["unverified"] == 0
    assert st["verify_rate"] == 1.0
    assert ev.of("reflex.verified")[0]["expect"] == "gone"
    assert rt.history("advance")[0]["verdict"] == "verified"
    assert res.status == "timed_out"  # max_fires 5 not reached; dialog gone


def test_expect_unverified_blocks_refire_until_window_then_trips():
    clock = VClock()
    src, sink = _dialog_world(clock, closes=False)  # presses do nothing
    ev = RecordingEventSink()
    rt = _rt(src, sink, clock, events=ev)
    spec = on_match(name="advance", when=_bright, do=["a"], max_fires=50, poll_hz=20, timeout_s=30)
    rt.register(spec.with_overrides(expect=expect.gone(within_ms=300), trip_after_unverified=3))
    res = rt.run(timeout_s=10.0)
    st = rt.stats()["advance"]
    # Exactly 3 presses: each waited out its 300 ms window, then the breaker tripped.
    assert st["fires"] == 3 and st["unverified"] == 3 and st["tripped"]
    assert len(sink.calls) == 3
    fire_ts = [e["t"] for e in sink.log]
    assert all(b - a >= 0.3 for a, b in zip(fire_ts, fire_ts[1:], strict=False))
    assert ev.of("reflex.tripped")[0]["consecutive_unverified"] == 3
    assert res.status == "all_done"
    rt.enable("advance")
    assert rt.stats()["advance"]["status"] == "armed"


def test_expect_changed_uses_prefire_frame():
    clock = VClock()
    src = MutableSource(_frame(200))

    class Sink(RecordingActionSink):
        def press(self, button, *, hold_ms):
            f = src.frame.copy()
            f[:10] = 90  # half the screen changes, still "bright" overall? no → mean 145
            src.frame = f
            return super().press(button, hold_ms=hold_ms)

    rt = _rt(src, Sink(clock=clock.now), clock)
    spec = on_match(name="scroll", when=_bright, do=["right"], poll_hz=20)
    rt.register(spec.with_overrides(expect=expect.changed(min_changed_fraction=0.3, within_ms=400)))
    rt.run(timeout_s=1.0)
    assert rt.stats()["scroll"]["verified"] == 1


def test_expect_appears_with_all_of():
    clock = VClock()
    src = MutableSource(_frame(200))

    class Sink(RecordingActionSink):
        def press(self, button, *, hold_ms):
            src.frame = _frame(40)
            return super().press(button, hold_ms=hold_ms)

    dim = lambda v: Match(meta={}) if 20 < v.frame_bgr.mean() < 60 else None  # noqa: E731
    rt = _rt(src, Sink(clock=clock.now), clock)
    spec = on_match(name="open", when=_bright, do=["a"], poll_hz=20)
    rt.register(spec.with_overrides(expect=expect.all_of([expect.gone(), expect.appears(predicate=dim)])))
    rt.run(timeout_s=1.0)
    assert rt.stats()["open"]["verified"] == 1


# --------------------------------------------------------------------------- gates, guards, context


def test_gate_on_context_provider_scene():
    clock = VClock()
    scene = {"v": "overworld"}
    sink = RecordingActionSink(clock=clock.now)
    rt = _rt(MutableSource(_frame(200)), sink, clock,
             context_providers={"scene": lambda view: {"scene": scene["v"]}})
    spec = on_match(name="battle_only", when=_bright, do=["a"], poll_hz=1000)
    rt.register(spec.with_overrides(gates=(extras_equals(key="scene", value="battle"),)))
    assert rt.tick().fired is None
    assert rt.stats()["battle_only"]["gate_blocked"] == 1
    scene["v"] = "battle"
    clock.sleep(0.01)
    assert rt.tick().fired == "battle_only"


def test_guard_suppresses_all_firing_and_emits_once():
    clock = VClock()
    shop = {"v": True}
    ev = RecordingEventSink()
    sink = RecordingActionSink(clock=clock.now)
    rt = _rt(MutableSource(_frame(200)), sink, clock, events=ev,
             guards={"payment_screen": lambda v: Match(meta={"why": "shop"}) if shop["v"] else None})
    rt.register(on_match(name="spam", when=_bright, do=["a"], max_fires=10, poll_hz=1000))
    for _ in range(5):
        rep = rt.tick()
        clock.sleep(0.01)
        assert rep.suppressed_reason == "guard=payment_screen"
    assert sink.log == []
    assert len(ev.of("reflex.guarded")) == 1
    assert rt.guard_hits() == {"payment_screen": 5}
    shop["v"] = False
    assert rt.tick().fired == "spam"
    assert len(ev.of("reflex.guard_cleared")) == 1


def test_guard_error_fails_closed():
    clock = VClock()
    sink = RecordingActionSink(clock=clock.now)

    def broken(_v):
        raise RuntimeError("detector crashed")

    rt = _rt(MutableSource(_frame(200)), sink, clock, guards={"g": broken})
    rt.register(on_match(name="x", when=_bright, do=["a"]))
    assert rt.tick().suppressed_reason == "guard=g"
    assert sink.log == []


def test_runtime_seeds_prev_frame_so_motion_present_works():
    clock = VClock()
    src = MutableSource(_frame(0))
    sink = RecordingActionSink(clock=clock.now)
    rt = _rt(src, sink, clock)
    rt.register(on_match(name="moved", when=motion_present(region=Region(0, 0, 20, 20)), do=["b"],
                         poll_hz=1000, reverify_on_fire=False))
    assert rt.tick().fired is None  # no previous frame yet
    src.frame = _frame(255)
    clock.sleep(0.01)
    assert rt.tick().fired == "moved"


# --------------------------------------------------------------------------- run loop + engine


def test_run_until_goal_reached():
    clock = VClock()
    src = MutableSource(_frame(0))
    rt = _rt(src, RecordingActionSink(clock=clock.now), clock)
    rt.register(on_match(name="idle", when=lambda v: None, do=["a"]))
    ticks = {"n": 0}

    def goal(view):
        ticks["n"] += 1
        if ticks["n"] == 4:
            src.frame = _frame(200)
        return _bright(view)

    res = rt.run(timeout_s=5.0, until=goal)
    assert res.status == "goal_reached"
    assert res.goal_match == {"centroid_xy": (5, 6)}


def test_run_requires_bounded_timeout():
    clock = VClock()
    rt = _rt(MutableSource(_frame(0)), RecordingActionSink(), clock)
    with pytest.raises(ValueError):
        rt.run(timeout_s=0)


def test_macro_engine_reports_tripped_and_verdicts():
    clock = VClock()
    src, sink = _dialog_world(clock, closes=False)
    spec = on_match(name="advance", when=_bright, do=["a"], max_fires=10, poll_hz=20, timeout_s=10)
    spec = spec.with_overrides(expect=expect.gone(within_ms=200), trip_after_unverified=2)
    out = MacroEngine(src, sink, clock=clock.now, sleep=clock.sleep).run(spec)
    assert out.status == "tripped"
    assert out.fires == 2 and out.unverified == 2 and out.verified == 0
    assert [h["verdict"] for h in out.fire_history] == ["unverified", "unverified"]


def test_tick_stats_emitted_with_percentiles():
    clock = VClock()
    ev = RecordingEventSink()
    rt = _rt(MutableSource(_frame(0)), RecordingActionSink(), clock, events=ev, tick_stats_every_s=1.0)
    rt.register(on_match(name="n", when=lambda v: None, do=["a"], timeout_s=100))
    for _ in range(30):
        rt.tick()
        clock.sleep(0.05)
    stats = ev.of("reflex.tick_stats")
    assert stats, "expected a tick_stats window"
    s0 = stats[0]
    assert s0["ticks"] >= 19 and "tick_ms_p95" in s0 and s0["specs_armed"] == 1


def test_trigger_spec_validation():
    ok = dict(name="x", predicate=lambda v: None, action_factory=lambda m: ())
    for bad in ({"timeout_s": 0}, {"poll_hz": 0}, {"max_fires": 0}, {"hold_ms": -1}, {"trip_after_unverified": -1}):
        with pytest.raises(ValueError):
            TriggerSpec(**ok, **bad)
    with pytest.raises(ValueError):
        expect.gone(within_ms=0)


def test_hsv_cache_gives_identical_results():
    import reflex.conditions as C

    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    preds = [
        C.sprite_blob(region=Region(0, 0, 160, 120), hsv_lo=(0, 0, 200), hsv_hi=(179, 80, 255), min_area_px=1),
        C.region_color_ratio(region=Region(0, 40, 160, 80), hsv_lo=(0, 0, 100), hsv_hi=(179, 255, 255), ratio_min=0.0),
        C.region_color_ratio(region=Region(5, 5, 10, 10), hsv_lo=(0, 0, 100), hsv_hi=(179, 255, 255), ratio_min=0.0),
    ]
    from reflex.core import FrameView

    cached_view = FrameView(frame_bgr=frame, ts_ns=0, extras={})
    cached = [p(cached_view).meta for p in preds]
    assert "_hsv_full" in cached_view.extras  # the large region populated the shared cache
    fresh = [p(FrameView(frame_bgr=frame, ts_ns=0, extras={})).meta for p in reversed(preds)][::-1]
    assert cached == fresh


def test_template_grayscale_finds_same_target():
    from reflex.conditions import template_match
    from reflex.core import FrameView

    rng = np.random.default_rng(5)
    frame = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    tpl = frame[40:52, 30:46].copy()
    for gs in (False, True):
        m = template_match(region=Region(0, 0, 100, 100), templates=[tpl], threshold=0.95, grayscale=gs)(
            FrameView(frame_bgr=frame, ts_ns=0))
        assert m is not None and m.meta["bbox_xywh"] == (30, 40, 16, 12)


def test_hsv_cache_not_reused_across_frames_sharing_extras():
    """Reverify views share extras with the tick view but hold a fresh frame."""
    import reflex.conditions as C
    from reflex.core import FrameView

    p = C.region_color_ratio(region=Region(0, 0, 20, 20), hsv_lo=(0, 0, 200), hsv_hi=(179, 255, 255), ratio_min=0.5)
    extras: dict = {}
    assert p(FrameView(frame_bgr=_frame(255), ts_ns=0, extras=extras)) is not None
    assert p(FrameView(frame_bgr=_frame(0), ts_ns=1, extras=extras)) is None
