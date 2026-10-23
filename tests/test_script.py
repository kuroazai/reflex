"""Scripts: steps, control flow, variables, watchers, validation, stop/pause."""

from __future__ import annotations

import sys
import threading

import cv2
import numpy as np
import pytest
import yaml

from reflex.io import RecordingActionSink, RecordingEventSink
from reflex.script import Control, ScriptError, ScriptRunner, Target, build_script, run_script, schedule_times


class World:
    """Virtual clock + frames that change at set times."""

    def __init__(self, timeline=None):
        self.t = 0.0
        self.timeline = sorted(timeline or [(0.0, _blank())], key=lambda p: p[0])

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += max(float(dt), 0.0005)

    def ts_ns(self):
        return int(self.t * 1e9)

    def grab(self):
        frame = self.timeline[0][1]
        for t, f in self.timeline:
            if t <= self.t:
                frame = f
        return frame


def _blank(color=(0, 0, 0)):
    f = np.zeros((120, 160, 3), dtype=np.uint8)
    f[:] = color
    return f


def _with_box(color=(0, 0, 255), at=(100, 40)):
    f = _blank()
    x, y = at
    f[y : y + 10, x : x + 10] = color
    return f


def run(src, world=None, *, sink=None, control=None, base_dir=None, **kw):
    world = world or World()
    sink = sink or RecordingActionSink(clock=world.now)
    script = build_script(yaml.safe_load(src) if isinstance(src, str) else src, base_dir=base_dir,
                          check_keys=False)
    events = RecordingEventSink()
    runner = ScriptRunner(script, source=world, sink=sink, events=events, control=control, clock=world.now,
                          sleep=world.sleep, ts_ns=world.ts_ns, seed=1, **kw)
    return runner.run(), sink, events


def kinds(sink):
    return [(e["kind"], e.get("button"), e.get("x"), e.get("y")) for e in sink.log]


# --------------------------------------------------------------------------- basics


def test_press_type_click_and_vars():
    res, sink, _ = run("""
vars: {name: kuro}
steps:
  - press: [a, b]
  - type: "hi ${name}"
  - click: [10, 20]
  - right_click: [5, 5]
  - double_click: [1, 2]
  - chord: [ctrl, s]
""")
    assert res.status == "completed", res.error
    log = sink.log
    assert [e["button"] for e in log if e["kind"] == "key"] == ["a", "b"]
    assert [e["text"] for e in log if e["kind"] == "type"] == ["hi kuro"]
    clicks = [(e["x"], e["y"], e["button"]) for e in log if e["kind"] == "click"]
    assert clicks == [(10, 20, "left"), (5, 5, "right"), (1, 2, "left"), (1, 2, "left")]
    assert [e["kind"] for e in log if e["kind"] in ("key_down", "key_up")] == ["key_down"] * 2 + ["key_up"] * 2


def test_wait_for_stores_match_and_click_uses_it():
    world = World([(0.0, _blank()), (1.0, _with_box())])
    res, sink, events = run("""
steps:
  - wait_for: {color_search: {rgb: [255, 0, 0], tolerance: 5}}
    timeout_s: 5
    as: box
  - click: $box
  - log: "box at ${box.x},${box.y}"
""", world)
    assert res.status == "completed", res.error
    assert [(e["x"], e["y"]) for e in sink.log if e["kind"] == "click"] == [(100, 40)]
    assert world.t >= 1.0
    assert "box at 100,40" in res.log[-1]


def test_click_a_condition_finds_it_now():
    world = World([(0.0, _with_box(at=(30, 60)))])
    res, sink, _ = run("steps:\n  - click: {color_search: {rgb: [255, 0, 0]}}\n    offset: [2, 3]\n", world)
    assert res.status == "completed", res.error
    assert [(e["x"], e["y"]) for e in sink.log if e["kind"] == "click"] == [(32, 63)]


def test_wait_for_timeout_fails_with_location():
    res, _, _ = run("steps:\n  - name: find boss\n    wait_for: {color_search: {rgb: [0, 255, 0]}}\n    timeout_s: 2\n")
    assert res.status == "failed"
    assert "find boss" in res.failed_at and "not seen within 2s" in res.error


def test_on_timeout_continue_and_steps():
    res, sink, _ = run("""
steps:
  - wait_for: {color_search: {rgb: [0, 255, 0]}}
    timeout_s: 1
    on_timeout: continue
  - wait_for: {color_search: {rgb: [0, 255, 0]}}
    timeout_s: 1
    on_timeout: [{press: esc}]
  - press: enter
""")
    assert res.status == "completed"
    assert [e["button"] for e in sink.log if e["kind"] == "key"] == ["esc", "enter"]


def test_wait_gone():
    world = World([(0.0, _with_box()), (2.0, _blank())])
    res, _, _ = run("steps:\n  - wait_gone: {color_search: {rgb: [255, 0, 0]}}\n    timeout_s: 5\n", world)
    assert res.status == "completed" and world.t >= 2.0


def test_find_null_and_optional_click():
    res, sink, _ = run("""
steps:
  - find: {color_search: {rgb: [0, 255, 0]}}
    as: thing
  - if: "${thing == null}"
    then: [{log: nothing}]
  - click: $thing
    optional: true
  - click: {color_search: {rgb: [0, 255, 0]}}
    optional: true
""")
    assert res.status == "completed", res.error
    assert not [e for e in sink.log if e["kind"] == "click"]
    assert any("nothing" in line for line in res.log)


# --------------------------------------------------------------------------- control flow


def test_repeat_times_with_counter_and_set():
    res, sink, _ = run("""
vars: {total: 0}
steps:
  - repeat: {times: 4, as: i}
    steps:
      - set: {total: "${total + i}"}
      - press: "${'x' if i % 2 else 'y'}"
""")
    assert res.status == "completed", res.error
    assert res.vars["total"] == 6
    assert [e["button"] for e in sink.log if e["kind"] == "key"] == ["y", "x", "y", "x"]


def test_repeat_until_screen_changes_and_break():
    world = World([(0.0, _blank()), (3.0, _with_box())])
    res, sink, _ = run("""
vars: {n: 0}
steps:
  - repeat: {until: {color_search: {rgb: [255, 0, 0]}}}
    steps:
      - press: space
      - wait: 500
  - repeat: {forever: true}
    steps:
      - set: {n: "${n + 1}"}
      - if: "${n >= 3}"
        then: [break]
""", world)
    assert res.status == "completed", res.error
    presses = [e for e in sink.log if e["kind"] == "key"]
    assert 5 <= len(presses) <= 7
    assert res.vars["n"] == 3


def test_repeat_while_and_continue():
    res, sink, _ = run("""
vars: {i: 0}
steps:
  - repeat: {while: "${i < 5}"}
    steps:
      - set: {i: "${i + 1}"}
      - if: "${i % 2 == 0}"
        then: [continue]
      - press: "${str(i)}"
""")
    assert [e["button"] for e in sink.log if e["kind"] == "key"] == ["1", "3", "5"]


def test_if_else_on_screen_condition_and_when():
    world = World([(0.0, _with_box())])
    res, sink, _ = run("""
steps:
  - if: {color_search: {rgb: [255, 0, 0]}}
    then: [{press: a}]
    else: [{press: b}]
  - if: {not: {color_search: {rgb: [255, 0, 0]}}}
    then: [{press: c}]
  - press: d
    when: {any_of: ["${false}", {color_search: {rgb: [255, 0, 0]}}]}
  - press: e
    when: "${1 > 2}"
""", world)
    assert [e["button"] for e in sink.log if e["kind"] == "key"] == ["a", "d"]


def test_functions_with_args():
    res, sink, _ = run("""
functions:
  hit:
    - press: "${key}"
    - wait: 100
steps:
  - call: hit
    with: {key: q}
  - call: hit
    with: {key: w}
""")
    assert [e["button"] for e in sink.log if e["kind"] == "key"] == ["q", "w"]


def test_stop_and_fail_steps():
    ok, sink, _ = run("steps:\n  - press: a\n  - stop: done early\n  - press: b\n")
    assert ok.status == "completed" and [e["button"] for e in sink.log if e["kind"] == "key"] == ["a"]
    bad, _, _ = run("steps:\n  - fail: \"no gold\"\n")
    assert bad.status == "failed" and "no gold" in bad.error


def test_drag_and_scroll():
    res, sink, _ = run("""
steps:
  - drag: [10, 10]
    to: [50, 30]
    steps: 4
  - scroll: -3
    at: center
""")
    assert res.status == "completed", res.error
    ks = [e["kind"] for e in sink.log]
    assert ks[0] == "mouse_down" and "mouse_up" in ks and ks.count("move") == 4
    up = [e for e in sink.log if e["kind"] == "mouse_up"][0]
    assert (up["x"], up["y"]) == (50, 30)
    sc = [e for e in sink.log if e["kind"] == "scroll"][0]
    assert (sc["x"], sc["y"], sc["clicks"]) == (80, 60, -3)


# --------------------------------------------------------------------------- watchers, timeouts, control


def test_watchers_fire_during_waits():
    world = World([(0.0, _blank()), (1.0, _with_box()), (1.5, _blank())])
    res, sink, events = run("""
watchers:
  - id: popup
    kind: on_match
    params:
      when: {predicate: color_search, rgb: [255, 0, 0]}
      do: [esc]
steps:
  - wait: 3000
  - press: enter
""", world)
    assert res.status == "completed", res.error
    keys = [e["button"] for e in sink.log if e["kind"] == "key"]
    assert keys == ["esc", "enter"]
    assert any(k == "reflex.fired" for k, _ in events.events)


def test_script_timeout_is_enforced():
    res, _, _ = run("timeout_s: 2\nsteps:\n  - repeat: {forever: true}\n    steps: [{wait: 100}]\n")
    assert res.status == "timed_out" and "2s" in res.error


def test_control_stop_releases_held_keys():
    world = World()
    control = Control()
    sink = RecordingActionSink(clock=world.now)
    orig = world.sleep

    def sleep(dt):
        orig(dt)
        if world.t > 1.0:
            control.stop("test")

    world.sleep = sleep
    res, sink, _ = run("steps:\n  - key_down: w\n  - wait: 10000\n", world, sink=sink, control=control)
    assert res.status == "stopped" and res.error == "test"
    assert sink.log[-1]["kind"] == "release_all"


def test_pause_holds_until_resumed():
    world = World()
    control = Control()
    control.pause_event.set()
    orig = world.sleep

    def sleep(dt):
        orig(dt)
        if world.t > 2.0:
            control.pause_event.clear()

    world.sleep = sleep
    res, sink, events = run("steps:\n  - press: a\n", world, control=control)
    assert res.status == "completed" and world.t > 2.0
    assert "script.paused" in [k for k, _ in events.events]


def test_two_targets_run_concurrently():
    script = build_script(yaml.safe_load("steps:\n  - press: '${who}'\n  - wait: 50\n"), check_keys=False)
    a, b = RecordingActionSink(), RecordingActionSink()
    targets = [Target("a", World(), a, {"who": "a"}), Target("b", World(), b, {"who": "b"})]
    results = run_script(script, targets=targets, hotkeys=False)
    assert sorted(r.instance for r in results) == ["a", "b"] and all(r.ok for r in results)
    assert [e["button"] for e in a.log if e["kind"] == "key"] == ["a"]
    assert [e["button"] for e in b.log if e["kind"] == "key"] == ["b"]


def test_image_shorthand_with_template_file(tmp_path):
    tpl = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.circle(tpl, (4, 4), 3, (255, 255, 255), -1)
    cv2.imwrite(str(tmp_path / "dot.png"), tpl)
    frame = _blank()
    frame[50:58, 70:78] = tpl
    world = World([(0.0, frame)])
    res, sink, _ = run("steps:\n  - click: {image: dot.png}\n", world, base_dir=tmp_path)
    assert res.status == "completed", res.error
    assert [(e["x"], e["y"]) for e in sink.log if e["kind"] == "click"] == [(74, 54)]


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize("src,needle", [
    ("steps:\n  - prss: a\n", "did you mean 'press'"),
    ("steps:\n  - press: a\n    hold: 5\n", "press does not take ['hold']"),
    ("steps:\n  - wait_for: {imagee: a.png}\n", "no predicate plug-in named 'imagee'"),
    ("steps:\n  - wait_for: {image: missing.png}\n", "missing.png"),
    ("steps:\n  - if: '${1 +}'\n    then: [{press: a}]\n", "bad expression"),
    ("steps:\n  - call: nope\n", "no function named 'nope'"),
    ("steps:\n  - find: {color_search: {rgb: [1, 2, 3]}}\n", "find needs as"),
    ("steps:\n  - repeat: {times: 2}\n", "repeat needs steps"),
    ("timeout_s: 0\nsteps: []\n", "never run unbounded"),
    ("stepz: []\n", "unknown top-level key"),
    ("target: {input: magic}\nsteps: []\n", "input is postmessage"),
    ("steps:\n  - wait: \"${1 +}\"\n", "bad expression"),
    ("steps:\n  - press: a\n    times: \"${2 *}\"\n", "bad expression"),
])
def test_validation_errors_are_located(src, needle, tmp_path):
    with pytest.raises(ScriptError) as exc:
        build_script(yaml.safe_load(src), base_dir=tmp_path, check_keys=False)
    assert needle in str(exc.value)


@pytest.mark.skipif(sys.platform != "win32", reason="key names resolve through the Windows layout")
def test_unknown_key_is_caught_at_load():
    with pytest.raises(ScriptError, match="unknown key 'spacee'"):
        build_script(yaml.safe_load("steps:\n  - press: spacee\n"))


def test_expression_sandbox_blocks_code():
    res, _, _ = run("steps:\n  - log: \"${__import__('os')}\"\n")
    assert res.status == "failed" and "not allowed" in res.error


def test_schedule_times():
    import datetime as dt

    base = dt.datetime(2026, 10, 23, 8, 30).timestamp()
    assert schedule_times(every_s=600, at=None, now=base) == 600
    assert schedule_times(every_s=None, at=["09:00"], now=base) == 1800
    assert schedule_times(every_s=None, at=["08:00"], now=base) == 23.5 * 3600


def test_stop_from_another_thread_ends_a_long_wait_promptly():
    import time

    control = Control()
    script = build_script({"timeout_s": 30, "steps": [{"wait": 100000}]}, check_keys=False)
    runner = ScriptRunner(script, source=World(), sink=RecordingActionSink(), control=control)
    threading.Timer(0.2, lambda: control.stop("hotkey")).start()
    t0 = time.monotonic()
    res = runner.run()
    assert res.status == "stopped" and res.error == "hotkey"
    assert time.monotonic() - t0 < 2.0


def test_example_scripts_load():
    from pathlib import Path

    from reflex.script import load_script

    scripts = sorted((Path(__file__).resolve().parents[1] / "examples" / "scripts").glob("*.yaml"))
    assert scripts
    for path in scripts:
        load_script(path, check_keys=False)


@pytest.mark.skipif(sys.platform != "win32", reason="key names resolve through the Windows layout")
@pytest.mark.parametrize("watcher_do", ["[notakey]", "[{key: notakey}]", "[{chord: [ctrl, notakey]}]", "[{down: notakey}]"])
def test_unknown_keys_in_watcher_actions_are_caught_at_load(watcher_do):
    src = f"""
watchers:
  - id: w
    kind: on_match
    params:
      when: {{predicate: always}}
      do: {watcher_do}
steps:
  - wait: 10
"""
    with pytest.raises(ScriptError, match="unknown key 'notakey'"):
        build_script(yaml.safe_load(src))


@pytest.mark.skipif(sys.platform != "win32", reason="key names resolve through the Windows layout")
def test_unknown_button_param_is_caught_at_load():
    from reflex.dsl import Builder, DSLError, build_macro_set

    raw = yaml.safe_load("""
macro:
  id: d
  kind: dialog_advance_loop
  params:
    dialog: {predicate: always}
    advance_button: notakey
""")
    with pytest.raises(DSLError, match="unknown key 'notakey'"):
        build_macro_set(raw, builder=Builder())
