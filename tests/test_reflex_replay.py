"""Replay validation harness: real-frame scenarios, failure detection, record → replay round trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reflex.cli import main as cli_main
from reflex.dsl import load_macro_set
from reflex.io import FanoutEventSink, RecordingActionSink, RecordingEventSink
from reflex.replay import SessionRecorder, run_scenario, run_scenario_file
from reflex.runtime import ReflexRuntime

FIX = Path(__file__).resolve().parent / "fixtures" / "dialog"
SCENARIOS = sorted(FIX.glob("*.scenario.yaml"))


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_shipped_real_frame_scenarios_pass(path):
    res = run_scenario_file(path)
    assert res.passed, res.render()


def test_advance_scenario_fires_on_the_frame_it_saw():
    res = run_scenario_file(FIX / "advance_works.scenario.yaml")
    assert [f["frame"] for f in res.fires] == [0, 1, 2, 3]
    assert all(f["verdict"] == "verified" for f in res.fires)


def test_replay_is_deterministic():
    a = run_scenario_file(FIX / "advance_works.scenario.yaml")
    b = run_scenario_file(FIX / "advance_works.scenario.yaml")
    assert [(f["t"], f["frame"]) for f in a.fires] == [(f["t"], f["frame"]) for f in b.fires]


def _scn(**over):
    base = yaml.safe_load((FIX / "advance_works.scenario.yaml").read_text(encoding="utf-8"))
    base.update(over)
    return base


def test_wrong_expectation_is_reported_not_passed():
    res = run_scenario(_scn(expect=[{"fires": {"macro": "dialog_advance", "count": 2}}]), base_dir=FIX)
    assert not res.passed
    assert "got 4, want ==2" in res.checks[0].detail


def test_scenario_without_expectations_fails():
    res = run_scenario(_scn(expect=[]), base_dir=FIX)
    assert not res.passed
    assert any("no expectations" in c.detail for c in res.checks)


def test_unknown_macro_in_expectation_fails():
    res = run_scenario(_scn(expect=[{"never_fires": "does_not_exist"}]), base_dir=FIX)
    assert not res.passed


def test_bad_scenario_file_reports_load_error(tmp_path):
    p = tmp_path / "bad.scenario.yaml"
    p.write_text("name: x\nmacros: nope.yaml\nframes: {dir: frames}\nexpect: [{status: all_done}]\n", encoding="utf-8")
    res = run_scenario_file(p)
    assert not res.passed and res.checks[0].check == "load"


def test_cli_replay_exit_codes(tmp_path, capsys):
    assert cli_main(["replay", *map(str, SCENARIOS)]) == 0
    bad = tmp_path / "wrong.scenario.yaml"
    scn = _scn(expect=[{"fires": {"macro": "dialog_advance", "count": 99}}], macros=str(FIX / "macros.yaml"),
               frames={"dir": str(FIX / "frames"), "fps": 0.2})
    bad.write_text(yaml.safe_dump(scn), encoding="utf-8")
    assert cli_main(["replay", str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_record_live_session_then_replay_golden_scenario(tmp_path):
    """A 'live' run (synthetic world on a virtual clock) is recorded; the generated
    golden scenario replays against the recording and passes."""
    import cv2

    frames = [cv2.imread(str(p)) for p in sorted((FIX / "frames").glob("*.png"))]

    class LiveWorld:
        def __init__(self):
            self.t = 0.0
            self.i = 0

        def now(self):
            return self.t

        def sleep(self, dt):
            self.t += dt

        def grab(self):
            return frames[self.i]

    world = LiveWorld()

    class Sink(RecordingActionSink):
        def press(self, button, *, hold_ms):
            world.i = min(world.i + 1, len(frames) - 1)  # A advances the dialog, like the real game
            return super().press(button, hold_ms=hold_ms)

    out = tmp_path / "session"
    rec = SessionRecorder(out, clock=world.now)
    ms = load_macro_set(FIX / "macros.yaml")
    ev = RecordingEventSink()
    rt = ReflexRuntime(source=rec.source(world), sink=Sink(clock=world.now), events=FanoutEventSink(ev, rec.events),
                       clock=world.now, sleep=world.sleep, ts_ns=lambda: int(world.t * 1e9), guards=ms.guards)
    for s in ms.specs:
        rt.register(s)
    rt.run(timeout_s=3.0, stop_when_all_done=False)
    scenario = rec.close(macros_path=str(FIX / "macros.yaml"), name="golden")

    assert rt.stats()["dialog_advance"]["fires"] == 4
    assert (out / "events.jsonl").exists() and (out / "frames" / "index.jsonl").exists()
    # only changed frames are written: 5 distinct screens (+ keyframes while idle)
    assert 5 <= rec.n <= 8
    golden = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    assert {"fires": {"macro": "dialog_advance", "count": 4}} in golden["expect"]

    # Open-loop replay of the recording: the recorded frames already contain the
    # effect of each press, so the same macros must make the same decisions.
    res = run_scenario_file(scenario)
    assert res.passed, res.render()
    assert res.stats["dialog_advance"]["verified"] == 4
