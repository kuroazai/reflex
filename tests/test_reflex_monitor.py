"""Monitoring: JSONL contract, stats/health, perf bridge, dashboard API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflex.io import JsonlEventSink
from reflex.replay import run_scenario_file
from reflex.stats import read_events, render_summary, summarize, summarize_file

FIX = Path(__file__).resolve().parent / "fixtures" / "dialog"


@pytest.fixture()
def events_file(tmp_path) -> Path:
    ev = tmp_path / "events.jsonl"
    for s in sorted(FIX.glob("*.scenario.yaml")):
        run_scenario_file(s, write_events=str(ev))
    return ev


def test_jsonl_envelope_is_never_overwritten_by_payload(tmp_path):
    p = tmp_path / "e.jsonl"
    sink = JsonlEventSink(p, session="s1")
    sink.emit("reflex.registered", {"name": "x", "kind": "on_match", "session": "evil", "ts_ns": 1})
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["kind"] == "reflex.registered" and rec["session"] == "s1"
    assert rec["spec_kind"] == "on_match" and rec["payload_session"] == "evil" and rec["payload_ts_ns"] == 1


def test_read_events_tolerates_torn_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"kind":"reflex.fired","name":"a","session":"s"}\n{"kind": "reflex.fi', encoding="utf-8")
    assert len(read_events(p)) == 1


def test_health_per_session(events_file):
    good = summarize_file(events_file, session="replay:advance_works.scenario")
    assert good["health"] == "ok"
    assert good["totals"]["verify_rate"] == 1.0 and good["totals"]["fires"] == 4
    (row,) = good["specs"]
    assert row["expect"] == "any_of" and row["kind"] == "dialog_advance_loop"
    assert row["verify_latency_ms_p50"] is not None

    bad = summarize_file(events_file, session="replay:dead_input_trips.scenario")
    assert bad["health"] == "bad" and bad["totals"]["tripped"] == 1
    assert "TRIPPED" in render_summary(bad)

    guarded = summarize_file(events_file, session="replay:guard_blocks.scenario")
    assert guarded["totals"]["guard_engagements"] == 1 and guarded["guarded_now"] == "spend_screen"


def test_latest_session_is_default(events_file):
    s = summarize_file(events_file)
    assert s["session"] == "replay:guard_blocks.scenario"
    assert [x["session"] for x in s["sessions"]][-1] == s["session"]


def test_unvalidated_fires_warn():
    evs = [
        {"kind": "reflex.registered", "name": "blind", "session": "s", "priority": 5, "expect": None},
        {"kind": "reflex.fired", "name": "blind", "session": "s", "action": ["a"], "all_ok": True},
    ]
    s = summarize(evs)
    assert s["health"] == "warn" and s["totals"]["unvalidated_fires"] == 1
