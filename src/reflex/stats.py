"""
Summarise a reflex events.jsonl stream (live or replayed).

The JSONL written by `JsonlEventSink` is the monitoring contract: this
module, the `reflex stats` CLI and any dashboard read it.
Health is derived, not asserted:

    bad   — a reflex tripped, a spec was disabled (unsupported action), or
            the verify rate of any checked reflex fell below 50 %
    warn  — any unverified fire, predicate/input errors, or reflexes that
            fire with NO expectation (unvalidated behaviour)
    ok    — otherwise
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ERROR_KINDS = {
    "reflex.predicate_error",
    "reflex.gate_error",
    "reflex.guard_error",
    "reflex.action_factory_error",
    "reflex.input_error",
    "reflex.context_error",
    "reflex.expect_error",
    "reflex.goal_error",
}


def read_events(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> list[dict[str, Any]]:
    """Read (the tail of) an events file; tolerant of a torn last line."""
    path = Path(path)
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            fh.readline()  # drop the partial first line
        data = fh.read()
    out: list[dict[str, Any]] = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and "kind" in rec:
            out.append(rec)
    return out


def sessions(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, dict[str, Any]] = {}
    for e in events:
        s = str(e.get("session", "?"))
        d = by.setdefault(s, {"session": s, "first_ts_ns": e.get("ts_ns"), "last_ts_ns": e.get("ts_ns"), "events": 0, "fires": 0})
        d["events"] += 1
        d["last_ts_ns"] = e.get("ts_ns", d["last_ts_ns"])
        if e["kind"] == "reflex.fired":
            d["fires"] += 1
    return sorted(by.values(), key=lambda d: d["last_ts_ns"] or 0)


def summarize(events: list[dict[str, Any]], *, session: str | None = None, last: int = 20) -> dict[str, Any]:
    all_sessions = sessions(events)
    if session is None and all_sessions:
        session = all_sessions[-1]["session"]
    evs = [e for e in events if str(e.get("session", "?")) == session] if session else []

    specs: dict[str, dict[str, Any]] = {}

    def spec(name: str) -> dict[str, Any]:
        return specs.setdefault(
            name,
            {
                "name": name, "kind": None, "priority": None, "expect": None, "fires": 0, "verified": 0,
                "unverified": 0, "verify_latency_ms": [], "tripped": False, "disabled": None, "timed_out": False,
                "errors": 0, "last_fire_ts_ns": None, "last_action": None, "all_ok_false": 0,
            },
        )

    guards: dict[str, dict[str, Any]] = {}
    guarded_now: str | None = None
    errors: dict[str, int] = {}
    tick_windows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []

    for e in evs:
        k = e["kind"]
        name = e.get("name")
        if k == "reflex.registered" and name:
            s = spec(name)
            s["kind"] = e.get("kind_") or e.get("spec_kind") or s["kind"]
            s["priority"] = e.get("priority")
            s["expect"] = (e.get("expect") or {}).get("name") if isinstance(e.get("expect"), dict) else e.get("expect")
        elif k == "reflex.fired" and name:
            s = spec(name)
            s["fires"] += 1
            s["last_fire_ts_ns"] = e.get("ts_ns")
            s["last_action"] = e.get("action")
            if e.get("all_ok") is False:
                s["all_ok_false"] += 1
        elif k == "reflex.verified" and name:
            s = spec(name)
            s["verified"] += 1
            if e.get("latency_ms") is not None:
                s["verify_latency_ms"].append(float(e["latency_ms"]))
        elif k == "reflex.unverified" and name:
            spec(name)["unverified"] += 1
        elif k == "reflex.tripped" and name:
            spec(name)["tripped"] = True
        elif k == "reflex.unsupported_action" and name:
            spec(name)["disabled"] = f"unsupported_action needs {e.get('needs')}"
        elif k == "reflex.timeout" and name:
            spec(name)["timed_out"] = True
        elif k == "reflex.guarded":
            g = guards.setdefault(e.get("guard", "?"), {"guard": e.get("guard"), "engaged": 0, "last_ts_ns": None})
            g["engaged"] += 1
            g["last_ts_ns"] = e.get("ts_ns")
            guarded_now = e.get("guard")
        elif k == "reflex.guard_cleared":
            guarded_now = None
        elif k == "reflex.tick_stats":
            tick_windows.append(
                {key: e.get(key) for key in ("ts_ns", "window_s", "ticks", "tick_hz", "tick_ms_p50", "tick_ms_p95",
                                              "tick_ms_max", "eval_ms_p50", "eval_ms_p95", "suppressed", "specs_armed")}
            )
        elif k == "reflex.run_end":
            runs.append({"ts_ns": e.get("ts_ns"), "status": e.get("status"), "elapsed_s": e.get("elapsed_s"),
                         "fires": e.get("fires")})
        if k in ERROR_KINDS:
            errors[k] = errors.get(k, 0) + 1
            if name:
                spec(name)["errors"] += 1

    rows = []
    for s in specs.values():
        checked = s["verified"] + s["unverified"]
        lat = sorted(s.pop("verify_latency_ms"))
        s["verify_rate"] = round(s["verified"] / checked, 4) if checked else None
        s["verify_latency_ms_p50"] = lat[len(lat) // 2] if lat else None
        s["unvalidated"] = s["fires"] > 0 and s["expect"] is None
        if s["tripped"] or s["disabled"] or s["verify_rate"] is not None and s["verify_rate"] < 0.5:
            s["health"] = "bad"
        elif s["unverified"] or s["errors"] or s["unvalidated"] or s["all_ok_false"]:
            s["health"] = "warn"
        else:
            s["health"] = "ok"
        rows.append(s)
    rows.sort(key=lambda r: (-(r["priority"] or 0), r["name"]))

    fires = sum(r["fires"] for r in rows)
    verified = sum(r["verified"] for r in rows)
    unverified = sum(r["unverified"] for r in rows)
    checked = verified + unverified
    order = {"ok": 0, "warn": 1, "bad": 2}
    health = "ok"
    for r in rows:
        if order[r["health"]] > order[health]:
            health = r["health"]
    if errors and health == "ok":
        health = "warn"

    latest_tick = tick_windows[-1] if tick_windows else None
    recent = [
        {k: v for k, v in e.items() if k not in ("session", "match_meta", "evidence", "candidate")}
        for e in evs[-last:]
    ] if last else []

    return {
        "session": session,
        "sessions": all_sessions[-20:],
        "health": health,
        "totals": {
            "events": len(evs),
            "fires": fires,
            "verified": verified,
            "unverified": unverified,
            "verify_rate": round(verified / checked, 4) if checked else None,
            "unvalidated_fires": sum(r["fires"] for r in rows if r["unvalidated"]),
            "tripped": sum(1 for r in rows if r["tripped"]),
            "errors": sum(errors.values()),
            "guard_engagements": sum(g["engaged"] for g in guards.values()),
        },
        "specs": rows,
        "guards": sorted(guards.values(), key=lambda g: g["guard"] or ""),
        "guarded_now": guarded_now,
        "errors": errors,
        "tick": {
            "latest": latest_tick,
            "windows": tick_windows[-120:],
            "p95_max": max((w.get("tick_ms_p95") or 0.0) for w in tick_windows) if tick_windows else None,
        },
        "runs": runs[-10:],
        "recent": recent,
    }


def summarize_file(path: Path, *, session: str | None = None, last: int = 20) -> dict[str, Any]:
    out = summarize(read_events(Path(path)), session=session, last=last)
    p = Path(path)
    out["file"] = str(p)
    out["file_bytes"] = p.stat().st_size if p.exists() else 0
    out["file_mtime"] = os.path.getmtime(p) if p.exists() else None
    return out


def render_summary(s: dict[str, Any]) -> str:
    t = s["totals"]
    vr = "n/a" if t["verify_rate"] is None else f"{t['verify_rate'] * 100:.0f}%"
    lines = [
        f"session {s['session']}  health={s['health'].upper()}",
        f"  fires={t['fires']} verified={t['verified']} unverified={t['unverified']} verify_rate={vr} "
        f"unvalidated={t['unvalidated_fires']} tripped={t['tripped']} errors={t['errors']} guard_engagements={t['guard_engagements']}",
    ]
    lt = s["tick"]["latest"]
    if lt:
        lines.append(f"  tick p50={lt['tick_ms_p50']}ms p95={lt['tick_ms_p95']}ms hz={lt['tick_hz']} armed={lt['specs_armed']}")
    if s["guarded_now"]:
        lines.append(f"  GUARDED NOW by {s['guarded_now']}")
    lines.append(f"  {'reflex':<24}{'health':<7}{'fires':>6}{'ver':>5}{'unv':>5}{'rate':>7}  expect / notes")
    for r in s["specs"]:
        rate = "-" if r["verify_rate"] is None else f"{r['verify_rate'] * 100:.0f}%"
        notes = r["expect"] or "NO EXPECT"
        if r["tripped"]:
            notes += "  TRIPPED"
        if r["disabled"]:
            notes += f"  DISABLED ({r['disabled']})"
        if r["errors"]:
            notes += f"  errors={r['errors']}"
        lines.append(f"  {r['name']:<24}{r['health']:<7}{r['fires']:>6}{r['verified']:>5}{r['unverified']:>5}{rate:>7}  {notes}")
    if s["errors"]:
        lines.append(f"  errors: {s['errors']}")
    return "\n".join(lines)
