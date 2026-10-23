"""
Replay validation — prove macro behaviour on recorded frames, in CI.

A scenario couples macros with a frame sequence and states what must happen:

    name: dialog advance closes the box
    macros: ../macros/dialog.yaml          # file/dir, relative to the scenario;
                                           # or an inline `macros:` list / {macros:, guards:, context:}
    frames: {dir: frames/dialog, fps: 10}  # PNGs (sorted) or a recorded dir with index.jsonl
    duration_s: 4                          # default: last frame time + 1 s
    context: {scene: dialog}               # static view.extras for every tick
    react:                                 # optional closed-loop: an input moves the playhead
      - {input: a, advance: 1}             # or {input: a, goto_frame: 3}
    expect:
      - fires: {macro: dialog_advance, at_least: 1, at_most: 3}
      - first_fire: {macro: dialog_advance, action: [a], frame: 1}   # or within_s: [0.0, 0.5]
      - never_fires: block_on_red
      - verified: {macro: dialog_advance, at_least: 1}
      - unverified: {macro: dialog_advance, at_most: 0}
      - tripped: {macro: dialog_advance, is: false}
      - guarded: {guard: shop, at_least: 1}
      - status: all_done

Everything runs on a virtual clock, so a 60 s recording replays in
milliseconds and results are bit-for-bit deterministic.

Recording: `SessionRecorder` wraps a live frame source + event sink and
writes frames (only when they change), index.jsonl, events.jsonl and a
golden scenario.yaml whose expectations are the fires that actually
happened — so a live session becomes a regression test.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from reflex.actions import Action, labels
from reflex.dsl import Builder, DSLError, MacroSet, build_macro_set, load_macro_set
from reflex.expect import frame_diff_fraction
from reflex.io import JsonlEventSink, RecordingActionSink, RecordingEventSink, check_frame, jsonable
from reflex.runtime import ReflexRuntime


class ScenarioError(ValueError):
    pass


# --------------------------------------------------------------------------- frames


def load_frames(spec: Any, *, base_dir: Path) -> list[tuple[float, np.ndarray]]:
    """Load a frame sequence.

    spec: "dir" | {dir, fps?} | {files: [...], fps?}. A recorded dir has
    index.jsonl lines {"i", "t", "file"} and those timestamps win.
    """
    if isinstance(spec, str):
        spec = {"dir": spec}
    if not isinstance(spec, dict):
        raise ScenarioError(f"frames must be a dir or mapping, got {spec!r}")
    fps = float(spec.get("fps", 10.0))
    if fps <= 0:
        raise ScenarioError("frames.fps must be > 0")
    files: list[Path]
    times: list[float] | None = None
    if "files" in spec:
        files = [(base_dir / f) if not Path(f).is_absolute() else Path(f) for f in spec["files"]]
    elif "dir" in spec:
        d = Path(spec["dir"])
        d = d if d.is_absolute() else base_dir / d
        if not d.is_dir():
            raise ScenarioError(f"frames dir not found: {d}")
        index = d / "index.jsonl"
        if index.exists():
            rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
            files = [d / r["file"] for r in rows]
            times = [float(r["t"]) for r in rows]
        else:
            files = sorted([*d.glob("*.png"), *d.glob("*.jpg")])
    else:
        raise ScenarioError("frames needs `dir` or `files`")
    if not files:
        raise ScenarioError("frame sequence is empty")
    out: list[tuple[float, np.ndarray]] = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            raise ScenarioError(f"cannot read frame {f}")
        t = times[i] if times is not None else i / fps
        out.append((t, img))
    return out


class ReplayWorld:
    """Frames on a virtual clock; optional react rules move the playhead."""

    def __init__(self, frames: Sequence[tuple[float, np.ndarray]]) -> None:
        self.frames = sorted(((float(t), check_frame(f, who="replay frame")) for t, f in frames), key=lambda p: p[0])
        self.t = 0.0
        self._offset = 0.0
        self.index = 0
        self.decision_index = 0
        """Frame index of the last grab — the frame a fire was decided on."""
        self.jumps: list[dict[str, Any]] = []

    # clock
    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(0.0, float(dt))

    def ts_ns(self) -> int:
        return int(self.t * 1e9)

    # source
    def _index_at(self, tt: float) -> int:
        idx = 0
        for i, (ts, _f) in enumerate(self.frames):
            if ts <= tt + 1e-9:
                idx = i
            else:
                break
        return idx

    def grab(self) -> np.ndarray:
        self.index = self._index_at(self.t + self._offset)
        self.decision_index = self.index
        return self.frames[self.index][1]

    def goto(self, frame_index: int, *, reason: str) -> None:
        k = max(0, min(len(self.frames) - 1, int(frame_index)))
        self._offset = self.frames[k][0] - self.t
        self.index = k
        self.jumps.append({"t": round(self.t, 4), "to": k, "reason": reason})

    @property
    def end_t(self) -> float:
        return self.frames[-1][0]


class _ReactiveSink(RecordingActionSink):
    """Records actions; `react` rules turn an input into a playhead jump."""

    def __init__(self, world: ReplayWorld, rules: list[dict[str, Any]]) -> None:
        super().__init__(clock=world.now)
        self.world = world
        self.rules = rules

    def _rec(self, kind: str, **kw: Any) -> bool:
        kw["frame"] = self.world.decision_index
        ok = super()._rec(kind, **kw)
        label = kw.get("button") if kind in ("key", "key_down") else (f"click:{kw.get('button')}" if kind == "click" else None)
        for r in self.rules:
            if r["on"] != label:
                continue
            if "goto_frame" in r:
                self.world.goto(int(r["goto_frame"]), reason=f"react:{label}")
            elif "advance" in r:
                self.world.goto(self.world.index + int(r["advance"]), reason=f"react:{label}")
            break
        return ok


class _AnnotatingEvents:
    """Stamps each runtime event with virtual time + frame index."""

    def __init__(self, world: ReplayWorld, inner: RecordingEventSink, extra: Any | None = None) -> None:
        self.world = world
        self.inner = inner
        self.extra = extra

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        body = dict(payload)
        body.setdefault("t", round(self.world.t, 4))
        body.setdefault("frame", self.world.decision_index)
        self.inner.emit(kind, body)
        if self.extra is not None:
            self.extra.emit(kind, body)


# --------------------------------------------------------------------------- scenario


@dataclass
class Check:
    check: str
    ok: bool
    detail: str


@dataclass
class ScenarioResult:
    name: str
    path: str
    passed: bool
    checks: list[Check]
    status: str
    elapsed_virtual_s: float
    wall_ms: float
    fires: list[dict[str, Any]]
    stats: dict[str, dict[str, Any]]
    guard_hits: dict[str, int]
    frames: int
    error: str | None = None
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(
            {
                "name": self.name,
                "path": self.path,
                "passed": self.passed,
                "status": self.status,
                "error": self.error,
                "frames": self.frames,
                "elapsed_virtual_s": self.elapsed_virtual_s,
                "wall_ms": self.wall_ms,
                "checks": [c.__dict__ for c in self.checks],
                "fires": self.fires,
                "stats": self.stats,
                "guard_hits": self.guard_hits,
            }
        )

    def render(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        lines = [f"{head} {self.name}  ({self.path})",
                 f"     status={self.status} frames={self.frames} virtual={self.elapsed_virtual_s:.2f}s wall={self.wall_ms:.0f}ms"]
        if self.error:
            lines.append(f"     error: {self.error}")
        for c in self.checks:
            lines.append(f"     [{'ok' if c.ok else 'XX'}] {c.check}: {c.detail}")
        for f in self.fires[:12]:
            lines.append(f"       fire t={f['t']:.2f}s frame={f['frame']} {f['name']} {f['action']} -> {f.get('verdict', '-')}")
        if len(self.fires) > 12:
            lines.append(f"       … {len(self.fires) - 12} more fires")
        return "\n".join(lines)


def _count_check(label: str, n: int, rule: dict[str, Any]) -> Check:
    lo = rule.get("at_least")
    hi = rule.get("at_most")
    eq = rule.get("count")
    ok = True
    want = []
    if eq is not None:
        ok &= n == int(eq)
        want.append(f"=={eq}")
    if lo is not None:
        ok &= n >= int(lo)
        want.append(f">={lo}")
    if hi is not None:
        ok &= n <= int(hi)
        want.append(f"<={hi}")
    if not want:
        ok = n >= 1
        want.append(">=1")
    return Check(label, bool(ok), f"got {n}, want {' and '.join(want)}")


def _evaluate(expectations: list[Any], *, fires: list[dict[str, Any]], stats: dict[str, dict[str, Any]],
              guard_hits: dict[str, int], status: str, macro_names: list[str]) -> list[Check]:
    checks: list[Check] = []

    def need_macro(rule: dict[str, Any], label: str) -> str | None:
        m = rule.get("macro")
        if m not in macro_names:
            checks.append(Check(label, False, f"unknown macro {m!r} (have {macro_names})"))
            return None
        return m

    for i, raw in enumerate(expectations):
        if not isinstance(raw, dict) or len(raw) != 1:
            checks.append(Check(f"expect[{i}]", False, f"each expectation is a one-key mapping, got {raw!r}"))
            continue
        (key, rule), = raw.items()
        if key == "fires":
            m = need_macro(rule, f"fires[{rule.get('macro')}]")
            if m:
                checks.append(_count_check(f"fires[{m}]", stats[m]["fires"], rule))
        elif key in ("verified", "unverified"):
            m = need_macro(rule, f"{key}[{rule.get('macro')}]")
            if m:
                checks.append(_count_check(f"{key}[{m}]", stats[m][key], rule))
        elif key == "never_fires":
            names = rule if isinstance(rule, list) else [rule]
            for m in names:
                if m not in macro_names:
                    checks.append(Check(f"never_fires[{m}]", False, f"unknown macro {m!r}"))
                    continue
                n = stats[m]["fires"]
                checks.append(Check(f"never_fires[{m}]", n == 0, f"got {n} fires"))
        elif key == "tripped":
            m = need_macro(rule, f"tripped[{rule.get('macro')}]")
            if m:
                want = bool(rule.get("is", True))
                got = bool(stats[m]["tripped"])
                checks.append(Check(f"tripped[{m}]", got == want, f"got {got}, want {want}"))
        elif key == "guarded":
            g = rule.get("guard")
            checks.append(_count_check(f"guarded[{g}]", int(guard_hits.get(g, 0)), rule))
        elif key == "status":
            checks.append(Check("status", status == rule, f"got {status!r}, want {rule!r}"))
        elif key == "first_fire":
            m = need_macro(rule, f"first_fire[{rule.get('macro')}]")
            if not m:
                continue
            mine = [f for f in fires if f["name"] == m]
            if not mine:
                checks.append(Check(f"first_fire[{m}]", False, "never fired"))
                continue
            f0 = mine[0]
            problems = []
            if "action" in rule and list(f0["action"]) != [str(a) for a in rule["action"]]:
                problems.append(f"action {f0['action']} != {rule['action']}")
            if "frame" in rule and f0["frame"] != int(rule["frame"]):
                problems.append(f"frame {f0['frame']} != {rule['frame']}")
            if "within_s" in rule:
                lo, hi = (float(x) for x in rule["within_s"])
                if not (lo <= f0["t"] <= hi):
                    problems.append(f"t {f0['t']:.3f}s not in [{lo}, {hi}]")
            detail = "; ".join(problems) if problems else f"t={f0['t']:.3f}s frame={f0['frame']} action={f0['action']}"
            checks.append(Check(f"first_fire[{m}]", not problems, detail))
        elif key == "order":
            want = [str(x) for x in rule]
            got = [f["name"] for f in fires if f["name"] in want]
            dedup = [g for j, g in enumerate(got) if j == 0 or got[j - 1] != g]
            ok = dedup[: len(want)] == want
            checks.append(Check("order", ok, f"fire order {dedup} vs {want}"))
        else:
            checks.append(Check(f"expect[{i}]", False, f"unknown expectation {key!r}"))
    return checks


def _load_macros(raw: Any, *, base_dir: Path, where: str, bindings: dict[str, Any] | None = None) -> MacroSet:
    if isinstance(raw, str):
        p = Path(raw)
        return load_macro_set(p if p.is_absolute() else base_dir / p, bindings=bindings)
    if isinstance(raw, list):
        raw = {"macros": raw}
    if isinstance(raw, dict):
        return build_macro_set(raw, builder=Builder(base_dir=base_dir, where=where, bindings=bindings))
    raise ScenarioError("scenario `macros` must be a path, a list, or a mapping")


def run_scenario(scn: dict[str, Any], *, base_dir: Path, where: str = "<scenario>",
                 event_sink: Any | None = None, bindings: dict[str, Any] | None = None) -> ScenarioResult:
    w0 = time.perf_counter()
    name = str(scn.get("name") or Path(where).stem)
    allowed = {"name", "description", "macros", "frames", "duration_s", "context", "react", "expect",
               "stop_when_all_done", "bindings_required"}
    unknown = sorted(set(scn) - allowed)
    if unknown:
        raise ScenarioError(f"{where}: unknown scenario key(s) {unknown}; allowed {sorted(allowed)}")
    try:
        ms = _load_macros(scn.get("macros"), base_dir=base_dir, where=where, bindings=bindings)
    except DSLError as exc:
        raise ScenarioError(str(exc)) from None
    if not ms.specs:
        raise ScenarioError(f"{where}: scenario has no macros")
    world = ReplayWorld(load_frames(scn.get("frames"), base_dir=base_dir))
    rules = []
    for r in scn.get("react") or []:
        # `input:` names the action label; a bare YAML `on:` key parses as True, accept it too.
        if isinstance(r, dict) and True in r and "input" not in r:
            r = {("input" if k is True else k): v for k, v in r.items()}
        if not isinstance(r, dict) or "input" not in r:
            raise ScenarioError(f"{where}: react rules need `input:` (got {r!r})")
        if not ({"goto_frame", "advance"} & set(r)):
            raise ScenarioError(f"{where}: react rule {r!r} needs goto_frame or advance")
        rules.append({"on": str(r["input"]), **{k: v for k, v in r.items() if k != "input"}})
    sink = _ReactiveSink(world, rules)
    rec = RecordingEventSink()
    events = _AnnotatingEvents(world, rec, event_sink)
    context = dict(ms.context)
    static = scn.get("context")
    if static:
        fixed = dict(static)
        context["_scenario_static"] = lambda _v: dict(fixed)
    rt = ReflexRuntime(source=world, sink=sink, events=events, clock=world.now, sleep=world.sleep,
                       ts_ns=world.ts_ns, guards=ms.guards, context_providers=context,
                       tick_stats_every_s=0)
    for s in ms.specs:
        rt.register(s)
    duration = float(scn.get("duration_s", world.end_t + 1.0))
    error = None
    status = "error"
    try:
        res = rt.run(timeout_s=duration, stop_when_all_done=bool(scn.get("stop_when_all_done", True)))
        status = res.status
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)

    stats = rt.stats()
    fires = []
    for k, p in rec.events:
        if k == "reflex.fired":
            fires.append({"t": p["t"], "frame": p["frame"], "name": p["name"], "action": p["action"],
                          "fire_index": p["fires_total"]})
    verdicts = {(p["name"], p["fire_index"]): k.split(".")[1] for k, p in rec.events
                if k in ("reflex.verified", "reflex.unverified")}
    for f in fires:
        f["verdict"] = verdicts.get((f["name"], f["fire_index"]), "unchecked" if stats[f["name"]].get("pending") is False else "pending")
    checks = _evaluate(list(scn.get("expect") or []), fires=fires, stats=stats, guard_hits=rt.guard_hits(),
                       status=status, macro_names=[s.name for s in ms.specs])
    if not scn.get("expect"):
        checks.append(Check("expect", False, "scenario declares no expectations — nothing is validated"))
    passed = error is None and all(c.ok for c in checks)
    return ScenarioResult(
        name=name, path=where, passed=passed, checks=checks, status=status,
        elapsed_virtual_s=round(world.t, 4), wall_ms=round((time.perf_counter() - w0) * 1000.0, 2),
        fires=fires, stats=stats, guard_hits=rt.guard_hits(), frames=len(world.frames), error=error,
        events=rec.events,
    )


def run_scenario_file(path: Path, *, write_events: str | None = None,
                      bindings: dict[str, Any] | None = None) -> ScenarioResult:
    path = Path(path)
    try:
        scn = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(scn, dict):
            raise ScenarioError(f"{path}: scenario must be a mapping")
        sink = JsonlEventSink(write_events, session=f"replay:{path.stem}") if write_events else None
        return run_scenario(scn, base_dir=path.parent, where=str(path), event_sink=sink, bindings=bindings)
    except (ScenarioError, yaml.YAMLError, OSError) as exc:
        return ScenarioResult(name=path.stem, path=str(path), passed=False,
                              checks=[Check("load", False, str(exc))], status="error",
                              elapsed_virtual_s=0.0, wall_ms=0.0, fires=[], stats={}, guard_hits={},
                              frames=0, error=str(exc))


# --------------------------------------------------------------------------- recording


class SessionRecorder:
    """Record a live session so it can be replayed as a regression test.

    Wrap the live frame source (`recorder.source(inner)`) and add
    `recorder.events` to the runtime's event fan-out. Frames are written
    only when they differ from the last written one (or every
    `keyframe_s`), so an idle screen costs nothing.
    """

    def __init__(self, out_dir: Path, *, clock: Callable[[], float] = time.monotonic,
                 min_change_fraction: float = 0.002, keyframe_s: float = 2.0, max_frames: int = 5000,
                 jpeg: bool = False) -> None:
        self.dir = Path(out_dir)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._t0 = clock()
        self._last: np.ndarray | None = None
        self._last_t = -1e9
        self.min_change = float(min_change_fraction)
        self.keyframe_s = float(keyframe_s)
        self.max_frames = int(max_frames)
        self.ext = ".jpg" if jpeg else ".png"
        self.n = 0
        self.dropped = 0
        self._index = (self.dir / "frames" / "index.jsonl").open("a", encoding="utf-8")
        self.events = _RecorderEvents(self, JsonlEventSink(self.dir / "events.jsonl", session=f"rec:{self.dir.name}"))
        self.fired: list[dict[str, Any]] = []

    def t(self) -> float:
        return self._clock() - self._t0

    def observe(self, frame: np.ndarray) -> None:
        t = self.t()
        if self._last is not None:
            changed = frame_diff_fraction(frame, self._last) >= self.min_change
            if not changed and (t - self._last_t) < self.keyframe_s:
                return
        if self.n >= self.max_frames:
            self.dropped += 1
            return
        fname = f"{self.n:06d}{self.ext}"
        cv2.imwrite(str(self.dir / "frames" / fname), frame)
        self._index.write(json.dumps({"i": self.n, "t": round(t, 4), "file": fname}) + "\n")
        self._index.flush()
        self._last = frame.copy()
        self._last_t = t
        self.n += 1

    def source(self, inner: Any) -> _RecordingSource:
        return _RecordingSource(inner, self)

    def close(self, *, macros_path: str | None = None, name: str | None = None) -> Path:
        """Finish and write a golden scenario.yaml asserting what happened."""
        self._index.close()
        per: dict[str, int] = {}
        for f in self.fired:
            per[f["name"]] = per.get(f["name"], 0) + 1
        expect: list[dict[str, Any]] = [{"fires": {"macro": m, "count": n}} for m, n in sorted(per.items())]
        firsts: dict[str, dict[str, Any]] = {}
        for f in self.fired:
            firsts.setdefault(f["name"], f)
        for m, f in sorted(firsts.items()):
            expect.append({"first_fire": {"macro": m, "action": f["action"],
                                          "within_s": [round(max(0.0, f["t"] - 0.25), 3), round(f["t"] + 0.25, 3)]}})
        scn = {
            "name": name or f"recorded {self.dir.name}",
            "description": "Golden scenario generated by SessionRecorder; review before trusting.",
            "macros": macros_path or "REPLACE_ME.yaml",
            "frames": {"dir": "frames"},
            "duration_s": round(self.t() + 0.5, 3),
            "stop_when_all_done": False,
            "expect": expect or [{"status": "timed_out"}],
        }
        out = self.dir / "scenario.yaml"
        out.write_text(yaml.safe_dump(scn, sort_keys=False), encoding="utf-8")
        return out


class _RecordingSource:
    def __init__(self, inner: Any, rec: SessionRecorder) -> None:
        self.inner = inner
        self.rec = rec

    def grab(self) -> np.ndarray:
        f = self.inner.grab()
        try:
            self.rec.observe(f)
        except Exception:  # noqa: BLE001 — recording must never break the live loop
            self.rec.dropped += 1
        return f


class _RecorderEvents:
    def __init__(self, rec: SessionRecorder, sink: JsonlEventSink) -> None:
        self.rec = rec
        self.sink = sink

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        body = dict(payload)
        body.setdefault("t", round(self.rec.t(), 4))
        if kind == "reflex.fired":
            self.rec.fired.append({"name": body.get("name"), "action": body.get("action"), "t": body["t"]})
        self.sink.emit(kind, body)


def action_labels(actions: Sequence[Action]) -> tuple[str, ...]:
    return labels(actions)
