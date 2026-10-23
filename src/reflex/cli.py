"""
`reflex` command line (also `python -m reflex`).

    reflex run <script.yaml> [--title T] [--dry-run] [--every 30m]   run a script (or watcher file)
    reflex windows [--title T] [--json]       list windows you can target
    reflex capture --title T -o shot.png      save a window's frame (to cut templates from)
    reflex record --title T -o macro.yaml     record your input into a script
    reflex validate <file-or-dir> ...         check scripts and watcher files without running
    reflex schema [--json]                    every script step and predicate, for people and agents
    reflex doctor                             prove capture and input work on this PC
    reflex list [--kind recipe] [--json]      every plug-in + its params
    reflex replay <scenario.yaml> ... [--json] run scenarios over recorded frames
    reflex stats <events.jsonl> [--session S] [--json]  summarise a live/replayed run
    reflex mcp                                serve reflex to AI agents over MCP (stdio)

Stop any run with ctrl+alt+end; pause/resume with ctrl+alt+home.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from reflex.registry import KINDS, REGISTRY


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_list(args: argparse.Namespace) -> int:
    items = REGISTRY.describe(args.kind)
    if args.json:
        _print_json({"plugins": items, "load_errors": REGISTRY.load_errors})
        return 0
    cur = None
    for it in items:
        if it["kind"] != cur:
            cur = it["kind"]
            print(f"\n== {cur} ==")
        params = []
        for p in it["params"]:
            s = p["name"] + ("" if p["required"] else f"={p['default']!r}")
            params.append(s)
        params += [f"{n} (yaml)" for n in it["yaml_only_params"]]
        params += [f"{a}->{r}" for a, r in it.get("aliases", {}).items()]
        print(f"  {it['name']:<28} {it['doc'][:70]}")
        if params:
            print(f"  {'':<28} params: {', '.join(params)}")
        if not it["origin"].startswith("reflex."):
            print(f"  {'':<28} from: {it['origin']}")
    if REGISTRY.load_errors:
        print("\nplug-in load errors:")
        for e in REGISTRY.load_errors:
            print(f"  {e['entry_point']}: {e['error']}")
    return 0


def _yaml_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(list(p.glob("*.yaml")) + list(p.glob("*.yml"))))
        else:
            out.append(p)
    return out


def validate_file(path: Path) -> dict:
    """{'path', 'ok', 'type': script|watchers, ...} for one YAML file."""
    import yaml

    from reflex.dsl import DSLError, load_macro_set
    from reflex.script import ScriptError, is_script, load_script

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {"path": str(path), "ok": False, "error": f"cannot read: {exc}"}
    if is_script(raw):
        try:
            sc = load_script(path)
        except (ScriptError, DSLError, FileNotFoundError) as exc:
            return {"path": str(path), "ok": False, "type": "script", "error": str(exc)}
        return {"path": str(path), "ok": True, "type": "script", "name": sc.name, "steps": len(sc.steps),
                "functions": sorted(sc.functions), "watchers": [s.name for s in sc.watchers.specs],
                "target": sc.target, "timeout_s": sc.timeout_s}
    try:
        ms = load_macro_set(path)
    except DSLError as exc:
        return {"path": str(path), "ok": False, "type": "watchers", "error": str(exc)}
    return {
        "path": str(path),
        "ok": True,
        "type": "watchers",
        "macros": [
            {
                "id": s.name,
                "kind": s.extra.get("kind"),
                "priority": s.priority,
                "gates": len(s.gates),
                "expect": s.expect.describe() if s.expect else None,
                "max_fires": s.max_fires,
                "timeout_s": s.timeout_s,
            }
            for s in ms.specs
        ],
        "guards": sorted(ms.guards),
        "context": sorted(ms.context),
        "skipped": ms.skipped,
    }


def cmd_validate(args: argparse.Namespace) -> int:
    report = [validate_file(p) for p in _yaml_files(args.paths)]
    rc = 0 if all(r["ok"] for r in report) else 1
    if args.json:
        _print_json(report)
        return rc
    for r in report:
        if not r["ok"]:
            print(f"FAIL {r['path']}\n     {r['error']}")
            continue
        if r.get("type") == "script":
            print(f"OK   {r['path']}  (script {r['name']!r}: {r['steps']} steps, functions={r['functions']}, "
                  f"watchers={r['watchers']})")
            continue
        print(f"OK   {r['path']}  ({len(r['macros'])} macros, guards={r['guards']}, context={r['context']})")
        for m in r["macros"]:
            exp = m["expect"]["name"] if m["expect"] else "-- no expect (fires unverified)"
            print(f"     {m['id']:<24} {m['kind']:<26} prio={m['priority']:<3} gates={m['gates']} expect={exp}")
    return rc


def cmd_replay(args: argparse.Namespace) -> int:
    from reflex.replay import run_scenario_file

    results = [run_scenario_file(Path(p), write_events=args.events) for p in args.scenarios]
    if args.json:
        _print_json([r.to_dict() for r in results])
    else:
        for r in results:
            print(r.render())
    return 0 if all(r.passed for r in results) else 1


def cmd_stats(args: argparse.Namespace) -> int:
    from reflex.stats import render_summary, summarize_file

    summary = summarize_file(Path(args.events), session=args.session, last=args.last)
    if args.json:
        _print_json(summary)
    else:
        print(render_summary(summary))
    return 0


_DUR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*$")


def parse_duration(text: str) -> float:
    """'90' / '90s' / '30m' / '2h' / '500ms' -> seconds."""
    m = _DUR.match(str(text))
    if not m:
        raise argparse.ArgumentTypeError(f"not a duration: {text!r} (try 30s, 10m, 2h)")
    n = float(m.group(1))
    return n * {"ms": 0.001, "s": 1, None: 1, "m": 60, "h": 3600}[m.group(2)]


class _ConsoleEvents:
    """Prints the run as it happens: script logs, failures, watcher fires."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet

    def emit(self, kind: str, payload: dict) -> None:
        inst = payload.get("instance", "")
        tag = f"[{inst}] " if inst and inst != "main" else ""
        if kind == "script.log":
            print(f"{tag}{payload.get('message')}", flush=True)
        elif kind == "script.end" and payload.get("status") != "completed":
            print(f"{tag}{str(payload.get('status')).upper()}: {payload.get('error')}", flush=True)
        elif kind == "script.waiting_for_focus":
            print(f"{tag}waiting for the target window to get focus...", flush=True)
        elif kind == "script.paused":
            print(f"{tag}paused (ctrl+alt+home to resume)", flush=True)
        elif kind == "script.resumed":
            print(f"{tag}resumed", flush=True)
        elif not self.quiet and kind == "reflex.fired":
            print(f"{tag}watcher {payload.get('name')} -> {' '.join(payload.get('action') or [])}", flush=True)


def _target_overrides(args: argparse.Namespace) -> dict:
    window = {k: v for k, v in (("title", args.title), ("process", args.process), ("hwnd", args.hwnd)) if v}
    out: dict = {"input": args.input, "capture": args.capture}
    if window:
        out["window"] = window
    if args.all_windows:
        out["all_windows"] = True
    if args.screen:
        out["screen"] = "all" if args.screen == "all" else [int(v) for v in args.screen.split(",")]
    return out


def cmd_run(args: argparse.Namespace) -> int:
    import yaml

    from reflex.hotkeys import DEFAULT_PAUSE, DEFAULT_STOP, HotkeyWatcher
    from reflex.io import FanoutEventSink, JsonlEventSink, StaticFrameSource
    from reflex.script import (
        Control,
        ScriptError,
        is_script,
        load_script,
        open_targets,
        run_script,
        schedule_times,
    )

    path = Path(args.file)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    sinks: list = [_ConsoleEvents(quiet=args.json)]
    if args.events:
        sinks.append(JsonlEventSink(args.events))
    events = FanoutEventSink(*sinks)
    source = StaticFrameSource(_read_image(args.image)) if args.image else None
    control = Control()
    keys = HotkeyWatcher({DEFAULT_STOP: lambda: control.stop("hotkey ctrl+alt+end"),
                          DEFAULT_PAUSE: control.toggle_pause}).start()
    try:
        if not is_script(raw):
            return _run_watchers(args, path, events, control, source)
        try:
            script = load_script(path)
            targets = open_targets(script, overrides=_target_overrides(args), dry_run=args.dry_run, source=source)
        except (ScriptError, LookupError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not args.json:
            names = ", ".join(t.instance for t in targets)
            mode = "DRY RUN (no input sent)" if args.dry_run else f"input={type(targets[0].sink).__name__}"
            print(f"running {script.name!r} on {names}  [{mode}]  stop: ctrl+alt+end  pause: ctrl+alt+home",
                  flush=True)
        scheduled = bool(args.every or args.at)
        runs = args.times if args.times is not None else (None if scheduled else 1)
        all_results = []
        n = 0
        while runs is None or n < runs:
            if n > 0 or (scheduled and args.wait_first):
                wait = schedule_times(every_s=args.every, at=args.at) if scheduled else 0.2
                if scheduled and not args.json:
                    print(f"next run in {wait / 60:.1f} min", flush=True)
                if control.stop_event.wait(wait):
                    break
            results = run_script(script, targets=targets, events=events, control=control, hotkeys=False)
            all_results.extend(results)
            n += 1
            if control.stop_event.is_set():
                break
        if args.json:
            _print_json([r.to_dict() for r in all_results])
        else:
            for r in all_results:
                tail = f" ({r.error})" if r.error else ""
                print(f"{r.instance}: {r.status} after {r.elapsed_s:.1f}s, {r.steps_run} steps{tail}")
        return 0 if all(r.ok for r in all_results) else 1
    finally:
        keys.stop()


def _run_watchers(args, path, events, control, source) -> int:
    from reflex.dsl import load_macro_set
    from reflex.runtime import ReflexRuntime
    from reflex.script import Script, ScriptError, open_targets

    ms = load_macro_set(path)
    shell = Script(name=path.stem, steps=[], target={})
    try:
        targets = open_targets(shell, overrides=_target_overrides(args), dry_run=args.dry_run, source=source)
    except (ScriptError, LookupError) as exc:
        print(f"error: {exc} (watcher files take their target from --title/--process/--screen)", file=sys.stderr)
        return 2
    t = targets[0]
    rt = ReflexRuntime(source=t.source, sink=t.sink, events=events, guards=ms.guards,
                       context_providers=ms.context, foreground_ok=getattr(t.sink, "target_ok", None))
    for spec in ms.specs:
        rt.register(spec)
    print(f"watching with {len(ms.specs)} macros for {args.duration:g}s  stop: ctrl+alt+end", flush=True)
    try:
        res = rt.run(timeout_s=args.duration, should_stop=control.stop_event.is_set)
    finally:
        if hasattr(t.sink, "release_all"):
            t.sink.release_all()
    out = {"status": res.status, "elapsed_s": res.elapsed_s, "fires": res.fires}
    if args.json:
        _print_json(out)
    else:
        print(f"{res.status}: fires {res.fires}")
    return 0


def _read_image(path: str):
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read image {path}")
    return img


def cmd_windows(args: argparse.Namespace) -> int:
    from reflex.backends.windows import list_windows

    wins = list_windows(title=args.title, process=args.process)
    if args.json:
        _print_json([w.describe() for w in wins])
        return 0
    for w in wins:
        flag = " (minimized)" if w.minimized else ""
        print(f"{w.hwnd:>10}  {w.process[:22]:<22} {w.client[2]:>5}x{w.client[3]:<5} {w.title[:70]}{flag}")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    import cv2

    from reflex.backends.windows import ScreenSource, WindowSource

    if args.title or args.process or args.hwnd:
        find = {k: v for k, v in (("title", args.title), ("process", args.process)) if v}
        src = WindowSource(args.hwnd, method=args.capture or "auto", **find)
        what = repr(src.window.title)
    else:
        src = ScreenSource(None)
        what = "the screen"
    frame = src.grab()
    if args.region:
        x, y, w, h = (int(v) for v in args.region.split(","))
        frame = frame[y : y + h, x : x + w]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"saved {frame.shape[1]}x{frame.shape[0]} frame of {what} to {out}")
    return 0


def schema_doc() -> dict:
    from reflex.script import TOP_KEYS, describe_steps

    return {
        "script_keys": sorted(TOP_KEYS),
        "steps": describe_steps(),
        "conditions": "a predicate mapping ({image: f.png}, {text: {contains: Start}}, {predicate: name, ...}), "
                      "an expression string (\"${hp < 30}\"), or {not: c} / {any_of: [..]} / {all_of: [..]}",
        "targets": "click/move targets: [x, y] | $var (a stored match) | 'center' | a condition to find now",
        "expressions": "${...}: + - * / // % **, comparisons, and/or/not, x if c else y, a.b, a[0], "
                       "min max abs round int float str len bool",
        "predicates": REGISTRY.describe("predicate"),
    }


def cmd_record(args: argparse.Namespace) -> int:
    from reflex.backends.windows import find_window
    from reflex.recorder import record_to_file

    find = {k: v for k, v in (("title", args.title), ("process", args.process), ("hwnd", args.hwnd)) if v}
    if not find:
        print("error: say which window to record: --title, --process or --hwnd (see reflex windows)",
              file=sys.stderr)
        return 2
    window = find_window(**find)
    script = record_to_file(window, args.output, images=args.images, max_s=args.max, crop_px=args.crop)
    n = len(script["steps"])
    print(f"wrote {n} steps to {args.output}. Try it: reflex run {args.output} --dry-run")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from reflex.mcp_server import main as serve

    serve(base_dir=args.base_dir)
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    doc = schema_doc()
    if args.json:
        _print_json(doc)
        return 0
    print("Script steps (each step is one mapping; all take name:, comment:, when:)\n")
    for st in doc["steps"]:
        extra = f"  [{', '.join(st['fields'])}]" if st["fields"] else ""
        print(f"  {st['step']:<13} {st['doc']}{extra}")
    print(f"\nConditions: {doc['conditions']}\nTargets: {doc['targets']}\nExpressions: {doc['expressions']}")
    print("\nPredicates (use as {name: {params}}):")
    for it in doc["predicates"]:
        print(f"  {it['name']:<22} {it['doc'][:80]}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            checks.append((name, True, fn() or ""))
        except Exception as exc:  # noqa: BLE001
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))

    if sys.platform != "win32":
        print("reflex's capture and input backends are Windows-only; the engine itself runs anywhere.")
        return 1
    from reflex.backends.windows import PostMessageSink, WindowSource, benchmark, screen_capture_available
    from reflex.backends.windows.testwindow import BLUE_BGR, RED_BGR, TestWindow

    def near(px, bgr) -> bool:
        return all(abs(int(a) - b) < 8 for a, b in zip(px, bgr, strict=True))

    with TestWindow(title="reflex doctor") as tw:
        src = WindowSource(tw.hwnd)

        def cap() -> str:
            f = src.grab()
            if not (near(f[20, 10], RED_BGR) and near(f[20, -10], BLUE_BGR)):
                raise AssertionError("captured colours are wrong")
            b = benchmark(src, seconds=0.5)
            return f"{f.shape[1]}x{f.shape[0]}, {b['fps']} fps, p95 {b['ms_p95']} ms"

        def keys() -> str:
            PostMessageSink(tw.hwnd).press("a", hold_ms=5)
            if not tw.wait_for(lambda e: e[0] == "keyup"):
                raise AssertionError("the window got no key messages")
            return "key down/up delivered without focus"

        def clicks() -> str:
            tw.clear()
            PostMessageSink(tw.hwnd).click(src.grab().shape[1] - 10, 10)
            got = tw.wait_for(lambda e: e[0] == "lbuttonup")
            if not got or got[0][1] != "child":
                raise AssertionError("click did not reach the child window")
            return "click routed to the child window under the point"

        check("window capture (works while covered)", cap)
        check("background keys (PostMessage)", keys)
        check("background clicks (PostMessage)", clicks)

    def screen() -> str:
        if not screen_capture_available():
            raise RuntimeError("screen is locked; screen capture works once it is unlocked")
        return "available"

    def ocr() -> str:
        from reflex.ocr import get_engine

        eng = get_engine()
        return f"Windows OCR ({getattr(eng, 'language', '?')})"

    check("screen capture", screen)
    check("text recognition (OCR)", ocr)
    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print("\nSendInput (real input) is not tested here because it would move your cursor. "
          "Set REFLEX_LIVE_INPUT=1 and run the test suite to check it.")
    return 0 if all(ok for _, ok, _ in checks) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reflex", description="Reflex macro engine tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    def target_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--title", help="target window: title contains this")
        sp.add_argument("--process", help="target window: process name contains this (e.g. game.exe)")
        sp.add_argument("--hwnd", type=int, help="target window handle (from reflex windows)")
        sp.add_argument("--capture", choices=["auto", "printwindow", "screen"], default=None)

    s = sub.add_parser("run", help="run a script, or a watcher file against a window")
    s.add_argument("file")
    target_args(s)
    s.add_argument("--input", choices=["postmessage", "sendinput"], default=None,
                   help="postmessage: background, no focus needed; sendinput: real input to the focused window")
    s.add_argument("--all-windows", action="store_true", help="run once per matching window, in parallel")
    s.add_argument("--screen", help="target the screen instead: 'all' or x,y,w,h (uses sendinput)")
    s.add_argument("--dry-run", action="store_true", help="watch the screen but send no input")
    s.add_argument("--image", help="use this image as the screen (test a script offline)")
    s.add_argument("--events", help="append every event to this JSONL file (see reflex stats)")
    s.add_argument("--every", type=parse_duration, help="repeat on a schedule, e.g. 30m")
    s.add_argument("--at", action="append", help="run daily at HH:MM (repeatable)")
    s.add_argument("--times", type=int, help="number of runs (default 1, or unlimited when scheduled)")
    s.add_argument("--wait-first", action="store_true", help="with --every/--at: wait for the first slot")
    s.add_argument("--duration", type=parse_duration, default=600.0, help="watcher files: how long to watch")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("windows", help="list windows you can target")
    s.add_argument("--title")
    s.add_argument("--process")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_windows)

    s = sub.add_parser("capture", help="save a frame of a window or the screen as PNG")
    target_args(s)
    s.add_argument("-o", "--output", default="capture.png")
    s.add_argument("--region", help="crop x,y,w,h (frame pixels), handy for cutting a template")
    s.set_defaults(fn=cmd_capture)

    s = sub.add_parser("record", help="record your input in a window into a script (stop: ctrl+alt+end)")
    target_args(s)
    s.add_argument("-o", "--output", default="recorded.yaml")
    s.add_argument("--images", action="store_true",
                   help="save what was under each click and click that image, not a fixed pixel")
    s.add_argument("--crop", type=int, default=48, help="size in pixels of each click image")
    s.add_argument("--max", type=parse_duration, default=3600.0, help="stop recording after this long")
    s.set_defaults(fn=cmd_record)

    s = sub.add_parser("mcp", help="serve reflex's tools to AI agents over MCP (stdio)")
    s.add_argument("--base-dir", help="where relative template and script paths resolve (default: cwd)")
    s.set_defaults(fn=cmd_mcp)

    s = sub.add_parser("schema", help="every script step and predicate (use --json for agents)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_schema)

    s = sub.add_parser("doctor", help="check capture, background input and OCR on this PC")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("list", help="list registered plug-ins")
    s.add_argument("--kind", choices=KINDS)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("validate", help="check scripts and watcher files without running them")
    s.add_argument("paths", nargs="+")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("replay", help="run replay scenarios (exit 1 if any expectation fails)")
    s.add_argument("scenarios", nargs="+")
    s.add_argument("--json", action="store_true")
    s.add_argument("--events", help="also append replay events to this JSONL file")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("stats", help="summarise an events.jsonl stream")
    s.add_argument("events")
    s.add_argument("--session", help="only this session id (default: latest)")
    s.add_argument("--last", type=int, default=20, help="recent events to include")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
