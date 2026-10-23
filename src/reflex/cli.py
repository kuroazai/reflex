"""
`reflex` command line (also `python -m reflex`).

    reflex list [--kind recipe] [--json]      every plug-in + its params
    reflex validate <file-or-dir> ...         build macros, report errors
    reflex replay <scenario.yaml> ... [--json] run scenarios over recorded frames
    reflex stats <events.jsonl> [--session S] [--json]  summarise a live/replayed run
"""

from __future__ import annotations

import argparse
import json
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


def cmd_validate(args: argparse.Namespace) -> int:
    from reflex.dsl import DSLError, load_macro_set

    rc = 0
    report = []
    for raw in args.paths:
        try:
            ms = load_macro_set(Path(raw))
        except DSLError as exc:
            rc = 1
            report.append({"path": raw, "ok": False, "error": str(exc)})
            continue
        report.append(
            {
                "path": raw,
                "ok": True,
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
        )
    if args.json:
        _print_json(report)
        return rc
    for r in report:
        if not r["ok"]:
            print(f"FAIL {r['path']}\n     {r['error']}")
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reflex", description="Reflex macro engine tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list registered plug-ins")
    s.add_argument("--kind", choices=KINDS)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("validate", help="build macro YAML and report problems")
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
