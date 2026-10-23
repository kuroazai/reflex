"""Plug-in registry + generic YAML DSL."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from reflex.cli import main as cli_main
from reflex.core import FrameView, Match, TriggerSpec
from reflex.dsl import DSLError, load_macro_set
from reflex.io import RecordingActionSink, RecordingEventSink
from reflex.registry import REGISTRY, Registry, register_predicate, register_recipe
from reflex.runtime import ReflexRuntime

MACRO_DIR = Path(__file__).resolve().parents[1] / "examples"


def _write(tmp_path: Path, text: str, name: str = "m.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _view(v: int = 200, **extras) -> FrameView:
    return FrameView(frame_bgr=np.full((40, 40, 3), v, dtype=np.uint8), ts_ns=0, extras=extras)


# --------------------------------------------------------------------------- plug-ins


@register_predicate("test_mean_above", replace=True)
def _mean_above(*, threshold: float):
    """Mean brightness above a threshold (test-only plug-in)."""

    def p(view):
        m = float(view.frame_bgr.mean())
        return Match(meta={"mean": m}) if m > threshold else None

    return p


@register_recipe("test_double_tap", replace=True)
def _double_tap(*, name: str, when, button: str = "a", priority: int = 50) -> TriggerSpec:
    """Press a button twice when `when` matches (test-only plug-in)."""
    return TriggerSpec(name=name, predicate=when, action_factory=lambda m: (button, button),
                       priority=priority, extra={"kind": "test_double_tap"})


def test_plugin_defined_in_test_module_is_usable_from_yaml(tmp_path):
    p = _write(tmp_path, """
macro:
  id: tap
  kind: test_double_tap
  priority: 77
  params:
    when: {predicate: test_mean_above, threshold: 100}
    button: x
""")
    (spec,) = load_macro_set(p).specs
    assert spec.priority == 77 and spec.extra["kind"] == "test_double_tap"
    assert spec.predicate(_view(200)) is not None and spec.predicate(_view(50)) is None
    assert tuple(spec.action_factory(Match())) == ("x", "x")
    names = [d["name"] for d in REGISTRY.describe("recipe")]
    assert "test_double_tap" in names


def test_describe_reports_params_and_docs():
    d = {x["name"]: x for x in REGISTRY.describe("predicate")}
    sb = d["sprite_blob"]
    req = {p["name"] for p in sb["params"] if p["required"]}
    assert req == {"region", "hsv_lo", "hsv_hi"}
    assert d["template_match"]["yaml_only_params"] == ["template_paths"]
    assert d["test_mean_above"]["doc"].startswith("Mean brightness")


def test_duplicate_registration_rejected_without_replace():
    reg = Registry()
    reg.register("predicate", "x", lambda: None)
    with pytest.raises(ValueError):
        reg.register("predicate", "x", lambda: 1)


def test_entry_point_plugins_are_loaded_and_errors_contained(monkeypatch):
    reg = Registry()
    reg._builtins_loaded = True

    def good(r):
        r.register("predicate", "from_entry_point", lambda *, k=1: (lambda v: None))

    def bad(_r):
        raise RuntimeError("plugin exploded")

    eps = [SimpleNamespace(name="good", value="pkg:good", load=lambda: good),
           SimpleNamespace(name="bad", value="pkg:bad", load=lambda: bad)]
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda group: eps if group == "reflex.plugins" else [])
    assert reg.names("predicate") == ["from_entry_point"]
    assert reg.load_errors and "plugin exploded" in reg.load_errors[0]["error"]


# --------------------------------------------------------------------------- DSL validation


@pytest.mark.parametrize(
    "body, needle",
    [
        ("macro: {id: a, kind: nope}", "no recipe plug-in named 'nope'"),
        ("macro: {id: a, kind: on_match, params: {when: {predicate: always}, do: [a], bogus: 1}}",
         "unknown param(s) ['bogus']"),
        ("macro: {id: a, kind: on_match, params: {do: [a]}}", "missing required param(s) ['when']"),
        ("macro: {id: a, kind: on_match, params: {when: {predicate: sprit_blob}, do: [a]}}",
         "did you mean sprite_blob"),
        ("macro: {id: a, kind: on_match, colour: red, params: {when: {predicate: always}, do: [a]}}",
         "unknown key(s) ['colour']"),
        ("macro: {id: a, kind: on_match, params: {when: {predicate: always}, do: [{teleport: 1}]}}",
         "unknown action form"),
        ("macro: {id: a, kind: on_match, expect: {kind: gone, within_ms: 0}, params: {when: {predicate: always}, do: [a]}}",
         "within_ms must be > 0"),
        ("macro: {id: a, kind: on_match, params: {when: $missing, do: [a]}}", "unbound reference '$missing'"),
    ],
)
def test_dsl_errors_are_located_and_specific(tmp_path, body, needle):
    p = _write(tmp_path, body)
    with pytest.raises(DSLError) as ei:
        load_macro_set(p)
    msg = str(ei.value)
    assert needle in msg, msg
    assert "m.yaml" in msg


def test_full_featured_file_builds_and_runs(tmp_path):
    tpl = np.full((4, 4, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "buy.png"), tpl)
    p = _write(tmp_path, """
guards:
  shop:
    predicate: template_match
    region: [0, 0, 40, 40]
    template_paths: [buy.png]
    threshold: 0.99
context:
  scene: {kind: static, values: {scene: dialog}}
macros:
  - id: advance
    kind: on_match
    priority: 65
    params:
      when:
        all_of:
          - {predicate: test_mean_above, threshold: 100}
          - not: {predicate: extras_equals, key: scene, value: battle}
      do: [a, {wait: 30}, {click: [5, 6]}]
      max_fires: 3
    gates:
      - {predicate: extras_in, key: scene, values: [dialog, overworld]}
    expect: {kind: any_of, parts: [gone, {kind: changed, min_changed_fraction: 0.2}], within_ms: 400}
    trip_after_unverified: 2
    cooldown_s: 0.1
  - id: disabled_one
    kind: on_match
    enabled: false
    params: {when: {predicate: always}, do: [b]}
""")
    ms = load_macro_set(p)
    assert [s.name for s in ms.specs] == ["advance"] and ms.skipped == ["disabled_one"]
    spec = ms.specs[0]
    assert spec.priority == 65 and spec.cooldown_s == 0.1 and spec.trip_after_unverified == 2
    assert len(spec.gates) == 1 and spec.expect.name == "any_of" and spec.expect.within_ms == 400
    assert set(ms.guards) == {"shop"} and set(ms.context) == {"scene"}

    # Run it: the frame is a mid-grey (no pure-white 4x4 template) so the guard stays quiet.
    frame = np.full((40, 40, 3), 180, dtype=np.uint8)
    clock = {"t": 0.0}
    sink = RecordingActionSink(clock=lambda: clock["t"])
    ev = RecordingEventSink()

    def sleep(dt):
        clock["t"] += dt

    rt = ReflexRuntime(source=SimpleNamespace(grab=lambda: frame), sink=sink, events=ev,
                       clock=lambda: clock["t"], sleep=sleep, guards=ms.guards, context_providers=ms.context)
    for s in ms.specs:
        rt.register(s)
    rt.run(timeout_s=3.0)
    st = rt.stats()["advance"]
    # frame never changes → each fire unverified → trips after 2
    assert st["fires"] == 2 and st["tripped"] and st["unverified"] == 2
    assert [e["kind"] for e in sink.log[:2]] == ["key", "click"]

    # Now make the guard hit: paste the white template into the frame.
    frame2 = frame.copy()
    frame2[10:14, 10:14] = 255
    assert ms.guards["shop"](FrameView(frame_bgr=frame2, ts_ns=0)) is not None


def test_bindings_inject_host_objects(tmp_path):
    p = _write(tmp_path, """
macro:
  id: follow
  kind: on_match
  params:
    when: $scene_is_battle
    do: [a]
""")
    marker = lambda v: Match(meta={"bound": True})  # noqa: E731
    (spec,) = load_macro_set(p, bindings={"scene_is_battle": marker}).specs
    assert spec.predicate(_view()).meta == {"bound": True}


def test_shipped_macros_all_load_and_validate_cli(capsys):
    ms = load_macro_set(MACRO_DIR)
    assert {s.name for s in ms.specs} >= {"block_on_red", "dialog_advance", "npc_interact_green"}
    assert cli_main(["validate", str(MACRO_DIR)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "dialog_advance" in out


def test_cli_validate_fails_on_bad_file(tmp_path, capsys):
    p = _write(tmp_path, "macro: {id: a, kind: nope}")
    assert cli_main(["validate", str(p)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_list_json(capsys):
    import json

    assert cli_main(["list", "--kind", "expect", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {p["name"] for p in data["plugins"]} >= {"gone", "appears", "changed", "all_of", "any_of"}


def test_flat_template_does_not_match_everything():
    """Regression: TM_CCOEFF_NORMED on a zero-variance template scored 1.0 on any image."""
    from reflex.conditions import template_match
    from reflex.core import Region

    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    p = template_match(region=Region(0, 0, 40, 40), templates=[white], threshold=0.95)
    assert p(_view(180)) is None
    frame = np.full((40, 40, 3), 180, dtype=np.uint8)
    frame[10:14, 20:24] = 255
    m = p(FrameView(frame_bgr=frame, ts_ns=0))
    assert m is not None and m.meta["bbox_xywh"] == (20, 10, 4, 4)


def test_template_match_offsets_use_clipped_region():
    from reflex.conditions import template_match
    from reflex.core import Region

    tpl = np.zeros((4, 4, 3), dtype=np.uint8)
    tpl[:2] = 255  # textured
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[5:7, 3:7] = 255
    p = template_match(region=Region(-10, -10, 60, 60), templates=[tpl], threshold=0.9)
    m = p(FrameView(frame_bgr=frame, ts_ns=0))
    assert m is not None and m.meta["bbox_xywh"][:2] == (3, 5)
