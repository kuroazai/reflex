"""YAML DSL loader: per-kind round-trips and validation errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from reflex.dsl import DSLError, load_macros_yaml


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_dialog_advance_loop_loads(tmp_path: Path):
    p = _write(
        tmp_path,
        "dialog.yaml",
        """
macro:
  id: dialog_advance
  kind: dialog_advance_loop
  priority: 60
  params:
    dialog:
      predicate: color_at
      xy: [120, 384]
      hsv_lo: [0, 0, 220]
      hsv_hi: [179, 50, 255]
    advance_button: a
    cooldown_ms: 220
    max_advances: 32
    poll_hz: 12.0
    hold_ms: 35
""",
    )
    specs = load_macros_yaml(p)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "dialog_advance"
    assert spec.priority == 60
    assert spec.max_fires == 32
    assert spec.extra["kind"] == "dialog_advance_loop"


def test_color_armed_button_loads(tmp_path: Path):
    p = _write(
        tmp_path,
        "block.yaml",
        """
macro:
  id: block_on_red
  kind: color_armed_button
  priority: 90
  params:
    color:
      predicate: color_at
      xy: [640, 360]
      hsv_lo: [0, 180, 180]
      hsv_hi: [10, 255, 255]
    button: x
    cooldown_s: 0.5
    poll_hz: 60.0
    hold_ms: 30
""",
    )
    specs = load_macros_yaml(p)
    assert len(specs) == 1
    assert specs[0].priority == 90
    assert specs[0].cooldown_s == 0.5


def test_adjacent_sprite_face_talk_loads(tmp_path: Path):
    p = _write(
        tmp_path,
        "npc.yaml",
        """
macro:
  id: npc_interact
  kind: adjacent_sprite_face_talk
  priority: 70
  params:
    sprite:
      predicate: sprite_blob
      region: { x: 0, y: 0, w: 800, h: 800 }
      hsv_lo: [40, 60, 60]
      hsv_hi: [85, 255, 255]
      min_area_px: 80
      pick: nearest
      nearest_to_xy: [400, 400]
    player_screen_xy: [400, 400]
    tile_size_px: 32
    talk_button: a
""",
    )
    specs = load_macros_yaml(p)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.extra["kind"] == "adjacent_sprite_face_talk"
    assert spec.extra["tile_size_px"] == 32


def test_unknown_kind_raises(tmp_path: Path):
    p = _write(tmp_path, "bad.yaml", "macro:\n  id: x\n  kind: nope\n")
    with pytest.raises(DSLError):
        load_macros_yaml(p)


def test_missing_id_raises(tmp_path: Path):
    p = _write(tmp_path, "bad.yaml", "macro:\n  kind: dialog_advance_loop\n  params: {}\n")
    with pytest.raises(DSLError):
        load_macros_yaml(p)


def test_bundled_example_macros_load():
    """The example macros under examples/ must parse."""
    macros_dir = Path(__file__).resolve().parent.parent / "examples"
    expected = {"dialog_advance", "npc_interact_green", "block_on_red"}
    found = set()
    for f in macros_dir.glob("*.yaml"):
        for spec in load_macros_yaml(f):
            found.add(spec.name)
    assert expected.issubset(found), f"missing: {expected - found}"
