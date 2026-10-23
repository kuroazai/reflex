"""The command line, end to end."""

from __future__ import annotations

import json
import sys
import time

import pytest

from reflex.cli import main, parse_duration


def test_parse_duration():
    assert parse_duration("90") == 90
    assert parse_duration("30m") == 1800
    assert parse_duration("2h") == 7200
    assert parse_duration("250ms") == 0.25


def test_validate_reports_scripts_and_bad_files(tmp_path, capsys):
    good = tmp_path / "good.yaml"
    good.write_text("name: g\nsteps:\n  - wait: 10\n", encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text("steps:\n  - clik: [1, 2]\n", encoding="utf-8")
    assert main(["validate", str(good), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["type"] == "script" and out[0]["steps"] == 1
    assert main(["validate", str(tmp_path)]) == 1
    text = capsys.readouterr().out
    assert "FAIL" in text and "did you mean 'click'" in text


def test_schema_json_lists_steps_and_predicates(capsys):
    assert main(["schema", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    steps = {s["step"] for s in doc["steps"]}
    assert {"click", "wait_for", "repeat", "read_text"} <= steps
    assert {"image", "text", "color_search"} <= {p["name"] for p in doc["predicates"]}


def test_run_against_an_image_dry_run(tmp_path, capsys):
    import cv2
    import numpy as np

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[40:50, 60:70] = (0, 255, 0)
    img = tmp_path / "screen.png"
    cv2.imwrite(str(img), frame)
    script = tmp_path / "s.yaml"
    script.write_text("""
steps:
  - wait_for: {color_search: {rgb: [0, 255, 0]}}
    timeout_s: 2
    as: g
  - click: $g
  - log: "clicked ${g.x},${g.y}"
""", encoding="utf-8")
    assert main(["run", str(script), "--image", str(img), "--dry-run", "--json"]) == 0
    out = capsys.readouterr().out
    assert "clicked 60,40" in out  # pixel search returns the first (top-left) matching pixel
    res = json.loads(out[out.index("[\n"):])
    assert res[0]["status"] == "completed" and res[0]["vars"]["g"]["x"] == 60


@pytest.mark.skipif(sys.platform != "win32", reason="Windows backend")
def test_run_script_against_a_real_window_in_the_background(tmp_path, capsys):
    from reflex.backends.windows.testwindow import TestWindow

    title = f"reflex-cli-{time.monotonic_ns()}"
    script = tmp_path / "type.yaml"
    script.write_text(f"""
name: cli e2e
target:
  window: {{title: "{title}"}}
  input: postmessage
steps:
  - wait_for: {{color_search: {{rgb: [220, 40, 40], tolerance: 10}}}}
    timeout_s: 5
  - type: "typed by reflex"
  - log: done
""", encoding="utf-8")
    with TestWindow(title=title, edit=True) as tw:
        rc = main(["run", str(script), "--json"])
        out = capsys.readouterr().out
        deadline = time.monotonic() + 2
        while "typed by reflex" not in tw.edit_text() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert rc == 0, out
        assert tw.edit_text() == "typed by reflex"
        results = json.loads(out[out.index("[\n"):])
        assert results[0]["status"] == "completed"
