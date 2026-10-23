"""Agent tools and the MCP server, against real windows this test creates."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows backend")

if sys.platform == "win32":
    from reflex.agent import ReflexTools, ToolError
    from reflex.backends.windows.testwindow import TestWindow


@pytest.fixture
def tools(tmp_path):
    return ReflexTools(base_dir=tmp_path)


@pytest.fixture
def tw():
    with TestWindow(title=f"reflex-agent-{time.monotonic_ns()}", edit=True, size=(600, 300)) as win:
        yield win


def test_list_and_screenshot(tools, tw):
    assert any(w["hwnd"] == tw.hwnd for w in tools.list_windows(title="reflex-agent"))
    png, info = tools.screenshot(hwnd=tw.hwnd, max_width=200)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 200 and info["scale"] == round(200 / info["frame_width"], 4)
    png, info = tools.screenshot(hwnd=tw.hwnd, region=[0, 0, 50, 40])
    assert info["frame_width"] == 50 and info["offset"] == [0, 0]


def test_bad_window_is_a_tool_error(tools):
    with pytest.raises(ToolError, match="no visible window"):
        tools.screenshot(title="no such window 81c2")
    with pytest.raises(ToolError, match="say which window"):
        tools.click(1, 1)


def test_find_color_save_template_find_image(tools, tw):
    pts = tools.find_color([220, 40, 40], hwnd=tw.hwnd, tolerance=10, max_results=3)
    assert pts and pts[0] == [0, 0]
    tools.save_template([10, 10, 30, 20], "red.png", hwnd=tw.hwnd)
    saved = tools.save_template([10, 10, 30, 20], "red.png", hwnd=tw.hwnd)
    assert saved["warning"]  # a flat red patch matches anywhere red: the agent is told
    hits = tools.find_image("red.png", hwnd=tw.hwnd, max_results=1)
    assert hits and hits[0]["score"] > 0.95


def test_type_and_read_back(tools, tw):
    assert tools.type_text("AGENT 77", hwnd=tw.hwnd)["ok"]
    deadline = time.monotonic() + 2
    while tw.edit_text() != "AGENT 77" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert tw.edit_text() == "AGENT 77"
    found = None
    deadline = time.monotonic() + 3
    while found is None and time.monotonic() < deadline:  # the box repaints a moment after its text changes
        try:
            found = tools.find_text("77", hwnd=tw.hwnd)
        except ToolError:
            pytest.skip("no OCR")
        time.sleep(0.1)
    assert found is not None and found["x"] > 300  # in the text box on the right


def test_validate_and_run_script_dry_then_real(tools, tw):
    bad = tools.validate_script(script="steps:\n  - tpye: hi\n")
    assert not bad["ok"] and "did you mean 'type'" in bad["error"]
    script = "steps:\n  - wait_for: {color_search: {rgb: [220, 40, 40], tolerance: 10}}\n    timeout_s: 3\n" \
             "  - type: from a script\n"
    dry = tools.run_script(script=script, hwnd=tw.hwnd, dry_run=True, wait_s=5)
    assert dry["status"] == "completed", dry
    assert [a["kind"] for a in dry["results"][0]["dry_run_actions"]] == ["type", "release_all"]
    assert tw.edit_text() == ""
    real = tools.run_script(script=script, hwnd=tw.hwnd, wait_s=5)
    assert real["status"] == "completed", real
    deadline = time.monotonic() + 2
    while tw.edit_text() != "from a script" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert tw.edit_text() == "from a script"


def test_background_run_can_be_stopped(tools, tw):
    started = tools.run_script(script="steps:\n  - wait: 60000\n", hwnd=tw.hwnd)
    assert started["status"] == "running"
    stopped = tools.stop_run(started["run_id"])
    assert stopped["status"] == "stopped"
    assert any(r["run_id"] == started["run_id"] for r in tools.list_runs())


def test_watch_fires_in_the_background(tools, tw):
    watchers = """
macros:
  - id: on_red
    kind: on_match
    max_fires: 2
    cooldown_s: 0.05
    params:
      when: {predicate: color_search, rgb: [220, 40, 40], tolerance: 10}
      do: [{type: "x"}]
"""
    run = tools.watch(watchers, hwnd=tw.hwnd, timeout_s=5)
    deadline = time.monotonic() + 5
    while tools.run_status(run["run_id"])["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    st = tools.run_status(run["run_id"])
    assert st["results"][0]["fires"] == {"on_red": 2}
    assert any(e["kind"] == "reflex.fired" for e in st["events"])
    deadline = time.monotonic() + 2
    while tw.edit_text() != "xx" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert tw.edit_text() == "xx"


def test_mcp_server_lists_and_calls_tools(tools, tw):
    pytest.importorskip("mcp")
    from reflex.mcp_server import build_server

    server = build_server(tools)
    listed = asyncio.run(server.list_tools())
    names = {t.name for t in listed}
    assert {"screenshot", "click", "run_script", "watch", "run_status", "schema"} <= names
    out = asyncio.run(server.call_tool("screenshot", {"hwnd": tw.hwnd, "max_width": 100}))
    blob = json.dumps(out, default=lambda o: getattr(o, "model_dump", lambda: repr(o))())
    assert "image/png" in blob and "frame_width" in blob
    err = asyncio.run(server.call_tool("click", {"x": 1, "y": 1, "title": "no such window 0b7"}))
    assert "no visible window" in json.dumps(err, default=lambda o: getattr(o, "model_dump", lambda: repr(o))())
