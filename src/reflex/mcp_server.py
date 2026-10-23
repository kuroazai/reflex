"""MCP server: `reflex mcp` (stdio). Install with `pip install reflex[mcp]`.

Add to an MCP client config, e.g. Claude Code:

    claude mcp add reflex -- reflex mcp --base-dir C:/path/to/your/macros

The tools are thin wrappers over reflex.agent.ReflexTools; the docstrings
below are what the agent sees.
"""

from __future__ import annotations

import json
from typing import Any

from reflex.agent import ReflexTools, ToolError

INSTRUCTIONS = """\
reflex drives Windows programs (games, emulators, apps) by watching the screen and sending input.

Coordinates: every x/y is a pixel in the target window's client area. Screenshots may be downscaled;
divide by their `scale` (and add `offset` for a region) to get window pixels.

Input: input="postmessage" (default) works in the background without taking focus or moving the
user's cursor; most apps, launchers and emulators accept it. Many games read raw input and ignore it:
then use input="sendinput", which takes focus and moves the real cursor.

Fast reactions: you are too slow to react to things that last under a second. Write a script or
watchers (call `schema` for the format, `validate_script` to check it) and run them with run_script /
watch; they react at frame rate while you poll run_status. Every run is bounded by a timeout and can
be stopped with stop_run, and the user can stop everything with ctrl+alt+end.
"""


def _server_classes():
    try:
        from mcp.server.mcpserver import Image, MCPServer  # mcp >= 2

        return MCPServer, Image
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP, Image  # mcp 1.x

            return FastMCP, Image
        except ImportError as exc:
            raise SystemExit("the MCP server needs the mcp package: pip install reflex[mcp]") from exc


def _err(exc: ToolError) -> str:
    return json.dumps({"error": str(exc)})


def build_server(tools: ReflexTools | None = None):
    Server, Image = _server_classes()
    t = tools or ReflexTools()
    mcp = Server("reflex", instructions=INSTRUCTIONS)

    def wrap(fn, *args: Any, **kw: Any) -> Any:
        try:
            return fn(*args, **kw)
        except ToolError as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def list_windows(title: str | None = None, process: str | None = None) -> list[dict[str, Any]]:
        """List visible windows (hwnd, title, process, client size). Filter by title or process substring."""
        return t.list_windows(title=title, process=process)

    @mcp.tool(structured_output=False)
    def screenshot(title: str | None = None, hwnd: int | None = None, process: str | None = None,
                   region: list[int] | None = None, max_width: int = 1280) -> list[Any]:
        """See a window: its client area as an image (works while covered, not while minimized).
        region=[x, y, w, h] crops in window pixels. The text part gives the scale to map back."""
        try:
            png, info = t.screenshot(title=title, hwnd=hwnd, process=process, region=region, max_width=max_width)
        except ToolError as exc:
            return [_err(exc)]
        return [Image(data=png, format="png"), json.dumps(info)]

    @mcp.tool()
    def find_image(template: str, title: str | None = None, hwnd: int | None = None, process: str | None = None,
                   threshold: float = 0.85, region: list[int] | None = None, max_results: int = 5,
                   scales: list[float] | None = None) -> Any:
        """Find a template PNG (path) in the window. Returns hits with centre x/y, box and score."""
        return wrap(t.find_image, template, title=title, hwnd=hwnd, process=process, threshold=threshold,
                    region=region, max_results=max_results, scales=scales)

    @mcp.tool()
    def read_text(title: str | None = None, hwnd: int | None = None, process: str | None = None,
                  region: list[int] | None = None) -> Any:
        """Read text in the window or a region (OCR). Each word comes with its box and centre."""
        return wrap(t.read_text, title=title, hwnd=hwnd, process=process, region=region)

    @mcp.tool()
    def find_text(contains: str, title: str | None = None, hwnd: int | None = None, process: str | None = None,
                  region: list[int] | None = None) -> Any:
        """Where some text appears (case-insensitive): its box and centre, or null."""
        return wrap(t.find_text, contains, title=title, hwnd=hwnd, process=process, region=region)

    @mcp.tool()
    def find_color(rgb: list[int], title: str | None = None, hwnd: int | None = None, process: str | None = None,
                   tolerance: int = 8, region: list[int] | None = None, max_results: int = 20) -> Any:
        """Pixels matching an [r, g, b] colour within tolerance, top-left first."""
        return wrap(t.find_color, rgb, title=title, hwnd=hwnd, process=process, tolerance=tolerance,
                    region=region, max_results=max_results)

    @mcp.tool()
    def save_template(region: list[int], path: str, title: str | None = None, hwnd: int | None = None,
                      process: str | None = None) -> Any:
        """Save region [x, y, w, h] of the window as a PNG to use in {image: path} conditions."""
        return wrap(t.save_template, region, path, title=title, hwnd=hwnd, process=process)

    @mcp.tool()
    def click(x: int, y: int, title: str | None = None, hwnd: int | None = None, process: str | None = None,
              button: str = "left", double: bool = False, input: str | None = None) -> Any:
        """Click at window pixel (x, y). button: left | right | middle."""
        return wrap(t.click, x, y, title=title, hwnd=hwnd, process=process, button=button, double=double,
                    input=input)

    @mcp.tool()
    def press(keys: list[str], title: str | None = None, hwnd: int | None = None, process: str | None = None,
              hold_ms: int = 50, together: bool = False, input: str | None = None) -> Any:
        """Press keys in order, or together=true for a chord (["ctrl", "s"]). Names: a-z, 0-9, f1-f24,
        enter, esc, space, tab, up/down/left/right, shift, ctrl, alt, ..."""
        return wrap(t.press, keys, title=title, hwnd=hwnd, process=process, hold_ms=hold_ms, together=together,
                    input=input)

    @mcp.tool()
    def type_text(text: str, title: str | None = None, hwnd: int | None = None, process: str | None = None,
                  input: str | None = None) -> Any:
        """Type text into the window's focused control (any characters; \\n for new lines)."""
        return wrap(t.type_text, text, title=title, hwnd=hwnd, process=process, input=input)

    @mcp.tool()
    def scroll(x: int, y: int, clicks: int, title: str | None = None, hwnd: int | None = None,
               process: str | None = None, input: str | None = None) -> Any:
        """Scroll the wheel at (x, y) by `clicks` notches (negative scrolls down)."""
        return wrap(t.scroll, x, y, clicks, title=title, hwnd=hwnd, process=process, input=input)

    @mcp.tool()
    def drag(x1: int, y1: int, x2: int, y2: int, title: str | None = None, hwnd: int | None = None,
             process: str | None = None, duration_ms: int = 300, input: str | None = None) -> Any:
        """Press at (x1, y1), move to (x2, y2), release."""
        return wrap(t.drag, x1, y1, x2, y2, title=title, hwnd=hwnd, process=process, duration_ms=duration_ms,
                    input=input)

    @mcp.tool()
    def schema() -> Any:
        """The script format: every step, condition form, expression rule and predicate with its params."""
        return t.schema()

    @mcp.tool()
    def validate_script(script: str | None = None, path: str | None = None) -> Any:
        """Check a script (YAML text, or a path) without running it. Errors name the step and the problem."""
        return t.validate_script(script=script, path=path)

    @mcp.tool()
    def run_script(script: str | None = None, path: str | None = None, title: str | None = None,
                   hwnd: int | None = None, process: str | None = None, input: str | None = None,
                   all_windows: bool = False, dry_run: bool = False, wait_s: float = 0.0) -> Any:
        """Run a script in the background (YAML text or a path). Window args override its target.
        Returns a run_id immediately (or the result if it finishes within wait_s); poll run_status.
        dry_run watches the screen but sends no input and reports what it would have done."""
        return wrap(t.run_script, script=script, path=path, title=title, hwnd=hwnd, process=process,
                    input=input, all_windows=all_windows, dry_run=dry_run, wait_s=wait_s)

    @mcp.tool()
    def watch(watchers: str, title: str | None = None, hwnd: int | None = None, process: str | None = None,
              input: str | None = None, timeout_s: float = 300.0, dry_run: bool = False) -> Any:
        """Arm watchers (YAML with macros: [...]) that react at frame rate in the background, e.g.
        'press 4 whenever the heal icon lights up'. Poll run_status to see fires; stop_run to end."""
        return wrap(t.watch, watchers, title=title, hwnd=hwnd, process=process, input=input,
                    timeout_s=timeout_s, dry_run=dry_run)

    @mcp.tool()
    def run_status(run_id: str, events: int = 30) -> Any:
        """A run's status (running | completed | failed | stopped | timed_out), recent events and results."""
        return wrap(t.run_status, run_id, events=events)

    @mcp.tool()
    def stop_run(run_id: str) -> Any:
        """Stop a run. Held keys and buttons are released."""
        return wrap(t.stop_run, run_id)

    @mcp.tool()
    def list_runs() -> Any:
        """Every run started in this session and its status."""
        return t.list_runs()

    return mcp


def main(base_dir: str | None = None) -> None:
    build_server(ReflexTools(base_dir=base_dir)).run("stdio")
