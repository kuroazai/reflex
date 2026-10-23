"""reflex as a toolset for AI agents.

An agent reasons in seconds; a game moves in milliseconds. These tools give
an agent both halves:

  look and act directly   screenshot, find_image, read_text, find_color,
                          click, press, type_text, drag, scroll
  hand off fast work      validate a script, run it in the background, poll
                          its status and log, stop it. Watchers ("whenever the
                          heal prompt shows, press 4") react at frame rate
                          while the agent thinks about something else.

Every tool takes the target window by `title` (substring), `process` or
`hwnd`, and every coordinate is a pixel in that window's client area: the
same space the screenshots are in, before any downscaling (the screenshot
result says the scale).

`reflex.mcp_server` exposes these over MCP. They are plain methods, so any
other agent framework can call them directly.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from reflex.io import RecordingActionSink, jsonable
from reflex.vision import is_flat


class ToolError(ValueError):
    """A problem the agent can fix (bad window name, bad script)."""


@dataclass
class _Run:
    id: str
    kind: str
    name: str
    started: float
    thread: threading.Thread
    control: Any
    events: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def status(self) -> str:
        if not self.done.is_set():
            return "running"
        if self.error:
            return "error"
        statuses = {r.get("status") for r in self.results}
        return statuses.pop() if len(statuses) == 1 else "mixed"


class _EventBuffer:
    def __init__(self, run: _Run, limit: int = 400) -> None:
        self.run = run
        self.limit = limit

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        if kind in ("reflex.tick_stats", "script.step"):
            return  # too chatty for an agent to read
        ev = {"t": round(time.monotonic() - self.run.started, 3), "kind": kind, **jsonable(payload)}
        self.run.events.append(ev)
        if len(self.run.events) > self.limit:
            del self.run.events[: len(self.run.events) - self.limit]


class ReflexTools:
    def __init__(self, *, base_dir: str | Path | None = None, default_input: str = "postmessage") -> None:
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.default_input = default_input
        self._runs: dict[str, _Run] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ helpers

    def _window(self, title: str | None, hwnd: int | None, process: str | None):
        from reflex.backends.windows import WindowNotFound, find_window

        if not (title or hwnd or process):
            raise ToolError("say which window: title (part of its title), process (e.g. game.exe) or hwnd")
        try:
            return find_window(**{k: v for k, v in (("title", title), ("hwnd", hwnd), ("process", process)) if v})
        except WindowNotFound as exc:
            raise ToolError(str(exc)) from None

    def _frame(self, title: str | None, hwnd: int | None, process: str | None) -> tuple[Any, np.ndarray]:
        from reflex.backends.windows import CaptureError, WindowSource

        win = self._window(title, hwnd, process)
        try:
            return win, WindowSource(win).grab()
        except CaptureError as exc:
            raise ToolError(str(exc)) from None

    def _sink(self, win: Any, input_mode: str | None):
        from reflex.backends.windows import PostMessageSink, SendInputSink, focus

        mode = input_mode or self.default_input
        if mode == "postmessage":
            return PostMessageSink(win)
        if mode == "sendinput":
            focus(win.hwnd)
            time.sleep(0.1)
            return SendInputSink(win)
        raise ToolError("input is postmessage (background, no focus needed) or sendinput (real input; takes focus)")

    def _path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.base_dir / path

    @staticmethod
    def _region(region: list[int] | None) -> tuple[int, int, int, int] | None:
        if region is None:
            return None
        if len(region) != 4:
            raise ToolError("region is [x, y, width, height] in window pixels")
        return tuple(int(v) for v in region)  # type: ignore[return-value]

    # ------------------------------------------------------------ looking

    def list_windows(self, title: str | None = None, process: str | None = None) -> list[dict[str, Any]]:
        """Visible top-level windows: hwnd, title, process, client size."""
        from reflex.backends.windows import list_windows

        return [{"hwnd": w.hwnd, "title": w.title, "process": w.process, "width": w.client[2],
                 "height": w.client[3], "minimized": w.minimized} for w in list_windows(title=title, process=process)]

    def screenshot(self, title: str | None = None, hwnd: int | None = None, process: str | None = None,
                   region: list[int] | None = None, max_width: int = 1280) -> tuple[bytes, dict[str, Any]]:
        """PNG of the window's client area (works while it is covered) plus how it was scaled."""
        win, frame = self._frame(title, hwnd, process)
        ox = oy = 0
        reg = self._region(region)
        if reg is not None:
            ox, oy, w, h = reg
            frame = frame[oy : oy + h, ox : ox + w]
        fh, fw = frame.shape[:2]
        scale = 1.0
        if max_width and fw > max_width:
            scale = max_width / fw
            frame = cv2.resize(frame, (max_width, max(1, round(fh * scale))), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            raise ToolError("could not encode the screenshot")
        info = {"window": win.title, "hwnd": win.hwnd, "frame_width": fw, "frame_height": fh, "scale": round(scale, 4),
                "offset": [ox, oy],
                "note": "coordinates for other tools are window pixels: image_x / scale + offset_x"}
        return png.tobytes(), info

    def find_image(self, template: str, title: str | None = None, hwnd: int | None = None,
                   process: str | None = None, threshold: float = 0.85, region: list[int] | None = None,
                   max_results: int = 5, scales: list[float] | None = None) -> list[dict[str, Any]]:
        """Where a template PNG appears in the window, best first ([] if nowhere)."""
        from reflex.vision import find_image

        _win, frame = self._frame(title, hwnd, process)
        try:
            hits = find_image(frame, self._path(template), region=self._region(region), threshold=threshold,
                              max_results=max_results, scales=tuple(scales or (1.0,)))
        except FileNotFoundError as exc:
            raise ToolError(str(exc)) from None
        return [{"x": h.center[0], "y": h.center[1], "box": [h.x, h.y, h.w, h.h], "score": round(h.score, 3),
                 "scale": h.scale} for h in hits]

    def read_text(self, title: str | None = None, hwnd: int | None = None, process: str | None = None,
                  region: list[int] | None = None) -> dict[str, Any]:
        """OCR the window (or a region): full text plus each word with its box and centre."""
        from reflex.ocr import OcrUnavailable, read_text

        _win, frame = self._frame(title, hwnd, process)
        try:
            res = read_text(frame, self._region(region))
        except OcrUnavailable as exc:
            raise ToolError(str(exc)) from None
        return {"text": res.text,
                "words": [{"text": w.text, "box": list(w.bbox_xywh), "x": w.centroid_xy[0], "y": w.centroid_xy[1]}
                          for w in res.words]}

    def find_text(self, contains: str, title: str | None = None, hwnd: int | None = None,
                  process: str | None = None, region: list[int] | None = None) -> dict[str, Any] | None:
        """Where some text is on screen (case-insensitive), or null."""
        from reflex.ocr import read_text

        _win, frame = self._frame(title, hwnd, process)
        hit = read_text(frame, self._region(region)).find(contains)
        if hit is None:
            return None
        return {"text": hit.text, "box": list(hit.bbox_xywh), "x": hit.centroid_xy[0], "y": hit.centroid_xy[1]}

    def find_color(self, rgb: list[int], title: str | None = None, hwnd: int | None = None,
                   process: str | None = None, tolerance: int = 8, region: list[int] | None = None,
                   max_results: int = 20) -> list[list[int]]:
        """Pixels of a colour, top-left first (pixel search)."""
        from reflex.vision import find_color

        _win, frame = self._frame(title, hwnd, process)
        pts = find_color(frame, rgb, tolerance=tolerance, region=self._region(region), pick="all",
                         max_results=max_results)
        return [[x, y] for x, y in pts]

    def save_template(self, region: list[int], path: str, title: str | None = None, hwnd: int | None = None,
                      process: str | None = None) -> dict[str, Any]:
        """Crop a region of the window to a PNG, for use as an image condition in scripts."""
        _win, frame = self._frame(title, hwnd, process)
        x, y, w, h = self._region(region)  # type: ignore[misc]
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            raise ToolError(f"region {region} is outside the {frame.shape[1]}x{frame.shape[0]} window")
        out = self._path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop)
        return {"path": str(out), "width": int(crop.shape[1]), "height": int(crop.shape[0]),
                "warning": "flat region: it will match anywhere" if is_flat(crop) else None}

    # ------------------------------------------------------------ acting

    def click(self, x: int, y: int, title: str | None = None, hwnd: int | None = None, process: str | None = None,
              button: str = "left", double: bool = False, input: str | None = None) -> dict[str, Any]:
        win = self._window(title, hwnd, process)
        sink = self._sink(win, input)
        ok = sink.click(int(x), int(y), button=button)
        if double:
            time.sleep(0.06)
            ok = sink.click(int(x), int(y), button=button) and ok
        return {"ok": bool(ok), "window": win.title}

    def press(self, keys: list[str] | str, title: str | None = None, hwnd: int | None = None,
              process: str | None = None, hold_ms: int = 50, together: bool = False,
              input: str | None = None) -> dict[str, Any]:
        """Press keys one after another, or together (a chord like ctrl+s)."""
        from reflex.backends.windows import UnknownKey

        win = self._window(title, hwnd, process)
        sink = self._sink(win, input)
        seq = [keys] if isinstance(keys, str) else list(keys)
        try:
            if together:
                ok = all([sink.key_down(k) for k in seq])
                time.sleep(hold_ms / 1000.0)
                ok = all([sink.key_up(k) for k in reversed(seq)]) and ok
            else:
                ok = all([sink.press(k, hold_ms=hold_ms) for k in seq])
        except UnknownKey as exc:
            raise ToolError(str(exc)) from None
        return {"ok": bool(ok), "window": win.title}

    def type_text(self, text: str, title: str | None = None, hwnd: int | None = None, process: str | None = None,
                  input: str | None = None) -> dict[str, Any]:
        win = self._window(title, hwnd, process)
        return {"ok": bool(self._sink(win, input).type_text(text)), "window": win.title}

    def scroll(self, x: int, y: int, clicks: int, title: str | None = None, hwnd: int | None = None,
               process: str | None = None, input: str | None = None) -> dict[str, Any]:
        win = self._window(title, hwnd, process)
        return {"ok": bool(self._sink(win, input).scroll(int(x), int(y), clicks=int(clicks))), "window": win.title}

    def drag(self, x1: int, y1: int, x2: int, y2: int, title: str | None = None, hwnd: int | None = None,
             process: str | None = None, steps: int = 12, duration_ms: int = 300,
             input: str | None = None) -> dict[str, Any]:
        win = self._window(title, hwnd, process)
        sink = self._sink(win, input)
        ok = sink.mouse_down(int(x1), int(y1))
        for i in range(1, steps + 1):
            sink.move(round(x1 + (x2 - x1) * i / steps), round(y1 + (y2 - y1) * i / steps))
            time.sleep(duration_ms / 1000.0 / steps)
        ok = sink.mouse_up(int(x2), int(y2)) and ok
        return {"ok": bool(ok), "window": win.title}

    # ------------------------------------------------------------ scripts

    def schema(self) -> dict[str, Any]:
        """Every script step, condition form and predicate, for writing scripts."""
        from reflex.cli import schema_doc

        return schema_doc()

    def _load(self, script: str | None, path: str | None):
        from reflex.dsl import DSLError
        from reflex.script import ScriptError, build_script, load_script

        if bool(script) == bool(path):
            raise ToolError("pass exactly one of script (YAML text) or path (a .yaml file)")
        try:
            if path:
                return load_script(self._path(path))
            raw = yaml.safe_load(script)  # type: ignore[arg-type]
            return build_script(raw, base_dir=self.base_dir, where="<script>")
        except (ScriptError, DSLError, FileNotFoundError, yaml.YAMLError) as exc:
            raise ToolError(str(exc)) from None

    def validate_script(self, script: str | None = None, path: str | None = None) -> dict[str, Any]:
        """Check a script without running it: steps, key names, images, expressions."""
        try:
            sc = self._load(script, path)
        except ToolError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "name": sc.name, "steps": len(sc.steps), "functions": sorted(sc.functions),
                "watchers": [s.name for s in sc.watchers.specs], "target": sc.target}

    def run_script(self, script: str | None = None, path: str | None = None, title: str | None = None,
                   hwnd: int | None = None, process: str | None = None, input: str | None = None,
                   all_windows: bool = False, dry_run: bool = False, wait_s: float = 0.0) -> dict[str, Any]:
        """Start a script in the background. Returns a run id at once, or the result if it ends within wait_s."""
        from reflex.script import Control, ScriptError, open_targets, run_script

        sc = self._load(script, path)
        overrides: dict[str, Any] = {"input": input}
        win = {k: v for k, v in (("title", title), ("hwnd", hwnd), ("process", process)) if v}
        if win:
            overrides["window"] = win
        if all_windows:
            overrides["all_windows"] = True
        try:
            targets = open_targets(sc, overrides=overrides, dry_run=dry_run)
        except (ScriptError, LookupError) as exc:
            raise ToolError(str(exc)) from None
        control = Control()
        run = _Run(id=uuid.uuid4().hex[:8], kind="script", name=sc.name, started=time.monotonic(),
                   thread=threading.Thread(), control=control)
        events = _EventBuffer(run)

        def go() -> None:
            try:
                run.results = [r.to_dict() for r in run_script(sc, targets=targets, events=events, control=control,
                                                               hotkeys=True)]
                for t in targets:
                    if isinstance(t.sink, RecordingActionSink):
                        for r in run.results:
                            if r["instance"] == t.instance:
                                r["dry_run_actions"] = t.sink.log[-100:]
            except Exception as exc:  # noqa: BLE001
                run.error = f"{type(exc).__name__}: {exc}"
            finally:
                run.done.set()

        run.thread = threading.Thread(target=go, name=f"reflex-run-{run.id}", daemon=True)
        with self._lock:
            self._runs[run.id] = run
        run.thread.start()
        if wait_s > 0:
            run.done.wait(wait_s)
        return self.run_status(run.id)

    def watch(self, watchers: str, title: str | None = None, hwnd: int | None = None, process: str | None = None,
              input: str | None = None, timeout_s: float = 300.0, dry_run: bool = False) -> dict[str, Any]:
        """Arm watchers (YAML: a macros: list, or one macro:) on a window in the background.

        They react at frame rate until timeout_s or stop_run. Poll run_status for fires.
        """
        from reflex.dsl import Builder, DSLError, build_macro_set
        from reflex.runtime import ReflexRuntime
        from reflex.script import Control

        try:
            raw = yaml.safe_load(watchers)
            ms = build_macro_set(raw, builder=Builder(base_dir=self.base_dir, where="<watchers>"))
        except (DSLError, yaml.YAMLError, FileNotFoundError) as exc:
            raise ToolError(str(exc)) from None
        if not ms.specs:
            raise ToolError("no watchers: give macros: [...] or macro: {...}")
        win = self._window(title, hwnd, process)
        from reflex.backends.windows import WindowSource

        source = WindowSource(win)
        sink = RecordingActionSink() if dry_run else self._sink(win, input)
        control = Control()
        run = _Run(id=uuid.uuid4().hex[:8], kind="watch", name=",".join(s.name for s in ms.specs),
                   started=time.monotonic(), thread=threading.Thread(), control=control)
        rt = ReflexRuntime(source=source, sink=sink, events=_EventBuffer(run), guards=ms.guards,
                           context_providers=ms.context, foreground_ok=getattr(sink, "target_ok", None))
        for spec in ms.specs:
            rt.register(spec.with_overrides(timeout_s=max(spec.timeout_s, timeout_s)))

        def go() -> None:
            try:
                res = rt.run(timeout_s=timeout_s, should_stop=control.stop_event.is_set,
                             stop_when_all_done=True)
                run.results = [{"status": res.status if res.status != "stopped" else "stopped",
                                "elapsed_s": res.elapsed_s, "fires": res.fires, "stats": jsonable(res.stats)}]
            except Exception as exc:  # noqa: BLE001
                run.error = f"{type(exc).__name__}: {exc}"
            finally:
                if hasattr(sink, "release_all"):
                    sink.release_all()
                run.done.set()

        run.thread = threading.Thread(target=go, name=f"reflex-watch-{run.id}", daemon=True)
        with self._lock:
            self._runs[run.id] = run
        run.thread.start()
        return self.run_status(run.id)

    def run_status(self, run_id: str, events: int = 30) -> dict[str, Any]:
        """Status, the latest events/log lines and (when finished) results of a run."""
        run = self._runs.get(run_id)
        if run is None:
            raise ToolError(f"no run {run_id!r}; runs: {sorted(self._runs)}")
        return {"run_id": run.id, "kind": run.kind, "name": run.name, "status": run.status(),
                "elapsed_s": round(time.monotonic() - run.started, 2), "error": run.error,
                "results": run.results, "events": run.events[-int(events):]}

    def stop_run(self, run_id: str) -> dict[str, Any]:
        run = self._runs.get(run_id)
        if run is None:
            raise ToolError(f"no run {run_id!r}")
        run.control.stop("stopped by agent")
        run.done.wait(5.0)
        return self.run_status(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        return [{"run_id": r.id, "kind": r.kind, "name": r.name, "status": r.status()} for r in self._runs.values()]
