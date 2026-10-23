"""Registers the built-in plug-ins. Called once by the Registry on first use."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reflex import conditions as C
from reflex import expect as E
from reflex import recipes as R
from reflex import watchdog as W
from reflex.core import FrameView, Predicate
from reflex.io import RecordingActionSink, RecordingInputSink, StaticFrameSource

# --------------------------------------------------------------------------- combinator wrappers
# (named-keyword forms of the variadic helpers so YAML params line up)


def any_of(*, predicates: Sequence[Predicate]) -> Predicate:
    """Fires on the first sub-predicate that matches."""
    return C.any_of(predicates)


def all_of(*, predicates: Sequence[Predicate]) -> Predicate:
    """Fires only when every sub-predicate matches."""
    return C.all_of(predicates)


def not_(*, predicate: Predicate) -> Predicate:
    """Fires when the wrapped predicate does NOT match."""
    return C.not_(predicate)


def expect_all_of(*, parts: Sequence[E.Expectation], within_ms: int | None = None) -> E.Expectation:
    """Every part must hold at least once inside the window."""
    return E.all_of(parts, within_ms=within_ms)


def expect_any_of(*, parts: Sequence[E.Expectation], within_ms: int | None = None) -> E.Expectation:
    """Any part holding inside the window verifies the fire."""
    return E.any_of(parts, within_ms=within_ms)


def static_context(*, values: dict[str, Any]):
    """Context provider returning fixed extras (tests, replays, manual overrides)."""
    fixed = dict(values)

    def provider(_view: FrameView) -> dict[str, Any]:
        return dict(fixed)

    return provider


def recording_sink() -> RecordingActionSink:
    """Records every action; performs nothing. Full capabilities (dry runs)."""
    return RecordingActionSink()


def press_recording_sink() -> RecordingInputSink:
    """Records key presses only (legacy press-only capability)."""
    return RecordingInputSink()


def window_source(
    *,
    title: str | None = None,
    process: str | None = None,
    class_name: str | None = None,
    hwnd: int | None = None,
    method: str = "auto",
):
    """One window's client area, captured even while it is covered (Windows)."""
    from reflex.backends.windows import WindowSource

    return WindowSource(hwnd, method=method, **_find(title, process, class_name))  # type: ignore[arg-type]


def screen_source(*, region: Sequence[int] | None = None):
    """A fixed screen rectangle [x, y, w, h], or every monitor (Windows)."""
    from reflex.backends.windows import ScreenSource

    return ScreenSource(tuple(region) if region is not None else None)  # type: ignore[arg-type]


def postmessage_sink(
    *,
    title: str | None = None,
    process: str | None = None,
    class_name: str | None = None,
    hwnd: int | None = None,
):
    """Background input posted to one window: no focus, no cursor movement (Windows)."""
    from reflex.backends.windows import PostMessageSink

    return PostMessageSink(hwnd, **_find(title, process, class_name))


def sendinput_sink(
    *,
    title: str | None = None,
    process: str | None = None,
    class_name: str | None = None,
    hwnd: int | None = None,
    require_focus: bool = True,
):
    """Real keyboard and mouse input to the focused window (Windows)."""
    from reflex.backends.windows import SendInputSink

    return SendInputSink(hwnd, require_focus=require_focus, **_find(title, process, class_name))


def _find(title: str | None, process: str | None, class_name: str | None) -> dict[str, str]:
    return {k: v for k, v in (("title", title), ("process", process), ("class_name", class_name)) if v}


def image_source(*, path: str) -> StaticFrameSource:
    """A single image file as a frame source."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"image_source: cannot read {path}")
    return StaticFrameSource(img)


# --------------------------------------------------------------------------- adapters


def _template_adapter(params: dict[str, Any], builder: Any) -> dict[str, Any]:
    import cv2

    out = dict(params)
    paths = out.pop("template_paths", None)
    if paths is not None:
        if isinstance(paths, str):
            paths = [paths]
        templates = []
        for raw in paths:
            p = Path(str(raw))
            if not p.is_absolute() and getattr(builder, "base_dir", None) is not None:
                p = Path(builder.base_dir) / p
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"template_match: failed to read template {p}")
            templates.append(img)
        out["templates"] = templates
    return out


def register_all(reg) -> None:
    P = "predicate"
    reg.register(P, "sprite_blob", C.sprite_blob, replace=True)
    reg.register(P, "template_match", C.template_match, adapter=_template_adapter,
                 extra_params=("template_paths",), replace=True)
    reg.register(P, "color_at", C.color_at, replace=True)
    reg.register(P, "pixel_at", C.pixel_at, replace=True, doc="Exact-pixel RGB check with tolerance.")
    reg.register(P, "region_color_ratio", C.region_color_ratio, replace=True,
                 doc="Fraction of a region inside an HSV band is within [ratio_min, ratio_max].")
    reg.register(P, "tile_occupied", C.tile_occupied, replace=True,
                 doc="A blob in the HSV band occupies the tile at (col_offset, row_offset) from the player.")
    reg.register(P, "motion_present", C.motion_present, replace=True)
    reg.register(P, "extras_equals", C.extras_equals, replace=True, tags=("gate",))
    reg.register(P, "extras_in", C.extras_in, replace=True, tags=("gate",),
                 doc="Context gate: view.extras[key] is one of values.")
    reg.register(P, "always", C.always, replace=True)
    reg.register(P, "track_locked_to", C.track_locked_to, replace=True)
    reg.register(P, "any_of", any_of, replace=True)
    reg.register(P, "all_of", all_of, replace=True)
    reg.register(P, "not", not_, replace=True)

    RC = "recipe"
    reg.register(RC, "on_match", R.on_match, replace=True)
    reg.register(RC, "click_target", R.click_target, replace=True,
                 aliases={"target": "locator"})
    reg.register(RC, "dialog_advance_loop", R.dialog_advance_loop, replace=True,
                 aliases={"dialog": "dialog_present"})
    reg.register(RC, "color_armed_button", R.color_armed_button, replace=True,
                 aliases={"color": "color_predicate"})
    reg.register(RC, "adjacent_sprite_face_talk", R.adjacent_sprite_face_talk, replace=True,
                 aliases={"sprite": "sprite_locator"})
    reg.register(RC, "adjacent_track_face_talk", R.adjacent_track_face_talk, replace=True)
    reg.register(RC, "walk_to_track", R.walk_to_track, replace=True)
    reg.register(RC, "sprite_within_radius", R.sprite_within_radius, replace=True,
                 aliases={"sprite": "sprite_locator", "do": "on_match_action"})
    reg.register(RC, "anti_stuck", W.anti_stuck, replace=True)
    reg.register(RC, "deadman_release", W.deadman_release, replace=True)

    X = "expect"
    reg.register(X, "gone", E.gone, replace=True)
    reg.register(X, "appears", E.appears, replace=True)
    reg.register(X, "changed", E.changed, replace=True)
    reg.register(X, "all_of", expect_all_of, replace=True)
    reg.register(X, "any_of", expect_any_of, replace=True)

    reg.register("sink", "recording", recording_sink, replace=True)
    reg.register("sink", "press_recording", press_recording_sink, replace=True)
    reg.register("source", "image", image_source, replace=True)
    reg.register("source", "window", window_source, replace=True)
    reg.register("source", "screen", screen_source, replace=True)
    reg.register("sink", "postmessage", postmessage_sink, replace=True)
    reg.register("sink", "sendinput", sendinput_sink, replace=True)
    reg.register("context", "static", static_context, replace=True)
