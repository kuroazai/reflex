from __future__ import annotations

from collections.abc import Callable
from typing import Any

from reflex.core import FrameView, Match, Predicate, Region, TriggerSpec


_CARDINAL_DELTA_TO_BUTTON: dict[tuple[int, int], str] = {
    (1, 0): "right",
    (-1, 0): "left",
    (0, 1): "down",
    (0, -1): "up",
}


def adjacent_sprite_face_talk(
    *,
    name: str,
    sprite_locator: Predicate,
    player_screen_xy: tuple[int, int],
    tile_size_px: int,
    talk_button: str = "a",
    timeout_s: float = 30.0,
    poll_hz: float = 30.0,
    inter_press_ms: int = 50,
    hold_ms: int = 60,
    centroid_meta_key: str = "centroid_xy",
    cooldown_s: float = 0.0,
) -> TriggerSpec:
    px, py = int(player_screen_xy[0]), int(player_screen_xy[1])
    ts = int(tile_size_px)
    if ts <= 0:
        raise ValueError("tile_size_px must be > 0")

    def predicate(view: FrameView) -> Match | None:
        match = sprite_locator(view)
        if match is None:
            return None
        cx_cy = match.meta.get(centroid_meta_key)
        if not (isinstance(cx_cy, tuple) and len(cx_cy) == 2):
            return None
        cx, cy = int(cx_cy[0]), int(cx_cy[1])
        dx_px = cx - px
        dy_px = cy - py
        dc = int(round(dx_px / ts))
        dr = int(round(dy_px / ts))
        button = _CARDINAL_DELTA_TO_BUTTON.get((dc, dr))
        if button is None:
            return None
        return Match(
            meta={
                "predicate": "adjacent_sprite_face_talk",
                "sprite_centroid_xy": (cx, cy),
                "player_screen_xy": (px, py),
                "tile_delta_cr": (dc, dr),
                "tile_offset_px": (dx_px, dy_px),
                "face_button": button,
                "talk_button": talk_button,
                "child_match_meta": match.meta,
            }
        )

    def action_factory(match: Match) -> tuple[str, ...]:
        face = match.meta["face_button"]
        return (str(face), str(talk_button))

    return TriggerSpec(
        name=name,
        predicate=predicate,
        action_factory=action_factory,
        timeout_s=timeout_s,
        poll_hz=poll_hz,
        max_fires=1,
        cooldown_s=cooldown_s,
        inter_press_ms=inter_press_ms,
        hold_ms=hold_ms,
        reverify_on_fire=True,
        extra={
            "kind": "adjacent_sprite_face_talk",
            "player_screen_xy": (px, py),
            "tile_size_px": ts,
        },
    )


def sprite_within_radius(
    *,
    name: str,
    sprite_locator: Predicate,
    player_screen_xy: tuple[int, int],
    tile_size_px: int,
    radius_tiles: int,
    on_match_action: Callable[[Match], Any],
    timeout_s: float = 30.0,
    poll_hz: float = 30.0,
    centroid_meta_key: str = "centroid_xy",
) -> TriggerSpec:
    px, py = int(player_screen_xy[0]), int(player_screen_xy[1])
    ts = int(tile_size_px)
    r = int(radius_tiles)
    if r < 1:
        raise ValueError("radius_tiles must be >= 1")

    def predicate(view: FrameView) -> Match | None:
        match = sprite_locator(view)
        if match is None:
            return None
        cx_cy = match.meta.get(centroid_meta_key)
        if not (isinstance(cx_cy, tuple) and len(cx_cy) == 2):
            return None
        cx, cy = int(cx_cy[0]), int(cx_cy[1])
        dc = int(round((cx - px) / ts))
        dr = int(round((cy - py) / ts))
        if max(abs(dc), abs(dr)) > r:
            return None
        return Match(
            meta={
                "predicate": "sprite_within_radius",
                "sprite_centroid_xy": (cx, cy),
                "player_screen_xy": (px, py),
                "tile_delta_cr": (dc, dr),
                "radius_tiles": r,
                "child_match_meta": match.meta,
            }
        )

    return TriggerSpec(
        name=name,
        predicate=predicate,
        action_factory=on_match_action,
        timeout_s=timeout_s,
        poll_hz=poll_hz,
        max_fires=1,
        cooldown_s=0.0,
        inter_press_ms=50,
        hold_ms=60,
        reverify_on_fire=True,
        extra={
            "kind": "sprite_within_radius",
            "player_screen_xy": (px, py),
            "tile_size_px": ts,
            "radius_tiles": r,
        },
    )


def player_screen_xy_from_relative(
    *,
    centroid_relative_xy: tuple[float, float],
    frame_size_wh: tuple[int, int],
) -> tuple[int, int]:
    rx, ry = float(centroid_relative_xy[0]), float(centroid_relative_xy[1])
    if not (0.0 <= rx <= 1.0 and 0.0 <= ry <= 1.0):
        raise ValueError(f"centroid_relative_xy out of [0,1]: {(rx, ry)!r}")
    w, h = int(frame_size_wh[0]), int(frame_size_wh[1])
    return (int(round(rx * w)), int(round(ry * h)))
