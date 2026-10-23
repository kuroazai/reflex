"""
High-level prebuilt TriggerSpec recipes.

Recipes wrap a Predicate plus an action_factory plus sensible defaults
into a TriggerSpec ready to register with the runtime. Use these when
the reflex's shape is well-known (e.g. "approach an adjacent sprite and
press talk") and only the parameters change per game.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reflex import actions as _actions
from reflex.core import FrameView, Match, Predicate, TriggerSpec

_CARDINAL_TO_BUTTON = {
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
    max_fires: int = 1,
    cooldown_s: float = 0.0,
    inter_press_ms: int = 50,
    hold_ms: int = 60,
    priority: int = 70,
) -> TriggerSpec:
    """Locate a sprite, face it if cardinally adjacent, then press talk.

    Wraps any sprite-finding Predicate (sprite_blob, template_match, etc.)
    into the canonical "approach-and-talk" reflex used in tile-grid games.
    """
    if tile_size_px <= 0:
        raise ValueError("tile_size_px must be positive")
    px, py = int(player_screen_xy[0]), int(player_screen_xy[1])

    def predicate(view: FrameView) -> Match | None:
        m = sprite_locator(view)
        if m is None:
            return None
        cx, cy = m.meta["centroid_xy"]
        dc = int(round((int(cx) - px) / tile_size_px))
        dr = int(round((int(cy) - py) / tile_size_px))
        face = _CARDINAL_TO_BUTTON.get((dc, dr))
        if face is None:
            return None
        merged = dict(m.meta)
        merged.update(
            {
                "predicate": "adjacent_sprite_face_talk",
                "tile_delta_cr": (dc, dr),
                "face_button": face,
                "talk_button": talk_button,
            }
        )
        return Match(meta=merged)

    def action_factory(match: Match) -> tuple[str, ...]:
        return (str(match.meta["face_button"]), str(match.meta["talk_button"]))

    return TriggerSpec(
        name=name,
        predicate=predicate,
        action_factory=action_factory,
        timeout_s=timeout_s,
        poll_hz=poll_hz,
        max_fires=max_fires,
        cooldown_s=cooldown_s,
        inter_press_ms=inter_press_ms,
        hold_ms=hold_ms,
        reverify_on_fire=True,
        priority=priority,
        extra={
            "kind": "adjacent_sprite_face_talk",
            "player_screen_xy": [px, py],
            "tile_size_px": int(tile_size_px),
            "talk_button": talk_button,
        },
    )


def dialog_advance_loop(
    *,
    name: str,
    dialog_present: Predicate,
    advance_button: str = "a",
    cooldown_ms: int = 220,
    max_advances: int = 32,
    poll_hz: float = 12.0,
    priority: int = 60,
    hold_ms: int = 35,
) -> TriggerSpec:
    """Press the advance button while a dialog box is on screen.

    Use a cheap dialog detector (white panel, OCR contains certain glyphs,
    color band at known coordinates) and let the runtime drive `max_advances`
    presses with `cooldown_ms` between them. Pairs naturally with a higher-
    priority guard that disables this trigger during cutscenes / battles.
    """

    def predicate(view: FrameView) -> Match | None:
        m = dialog_present(view)
        if m is None:
            return None
        merged = dict(m.meta)
        merged.update({"predicate": "dialog_advance_loop", "advance_button": advance_button})
        return Match(meta=merged)

    def action_factory(_match: Match) -> tuple[str, ...]:
        return (advance_button,)

    return TriggerSpec(
        name=name,
        predicate=predicate,
        action_factory=action_factory,
        timeout_s=120.0,
        poll_hz=poll_hz,
        max_fires=int(max_advances),
        cooldown_s=max(0.0, cooldown_ms / 1000.0),
        inter_press_ms=20,
        hold_ms=hold_ms,
        reverify_on_fire=True,
        priority=priority,
        extra={
            "kind": "dialog_advance_loop",
            "advance_button": advance_button,
            "cooldown_ms": int(cooldown_ms),
            "max_advances": int(max_advances),
        },
    )


def adjacent_track_face_talk(
    *,
    name: str,
    track_id: int,
    track_provider,
    player_xy_provider,
    tile_size_px: int,
    talk_button: str = "a",
    timeout_s: float = 30.0,
    poll_hz: float = 30.0,
    max_fires: int = 1,
    cooldown_s: float = 0.0,
    inter_press_ms: int = 50,
    hold_ms: int = 60,
    priority: int = 70,
) -> TriggerSpec:
    """Same as adjacent_sprite_face_talk but pinned to a specific track ID.

    Both `track_provider(view)` and `player_xy_provider(view)` are called
    every tick. The recipe fires only when the named track is CONFIRMED
    AND its centroid sits in a cardinally adjacent tile relative to the
    live player position. Unlike the sprite_blob version, this never
    flips to a different object on subsequent ticks — once Birch is
    track 7, he stays track 7 until the tracker drops him.
    """
    if tile_size_px <= 0:
        raise ValueError("tile_size_px must be positive")

    def predicate(view: FrameView) -> Match | None:
        info = track_provider(view)
        if info is None or int(info.get("track_id", -1)) != int(track_id):
            return None
        if str(info.get("state", "")).lower() != "confirmed":
            return None
        player = player_xy_provider(view)
        if player is None:
            return None
        cx, cy = info.get("centroid_xy", (0, 0))
        dc = int(round((int(cx) - int(player[0])) / tile_size_px))
        dr = int(round((int(cy) - int(player[1])) / tile_size_px))
        face = _CARDINAL_TO_BUTTON.get((dc, dr))
        if face is None:
            return None
        return Match(
            meta={
                "predicate": "adjacent_track_face_talk",
                "track_id": int(track_id),
                "centroid_xy": tuple(info.get("centroid_xy", (0, 0))),
                "bbox_xywh": tuple(info.get("bbox_xywh", (0, 0, 0, 0))),
                "tile_delta_cr": (dc, dr),
                "face_button": face,
                "talk_button": talk_button,
                "player_xy": tuple(player),
                "score": float(info.get("score", 0.0)),
            }
        )

    def action_factory(match: Match) -> tuple[str, ...]:
        return (str(match.meta["face_button"]), str(match.meta["talk_button"]))

    return TriggerSpec(
        name=name,
        predicate=predicate,
        action_factory=action_factory,
        timeout_s=timeout_s,
        poll_hz=poll_hz,
        max_fires=max_fires,
        cooldown_s=cooldown_s,
        inter_press_ms=inter_press_ms,
        hold_ms=hold_ms,
        reverify_on_fire=True,
        priority=priority,
        extra={
            "kind": "adjacent_track_face_talk",
            "track_id": int(track_id),
            "tile_size_px": int(tile_size_px),
            "talk_button": talk_button,
        },
    )


def walk_to_track(
    *,
    name: str,
    track_id: int,
    track_provider,
    player_xy_provider,
    tile_size_px: int,
    timeout_s: float = 30.0,
    poll_hz: float = 4.0,
    max_steps: int = 16,
    cooldown_s: float = 0.30,
    inter_press_ms: int = 0,
    hold_ms: int = 120,
    priority: int = 60,
) -> TriggerSpec:
    """Navigation reflex: press one cardinal key per fire toward a tracked
    target until that target is cardinally adjacent to the player.

    Pairs with `adjacent_track_face_talk` (default priority 70) — when both
    are registered for the same track_id, this recipe fires while the
    target is two-or-more tiles away and returns None once it's
    cardinally adjacent, letting the higher-priority talk recipe fire
    naturally on the same tick.

    Direction choice is the obvious Manhattan-reducing step: pick the
    larger axis first (so diagonal targets get squared up horizontally
    before vertically — change to suit your game by composing your own
    factory). If the predicted centroid (track.cx + vx*ahead_ticks) is
    available via the provider, intercept becomes free; for stationary
    NPCs the static centroid is fine.
    """
    if tile_size_px <= 0:
        raise ValueError("tile_size_px must be positive")

    def predicate(view: FrameView) -> Match | None:
        info = track_provider(view)
        if info is None or int(info.get("track_id", -1)) != int(track_id):
            return None
        if str(info.get("state", "")).lower() != "confirmed":
            return None
        player = player_xy_provider(view)
        if player is None:
            return None
        cx, cy = info.get("centroid_xy", (0, 0))
        dx = int(cx) - int(player[0])
        dy = int(cy) - int(player[1])
        tdc = int(round(dx / tile_size_px))
        tdr = int(round(dy / tile_size_px))
        if (tdc, tdr) in _CARDINAL_TO_BUTTON:
            return None
        if tdc == 0 and tdr == 0:
            return None
        if abs(tdc) >= abs(tdr) and tdc != 0:
            step = "right" if tdc > 0 else "left"
        else:
            step = "down" if tdr > 0 else "up"
        return Match(
            meta={
                "predicate": "walk_to_track",
                "track_id": int(track_id),
                "centroid_xy": tuple(info.get("centroid_xy", (0, 0))),
                "tile_delta_cr": (tdc, tdr),
                "step": step,
                "player_xy": tuple(player),
                "manhattan_tiles": abs(tdc) + abs(tdr),
            }
        )

    def action_factory(match: Match) -> tuple[str, ...]:
        return (str(match.meta["step"]),)

    return TriggerSpec(
        name=name,
        predicate=predicate,
        action_factory=action_factory,
        timeout_s=timeout_s,
        poll_hz=poll_hz,
        max_fires=int(max_steps),
        cooldown_s=cooldown_s,
        inter_press_ms=inter_press_ms,
        hold_ms=hold_ms,
        reverify_on_fire=False,
        priority=priority,
        extra={
            "kind": "walk_to_track",
            "track_id": int(track_id),
            "tile_size_px": int(tile_size_px),
            "max_steps": int(max_steps),
        },
    )


def color_armed_button(
    *,
    name: str,
    color_predicate: Predicate,
    button: str,
    cooldown_s: float = 0.5,
    poll_hz: float = 60.0,
    priority: int = 50,
    hold_ms: int = 30,
) -> TriggerSpec:
    """Press a button as soon as a color predicate becomes true.

    The classic 'pixel pattern → input' macro. Use for hard
    real-time reflexes — block on red flash, dodge on yellow telegraph,
    interact on prompt-glow appearance.
    """

    def predicate(view: FrameView) -> Match | None:
        m = color_predicate(view)
        if m is None:
            return None
        merged = dict(m.meta)
        merged.update({"predicate": "color_armed_button", "button": button})
        return Match(meta=merged)

    def action_factory(_match: Match) -> tuple[str, ...]:
        return (button,)

    return TriggerSpec(
        name=name,
        predicate=predicate,
        action_factory=action_factory,
        timeout_s=600.0,
        poll_hz=poll_hz,
        max_fires=10_000,
        cooldown_s=cooldown_s,
        inter_press_ms=10,
        hold_ms=hold_ms,
        reverify_on_fire=False,
        priority=priority,
        extra={"kind": "color_armed_button", "button": button},
    )


def sprite_within_radius(
    *,
    name: str,
    sprite_locator: Predicate,
    player_screen_xy: tuple[int, int],
    tile_size_px: int,
    radius_tiles: int,
    on_match_action,
    timeout_s: float = 30.0,
    poll_hz: float = 30.0,
    centroid_meta_key: str = "centroid_xy",
    priority: int = 50,
) -> TriggerSpec:
    """Fire `on_match_action` when a sprite is within `radius_tiles` (Chebyshev)."""
    px, py = int(player_screen_xy[0]), int(player_screen_xy[1])
    ts = int(tile_size_px)
    r = int(radius_tiles)
    if ts <= 0:
        raise ValueError("tile_size_px must be positive")
    if r < 1:
        raise ValueError("radius_tiles must be >= 1")

    def predicate(view: FrameView) -> Match | None:
        match = sprite_locator(view)
        if match is None:
            return None
        cx_cy = match.meta.get(centroid_meta_key)
        if not (isinstance(cx_cy, (tuple, list)) and len(cx_cy) == 2):
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
                "centroid_xy": (cx, cy),
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
        priority=priority,
        extra={"kind": "sprite_within_radius", "player_screen_xy": (px, py), "tile_size_px": ts, "radius_tiles": r},
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


def on_match(
    *,
    name: str,
    when: Predicate,
    do: Sequence[Any],
    timeout_s: float = 30.0,
    poll_hz: float = 30.0,
    max_fires: int = 1,
    cooldown_s: float = 0.0,
    inter_press_ms: int = 50,
    hold_ms: int = 60,
    priority: int = 50,
    reverify_on_fire: bool = True,
) -> TriggerSpec:
    """The generic reflex: when `when` matches, perform `do` (fixed actions).

    `do` items are action strings or typed actions (see reflex.actions) —
    in YAML: ["a", {click: [x, y]}, {wait: 120}, {chord: [ctrl, s]}].
    """
    seq = tuple(_actions.from_spec(a) if not isinstance(a, _actions.ACTION_TYPES) else a for a in do)
    if not seq:
        raise ValueError(f"{name}: on_match needs at least one action in `do`")

    def predicate(view: FrameView) -> Match | None:
        return when(view)

    return TriggerSpec(
        name=name,
        predicate=predicate,
        action_factory=_actions.press_keys(seq),
        timeout_s=timeout_s,
        poll_hz=poll_hz,
        max_fires=max_fires,
        cooldown_s=cooldown_s,
        inter_press_ms=inter_press_ms,
        hold_ms=hold_ms,
        reverify_on_fire=reverify_on_fire,
        priority=priority,
        extra={"kind": "on_match", "do": list(_actions.labels(seq))},
    )


def click_target(
    *,
    name: str,
    locator: Predicate,
    button: str = "left",
    offset_xy: tuple[int, int] = (0, 0),
    centroid_meta_key: str = "centroid_xy",
    timeout_s: float = 30.0,
    poll_hz: float = 15.0,
    max_fires: int = 1,
    cooldown_s: float = 0.5,
    priority: int = 60,
) -> TriggerSpec:
    """Click wherever `locator` finds its target (template/blob centroid).

    The mouse-UI analogue of face-and-talk: find the button, click it.
    Pair with `expect: gone` so a click that didn't take is caught.
    """
    return TriggerSpec(
        name=name,
        predicate=locator,
        action_factory=_actions.click_match(meta_key=centroid_meta_key, button=button, offset_xy=offset_xy),
        timeout_s=timeout_s,
        poll_hz=poll_hz,
        max_fires=max_fires,
        cooldown_s=cooldown_s,
        inter_press_ms=0,
        hold_ms=0,
        priority=priority,
        extra={"kind": "click_target", "button": button},
    )
