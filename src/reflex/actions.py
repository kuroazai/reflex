"""
Typed input actions + action-factory helpers.

An action factory returns a sequence of action items. A plain string is
shorthand for `Key(name)` so every pre-existing factory keeps working.
Typed actions cover what a press-only interface cannot express:

    Key(name, hold_ms=None)      press + release (hold defaults to spec.hold_ms)
    KeyDown(name) / KeyUp(name)  hold a direction across later actions
    Chord(keys, hold_ms=None)    press several keys together, release together
    Click(x, y, button="left")   mouse click at window-client coordinates
    Wait(ms)                     pause inside the sequence

Every action names the sink capability it needs (`Action.capability`);
the runtime refuses an action the sink cannot perform instead of silently
dropping it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from reflex.core import Match


@dataclass(frozen=True)
class Key:
    name: str
    hold_ms: int | None = None
    capability = "key"

    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class KeyDown:
    name: str
    capability = "key_down"

    def label(self) -> str:
        return f"{self.name}:down"


@dataclass(frozen=True)
class KeyUp:
    name: str
    capability = "key_up"

    def label(self) -> str:
        return f"{self.name}:up"


@dataclass(frozen=True)
class Chord:
    keys: tuple[str, ...]
    hold_ms: int | None = None
    capability = "chord"

    def label(self) -> str:
        return "+".join(self.keys)


@dataclass(frozen=True)
class Click:
    x: int
    y: int
    button: str = "left"
    capability = "click"

    def label(self) -> str:
        return f"click:{self.button}@{int(self.x)},{int(self.y)}"


@dataclass(frozen=True)
class Wait:
    ms: int
    capability = "wait"

    def label(self) -> str:
        return f"wait:{int(self.ms)}ms"


Action = Key | KeyDown | KeyUp | Chord | Click | Wait
ACTION_TYPES = (Key, KeyDown, KeyUp, Chord, Click, Wait)


def normalize(item: Any) -> Action:
    """Coerce a factory output item into a typed Action."""
    if isinstance(item, ACTION_TYPES):
        return item
    if isinstance(item, str):
        if not item:
            raise ValueError("empty action string")
        return Key(item)
    raise TypeError(f"not an action: {item!r}")


def normalize_seq(items: Iterable[Any]) -> tuple[Action, ...]:
    return tuple(normalize(i) for i in items)


def from_spec(value: Any) -> Action:
    """Build an Action from its YAML form.

    "a"                         -> Key("a")
    {key: a, hold_ms: 80}       -> Key
    {down: right} / {up: right} -> KeyDown / KeyUp
    {chord: [a, b]}             -> Chord
    {click: [x, y], button: r}  -> Click
    {wait: 120}                 -> Wait
    """
    if isinstance(value, str):
        return Key(value)
    if not isinstance(value, dict):
        raise ValueError(f"action must be a string or mapping, got {value!r}")
    if "key" in value:
        return Key(str(value["key"]), hold_ms=_opt_int(value.get("hold_ms")))
    if "down" in value:
        return KeyDown(str(value["down"]))
    if "up" in value:
        return KeyUp(str(value["up"]))
    if "chord" in value:
        keys = value["chord"]
        if not isinstance(keys, (list, tuple)) or not keys:
            raise ValueError(f"chord needs a non-empty key list, got {keys!r}")
        return Chord(tuple(str(k) for k in keys), hold_ms=_opt_int(value.get("hold_ms")))
    if "click" in value:
        xy = value["click"]
        if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
            raise ValueError(f"click needs [x, y], got {xy!r}")
        return Click(int(xy[0]), int(xy[1]), button=str(value.get("button", "left")))
    if "wait" in value:
        return Wait(int(value["wait"]))
    raise ValueError(f"unknown action form {value!r}")


def _opt_int(v: Any) -> int | None:
    return None if v is None else int(v)


def labels(actions: Sequence[Action]) -> tuple[str, ...]:
    return tuple(a.label() for a in actions)


# ---------------------------------------------------------------------------
# Action-factory helpers
# ---------------------------------------------------------------------------

Factory = Callable[[Match], Sequence[Any]]


def press_keys(buttons: Sequence[Any]) -> Factory:
    """Constant factory: always emit the same sequence."""
    seq = normalize_seq(buttons)

    def factory(_match: Match) -> Sequence[Action]:
        return seq

    return factory


def compute_then_press(fn: Callable[[Match], Sequence[Any]]) -> Factory:
    """Factory whose sequence depends on the match (e.g. its location)."""

    def factory(match: Match) -> Sequence[Action]:
        return normalize_seq(fn(match))

    return factory


def click_match(*, meta_key: str = "centroid_xy", button: str = "left", offset_xy: tuple[int, int] = (0, 0)) -> Factory:
    """Click wherever the predicate found its target."""

    def factory(match: Match) -> Sequence[Action]:
        xy = match.meta.get(meta_key)
        if not (isinstance(xy, (tuple, list)) and len(xy) == 2):
            raise ValueError(f"click_match: match.meta[{meta_key!r}] missing: {match.meta!r}")
        return (Click(int(xy[0]) + int(offset_xy[0]), int(xy[1]) + int(offset_xy[1]), button=button),)

    return factory


def face_then_talk(*, talk_button: str = "a", direction_meta_key: str = "face_button") -> Factory:
    def factory(match: Match) -> Sequence[Action]:
        face = match.meta.get(direction_meta_key)
        if not isinstance(face, str):
            raise ValueError(
                f"face_then_talk: match.meta['{direction_meta_key}'] missing or non-str: {match.meta!r}"
            )
        return (Key(face), Key(talk_button))

    return factory


def chained(*factories: Factory) -> Factory:
    def factory(match: Match) -> Sequence[Action]:
        out = []
        for f in factories:
            out.extend(normalize_seq(f(match)))
        return tuple(out)

    return factory


_CARDINALS: dict[tuple[int, int], str] = {
    (1, 0): "right",
    (-1, 0): "left",
    (0, 1): "down",
    (0, -1): "up",
}


def relative_button_from_delta(
    *,
    cardinal_only: bool = True,
    delta_meta_key: str = "tile_delta_cr",
) -> Callable[[Match], dict[str, Any]]:
    def transform(match: Match) -> dict[str, Any]:
        delta = match.meta.get(delta_meta_key)
        if not (isinstance(delta, (tuple, list)) and len(delta) == 2):
            return {}
        dc, dr = int(delta[0]), int(delta[1])
        if cardinal_only:
            face = _CARDINALS.get((dc, dr))
            return {} if face is None else {"face_button": face}
        if dc == 0 and dr == 0:
            return {}
        if abs(dc) >= abs(dr):
            face = "right" if dc > 0 else "left"
        else:
            face = "down" if dr > 0 else "up"
        return {"face_button": face}

    return transform
