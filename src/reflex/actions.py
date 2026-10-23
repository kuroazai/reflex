from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from reflex.core import Match


def press_keys(buttons: Sequence[str]) -> Callable[[Match], Sequence[str]]:
    seq = tuple(str(b) for b in buttons)

    def factory(_match: Match) -> Sequence[str]:
        return seq

    return factory


def compute_then_press(fn: Callable[[Match], Sequence[str]]) -> Callable[[Match], Sequence[str]]:
    def factory(match: Match) -> Sequence[str]:
        return tuple(str(b) for b in fn(match))

    return factory


def face_then_talk(
    *,
    talk_button: str = "a",
    direction_meta_key: str = "face_button",
) -> Callable[[Match], Sequence[str]]:
    def factory(match: Match) -> Sequence[str]:
        face = match.meta.get(direction_meta_key)
        if not isinstance(face, str):
            raise ValueError(
                f"face_then_talk: match.meta['{direction_meta_key}'] missing or non-str: {match.meta!r}"
            )
        return (face, talk_button)

    return factory


def chained(
    *factories: Callable[[Match], Sequence[str]],
) -> Callable[[Match], Sequence[str]]:
    def factory(match: Match) -> Sequence[str]:
        out: list[str] = []
        for f in factories:
            out.extend(str(b) for b in f(match))
        return tuple(out)

    return factory


def relative_button_from_delta(
    *,
    cardinal_only: bool = True,
    delta_meta_key: str = "tile_delta_cr",
) -> Callable[[Match], dict[str, Any]]:
    cardinals: dict[tuple[int, int], str] = {
        (1, 0): "right",
        (-1, 0): "left",
        (0, 1): "down",
        (0, -1): "up",
    }

    def transform(match: Match) -> dict[str, Any]:
        delta = match.meta.get(delta_meta_key)
        if not (isinstance(delta, (tuple, list)) and len(delta) == 2):
            return {}
        dc, dr = int(delta[0]), int(delta[1])
        if cardinal_only:
            face = cardinals.get((dc, dr))
            if face is None:
                return {}
            return {"face_button": face}
        # Diagonal handling: prefer larger axis
        if abs(dc) >= abs(dr):
            face = "right" if dc > 0 else "left"
        else:
            face = "down" if dr > 0 else "up"
        return {"face_button": face}

    return transform
