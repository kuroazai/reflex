"""track_locked_to + adjacent_track_face_talk: pin reflex to a track id."""

from __future__ import annotations

import numpy as np

from reflex.conditions import track_locked_to
from reflex.core import FrameView
from reflex.recipes import adjacent_track_face_talk


def _view():
    return FrameView(frame_bgr=np.zeros((10, 10, 3), dtype=np.uint8), ts_ns=0)


def _track(track_id, cx, cy, *, state="confirmed", bbox=(0, 0, 24, 24), score=0.9):
    return {
        "track_id": track_id,
        "centroid_xy": (cx, cy),
        "bbox_xywh": bbox,
        "state": state,
        "score": score,
        "hits": 5,
    }


def test_track_locked_to_fires_when_id_matches_and_confirmed():
    pred = track_locked_to(track_id=7, track_provider=lambda v: _track(7, 100, 200))
    m = pred(_view())
    assert m is not None
    assert m.meta["track_id"] == 7
    assert m.meta["centroid_xy"] == (100, 200)
    assert m.meta["state"] == "confirmed"


def test_track_locked_to_does_not_fire_when_id_differs():
    pred = track_locked_to(track_id=7, track_provider=lambda v: _track(99, 100, 200))
    assert pred(_view()) is None


def test_track_locked_to_does_not_fire_when_track_lost():
    pred = track_locked_to(track_id=7, track_provider=lambda v: _track(7, 100, 200, state="lost"))
    assert pred(_view()) is None


def test_track_locked_to_does_not_fire_when_provider_returns_none():
    pred = track_locked_to(track_id=7, track_provider=lambda v: None)
    assert pred(_view()) is None


def test_adjacent_track_face_talk_fires_when_track_adjacent_right():
    spec = adjacent_track_face_talk(
        name="lock_npc",
        track_id=42,
        track_provider=lambda v: _track(42, cx=200, cy=100),
        player_xy_provider=lambda v: (104, 100),  # 96 px left of track = +1 tile right
        tile_size_px=96,
        talk_button="a",
    )
    m = spec.predicate(_view())
    assert m is not None
    assert m.meta["face_button"] == "right"
    assert m.meta["tile_delta_cr"] == (1, 0)
    assert spec.action_factory(m) == ("right", "a")


def test_adjacent_track_face_talk_does_not_fire_when_diagonal():
    spec = adjacent_track_face_talk(
        name="lock_npc",
        track_id=42,
        track_provider=lambda v: _track(42, cx=200, cy=200),
        player_xy_provider=lambda v: (104, 104),  # diagonal +1, +1 -> not cardinal
        tile_size_px=96,
    )
    assert spec.predicate(_view()) is None


def test_adjacent_track_face_talk_does_not_fire_when_id_flips():
    """If the tracker reassigned the id (shouldn't happen but tested anyway),
    the locked spec must refuse — that's the whole point of locking."""
    state = {"id": 42}

    def provider(v):
        return _track(state["id"], cx=200, cy=100)

    spec = adjacent_track_face_talk(
        name="lock_npc",
        track_id=42,
        track_provider=provider,
        player_xy_provider=lambda v: (104, 100),
        tile_size_px=96,
    )
    assert spec.predicate(_view()) is not None  # initially fires
    state["id"] = 99  # tracker swapped to a different id
    assert spec.predicate(_view()) is None  # now refuses


def test_adjacent_track_face_talk_does_not_fire_when_player_unknown():
    spec = adjacent_track_face_talk(
        name="lock_npc",
        track_id=42,
        track_provider=lambda v: _track(42, cx=200, cy=100),
        player_xy_provider=lambda v: None,
        tile_size_px=96,
    )
    assert spec.predicate(_view()) is None
