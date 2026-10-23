"""walk_to_track navigation recipe + composition with adjacent_track_face_talk."""

from __future__ import annotations

import numpy as np

from reflex.core import FrameView
from reflex.recipes import adjacent_track_face_talk, walk_to_track


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


def test_walk_to_track_picks_horizontal_step_when_dx_dominates():
    spec = walk_to_track(
        name="nav",
        track_id=7,
        track_provider=lambda v: _track(7, cx=500, cy=110),  # +400 right, +10 down (basically pure right)
        player_xy_provider=lambda v: (100, 100),
        tile_size_px=96,
    )
    m = spec.predicate(_view())
    assert m is not None
    assert m.meta["step"] == "right"
    assert spec.action_factory(m) == ("right",)


def test_walk_to_track_picks_vertical_step_when_dy_dominates():
    spec = walk_to_track(
        name="nav",
        track_id=7,
        track_provider=lambda v: _track(7, cx=110, cy=500),
        player_xy_provider=lambda v: (100, 100),
        tile_size_px=96,
    )
    assert spec.predicate(_view()).meta["step"] == "down"


def test_walk_to_track_picks_left_for_negative_dx():
    spec = walk_to_track(
        name="nav",
        track_id=7,
        track_provider=lambda v: _track(7, cx=100, cy=100),
        player_xy_provider=lambda v: (500, 110),
        tile_size_px=96,
    )
    assert spec.predicate(_view()).meta["step"] == "left"


def test_walk_to_track_picks_up_for_negative_dy():
    spec = walk_to_track(
        name="nav",
        track_id=7,
        track_provider=lambda v: _track(7, cx=100, cy=100),
        player_xy_provider=lambda v: (110, 500),
        tile_size_px=96,
    )
    assert spec.predicate(_view()).meta["step"] == "up"


def test_walk_to_track_no_fire_when_adjacent():
    """When target is one cardinal tile away, walk_to_track must yield —
    that's how the talk recipe takes over."""
    spec = walk_to_track(
        name="nav",
        track_id=7,
        track_provider=lambda v: _track(7, cx=196, cy=100),  # +96 right = +1 tile
        player_xy_provider=lambda v: (100, 100),
        tile_size_px=96,
    )
    assert spec.predicate(_view()) is None


def test_walk_to_track_no_fire_when_overlapping():
    spec = walk_to_track(
        name="nav",
        track_id=7,
        track_provider=lambda v: _track(7, cx=100, cy=100),
        player_xy_provider=lambda v: (100, 100),
        tile_size_px=96,
    )
    assert spec.predicate(_view()) is None


def test_walk_to_track_no_fire_when_track_lost():
    spec = walk_to_track(
        name="nav",
        track_id=7,
        track_provider=lambda v: _track(7, cx=500, cy=100, state="lost"),
        player_xy_provider=lambda v: (100, 100),
        tile_size_px=96,
    )
    assert spec.predicate(_view()) is None


def test_walk_to_track_no_fire_when_id_mismatch():
    spec = walk_to_track(
        name="nav",
        track_id=7,
        track_provider=lambda v: _track(99, cx=500, cy=100),
        player_xy_provider=lambda v: (100, 100),
        tile_size_px=96,
    )
    assert spec.predicate(_view()) is None


def test_walk_to_track_max_steps_via_max_fires():
    """max_steps maps directly onto max_fires so the runtime caps the
    navigation budget without per-recipe state."""
    spec = walk_to_track(
        name="nav",
        track_id=7,
        track_provider=lambda v: _track(7, cx=500, cy=100),
        player_xy_provider=lambda v: (100, 100),
        tile_size_px=96,
        max_steps=4,
    )
    assert spec.max_fires == 4


def test_composition_walk_then_talk_handoff():
    """Two reflexes registered for the same track_id should hand off
    cleanly: walk fires while far, talk fires once adjacent."""
    state = {"target_cx": 500, "target_cy": 100, "player_x": 100, "player_y": 100}

    def tp(v):
        return _track(7, cx=state["target_cx"], cy=state["target_cy"])

    def pp(v):
        return (state["player_x"], state["player_y"])

    walk = walk_to_track(name="walk", track_id=7,
                        track_provider=tp, player_xy_provider=pp,
                        tile_size_px=96, max_steps=10)
    talk = adjacent_track_face_talk(name="talk", track_id=7,
                                    track_provider=tp, player_xy_provider=pp,
                                    tile_size_px=96, talk_button="a")

    # Step 1: target far. walk fires "right", talk does NOT.
    walk_match = walk.predicate(_view())
    talk_match = talk.predicate(_view())
    assert walk_match is not None and walk_match.meta["step"] == "right"
    assert talk_match is None

    # Simulate one step of progress: player moves right by 96 px.
    state["player_x"] = 196

    walk_match = walk.predicate(_view())
    talk_match = talk.predicate(_view())
    assert walk_match is not None and walk_match.meta["step"] == "right"
    assert talk_match is None

    # Player keeps walking until adjacent (+96 from target left edge).
    state["player_x"] = 404  # target at 500 → +96 = +1 tile right (cardinal)
    walk_match = walk.predicate(_view())
    talk_match = talk.predicate(_view())
    # Now walk yields — runtime priority will fire talk instead.
    assert walk_match is None
    assert talk_match is not None and talk_match.meta["face_button"] == "right"
    assert talk.action_factory(talk_match) == ("right", "a")
