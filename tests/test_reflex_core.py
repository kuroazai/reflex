"""Region geometry, FrameView, Match, TriggerSpec basics."""

from __future__ import annotations

import numpy as np

from reflex.core import FrameView, Match, Region, TriggerSpec


def test_region_as_xywh_and_xyxy():
    r = Region(10, 20, 30, 40)
    assert r.as_xywh() == (10, 20, 30, 40)
    assert r.as_xyxy() == (10, 20, 40, 60)


def test_region_crop_within_bounds():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:60, 30:80] = 255
    crop = Region(30, 20, 50, 40).crop(frame)
    assert crop.shape == (40, 50, 3)
    assert int(crop.mean()) == 255


def test_region_crop_clipped_to_frame():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    crop = Region(40, 40, 100, 100).crop(frame)
    assert crop.shape == (10, 10, 3)


def test_region_crop_completely_outside():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    crop = Region(200, 200, 50, 50).crop(frame)
    assert crop.size == 0


def test_region_clip_to_returns_clipped_region():
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    clipped = Region(20, 20, 100, 100).clip_to(frame)
    assert clipped.as_xywh() == (20, 20, 20, 10)


def test_region_contains_xy():
    r = Region(10, 10, 20, 20)
    assert r.contains_xy(10, 10) is True
    assert r.contains_xy(29, 29) is True
    assert r.contains_xy(30, 30) is False
    assert r.contains_xy(9, 9) is False


def test_match_meta_round_trip():
    m = Match(meta={"score": 0.84, "centroid_xy": (10, 20)})
    assert m.meta["score"] == 0.84
    assert m.meta["centroid_xy"] == (10, 20)


def test_trigger_spec_with_priority_returns_new_spec():
    def predicate(_v):
        return None

    def af(_m):
        return ("a",)

    spec = TriggerSpec(name="x", predicate=predicate, action_factory=af, priority=50)
    bumped = spec.with_priority(95)
    assert spec.priority == 50
    assert bumped.priority == 95
    assert bumped.name == "x"


def test_frame_view_holds_extras():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    view = FrameView(frame_bgr=frame, ts_ns=12345, extras={"prev_frame_bgr": frame})
    assert view.ts_ns == 12345
    assert "prev_frame_bgr" in view.extras
