"""Predicate factories: sprite_blob, color_at, motion_present, any_of, all_of."""

from __future__ import annotations

import numpy as np

from reflex.conditions import all_of, any_of, color_at, motion_present, sprite_blob
from reflex.core import FrameView, Region


def _frame_with_green_blob(*, w=200, h=200, blob_xywh=(60, 60, 30, 30)):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    x, y, bw, bh = blob_xywh
    frame[y : y + bh, x : x + bw] = (40, 200, 40)  # BGR green
    return frame


def _view(frame, *, ts_ns=0, extras=None):
    return FrameView(frame_bgr=frame, ts_ns=ts_ns, extras=extras or {})


def test_sprite_blob_finds_green_blob_largest():
    frame = _frame_with_green_blob()
    pred = sprite_blob(
        region=Region(0, 0, 200, 200),
        hsv_lo=(40, 80, 80),
        hsv_hi=(85, 255, 255),
        min_area_px=100,
    )
    m = pred(_view(frame))
    assert m is not None
    cx, cy = m.meta["centroid_xy"]
    assert 70 <= cx <= 80
    assert 70 <= cy <= 80
    assert m.meta["area_px"] >= 800


def test_sprite_blob_returns_none_when_below_min_area():
    frame = _frame_with_green_blob(blob_xywh=(60, 60, 5, 5))
    pred = sprite_blob(
        region=Region(0, 0, 200, 200),
        hsv_lo=(40, 80, 80),
        hsv_hi=(85, 255, 255),
        min_area_px=100,
    )
    assert pred(_view(frame)) is None


def test_sprite_blob_pick_nearest_chooses_closer_blob():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[20:50, 20:50] = (40, 200, 40)     # close to (10,10)
    frame[200:240, 200:240] = (40, 200, 40)  # far from (10,10)
    pred = sprite_blob(
        region=Region(0, 0, 300, 300),
        hsv_lo=(40, 80, 80),
        hsv_hi=(85, 255, 255),
        min_area_px=100,
        pick="nearest",
        nearest_to_xy=(10, 10),
    )
    m = pred(_view(frame))
    assert m is not None
    cx, cy = m.meta["centroid_xy"]
    assert cx < 100 and cy < 100


def test_color_at_inside_band():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[50, 50] = (0, 0, 255)  # red BGR
    pred = color_at(xy=(50, 50), hsv_lo=(0, 180, 180), hsv_hi=(10, 255, 255))
    assert pred(_view(frame)) is not None


def test_color_at_outside_band():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[50, 50] = (0, 255, 0)
    pred = color_at(xy=(50, 50), hsv_lo=(0, 180, 180), hsv_hi=(10, 255, 255))
    assert pred(_view(frame)) is None


def test_color_at_out_of_bounds_xy_returns_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pred = color_at(xy=(500, 500), hsv_lo=(0, 0, 0), hsv_hi=(179, 255, 255))
    assert pred(_view(frame)) is None


def test_motion_present_no_prev_returns_none():
    frame = _frame_with_green_blob()
    pred = motion_present(region=Region(0, 0, 200, 200))
    assert pred(_view(frame)) is None


def test_motion_present_detects_change():
    a = _frame_with_green_blob()
    b = _frame_with_green_blob(blob_xywh=(120, 120, 30, 30))
    pred = motion_present(region=Region(0, 0, 200, 200))
    m = pred(_view(b, extras={"prev_frame_bgr": a}))
    assert m is not None
    assert m.meta["changed_fraction"] > 0.01


def test_any_of_fires_on_first_hit():
    frame = _frame_with_green_blob()
    p1 = sprite_blob(
        region=Region(0, 0, 200, 200),
        hsv_lo=(0, 0, 0),
        hsv_hi=(5, 5, 5),  # impossible band
        min_area_px=100,
    )
    p2 = sprite_blob(
        region=Region(0, 0, 200, 200),
        hsv_lo=(40, 80, 80),
        hsv_hi=(85, 255, 255),
        min_area_px=100,
    )
    m = any_of([p1, p2])(_view(frame))
    assert m is not None


def test_all_of_requires_every_predicate():
    frame = _frame_with_green_blob()
    p_match = sprite_blob(
        region=Region(0, 0, 200, 200),
        hsv_lo=(40, 80, 80),
        hsv_hi=(85, 255, 255),
        min_area_px=100,
    )
    p_miss = sprite_blob(
        region=Region(0, 0, 200, 200),
        hsv_lo=(0, 0, 0),
        hsv_hi=(5, 5, 5),
        min_area_px=100,
    )
    assert all_of([p_match, p_miss])(_view(frame)) is None
    assert all_of([p_match, p_match])(_view(frame)) is not None
