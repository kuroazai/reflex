from __future__ import annotations

import numpy as np

from reflex.conditions import (
    all_,
    any_,
    not_,
    pixel_at,
    region_color_ratio,
    sprite_blob,
)
from reflex.core import FrameView, Region


def _frame_with_white_blob() -> np.ndarray:
    img = np.zeros((64, 96, 3), dtype=np.uint8)
    img[20:40, 60:80, :] = 255
    return img


def _view(img: np.ndarray) -> FrameView:
    return FrameView(frame_bgr=img, ts_ns=0)


def test_pixel_at_matches_within_tolerance():
    img = _frame_with_white_blob()
    pred = pixel_at(x=70, y=30, rgb=(250, 250, 250), tolerance=10)
    assert pred(_view(img)) is not None


def test_pixel_at_misses_off_tolerance():
    img = _frame_with_white_blob()
    pred = pixel_at(x=10, y=10, rgb=(255, 255, 255), tolerance=10)
    assert pred(_view(img)) is None


def test_region_color_ratio_finds_white_strip():
    img = _frame_with_white_blob()
    pred = region_color_ratio(
        region=Region(50, 15, 40, 30),
        hsv_lo=(0, 0, 200),
        hsv_hi=(179, 50, 255),
        ratio_min=0.05,
    )
    m = pred(_view(img))
    assert m is not None
    assert m.meta["ratio"] > 0.05


def test_sprite_blob_returns_centroid_in_absolute_coords():
    img = _frame_with_white_blob()
    pred = sprite_blob(
        region=Region(0, 0, 96, 64),
        hsv_lo=(0, 0, 200),
        hsv_hi=(179, 50, 255),
        min_area_px=50,
        max_area_px=10000,
    )
    m = pred(_view(img))
    assert m is not None
    cx, cy = m.meta["centroid_xy"]
    assert 65 <= cx <= 75
    assert 25 <= cy <= 35


def test_sprite_blob_shape_filter_rejects_uniform_squares():
    img = np.zeros((96, 144, 3), dtype=np.uint8)
    img[10:30, 10:30, :] = 255
    img[10:30, 50:70, :] = 255
    img[50:90, 100:120, :] = 255

    locator_no_filter = sprite_blob(
        region=Region(0, 0, 144, 96),
        hsv_lo=(0, 0, 200),
        hsv_hi=(179, 50, 255),
        min_area_px=100,
        max_area_px=10000,
        pick="largest",
    )
    m_unfiltered = locator_no_filter(_view(img))
    assert m_unfiltered is not None

    locator_tall_only = sprite_blob(
        region=Region(0, 0, 144, 96),
        hsv_lo=(0, 0, 200),
        hsv_hi=(179, 50, 255),
        min_area_px=100,
        max_area_px=10000,
        aspect_ratio_range=(0.0, 0.85),
        min_bbox_height_px=30,
        pick="largest",
    )
    m_tall = locator_tall_only(_view(img))
    assert m_tall is not None
    cx, cy = m_tall.meta["centroid_xy"]
    assert 100 <= cx <= 120
    assert 50 <= cy <= 90


def test_sprite_blob_pick_nearest_to_target():
    img = np.zeros((64, 192, 3), dtype=np.uint8)
    img[20:40, 20:40, :] = 255
    img[20:40, 150:170, :] = 255

    locator = sprite_blob(
        region=Region(0, 0, 192, 64),
        hsv_lo=(0, 0, 200),
        hsv_hi=(179, 50, 255),
        min_area_px=50,
        max_area_px=10_000,
        pick="nearest",
        nearest_to_xy=(140, 30),
    )
    m = locator(_view(img))
    assert m is not None
    cx, _ = m.meta["centroid_xy"]
    assert 140 <= cx <= 180


def test_combinators_all_any_not():
    img = _frame_with_white_blob()
    p_white = pixel_at(x=70, y=30, rgb=(255, 255, 255), tolerance=10)
    p_black = pixel_at(x=5, y=5, rgb=(0, 0, 0), tolerance=10)
    p_red = pixel_at(x=70, y=30, rgb=(255, 0, 0), tolerance=10)

    assert all_(p_white, p_black)(_view(img)) is not None
    assert all_(p_white, p_red)(_view(img)) is None
    assert any_(p_white, p_red)(_view(img)) is not None
    assert any_(p_red, p_red)(_view(img)) is None
    assert not_(p_red)(_view(img)) is not None
    assert not_(p_white)(_view(img)) is None
