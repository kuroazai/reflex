from __future__ import annotations

import numpy as np

from reflex import FrameView, tile_occupied


def _frame_with_blob(*, w: int, h: int, blob_xywh: tuple[int, int, int, int], color_bgr: tuple[int, int, int]) -> FrameView:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 200
    img[..., 1] = 200
    img[..., 2] = 200
    bx, by, bw, bh = blob_xywh
    img[by : by + bh, bx : bx + bw] = np.array(color_bgr, dtype=np.uint8)
    return FrameView(frame_bgr=img, ts_ns=0)


def test_tile_occupied_fires_when_blob_in_target_tile() -> None:
    pred = tile_occupied(
        player_xy=(200, 200),
        tile_px=96,
        col_offset=-1,
        row_offset=0,
        hsv_lo=(50, 80, 80),
        hsv_hi=(80, 255, 255),
        min_area_px=64,
        min_fill_fraction=0.05,
    )
    view = _frame_with_blob(
        w=400,
        h=400,
        blob_xywh=(80, 170, 50, 50),
        color_bgr=(40, 200, 40),
    )
    m = pred(view)
    assert m is not None
    assert m.meta["predicate"] == "tile_occupied"
    assert m.meta["col_offset"] == -1
    assert m.meta["row_offset"] == 0
    assert m.meta["blob_area_px"] >= 64


def test_tile_occupied_skips_when_blob_in_wrong_tile() -> None:
    pred = tile_occupied(
        player_xy=(200, 200),
        tile_px=96,
        col_offset=0,
        row_offset=0,
        hsv_lo=(50, 80, 80),
        hsv_hi=(80, 255, 255),
        min_area_px=64,
        min_fill_fraction=0.05,
    )
    view = _frame_with_blob(
        w=400,
        h=400,
        blob_xywh=(20, 20, 30, 30),
        color_bgr=(40, 200, 40),
    )
    assert pred(view) is None


def test_tile_occupied_respects_aspect_ratio_filter() -> None:
    pred = tile_occupied(
        player_xy=(200, 200),
        tile_px=96,
        col_offset=0,
        row_offset=0,
        hsv_lo=(50, 80, 80),
        hsv_hi=(80, 255, 255),
        min_area_px=64,
        min_fill_fraction=0.02,
        aspect_ratio_range=(0.4, 0.9),
    )
    view = _frame_with_blob(
        w=400,
        h=400,
        blob_xywh=(170, 170, 80, 12),
        color_bgr=(40, 200, 40),
    )
    assert pred(view) is None
    view2 = _frame_with_blob(
        w=400,
        h=400,
        blob_xywh=(180, 170, 30, 60),
        color_bgr=(40, 200, 40),
    )
    m = pred(view2)
    assert m is not None
