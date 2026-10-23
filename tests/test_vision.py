"""Screen checks on drawn frames: image search, pixel search, OCR, stillness."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from reflex.core import FrameView, Region
from reflex.dsl import build_predicate
from reflex.vision import Template, color_search, find_color, find_image, image, load_template, stable, text


def _noise(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60, size=(h, w, 3), dtype=np.uint8)


def _sprite():
    s = np.zeros((20, 16, 3), dtype=np.uint8)
    s[:] = (30, 30, 200)
    cv2.circle(s, (8, 10), 6, (240, 240, 240), -1)
    cv2.line(s, (0, 0), (15, 19), (0, 200, 0), 2)
    return s


def _view(frame, t_ms=0):
    return FrameView(frame_bgr=frame, ts_ns=int(t_ms * 1e6))


# --------------------------------------------------------------------------- find_image


def test_find_image_locates_the_sprite():
    frame = _noise()
    frame[100:120, 50:66] = _sprite()
    hits = find_image(frame, _sprite())
    assert hits and (hits[0].x, hits[0].y) == (50, 100) and hits[0].score > 0.99
    assert hits[0].center == (58, 110)


def test_find_image_respects_region_and_threshold():
    frame = _noise()
    frame[100:120, 50:66] = _sprite()
    assert find_image(frame, _sprite(), region=(200, 0, 120, 240)) == []
    assert find_image(frame, _sprite(), region=(40, 90, 40, 40))[0].x == 50


def test_find_image_finds_every_copy():
    frame = _noise()
    spots = [(10, 10), (100, 40), (200, 150), (280, 200)]
    for x, y in spots:
        frame[y : y + 20, x : x + 16] = _sprite()
    hits = find_image(frame, _sprite(), max_results=10)
    assert sorted((h.x, h.y) for h in hits) == sorted(spots)


def test_find_image_across_scales():
    frame = _noise()
    big = cv2.resize(_sprite(), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    frame[60 : 60 + big.shape[0], 90 : 90 + big.shape[1]] = big
    assert find_image(frame, _sprite(), threshold=0.9) == []
    hit = find_image(frame, _sprite(), threshold=0.9, scales=(1.0, 1.5))[0]
    assert hit.scale == 1.5 and abs(hit.x - 90) <= 1 and abs(hit.y - 60) <= 1


def test_transparent_png_matches_over_any_background(tmp_path):
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    cv2.circle(rgba, (10, 10), 7, (0, 220, 255, 255), -1)  # opaque yellow disc, transparent corners
    path = tmp_path / "coin.png"
    cv2.imwrite(str(path), rgba)
    tpl = load_template(path)
    assert tpl.mask is not None
    frame = _noise(seed=3)
    frame[:] = (90, 30, 30)
    frame[150:170, 200:220][rgba[:, :, 3] > 0] = (0, 220, 255)
    frame[150:170, 200:220][rgba[:, :, 3] == 0] = (10, 200, 10)  # background unlike the template's black
    hit = find_image(frame, tpl, threshold=0.9)[0]
    assert (hit.x, hit.y) == (200, 150)


def test_flat_template_only_matches_its_colour():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[10:20, 10:20] = (0, 0, 255)
    red = np.zeros((10, 10, 3), dtype=np.uint8)
    red[:] = (0, 0, 255)
    green = red.copy()
    green[:] = (0, 255, 0)
    assert find_image(frame, red)[0].x == 10
    assert find_image(frame, green) == []


def test_image_predicate_count_and_pick():
    frame = _noise()
    for x, y in [(10, 10), (200, 150)]:
        frame[y : y + 20, x : x + 16] = _sprite()
    view = _view(frame)
    assert image(templates=[_sprite()], count_min=3)(view) is None
    m = image(templates=[_sprite()], count_min=2)(view)
    assert m.meta["count"] == 2
    near = image(templates=[_sprite()], pick="nearest", near_xy=(210, 160))(view)
    assert near.meta["centroid_xy"] == (208, 160)


def test_image_predicate_from_yaml(tmp_path):
    cv2.imwrite(str(tmp_path / "s.png"), _sprite())
    from reflex.dsl import Builder

    pred = Builder(base_dir=tmp_path).predicate({"predicate": "image", "template": "s.png", "threshold": 0.9})
    frame = _noise()
    frame[30:50, 40:56] = _sprite()
    assert pred(_view(frame)).meta["bbox_xywh"] == (40, 30, 16, 20)


# --------------------------------------------------------------------------- colour


def test_find_color_first_nearest_all():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[5, 30] = (0, 0, 250)  # BGR red-ish
    frame[35, 2] = (0, 0, 255)
    assert find_color(frame, (255, 0, 0), tolerance=10) == [(30, 5)]
    assert find_color(frame, (255, 0, 0), tolerance=10, pick="nearest", near_xy=(0, 39)) == [(2, 35)]
    assert len(find_color(frame, (255, 0, 0), tolerance=10, pick="all")) == 2
    assert find_color(frame, (255, 0, 0), tolerance=2) == [(2, 35)]


def test_color_search_predicate_and_min_pixels():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[10:12, 10:12] = (0, 255, 0)
    view = _view(frame)
    assert color_search(rgb=(0, 255, 0))(view).meta["xy"] == (10, 10)
    assert color_search(rgb=(0, 255, 0), min_pixels=5)(view) is None
    assert color_search(rgb=(0, 255, 0), min_pixels=4)(view) is not None
    pred = build_predicate({"predicate": "color_search", "rgb": [0, 255, 0], "region": [20, 20, 20, 20]})
    assert pred(view) is None


# --------------------------------------------------------------------------- stable


def test_stable_waits_for_stillness():
    pred = stable(for_ms=300)
    a = _noise(seed=1)
    b = _noise(seed=2)
    assert pred(_view(a, 0)) is None
    assert pred(_view(b, 100)) is None  # changed: clock restarts at 100
    assert pred(_view(b, 300)) is None  # still for 200 ms
    m = pred(_view(b, 450))
    assert m is not None and m.meta["still_ms"] == 350.0


def test_stable_ignores_change_outside_region():
    pred = stable(region=Region(0, 0, 50, 50), for_ms=100)
    a = _noise(seed=1)
    b = a.copy()
    b[200:, 200:] = 255
    pred(_view(a, 0))
    assert pred(_view(b, 150)) is not None


# --------------------------------------------------------------------------- OCR


def _ocr_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        from reflex.ocr import get_engine

        get_engine()
        return True
    except Exception:  # noqa: BLE001
        return False


ocr = pytest.mark.skipif(not _ocr_available(), reason="needs Windows OCR (pip install reflex[ocr])")


def _label(text_str, *, scale=1.0, size=(80, 400), at=(10, 50)):
    img = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    cv2.putText(img, text_str, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return img


@ocr
def test_read_text_and_numbers():
    from reflex.ocr import numbers_in, read_text

    res = read_text(_label("Gold 1250"))
    assert "1250" in res.text
    assert numbers_in("HP 42/100") == [42.0, 100.0]


@ocr
def test_text_predicate_contains_and_points_at_the_word():
    frame = _noise(h=200, w=500)
    frame[100:180, 50:450] = _label("START GAME")
    m = text(contains="start")(_view(frame))
    assert m is not None
    x, y = m.meta["centroid_xy"]
    assert 50 <= x <= 450 and 100 <= y <= 180


@ocr
def test_text_predicate_number_threshold():
    frame = _label("HP 42/100")
    assert text(number_below=50)(_view(frame)) is not None
    assert text(number_below=40)(_view(frame)) is None
    assert text(number_above=10)(_view(frame)).meta["number"] == 42.0


@ocr
def test_text_predicate_absent():
    frame = _label("Loading")
    assert text(contains="loading", absent=True)(_view(frame)) is None
    assert text(contains="victory", absent=True)(_view(frame)) is not None


def test_text_needs_a_condition():
    with pytest.raises(ValueError, match="text needs"):
        text()


def test_template_dataclass_size():
    assert Template(_sprite()).size == (16, 20)


def test_legacy_template_match_flat_template_on_black_is_not_a_match():
    from reflex.conditions import template_match

    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    red = np.zeros((10, 10, 3), dtype=np.uint8)
    red[:] = (0, 0, 255)
    pred = template_match(region=Region(0, 0, 50, 50), templates=[red])
    assert pred(_view(frame)) is None
    frame[30:40, 5:15] = (0, 0, 255)
    assert pred(_view(frame)).meta["bbox_xywh"] == (5, 30, 10, 10)
