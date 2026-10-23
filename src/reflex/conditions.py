"""
Reusable Predicate factories.

A Predicate is `Callable[[FrameView], Optional[Match]]`. This module
provides the building blocks that reflex specs assemble into typed
triggers — sprite blobs, template matches, OCR contains, motion
presence, and color-band hits.

All predicates here are pure functions of the FrameView. Side effects
(logging, debug image dumps, telemetry) belong in higher-level wrappers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from reflex.core import FrameView, Match, Predicate, Region


@dataclass(frozen=True)
class BlobInfo:
    centroid_xy: tuple[int, int]
    bbox_xywh: tuple[int, int, int, int]
    area_px: int


_HSV_CACHE_KEY = "_hsv_full"
_HSV_CACHE_MIN_FRACTION = 0.25


def hsv_roi(view: FrameView, clipped: Region) -> np.ndarray:
    """HSV of a (clipped) region, sharing one full-frame conversion per tick.

    Large regions read from a full-frame HSV computed once per FrameView (the
    runtime shares one view across every predicate in a tick), so N blob /
    ratio predicates cost one cvtColor instead of N. Small regions convert
    just their ROI, which is cheaper than a full-frame pass.
    """
    frame = view.frame_bgr
    cached = view.extras.get(_HSV_CACHE_KEY)
    # Keyed on the frame object: a reverify view shares extras but carries a fresh frame.
    full = cached[1] if cached is not None and cached[0] is frame else None
    area_frac = (clipped.w * clipped.h) / float(max(1, frame.shape[0] * frame.shape[1]))
    if full is None and area_frac >= _HSV_CACHE_MIN_FRACTION:
        full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        view.extras[_HSV_CACHE_KEY] = (frame, full)
    if full is not None:
        return full[clipped.y : clipped.y + clipped.h, clipped.x : clipped.x + clipped.w]
    return cv2.cvtColor(frame[clipped.y : clipped.y + clipped.h, clipped.x : clipped.x + clipped.w], cv2.COLOR_BGR2HSV)


def sprite_blob(
    *,
    region: Region,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    min_area_px: int = 24,
    max_area_px: int = 4096,
    aspect_ratio_range: tuple[float, float] | None = None,
    min_bbox_height_px: int | None = None,
    min_bbox_width_px: int | None = None,
    require_solidity_below: float | None = None,
    pick: str = "largest",
    nearest_to_xy: tuple[int, int] | None = None,
) -> Predicate:
    """HSV-banded connected-component blob detector.

    pick:
      "largest"  — biggest passing blob by area
      "first"    — first in scan order
      "nearest"  — closest passing blob to nearest_to_xy (required)
    """
    lo = np.array(hsv_lo, dtype=np.uint8)
    hi = np.array(hsv_hi, dtype=np.uint8)
    if pick == "nearest" and nearest_to_xy is None:
        raise ValueError('sprite_blob(pick="nearest") requires nearest_to_xy')

    def _classify(area_px: int, h: int, w: int) -> bool:
        if not (min_area_px <= area_px <= max_area_px):
            return False
        if min_bbox_height_px is not None and h < min_bbox_height_px:
            return False
        if min_bbox_width_px is not None and w < min_bbox_width_px:
            return False
        if aspect_ratio_range is not None and h > 0:
            ar = float(w) / float(h)
            if not (aspect_ratio_range[0] <= ar <= aspect_ratio_range[1]):
                return False
        if require_solidity_below is not None and (w * h) > 0:
            solidity = float(area_px) / float(w * h)
            if solidity >= require_solidity_below:
                return False
        return True

    def predicate(view: FrameView) -> Match | None:
        clipped = region.clip_to(view.frame_bgr)
        if clipped.w == 0 or clipped.h == 0:
            return None
        hsv = hsv_roi(view, clipped)
        mask = cv2.inRange(hsv, lo, hi)
        n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        passing = []
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            left = int(stats[i, cv2.CC_STAT_LEFT])
            top = int(stats[i, cv2.CC_STAT_TOP])
            w_b = int(stats[i, cv2.CC_STAT_WIDTH])
            h_b = int(stats[i, cv2.CC_STAT_HEIGHT])
            if not _classify(area, h_b, w_b):
                continue
            cx = int(centroids[i][0]) + clipped.x
            cy = int(centroids[i][1]) + clipped.y
            passing.append(
                {
                    "centroid_xy": (cx, cy),
                    "bbox_xywh": (left + clipped.x, top + clipped.y, w_b, h_b),
                    "area_px": area,
                }
            )

        if not passing:
            return None

        if pick == "first":
            chosen = passing[0]
        elif pick == "nearest":
            tx, ty = nearest_to_xy  # type: ignore[misc]
            chosen = min(
                passing,
                key=lambda r: (r["centroid_xy"][0] - tx) ** 2
                + (r["centroid_xy"][1] - ty) ** 2,
            )
        else:
            chosen = max(passing, key=lambda r: r["area_px"])

        return Match(
            meta={
                "predicate": "sprite_blob",
                "centroid_xy": chosen["centroid_xy"],
                "bbox_xywh": chosen["bbox_xywh"],
                "area_px": chosen["area_px"],
                "candidate_count": len(passing),
            }
        )

    return predicate


def template_match(
    *,
    region: Region,
    templates: Sequence[np.ndarray],
    threshold: float = 0.65,
    method: int = cv2.TM_CCOEFF_NORMED,
    grayscale: bool = False,
) -> Predicate:
    """Multi-template best-of-N matcher within a region.

    grayscale=True matches on luminance only — several times faster, fine
    for UI glyphs/buttons whose colour isn't the distinguishing feature.

    Returns the highest-scoring template hit if any clears `threshold`.
    Scores are normalised to "higher is better" in [0, 1].

    Flat (single-colour) templates have zero variance, which makes
    TM_CCOEFF_NORMED return 1.0 *everywhere* — a guaranteed false positive.
    Those templates are matched with TM_SQDIFF_NORMED instead (score =
    1 - sqdiff) so a solid-colour button crop only matches that colour.
    """
    if not templates:
        raise ValueError("template_match requires at least one template")
    tpl_list = [np.ascontiguousarray(t) for t in templates]
    if grayscale:
        tpl_list = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if t.ndim == 3 else t for t in tpl_list]
    methods = []
    for t in tpl_list:
        flat = float(t.reshape(-1, t.shape[-1] if t.ndim == 3 else 1).std(axis=0).max()) < 1e-6
        methods.append(cv2.TM_SQDIFF_NORMED if (flat and method == cv2.TM_CCOEFF_NORMED) else method)

    def _score(res: np.ndarray, m: int) -> tuple[float, tuple[int, int]]:
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(res)
        if m in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            return 1.0 - float(min_v), (int(min_l[0]), int(min_l[1]))
        return float(max_v), (int(max_l[0]), int(max_l[1]))

    def predicate(view: FrameView) -> Match | None:
        clipped = region.clip_to(view.frame_bgr)
        haystack = clipped.crop(view.frame_bgr)
        if haystack.size == 0:
            return None
        if grayscale:
            haystack = cv2.cvtColor(haystack, cv2.COLOR_BGR2GRAY)
        best_score = -1.0
        best_loc: tuple[int, int] | None = None
        best_idx = -1
        for i, tpl in enumerate(tpl_list):
            if haystack.shape[0] < tpl.shape[0] or haystack.shape[1] < tpl.shape[1]:
                continue
            score, loc = _score(cv2.matchTemplate(haystack, tpl, methods[i]), methods[i])
            if score > best_score:
                best_score = score
                best_loc = (loc[0] + clipped.x, loc[1] + clipped.y)
                best_idx = i
        if best_loc is None or best_score < threshold:
            return None
        tpl = tpl_list[best_idx]
        bx, by = best_loc
        bh, bw = int(tpl.shape[0]), int(tpl.shape[1])
        return Match(
            meta={
                "predicate": "template_match",
                "score": round(best_score, 4),
                "template_index": best_idx,
                "bbox_xywh": (bx, by, bw, bh),
                "centroid_xy": (bx + bw // 2, by + bh // 2),
            }
        )

    return predicate


def color_at(
    *,
    xy: tuple[int, int],
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
) -> Predicate:
    """Single-pixel HSV-range check at exact (x, y).

    The lightweight Blue Eye Macro analogue: deterministic, sub-millisecond,
    perfect for static HUD bar tests. Use for stable indicators where
    the band you care about never moves.
    """
    lo = np.array(hsv_lo, dtype=np.uint8)
    hi = np.array(hsv_hi, dtype=np.uint8)
    px, py = int(xy[0]), int(xy[1])

    def predicate(view: FrameView) -> Match | None:
        h, w = view.frame_bgr.shape[:2]
        if not (0 <= px < w and 0 <= py < h):
            return None
        bgr = view.frame_bgr[py : py + 1, px : px + 1]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
        if all(int(lo[c]) <= int(hsv[c]) <= int(hi[c]) for c in range(3)):
            return Match(meta={"predicate": "color_at", "xy": (px, py), "hsv": tuple(int(v) for v in hsv)})
        return None

    return predicate


def motion_present(
    *,
    region: Region,
    last_frame_extra_key: str = "prev_frame_bgr",
    diff_threshold: float = 6.0,
    min_changed_fraction: float = 0.01,
) -> Predicate:
    """Frame-diff-based motion detector for armed reflexes.

    Requires the runtime to seed view.extras[last_frame_extra_key] with the
    previous frame. Useful for "fire only after something visibly changed"
    gating — pairs well with a slow heavy detector running below.
    """

    def predicate(view: FrameView) -> Match | None:
        prev = view.extras.get(last_frame_extra_key)
        if not isinstance(prev, np.ndarray) or prev.shape != view.frame_bgr.shape:
            return None
        clipped = region.clip_to(view.frame_bgr)
        if clipped.w == 0 or clipped.h == 0:
            return None
        a = view.frame_bgr[clipped.y : clipped.y + clipped.h, clipped.x : clipped.x + clipped.w]
        b = prev[clipped.y : clipped.y + clipped.h, clipped.x : clipped.x + clipped.w]
        diff = cv2.absdiff(a, b).mean(axis=2)
        changed = float((diff > diff_threshold).mean())
        if changed < min_changed_fraction:
            return None
        return Match(
            meta={
                "predicate": "motion_present",
                "changed_fraction": round(changed, 4),
                "region_xywh": list(clipped.as_xywh()),
            }
        )

    return predicate


def any_of(predicates: Iterable[Predicate]) -> Predicate:
    """Disjunctive composition: fire on the first sub-predicate that hits."""
    parts = list(predicates)

    def predicate(view: FrameView) -> Match | None:
        for p in parts:
            m = p(view)
            if m is not None:
                return Match(meta={"predicate": "any_of", "from": m.meta})
        return None

    return predicate


def all_of(predicates: Iterable[Predicate]) -> Predicate:
    """Conjunctive composition: fire only when every sub-predicate hits.

    Returns the merged meta of every sub-match under "parts".
    """
    parts = list(predicates)

    def predicate(view: FrameView) -> Match | None:
        metas = []
        for p in parts:
            m = p(view)
            if m is None:
                return None
            metas.append(m.meta)
        return Match(meta={"predicate": "all_of", "parts": metas})

    return predicate


def track_locked_to(
    *,
    track_id: int,
    track_provider,
) -> Predicate:
    """Pin a reflex to a specific tracker ID.

    `track_provider(view)` is called on every tick and must return either
    None (track is currently lost / not in scene) or a dict with at least
    {bbox_xywh, centroid_xy, score, state}. Typical implementation: a
    closure over a multi-object tracker (e.g. ByteTrack) plus a
    detector that produces the per-tick detection list.

    The predicate fires (returns Match) only when the named track is
    present and CONFIRMED. Use this in place of nearest-blob heuristics
    when identity stability matters more than re-discovery (e.g. follow
    a specific NPC across an entire scene without ID flips).
    """
    target_id = int(track_id)

    def predicate(view: FrameView) -> Match | None:
        info = track_provider(view)
        if info is None:
            return None
        if int(info.get("track_id", -1)) != target_id:
            return None
        if str(info.get("state", "")).lower() != "confirmed":
            return None
        return Match(
            meta={
                "predicate": "track_locked_to",
                "track_id": target_id,
                "centroid_xy": tuple(info.get("centroid_xy", (0, 0))),
                "bbox_xywh": tuple(info.get("bbox_xywh", (0, 0, 0, 0))),
                "score": float(info.get("score", 0.0)),
                "state": str(info.get("state", "")),
                "hits": int(info.get("hits", 0)),
            }
        )

    return predicate


def pixel_at(
    *,
    x: int,
    y: int,
    rgb: tuple[int, int, int],
    tolerance: int = 10,
) -> Predicate:
    target = np.array(rgb, dtype=np.int16)
    tol = int(tolerance)

    def predicate(view: FrameView) -> Match | None:
        H, W = view.height, view.width
        if not (0 <= x < W and 0 <= y < H):
            return None
        bgr = view.frame_bgr[y, x]
        sample = np.array([int(bgr[2]), int(bgr[1]), int(bgr[0])], dtype=np.int16)
        if int(np.max(np.abs(sample - target))) <= tol:
            return Match(
                meta={
                    "predicate": "pixel_at",
                    "x": x,
                    "y": y,
                    "sample_rgb": (int(sample[0]), int(sample[1]), int(sample[2])),
                    "target_rgb": tuple(int(v) for v in rgb),
                    "tolerance": tol,
                }
            )
        return None

    return predicate


def region_color_ratio(
    *,
    region: Region,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    ratio_min: float = 0.05,
    ratio_max: float = 1.0,
) -> Predicate:
    lo = np.array(hsv_lo, dtype=np.uint8)
    hi = np.array(hsv_hi, dtype=np.uint8)
    rmin = float(ratio_min)
    rmax = float(ratio_max)

    def predicate(view: FrameView) -> Match | None:
        clipped = region.clip_to(view.frame_bgr)
        if clipped.w == 0 or clipped.h == 0:
            return None
        hsv = hsv_roi(view, clipped)
        mask = cv2.inRange(hsv, lo, hi)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if rmin <= ratio <= rmax:
            return Match(
                meta={
                    "predicate": "region_color_ratio",
                    "region_xywh": region.as_xywh(),
                    "ratio": round(ratio, 4),
                    "hsv_lo": tuple(int(v) for v in hsv_lo),
                    "hsv_hi": tuple(int(v) for v in hsv_hi),
                }
            )
        return None

    return predicate


def tile_occupied(
    *,
    player_xy: tuple[int, int],
    tile_px: int,
    col_offset: int,
    row_offset: int,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    min_area_px: int = 24,
    min_fill_fraction: float = 0.04,
    aspect_ratio_range: tuple[float, float] | None = None,
    require_solidity_below: float | None = None,
) -> Predicate:
    px, py = int(player_xy[0]), int(player_xy[1])
    tp = max(1, int(tile_px))
    cx_t = px + int(col_offset) * tp
    cy_t = py + int(row_offset) * tp
    half = tp // 2
    region = Region(
        x=max(0, cx_t - half),
        y=max(0, cy_t - half),
        w=tp,
        h=tp,
    )
    lo = np.array(hsv_lo, dtype=np.uint8)
    hi = np.array(hsv_hi, dtype=np.uint8)
    fill_min = float(min_fill_fraction)
    min_area = int(min_area_px)

    def predicate(view: FrameView) -> Match | None:
        clipped = region.clip_to(view.frame_bgr)
        if clipped.w == 0 or clipped.h == 0:
            return None
        hsv = hsv_roi(view, clipped)
        mask = cv2.inRange(hsv, lo, hi)
        fill = float(np.count_nonzero(mask)) / float(max(1, mask.size))
        if fill < fill_min:
            return None
        n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best_blob: BlobInfo | None = None
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            left = int(stats[i, cv2.CC_STAT_LEFT])
            top = int(stats[i, cv2.CC_STAT_TOP])
            w_b = int(stats[i, cv2.CC_STAT_WIDTH])
            h_b = int(stats[i, cv2.CC_STAT_HEIGHT])
            if aspect_ratio_range is not None and h_b > 0:
                ar = float(w_b) / float(h_b)
                if not (float(aspect_ratio_range[0]) <= ar <= float(aspect_ratio_range[1])):
                    continue
            if require_solidity_below is not None and (w_b * h_b) > 0:
                solidity = float(area) / float(w_b * h_b)
                if solidity >= float(require_solidity_below):
                    continue
            cx = float(centroids[i][0]) + clipped.x
            cy = float(centroids[i][1]) + clipped.y
            blob = BlobInfo(
                centroid_xy=(int(cx), int(cy)),
                bbox_xywh=(int(left + clipped.x), int(top + clipped.y), w_b, h_b),
                area_px=area,
            )
            if best_blob is None or area > best_blob.area_px:
                best_blob = blob
        if best_blob is None:
            return None
        return Match(
            meta={
                "predicate": "tile_occupied",
                "player_xy": (px, py),
                "tile_px": tp,
                "col_offset": int(col_offset),
                "row_offset": int(row_offset),
                "tile_region_xywh": clipped.as_xywh(),
                "fill_fraction": round(fill, 4),
                "blob_centroid_xy": best_blob.centroid_xy,
                "blob_bbox_xywh": best_blob.bbox_xywh,
                "blob_area_px": best_blob.area_px,
            }
        )

    return predicate



def not_(predicate_fn: Predicate) -> Predicate:
    """Fires when the wrapped predicate does NOT match."""

    def predicate(view: FrameView) -> Match | None:
        if predicate_fn(view) is None:
            return Match(meta={"predicate": "not_", "child_was_none": True})
        return None

    return predicate


def all_(*predicates: Predicate) -> Predicate:
    if not predicates:
        raise ValueError("all_() requires at least one predicate")
    return all_of(predicates)


def any_(*predicates: Predicate) -> Predicate:
    if not predicates:
        raise ValueError("any_() requires at least one predicate")
    return any_of(predicates)


first_match = any_


def in_sequence(predicates: Sequence[Predicate]) -> Predicate:
    return all_(*predicates)


def extras_equals(*, key: str, value: Any) -> Predicate:
    """Context gate: view.extras[key] == value (e.g. scene == "battle").

    Context comes from runtime context providers, so reflexes can be gated
    on game state the harness already computes instead of re-deriving it
    from pixels.
    """

    def predicate(view: FrameView) -> Match | None:
        got = view.extras.get(key)
        if got == value:
            return Match(meta={"predicate": "extras_equals", "key": key, "value": got})
        return None

    return predicate


def extras_in(*, key: str, values: Sequence[Any]) -> Predicate:
    allowed = list(values)

    def predicate(view: FrameView) -> Match | None:
        got = view.extras.get(key)
        if got in allowed:
            return Match(meta={"predicate": "extras_in", "key": key, "value": got})
        return None

    return predicate


def always() -> Predicate:
    """Always matches (use with gates, or for unconditional timed actions)."""

    def predicate(_view: FrameView) -> Match | None:
        return Match(meta={"predicate": "always"})

    return predicate
