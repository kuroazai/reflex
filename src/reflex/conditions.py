from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from reflex.core import FrameView, Match, Predicate, Region


@dataclass(frozen=True)
class BlobInfo:
    centroid_xy: tuple[int, int]
    bbox_xywh: tuple[int, int, int, int]
    area_px: int


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
        roi = region.crop(view.frame_bgr)
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
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


def sprite_blob(
    *,
    region: Region,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    min_area_px: int = 24,
    max_area_px: int = 4096,
    pick: str = "largest",
    nearest_to_xy: tuple[int, int] | None = None,
    aspect_ratio_range: tuple[float, float] | None = None,
    min_bbox_height_px: int | None = None,
    min_bbox_width_px: int | None = None,
    require_solidity_below: float | None = None,
) -> Predicate:
    lo = np.array(hsv_lo, dtype=np.uint8)
    hi = np.array(hsv_hi, dtype=np.uint8)
    min_area = int(min_area_px)
    max_area = int(max_area_px)
    if pick not in {"largest", "first", "nearest"}:
        raise ValueError("pick must be 'largest', 'first', or 'nearest'")
    if pick == "nearest" and nearest_to_xy is None:
        raise ValueError("pick='nearest' requires nearest_to_xy=(x, y)")
    target_xy = (int(nearest_to_xy[0]), int(nearest_to_xy[1])) if nearest_to_xy else None

    def predicate(view: FrameView) -> Match | None:
        clipped = region.clip_to(view.frame_bgr)
        roi = clipped.crop(view.frame_bgr)
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lo, hi)
        n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        candidates: list[BlobInfo] = []
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue
            left = int(stats[i, cv2.CC_STAT_LEFT])
            top = int(stats[i, cv2.CC_STAT_TOP])
            w_b = int(stats[i, cv2.CC_STAT_WIDTH])
            h_b = int(stats[i, cv2.CC_STAT_HEIGHT])
            if min_bbox_height_px is not None and h_b < int(min_bbox_height_px):
                continue
            if min_bbox_width_px is not None and w_b < int(min_bbox_width_px):
                continue
            if aspect_ratio_range is not None and h_b > 0:
                ar = float(w_b) / float(h_b)
                lo_ar, hi_ar = float(aspect_ratio_range[0]), float(aspect_ratio_range[1])
                if not (lo_ar <= ar <= hi_ar):
                    continue
            if require_solidity_below is not None and (w_b * h_b) > 0:
                solidity = float(area) / float(w_b * h_b)
                if solidity >= float(require_solidity_below):
                    continue
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            candidates.append(
                BlobInfo(
                    centroid_xy=(int(cx + clipped.x), int(cy + clipped.y)),
                    bbox_xywh=(left + clipped.x, top + clipped.y, w_b, h_b),
                    area_px=area,
                )
            )
            if pick == "first":
                break

        if not candidates:
            return None

        if pick == "first":
            best = candidates[0]
        elif pick == "nearest":
            tx, ty = target_xy  # type: ignore[misc]
            best = min(
                candidates,
                key=lambda b: (b.centroid_xy[0] - tx) ** 2 + (b.centroid_xy[1] - ty) ** 2,
            )
        else:
            best = max(candidates, key=lambda b: b.area_px)

        return Match(
            meta={
                "predicate": "sprite_blob",
                "centroid_xy": best.centroid_xy,
                "bbox_xywh": best.bbox_xywh,
                "area_px": best.area_px,
                "region_xywh": region.as_xywh(),
                "candidate_count": len(candidates),
                "pick_mode": pick,
            }
        )

    return predicate


def template_match(
    *,
    template_bgr: np.ndarray,
    region: Region | None = None,
    threshold: float = 0.85,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Predicate:
    if template_bgr.ndim != 3 or template_bgr.shape[2] != 3:
        raise ValueError("template_bgr must be (h, w, 3)")
    th = float(threshold)

    def predicate(view: FrameView) -> Match | None:
        haystack = (
            region.crop(view.frame_bgr) if region is not None else view.frame_bgr
        )
        offset_x = region.x if region is not None else 0
        offset_y = region.y if region is not None else 0
        if haystack.shape[0] < template_bgr.shape[0] or haystack.shape[1] < template_bgr.shape[1]:
            return None
        result = cv2.matchTemplate(haystack, template_bgr, method)
        _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(result)
        score = float(max_v)
        if score < th:
            return None
        x = int(max_l[0]) + offset_x
        y = int(max_l[1]) + offset_y
        h, w = template_bgr.shape[0], template_bgr.shape[1]
        return Match(
            meta={
                "predicate": "template_match",
                "score": round(score, 4),
                "bbox_xywh": (x, y, int(w), int(h)),
                "centroid_xy": (x + w // 2, y + h // 2),
            }
        )

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
        roi = clipped.crop(view.frame_bgr)
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
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


def all_(*predicates: Predicate) -> Predicate:
    if not predicates:
        raise ValueError("all_() requires at least one predicate")
    preds: tuple[Predicate, ...] = predicates

    def predicate(view: FrameView) -> Match | None:
        merged: dict = {"predicate": "all_", "children": []}
        for p in preds:
            m = p(view)
            if m is None:
                return None
            merged["children"].append(m.meta)
        return Match(meta=merged)

    return predicate


def any_(*predicates: Predicate) -> Predicate:
    if not predicates:
        raise ValueError("any_() requires at least one predicate")
    preds: tuple[Predicate, ...] = predicates

    def predicate(view: FrameView) -> Match | None:
        for i, p in enumerate(preds):
            m = p(view)
            if m is not None:
                return Match(
                    meta={
                        "predicate": "any_",
                        "matched_index": i,
                        "child": m.meta,
                    }
                )
        return None

    return predicate


def not_(predicate_fn: Predicate) -> Predicate:
    def predicate(view: FrameView) -> Match | None:
        m = predicate_fn(view)
        if m is None:
            return Match(meta={"predicate": "not_", "child_was_none": True})
        return None

    return predicate


def first_match(*predicates: Predicate) -> Predicate:
    return any_(*predicates)


def in_sequence(predicates: Sequence[Predicate]) -> Predicate:
    return all_(*predicates)
