"""Screen checks: find an image, find a colour, read text, wait for stillness.

Each check comes twice: a plain function that answers a question about one
frame (`find_image`, `find_color`), for scripts and agents, and a predicate
factory (`image`, `color_search`, `text`, `stable`) for macros. Coordinates
are always frame pixels, the same space the input sinks click in.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from reflex.core import FrameView, Match, Predicate, Region

# --------------------------------------------------------------------------- templates


@dataclass(frozen=True)
class Template:
    """An image to look for. `mask` is set when the PNG had transparency:
    transparent pixels are ignored, so a sprite matches over any background."""

    bgr: np.ndarray
    mask: np.ndarray | None = None
    name: str = ""

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.bgr.shape[1]), int(self.bgr.shape[0]))


_CACHE: dict[tuple[str, float], Template] = {}


def load_template(path: str | Path) -> Template:
    """Load a template image (cached until the file changes)."""
    p = Path(path)
    try:
        key = (str(p.resolve()), p.stat().st_mtime)
    except OSError:
        raise FileNotFoundError(f"cannot read template image {p}") from None
    hit = _CACHE.get(key)
    if hit is None:
        hit = _read_template(p)
        if len(_CACHE) > 256:
            _CACHE.clear()
        _CACHE[key] = hit
    return hit


def _read_template(p: Path) -> Template:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read template image {p}")
    if img.ndim == 2:
        return Template(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, p.name)
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = np.ascontiguousarray(img[:, :, :3])
        if int(alpha.min()) == 255:
            return Template(bgr, None, p.name)
        mask = (alpha > 127).astype(np.uint8) * 255
        if not mask.any():
            raise ValueError(f"template {p} is fully transparent")
        return Template(bgr, cv2.merge([mask, mask, mask]), p.name)
    return Template(np.ascontiguousarray(img), None, p.name)


def is_flat(img: np.ndarray, *, tolerance: float = 4.0) -> bool:
    """True when an image is (nearly) one colour, so as a template it would match
    any area of that colour. Measured per channel: a plain red patch is flat even
    though its red and blue values differ."""
    if img.size == 0:
        return True
    px = img.reshape(-1, img.shape[2] if img.ndim == 3 else 1).astype(np.float32)
    return float(px.std(axis=0).max()) < tolerance


def as_template(t: Template | np.ndarray | str | Path) -> Template:
    if isinstance(t, Template):
        return t
    if isinstance(t, np.ndarray):
        if t.ndim != 3 or t.shape[2] != 3:
            raise ValueError("template arrays must be (h, w, 3) BGR")
        return Template(np.ascontiguousarray(t))
    return load_template(t)


@dataclass(frozen=True)
class Hit:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float = 1.0
    template: int = 0

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def as_meta(self) -> dict:
        return {
            "bbox_xywh": (self.x, self.y, self.w, self.h),
            "centroid_xy": self.center,
            "score": round(self.score, 4),
            "scale": self.scale,
            "template_index": self.template,
        }


def _clip(frame: np.ndarray, region: Region | Sequence[int] | None) -> tuple[np.ndarray, int, int]:
    if region is None:
        return frame, 0, 0
    r = region if isinstance(region, Region) else Region(*(int(v) for v in region))
    c = r.clip_to(frame)
    return c.crop(frame), c.x, c.y


def _peaks(score_map: np.ndarray, threshold: float, w: int, h: int, limit: int) -> list[tuple[int, int, float]]:
    """Local maxima above threshold, greedily suppressing overlaps."""
    out: list[tuple[int, int, float]] = []
    m = score_map.copy()
    for _ in range(limit):
        _mn, mx, _mnl, mxl = cv2.minMaxLoc(m)
        if mx < threshold:
            break
        x, y = int(mxl[0]), int(mxl[1])
        out.append((x, y, float(mx)))
        x0, y0 = max(0, x - w // 2), max(0, y - h // 2)
        m[y0 : y + h // 2 + 1, x0 : x + w // 2 + 1] = -1.0
    return out


def match_scores(haystack: np.ndarray, tpl: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Score map in [0, 1] for every placement of `tpl` in `haystack`.

    Textured templates use normalised cross-correlation. A template that is a
    single colour (inside its mask) has no variance, which makes correlation
    undefined, and normalised squared difference divides by zero on black
    areas: both report perfect matches where there are none. Flat templates
    are scored as 1 - rms colour difference / 255 instead.
    """
    chans = 1 if tpl.ndim == 2 else tpl.shape[2]
    px = tpl.reshape(-1, chans)
    if mask is not None:
        m2 = mask if mask.ndim == 2 else mask[:, :, 0]
        px = px[m2.reshape(-1) > 0]
        n = int(np.count_nonzero(m2)) * chans
    else:
        n = int(tpl.shape[0] * tpl.shape[1]) * chans
    flat = px.size == 0 or float(px.astype(np.float32).std(axis=0).max()) < 0.5
    if flat:
        sq = cv2.matchTemplate(haystack, tpl, cv2.TM_SQDIFF, mask=mask)
        res = 1.0 - np.sqrt(np.maximum(sq, 0) / max(1, n)) / 255.0
    else:
        res = cv2.matchTemplate(haystack, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def find_image(
    frame_bgr: np.ndarray,
    templates: Template | np.ndarray | str | Path | Sequence[Template | np.ndarray | str | Path],
    *,
    region: Region | Sequence[int] | None = None,
    threshold: float = 0.85,
    scales: Sequence[float] = (1.0,),
    grayscale: bool = False,
    max_results: int = 1,
) -> list[Hit]:
    """Where the template(s) appear, best first. Scores are in [0, 1].

    scales   — try the template at these sizes too (e.g. [0.8, 1.0, 1.25]) for
               UIs that resize with the window.
    grayscale — faster; fine when colour is not what tells things apart.
    max_results — >1 returns every non-overlapping match (count enemies, loot).
    """
    if isinstance(templates, (Template, np.ndarray, str, Path)):
        tpls = [as_template(templates)]
    else:
        tpls = [as_template(t) for t in templates]
    if not tpls:
        raise ValueError("find_image needs at least one template")
    hay, ox, oy = _clip(frame_bgr, region)
    if hay.size == 0:
        return []
    hay_m = cv2.cvtColor(hay, cv2.COLOR_BGR2GRAY) if grayscale else hay
    hits: list[Hit] = []
    for ti, tpl in enumerate(tpls):
        for s in scales:
            img, mask = tpl.bgr, tpl.mask
            if s != 1.0:
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
                if mask is not None:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            th, tw = img.shape[:2]
            if th < 2 or tw < 2 or th > hay.shape[0] or tw > hay.shape[1]:
                continue
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if mask is not None:
                    mask = mask[:, :, 0]
            res = match_scores(hay_m, img, mask)
            for x, y, sc in _peaks(res, threshold, tw, th, max(1, int(max_results))):
                hits.append(Hit(x + ox, y + oy, tw, th, min(1.0, sc), float(s), ti))
    hits.sort(key=lambda h: -h.score)
    kept: list[Hit] = []
    for h in hits:  # drop overlaps across scales/templates
        if all(_iou(h, k) < 0.3 for k in kept):
            kept.append(h)
        if len(kept) >= max_results:
            break
    return kept


def _iou(a: Hit, b: Hit) -> float:
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- colour


def find_color(
    frame_bgr: np.ndarray,
    rgb: Sequence[int],
    *,
    tolerance: int = 8,
    region: Region | Sequence[int] | None = None,
    pick: Literal["first", "nearest", "all"] = "first",
    near_xy: Sequence[int] | None = None,
    max_results: int = 500,
) -> list[tuple[int, int]]:
    """Pixels within `tolerance` of an RGB colour (per channel). 'first' scans
    rows top to bottom, left to right, like a classic pixel search."""
    hay, ox, oy = _clip(frame_bgr, region)
    if hay.size == 0:
        return []
    target = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.int16)
    close = np.all(np.abs(hay.astype(np.int16) - target) <= int(tolerance), axis=2)
    ys, xs = np.nonzero(close)
    if xs.size == 0:
        return []
    if pick == "first":
        return [(int(xs[0]) + ox, int(ys[0]) + oy)]
    pts = np.stack([xs + ox, ys + oy], axis=1)
    if pick == "nearest":
        if near_xy is None:
            raise ValueError("pick='nearest' needs near_xy")
        d = (pts[:, 0] - int(near_xy[0])) ** 2 + (pts[:, 1] - int(near_xy[1])) ** 2
        i = int(np.argmin(d))
        return [(int(pts[i, 0]), int(pts[i, 1]))]
    return [(int(x), int(y)) for x, y in pts[: int(max_results)]]


# --------------------------------------------------------------------------- predicates


def image(
    *,
    templates: Sequence[Template | np.ndarray],
    region: Region | None = None,
    threshold: float = 0.85,
    scales: Sequence[float] = (1.0,),
    grayscale: bool = False,
    count_min: int = 1,
    count_max: int | None = None,
    pick: str = "best",
    near_xy: tuple[int, int] | None = None,
) -> Predicate:
    """Matches when an image is on screen (`count_min` or more times).

    pick: best | nearest (to near_xy) | leftmost | topmost — which hit's
    centroid_xy a click action uses. Meta also has `count` and every `hits`.
    """
    if pick not in ("best", "nearest", "leftmost", "topmost"):
        raise ValueError("pick must be best, nearest, leftmost or topmost")
    if pick == "nearest" and near_xy is None:
        raise ValueError("pick='nearest' needs near_xy")
    tpls = [as_template(t) for t in templates]
    want = max(int(count_min), int(count_max or 0), 1)
    limit = max(want + 1, 20) if (count_max is not None or count_min > 1 or pick != "best") else 1

    def predicate(view: FrameView) -> Match | None:
        hits = find_image(view.frame_bgr, tpls, region=region, threshold=threshold, scales=scales,
                          grayscale=grayscale, max_results=limit)
        n = len(hits)
        if n < int(count_min) or (count_max is not None and n > int(count_max)):
            return None
        if pick == "nearest":
            nx, ny = near_xy  # type: ignore[misc]
            chosen = min(hits, key=lambda h: (h.center[0] - nx) ** 2 + (h.center[1] - ny) ** 2)
        elif pick == "leftmost":
            chosen = min(hits, key=lambda h: h.x)
        elif pick == "topmost":
            chosen = min(hits, key=lambda h: h.y)
        else:
            chosen = hits[0]
        return Match(meta={"predicate": "image", **chosen.as_meta(), "count": n,
                           "hits": [h.as_meta()["bbox_xywh"] for h in hits]})

    return predicate


def color_search(
    *,
    rgb: tuple[int, int, int],
    tolerance: int = 8,
    region: Region | None = None,
    pick: str = "first",
    near_xy: tuple[int, int] | None = None,
    min_pixels: int = 1,
) -> Predicate:
    """Matches when a colour appears anywhere in the region (pixel search)."""
    if pick not in ("first", "nearest"):
        raise ValueError("pick must be first or nearest")

    def predicate(view: FrameView) -> Match | None:
        if min_pixels > 1:
            pts = find_color(view.frame_bgr, rgb, tolerance=tolerance, region=region, pick="all",
                             max_results=10**9)
            if len(pts) < min_pixels:
                return None
            if pick == "nearest":
                pts = find_color(view.frame_bgr, rgb, tolerance=tolerance, region=region, pick="nearest",
                                 near_xy=near_xy)
            n = len(pts)
        else:
            pts = find_color(view.frame_bgr, rgb, tolerance=tolerance, region=region, pick=pick,  # type: ignore[arg-type]
                             near_xy=near_xy)
            if not pts:
                return None
            n = 1
        x, y = pts[0]
        return Match(meta={"predicate": "color_search", "centroid_xy": (x, y), "xy": (x, y), "pixels": n})

    return predicate


def text(
    *,
    region: Region | None = None,
    contains: str | None = None,
    regex: str | None = None,
    equals: str | None = None,
    case: bool = False,
    number_below: float | None = None,
    number_above: float | None = None,
    number_index: int = 0,
    absent: bool = False,
) -> Predicate:
    """Matches on text read from the screen (OCR).

    contains / equals / regex — what the text must show.
    number_below / number_above — compare the Nth number in the text, e.g.
      "HP 42/100" with number_below: 50 matches (42 < 50).
    absent — invert: match when the text is NOT there (wait for a label to go).
    centroid_xy points at the matched words, so a click lands on them.
    """
    from reflex.ocr import numbers_in, read_text

    if not any(v is not None for v in (contains, regex, equals, number_below, number_above)):
        raise ValueError("text needs contains, equals, regex, number_below or number_above")
    rx = re.compile(regex, 0 if case else re.I) if regex else None
    reg = region.as_xywh() if region is not None else None

    def predicate(view: FrameView) -> Match | None:
        res = read_text(view.frame_bgr, reg)
        body = res.text if case else res.text.lower()
        flat = " ".join(res.lines)
        ok = True
        where = None
        if contains is not None:
            ok = (contains if case else contains.lower()) in body
            where = res.find(contains, case=case) if ok else None
        if ok and equals is not None:
            ok = flat.strip() == equals.strip() if case else flat.strip().lower() == equals.strip().lower()
        if ok and rx is not None:
            m = rx.search(flat)
            ok = m is not None
            if m is not None and where is None:
                where = res.find(m.group(0), case=case)
        number = None
        if ok and (number_below is not None or number_above is not None):
            nums = numbers_in(flat)
            if len(nums) <= number_index:
                ok = False
            else:
                number = nums[number_index]
                if number_below is not None and not number < float(number_below):
                    ok = False
                if number_above is not None and not number > float(number_above):
                    ok = False
        if absent:
            ok = not ok
        if not ok:
            return None
        meta: dict = {"predicate": "text", "text": res.text}
        if number is not None:
            meta["number"] = number
        if where is not None:
            meta["bbox_xywh"] = where.bbox_xywh
            meta["centroid_xy"] = where.centroid_xy
        elif reg is not None:
            meta["centroid_xy"] = (reg[0] + reg[2] // 2, reg[1] + reg[3] // 2)
        return Match(meta=meta)

    return predicate


def stable(
    *,
    region: Region | None = None,
    for_ms: int = 500,
    diff_threshold: float = 6.0,
    max_changed_fraction: float = 0.002,
) -> Predicate:
    """Matches once the region has stopped changing for `for_ms`.

    Use it to wait for a loading screen, an animation or a scroll to settle
    before reading or clicking. Timing uses frame timestamps, so it behaves
    the same under replay as live.
    """
    state: dict = {"prev": None, "since_ns": None}

    def predicate(view: FrameView) -> Match | None:
        cur, _ox, _oy = _clip(view.frame_bgr, region)
        prev = state["prev"]
        state["prev"] = cur.copy()
        if prev is None or prev.shape != cur.shape:
            state["since_ns"] = view.ts_ns
            return None
        changed = float((cv2.absdiff(cur, prev).max(axis=2) > diff_threshold).mean()) if cur.size else 0.0
        if changed > max_changed_fraction:
            state["since_ns"] = view.ts_ns
            return None
        still_ms = (view.ts_ns - state["since_ns"]) / 1e6
        if still_ms < for_ms:
            return None
        return Match(meta={"predicate": "stable", "still_ms": round(still_ms, 1)})

    return predicate
