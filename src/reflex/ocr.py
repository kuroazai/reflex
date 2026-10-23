"""Read text off a frame.

The engine is pluggable. The default is the OCR engine built into Windows 10
and 11 (Windows.Media.Ocr): no model download and a few milliseconds per
read. Install it with `pip install reflex[ocr]`. Anything with
`read(image_bgr) -> OcrResult` can replace it via `set_engine()`.

Small game text reads far better enlarged, so the Windows engine upscales
crops by 2x before reading and maps boxes back to frame pixels.
"""

from __future__ import annotations

import asyncio
import re
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Word:
    text: str
    bbox_xywh: tuple[int, int, int, int]
    line: int

    @property
    def centroid_xy(self) -> tuple[int, int]:
        x, y, w, h = self.bbox_xywh
        return (x + w // 2, y + h // 2)


@dataclass(frozen=True)
class OcrResult:
    text: str
    lines: tuple[str, ...] = ()
    words: tuple[Word, ...] = field(default_factory=tuple)

    def find(self, needle: str, *, case: bool = False) -> Word | None:
        """The first word (or run of words on one line) containing `needle`."""
        n = needle if case else needle.lower()
        for wd in self.words:
            if n in (wd.text if case else wd.text.lower()):
                return wd
        for li, line in enumerate(self.lines):
            if n in (line if case else line.lower()):
                ws = [wd for wd in self.words if wd.line == li]
                if ws:
                    x0 = min(wd.bbox_xywh[0] for wd in ws)
                    y0 = min(wd.bbox_xywh[1] for wd in ws)
                    x1 = max(wd.bbox_xywh[0] + wd.bbox_xywh[2] for wd in ws)
                    y1 = max(wd.bbox_xywh[1] + wd.bbox_xywh[3] for wd in ws)
                    return Word(text=line, bbox_xywh=(x0, y0, x1 - x0, y1 - y0), line=li)
        return None


class OcrEngine(Protocol):
    def read(self, image_bgr: np.ndarray) -> OcrResult: ...


class OcrUnavailable(RuntimeError):
    pass


_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> list[float]:
    """Every number in the text, in order ("HP 42/100" -> [42.0, 100.0])."""
    return [float(m.group(0).replace(",", ".")) for m in _NUMBER.finditer(text)]


class WindowsOcr:
    """Windows.Media.Ocr via pywinrt. Calls are serialised on one worker thread
    that owns its own event loop, so it is safe from sync code, threads, and
    code already running inside an event loop (like the MCP server)."""

    def __init__(self, *, language: str | None = None, upscale: float = 2.0) -> None:
        try:
            from winrt.windows.globalization import Language
            from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
            from winrt.windows.media.ocr import OcrEngine as _Engine
        except ImportError as exc:
            raise OcrUnavailable(
                "Windows OCR needs the winrt packages: pip install reflex[ocr]"
            ) from exc
        self._SoftwareBitmap = SoftwareBitmap
        self._BGRA8 = BitmapPixelFormat.BGRA8
        engine = _Engine.try_create_from_language(Language(language)) if language else \
            _Engine.try_create_from_user_profile_languages()
        if engine is None:
            raise OcrUnavailable(
                f"no Windows OCR language pack for {language or 'your profile languages'}; "
                "add one under Settings > Time & language > Language"
            )
        self._engine = engine
        self.language = engine.recognizer_language.language_tag
        self.upscale = float(upscale)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="reflex-ocr", daemon=True)
        self._thread.start()

    async def _recognize(self, bgra: np.ndarray) -> Any:
        h, w = bgra.shape[:2]
        sb = self._SoftwareBitmap.create_copy_from_buffer(bgra.tobytes(), self._BGRA8, w, h)
        return await self._engine.recognize_async(sb)

    def read(self, image_bgr: np.ndarray) -> OcrResult:
        if image_bgr.size == 0:
            return OcrResult(text="")
        img = image_bgr
        s = self.upscale if self.upscale and self.upscale != 1.0 else 1.0
        if s != 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        bgra = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
        fut: Future = asyncio.run_coroutine_threadsafe(self._recognize(bgra), self._loop)
        res = fut.result(timeout=10.0)
        lines: list[str] = []
        words: list[Word] = []
        for li, line in enumerate(res.lines):
            lines.append(line.text)
            for wd in line.words:
                b = wd.bounding_rect
                words.append(Word(
                    text=wd.text,
                    bbox_xywh=(int(b.x / s), int(b.y / s), max(1, int(b.width / s)), max(1, int(b.height / s))),
                    line=li,
                ))
        return OcrResult(text="\n".join(lines), lines=tuple(lines), words=tuple(words))

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)


_engine: OcrEngine | None = None
_engine_lock = threading.Lock()


def set_engine(engine: OcrEngine | None) -> None:
    global _engine
    with _engine_lock:
        _engine = engine


def get_engine() -> OcrEngine:
    """The configured engine, creating the Windows one on first use."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = WindowsOcr()
        return _engine


def read_text(frame_bgr: np.ndarray, region: tuple[int, int, int, int] | None = None,
              *, engine: OcrEngine | None = None) -> OcrResult:
    """OCR a frame or a region of it; word boxes come back in frame pixels."""
    eng = engine or get_engine()
    if region is None:
        return eng.read(frame_bgr)
    x, y, w, h = (int(v) for v in region)
    H, W = frame_bgr.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return OcrResult(text="")
    res = eng.read(frame_bgr[y0:y1, x0:x1])
    words = tuple(
        Word(text=wd.text, bbox_xywh=(wd.bbox_xywh[0] + x0, wd.bbox_xywh[1] + y0, *wd.bbox_xywh[2:]), line=wd.line)
        for wd in res.words
    )
    return OcrResult(text=res.text, lines=res.lines, words=words)
