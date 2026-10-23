"""Generate the synthetic dialog fixture: four pages of a text box, then the box closes.

Run from the repo root to rebuild `frames/`:

    python tests/fixtures/dialog/make_frames.py

The frames are drawn, not captured, so the fixture carries no game art. They
are tuned to the same measurements the macros in `macros.yaml` check:

    dialog pages: white ratio ~0.85 in the text panel, teal border present
    closed box:   no panel, no border
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

W, H = 480, 330
OUT = Path(__file__).resolve().parent / "frames"

PAGES = (
    ("Welcome to the test world.", "This box has four pages."),
    ("Each press of A should turn", "the page or close the box."),
    ("If a press changes nothing,", "the reflex is told so."),
    ("Three misses in a row and", "it trips instead of spamming."),
)


def scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (60, 120, 70)  # green field, BGR
    for _ in range(40):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.rectangle(img, (x, y), (x + 12, y + 12), (40, 90, 50), -1)
    return img


def page(lines: tuple[str, str]) -> np.ndarray:
    img = scene(0)
    cv2.rectangle(img, (20, 244), (460, 322), (250, 250, 250), -1)  # white panel
    cv2.rectangle(img, (18, 243), (462, 324), (160, 160, 40), 2)  # teal border (BGR)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (34, 272 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, lines in enumerate(PAGES):
        cv2.imwrite(str(OUT / f"{i:02d}.png"), page(lines))
    cv2.imwrite(str(OUT / f"{len(PAGES):02d}.png"), scene(0))


if __name__ == "__main__":
    main()
