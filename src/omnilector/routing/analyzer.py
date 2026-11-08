# SPDX-License-Identifier: AGPL-3.0-or-later
"""Page analyzer: photo or scan, and how much ink there is, with cheap heuristics (~10 ms)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from .languages import main_script

KINDS = ("scan", "photo")
BLANK_INK = 5e-5  # below this fraction of ink pixels, the page is blank
_INK_DIFF = 20  # a pixel is ink if it is at least this much darker than the estimated background
_SAMPLE_SIDE = 512  # a thumbnail is analyzed: the cost does not depend on the image size


@dataclass
class PageProfile:
    """Situation of an image page. The Router uses it to choose engines and preprocessing."""

    kind: str = "scan"  # scan | photo
    script: str = "latin"  # main script (from config.languages)
    ink: float = 1.0  # fraction of ink pixels (0 = blank page)


def image_signals(image: np.ndarray) -> dict[str, float]:
    """Metrics over a grayscale thumbnail."""
    h, w = image.shape[:2]
    f = _SAMPLE_SIDE / max(h, w)
    small = (
        image
        if f >= 1
        else cv2.resize(image, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    )
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    # Estimated background: the morphological closing erases the dark strokes (the text)
    background = cv2.GaussianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), (0, 0), 8)
    return {
        "illumination": float(background.std()) / 255.0,  # unevenness of the background (shadows, gradients)
        "bright": float((gray > 180).mean()),  # share of light background (paper)
        "ink": float((background.astype(np.int16) - gray > _INK_DIFF).mean()),  # strokes darker than their background
    }


def classify(signals: dict[str, float]) -> str:
    """photo if the lighting is uneven or the background is not light; otherwise scan (scans, screenshots and
    rasterized PDFs use the same engines and variants)."""
    return "photo" if signals["illumination"] > 0.08 or signals["bright"] < 0.35 else "scan"


def analyze(image: np.ndarray, hint: str | None = None, languages: Iterable[str] = ()) -> PageProfile:
    """RGB image -> PageProfile. `hint` (from the loader: camera EXIF) overrides the kind."""
    signals = image_signals(image)
    kind = hint if hint in KINDS else classify(signals)
    return PageProfile(kind=kind, script=main_script(languages), ink=signals["ink"])
