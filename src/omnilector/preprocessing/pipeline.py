# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configurable preprocessing. Keeps the original, produces auxiliary variants (only those requested)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)


def sharpen_rgb(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)


def remove_shadows(gray: np.ndarray) -> np.ndarray:
    """Photos: divides by the estimated background (the text is erased by dilating) -> even lighting."""
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 21)
    diff = 255 - cv2.absdiff(gray, background)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


def deskew_gray(gray: np.ndarray) -> np.ndarray:
    """Corrects a slight skew via minAreaRect on dark pixels."""
    try:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(bw > 0))
        if coords.shape[0] < 100:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 0.5 or abs(angle) > 15:
            return gray
        (h, w) = gray.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:  # noqa: BLE001 - straightening is optional: if it fails, the image as it is
        return gray


def _rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# name -> (RGB image, gray) -> RGB variant. Strategies and the router request the ones they use; gray,
# denoised and threshold stay available for new strategies.
VARIANTS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "gray": lambda img, gray: _rgb(gray),
    "clahe": lambda img, gray: _rgb(clahe_gray(gray)),
    "sharpen": lambda img, gray: sharpen_rgb(img),
    "shadow": lambda img, gray: _rgb(remove_shadows(gray)),
    "denoised": lambda img, gray: _rgb(denoise_gray(gray)),
    "threshold": lambda img, gray: _rgb(adaptive_threshold(deskew_gray(gray))),
}


def iter_variants(image: np.ndarray, names: Iterable[str]) -> Iterator[tuple[str, np.ndarray]]:
    """(name, variant) one at a time, so that only the one being read exists in memory. 'original' is the
    image itself; unknown names are ignored."""
    gray = None
    for name in names:
        if name == "original":
            yield name, image
        elif name in VARIANTS:
            if gray is None:
                gray = to_gray(image)
            yield name, VARIANTS[name](image, gray)


def build_variants(image: np.ndarray, names: Iterable[str]) -> dict[str, np.ndarray]:
    """{name: variant} at once. Always includes 'original'."""
    return {"original": image} | dict(iter_variants(image, names))
