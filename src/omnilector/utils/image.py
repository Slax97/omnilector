# SPDX-License-Identifier: AGPL-3.0-or-later
"""Normalizes inputs: str|Path|PIL|ndarray|bytes -> RGB ndarray + (w,h)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np

from ..exceptions import DocumentLoadError

ImageInput = Union[str, Path, bytes, np.ndarray, "object"]
LIGHT_INK = 160  # if the opaque part of a transparent image is lighter than this (white text), black background


def _orient(pil):
    """Honors the EXIF of phone photos (otherwise they come out rotated and the OCR loses everything)."""
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(pil)
    except Exception:  # noqa: BLE001 - odd EXIF: the image as it is
        return pil


def _rgb(pil):
    """PIL -> RGB. Transparent areas (signatures, logos, cutouts) go on white, or on black if the opaque part
    is light: a plain convert("RGB") leaves the background black and black text disappears."""
    from PIL import Image

    if pil.mode not in ("RGBA", "LA", "PA") and "transparency" not in pil.info:
        return pil.convert("RGB")
    rgba = pil.convert("RGBA")
    opaque = np.asarray(rgba.getchannel("A")) > 127
    light = bool(opaque.any()) and float(np.asarray(rgba.convert("L"))[opaque].mean()) > LIGHT_INK
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255) if light else (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _uint8(arr: np.ndarray) -> np.ndarray:
    """Arrays from other libraries to 0-255: float 0-1 (scikit-image), 16 bits (scanners), masks."""
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    data = np.nan_to_num(arr.astype(np.float64))
    if np.issubdtype(arr.dtype, np.floating) and data.max(initial=0.0) <= 1.0:
        data = data * 255
    elif arr.dtype == np.uint16:
        data = data / 257
    elif data.max(initial=0.0) > 255:
        data = data * (255 / data.max())
    return np.clip(np.rint(data), 0, 255).astype(np.uint8)


def _from_array(arr: np.ndarray):
    from PIL import Image

    if arr.size == 0:
        raise DocumentLoadError("Empty image (ndarray size 0)")
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2 and not (arr.ndim == 3 and arr.shape[2] in (3, 4)):
        raise DocumentLoadError(f"ndarray with unsupported shape: {arr.shape}")
    return Image.fromarray(np.ascontiguousarray(_uint8(arr)))


def load_image(img: object) -> tuple[np.ndarray, tuple[int, int]]:
    """Returns (rgb_array, (w,h)). Does not modify the original outside."""
    from PIL import Image

    if isinstance(img, (str, Path)):
        p = Path(img)  # type: ignore[arg-type]
        if not p.exists():
            raise DocumentLoadError(f"Image not found: {p}")
        try:
            pil = _rgb(_orient(Image.open(p)))
        except Exception as e:
            raise DocumentLoadError(f"Corrupt image or unsupported format: {p}: {e}") from e
    elif isinstance(img, bytes):
        try:
            pil = _rgb(_orient(Image.open(io.BytesIO(img))))
        except Exception as e:
            raise DocumentLoadError(f"Bytes are not a valid image: {e}") from e
    elif isinstance(img, np.ndarray):
        pil = _rgb(_from_array(img))
        return np.array(pil), (pil.width, pil.height)
    elif hasattr(img, "convert") and hasattr(img, "size"):  # PIL.Image
        try:
            pil = _rgb(_orient(img))
        except Exception as e:
            raise DocumentLoadError(f"Invalid PIL image: {e}") from e
    else:
        raise DocumentLoadError(f"Unsupported input type: {type(img)}")

    arr = np.array(pil)
    if arr.size == 0:
        raise DocumentLoadError("Empty image after loading")
    w, h = pil.size
    return arr, (w, h)
