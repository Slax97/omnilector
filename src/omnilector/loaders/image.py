# SPDX-License-Identifier: AGPL-3.0-or-later
"""Images: jpg, png, webp, bmp, gif, multi-page tiff, jpeg2000 (+ heic/heif with pillow-heif)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..exceptions import DocumentLoadError
from ..utils.deps import require
from ..utils.image import load_image
from ..utils.resources import input_side
from .base import RawDocument, RawPage, Source
from .detect import is_heif

if TYPE_CHECKING:
    from ..config import OCRConfig


def _camera(pil: object) -> str | None:
    """Camera make and model (EXIF): if present, it is a photo."""
    try:
        exif = pil.getexif()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - corrupt EXIF: not a photo
        return None
    camera = " ".join(str(exif.get(tag)).strip("\x00 \t") for tag in (0x010F, 0x0110) if exif.get(tag))
    return camera or None


def _draft(pil: object, config: OCRConfig) -> None:
    """A huge JPEG is decoded already reduced (libjpeg does it while reading, to 1/2, 1/4 or 1/8), down to
    what the pipeline would reduce it to anyway: a 48 Mpx photo is never whole in memory."""
    side = input_side(config.memory_mb, config.max_image_side)
    w, h = pil.size  # type: ignore[attr-defined]
    if pil.format != "JPEG" or side is None or max(w, h) <= side:  # type: ignore[attr-defined]
        return
    scale = side / max(w, h)
    pil.draft(pil.mode, (math.ceil(w * scale), math.ceil(h * scale)))  # type: ignore[attr-defined]


def load(src: Source, config: OCRConfig) -> RawDocument:
    if src.obj is not None:  # PIL / ndarray already in memory
        arr, size = load_image(src.obj)
        pages = [RawPage(1, image=arr, size=size)] if not src.budget or src.budget.take() else []
        return RawDocument("image", iter(pages))

    from PIL import Image

    if is_heif(src.head(32)):
        require("pillow_heif", "pillow-heif", "heic").register_heif_opener()
    try:
        pil = Image.open(src.stream())
    except Exception as e:
        raise DocumentLoadError(f"Corrupt image or unsupported format: {src.name}: {e}") from e
    frames = getattr(pil, "n_frames", 1) if pil.format == "TIFF" else 1
    camera = _camera(pil)
    metadata = {"format": pil.format, "frames": frames} | ({"camera": camera} if camera else {})
    warnings = []
    count = frames if config.max_pages is None else min(frames, config.max_pages)
    if count < frames:
        warnings.append(f"TIFF with {frames} pages: only the first {count} are processed (max_pages)")

    def pages():
        for i in range(count):
            pil.seek(i)
            if min(pil.size) < src.min_image_side:
                continue  # icon or bullet inside another file
            if src.budget and not src.budget.take():
                continue  # no OCR budget left (max_ocr_pages): OCR.read() adds the warning
            _draft(pil, config)
            arr, size = load_image(pil.copy() if frames > 1 else pil)
            yield RawPage(i + 1, image=arr, size=size, hint="photo" if camera else None)

    return RawDocument("image", pages(), metadata=metadata, warnings=warnings, on_close=pil.close)
