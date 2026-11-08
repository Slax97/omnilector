# SPDX-License-Identifier: AGPL-3.0-or-later
"""Searchable PDF: the document as it is, with the recognized text invisible on top of each line.

It looks the same (no image is touched) and in any viewer it can be searched, selected and copied, as
with ocrmypdf. From a PDF, only the pages and images that went through the OCR (native text is already
searchable); from an image, one page per image (or per page of a TIFF).

The invisible text goes without shaping the letters, character by character: that way it is extracted
exactly also in Devanagari or Thai (shaping them lost letters). PyMuPDF fonts: Helvetica (Latin,
Cyrillic, Greek), Droid Sans Fallback (Chinese, Japanese, Korean) and Noto for the other scripts. In
Arabic it comes out with the presentation forms of each letter (NFKC turns them back into the normal ones).
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import TYPE_CHECKING

from .exceptions import DocumentLoadError
from .loaders.base import Source
from .result import TextBlock
from .utils.deps import require
from .utils.text import right_to_left

if TYPE_CHECKING:
    from .config import OCRConfig

# Noto fonts shipped with PyMuPDF (Font(script=N), UCDN script number): Hebrew, Arabic, Devanagari, Bengali, Tamil,
# Telugu, Kannada, Thai, Armenian, Gurmukhi, Gujarati, Malayalam, Lao, Burmese, Georgian, Ethiopic and Khmer
NOTO_SCRIPTS = (5, 6, 9, 10, 14, 15, 16, 19, 4, 11, 12, 17, 20, 22, 23, 25, 30)
SIZE = 0.85  # font size relative to the line height
BASELINE = 0.2  # the baseline, this share of the height above the bottom edge
STRAIGHT_DEG = 0.5  # lines straighter than this go together, without rotating
DEFAULT_DPI = 300  # images without a credible resolution (photos say 72): page size at 300 dpi

# What was read per page for the searchable PDF: number -> ((width, height) of the image read, its blocks
# in pixels of that image as it arrived, not rotated)
Overlay = dict[int, tuple[tuple[int, int], list[TextBlock]]]


class _Fonts:
    """The font for each text: the first one that has all its letters (or, if none, letter by letter)."""

    def __init__(self, pymupdf) -> None:
        self.pymupdf = pymupdf
        self.loaded: list = []
        self.pending = ["helv", "cjk", *NOTO_SCRIPTS]  # loaded when needed (the CJK one weighs 4 MB)

    def _fonts(self):
        yield from self.loaded
        while self.pending:
            name = self.pending.pop(0)
            try:
                font = self.pymupdf.Font(name) if isinstance(name, str) else self.pymupdf.Font(script=name)
            except Exception:  # noqa: BLE001, S112 - PyMuPDF without that font: the others are fine
                continue
            self.loaded.append(font)
            yield font

    def runs(self, text: str) -> list[tuple[str, object]]:
        letters = [c for c in text if not c.isspace()]
        for font in self._fonts():
            if all(font.has_glyph(ord(c)) for c in letters):
                return [(text, font)]
        out: list[tuple[str, object]] = []
        for c in text:  # none has all of them: each letter with the first one that has it
            font = next((f for f in self._fonts() if f.has_glyph(ord(c))), self.loaded[0])
            if out and out[-1][1] is font:
                out[-1] = (out[-1][0] + c, font)
            else:
                out.append((c, font))
        return out


def _corners(block: TextBlock, sx: float, sy: float, derotate, pymupdf) -> tuple:
    """The 4 corners of the line (top-left, top-right, bottom-right, bottom-left in its reading direction)
    on the unrotated page, which is where it is written."""
    x0, y0, x1, y1 = block.box
    points = block.quad if block.quad and len(block.quad) == 4 else ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return tuple(pymupdf.Point(x * sx, y * sy) * derotate for x, y in points)


def _write_lines(page, blocks: list[TextBlock], size: tuple[int, int], fonts: _Fonts, pymupdf) -> int:
    """Invisible text of each block on the page, as wide as its line. Returns how many lines."""
    width_pt, height_pt = page.rect.width, page.rect.height  # the page as displayed, like the image that was read
    sx, sy = width_pt / max(1, size[0]), height_pt / max(1, size[1])
    straight = pymupdf.TextWriter(page.rect)
    written = 0
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        ul, ur, _, ll = _corners(block, sx, sy, page.derotation_matrix, pymupdf)
        length, height = abs(ur - ul), abs(ll - ul)
        if length < 1 or height < 1:
            continue
        runs = fonts.runs(text)
        natural = sum(font.text_length(part, fontsize=1) for part, font in runs)
        fontsize = min(SIZE * height, length / natural) if natural else SIZE * height
        start = ll + (ul - ll) * BASELINE
        angle = math.degrees(math.atan2(ur.y - ul.y, ur.x - ul.x))
        writer = straight if abs(angle) <= STRAIGHT_DEG else pymupdf.TextWriter(page.rect)
        position, rtl = start, right_to_left(text)
        for part, font in runs:
            _, position = writer.append(position, part, font=font, fontsize=fontsize, right_to_left=rtl)
        if writer is not straight:  # rotated (skewed photo, sideways scan): around its start
            writer.write_text(page, render_mode=3, morph=(start, pymupdf.Matrix(-angle)))
        written += 1
    straight.write_text(page, render_mode=3)
    return written


def _image_pages(src: Source, pymupdf) -> object:
    """PDF with one page per image (or per page of a TIFF), the image filling it. An upright JPEG goes as it
    is (without compressing it again); the others, as PNG or, if they were photos, JPEG."""
    from PIL import Image

    from .loaders.detect import is_heif
    from .utils.image import load_image

    out = pymupdf.open()
    if src.obj is not None:
        frames = [None]
        pil = None
    else:
        if is_heif(src.head(32)):
            require("pillow_heif", "pillow-heif", "heic").register_heif_opener()
        pil = Image.open(src.stream())
        frames = list(range(getattr(pil, "n_frames", 1) if pil.format == "TIFF" else 1))
    for i in frames:
        if pil is None:
            arr, (w, h) = load_image(src.obj)
            dpi, data, photo = DEFAULT_DPI, None, False
        else:
            pil.seek(i)
            dpi = pil.info.get("dpi", (0, 0))[0] or 0
            dpi = dpi if 150 <= dpi <= 1200 else DEFAULT_DPI
            photo = pil.format in ("JPEG", "MPO", "HEIF", "WEBP", "AVIF")
            if pil.format == "JPEG" and pil.getexif().get(0x0112, 1) == 1:  # upright: as it is
                data, (w, h) = src.read_bytes(), pil.size
            else:  # rotated by its EXIF, transparent, another format: as the OCR read it
                (arr, (w, h)), data = load_image(pil.copy() if len(frames) > 1 else pil), None
        if data is None:
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "JPEG" if photo else "PNG", quality=90)
            data = buf.getvalue()
        page = out.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
        page.insert_image(page.rect, stream=data)
    return out


def write(src: Source, kind: str, overlay: Overlay, output: str | Path, config: OCRConfig) -> int:
    """Writes the searchable PDF to `output`. Returns how many lines of invisible text it has."""
    pymupdf = require("pymupdf")
    if kind == "pdf":
        from .loaders.pdf import open_document

        doc = open_document(src, config, pymupdf)
    elif kind == "image":
        doc = _image_pages(src, pymupdf)
    else:
        raise DocumentLoadError(f"Searchable PDF only from PDFs and images: {src.name} is {kind}")
    fonts = _Fonts(pymupdf)
    written = 0
    try:
        for number, (size, blocks) in sorted(overlay.items()):
            if number <= doc.page_count:
                written += _write_lines(doc[number - 1], blocks, size, fonts, pymupdf)
        # only the characters used (the whole CJK font is 4 MB); if it fails, the whole fonts go in
        with contextlib.suppress(Exception):
            doc.subset_fonts()
        doc.save(str(output), garbage=3, deflate=True)
    finally:
        doc.close()
    return written
