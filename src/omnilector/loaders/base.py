# SPDX-License-Identifier: AGPL-3.0-or-later
"""Common loader types: normalized input (Source) and raw pages (RawPage)."""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..exceptions import DocumentLoadError, OmnilectorError
from ..result import TextBlock

if TYPE_CHECKING:
    from ..config import OCRConfig

# Approximate grid (px) for the boxes of text without geometry: docx, xlsx, html, txt, email
ROW_H = 30
COL_W = 200
CHAR_W = 12
MAX_DEPTH = 3  # files inside files (the attachment of an attached email...)
MIN_IMAGE_SIDE = 32  # images inside another file that are smaller (icons, bullets): no OCR


@dataclass
class OCRBudget:
    """Pages that can still go through the OCR (config.max_ocr_pages) and until when (config.time_limit).
    Shared by the whole document, attachments and embedded images included; native pages do not spend it."""

    left: int | None
    skipped: int = 0
    deadline: float | None = None  # time.monotonic() from which no more OCR is done
    late: int = 0  # pages or images without OCR because time ran out
    hurried: int = 0  # read without the extra passes (variants, consensus, handwriting...) because of time

    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def take(self) -> bool:
        if self.expired():
            self.late += 1
            return False
        if self.left is None:
            return True
        if self.left > 0:
            self.left -= 1
            return True
        self.skipped += 1
        return False


@dataclass
class Source:
    """Normalized input: path on disk, bytes in memory or image object (PIL / ndarray)."""

    name: str
    path: Path | None = None
    data: bytes | None = None
    obj: object | None = None
    depth: int = 0  # nesting: attachments of an email, which in turn may bring another email...
    budget: OCRBudget | None = None  # set by loaders.load(); files inside inherit it
    min_image_side: int = 0  # images with a smaller side are not read (icons inside another file)

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix.lower()

    def head(self, n: int = 8192) -> bytes:
        if self.data is not None:
            return self.data[:n]
        if self.path is not None:
            try:
                with self.path.open("rb") as f:
                    return f.read(n)
            except OSError as e:  # no permission, network drive down...: an error of the file, not of omnilector
                raise DocumentLoadError(f"Could not read: {self.path}: {e}") from e
        return b""

    def read_bytes(self) -> bytes:
        if self.data is None and self.path is not None:
            try:
                self.data = self.path.read_bytes()
            except OSError as e:
                raise DocumentLoadError(f"Could not read: {self.path}: {e}") from e
        return self.data or b""

    def stream(self) -> str | io.BytesIO:
        """Path (if there is one) or BytesIO: what PIL, python-docx, openpyxl, zipfile... accept."""
        return str(self.path) if self.path is not None else io.BytesIO(self.data or b"")


def _named(value: object) -> bool:
    """(name, content): what arrives in memory with the name of its file (an upload)."""
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str)


def _is_image(value: object) -> bool:
    return isinstance(value, np.ndarray) or (hasattr(value, "convert") and hasattr(value, "size"))


def source_name(value: object) -> str:
    """The name of the input in Document.source, without reading it (also if it cannot be read)."""
    if isinstance(value, Source):
        return value.name
    if _named(value):
        return value[0]  # type: ignore[index]
    if isinstance(value, (str, Path)):
        return str(Path(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<bytes>"
    if _is_image(value):
        return "<image>"
    if hasattr(value, "read"):
        return str(getattr(value, "name", "<stream>"))
    return f"<{type(value).__name__}>"


def to_source(value: object) -> Source:
    """str | Path | bytes | (name, bytes or stream) | stream | PIL.Image | ndarray -> Source.

    The name of what arrives in memory is its Document.source, and its extension helps recognize the
    format (.doc, .xls or .ppt without olefile, .eml, .html)."""
    if isinstance(value, Source):
        return value
    name = source_name(value)
    if _named(value):
        src = to_source(value[1])  # type: ignore[index]
        src.name = name
        return src
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")
        if not path.is_file():
            raise DocumentLoadError(f"Not a file: {path}")
        return Source(name=name, path=path)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Source(name=name, data=bytes(value))
    if _is_image(value):
        return Source(name=name, obj=value)
    if hasattr(value, "read"):
        data = value.read()  # type: ignore[attr-defined]
        if isinstance(data, str):
            data = data.encode("utf-8")
        return Source(name=name, data=data)
    raise DocumentLoadError(f"Unsupported input type: {type(value)}")


@dataclass
class ImageRegion:
    """Image inside a page with native text (photo, stamp, signature, logo), with that text covered."""

    image: np.ndarray  # RGB, at the page's scale
    origin: tuple[int, int]  # top left corner on the page (px)
    key: object | None = None  # the same image on several pages (letterhead logo) is read once


@dataclass
class RawPage:
    """Page as it comes out of the file: native text by rows, image for OCR, or native text with images
    inside (hybrid: the OCR only reads those images)."""

    number: int
    image: np.ndarray | None = None  # RGB: the whole page goes to the OCR. None = not needed
    rows: list[list[TextBlock]] = field(default_factory=list)  # native text by rows, in reading order
    size: tuple[int, int] = (0, 0)  # (w, h) of the coordinate system of the boxes
    hint: str | None = None  # hint for the analyzer: photo (camera EXIF)
    origin: str | None = None  # "sheet Sales", "attachment invoice.pdf"...
    regions: list[ImageRegion] = field(default_factory=list)  # images to read on a page with text
    tables: list = field(default_factory=list)  # core.layout.TableGrid of the tables of the file itself

    @property
    def blocks(self) -> list[TextBlock]:
        return [b for row in self.rows for b in row]


@dataclass
class RawDocument:
    """Output of a loader. `pages` is an iterator: pages are produced (and rasterized) one at a time.
    Used with `with`: closing it frees the file even if not every page was visited."""

    kind: str
    pages: Iterator[RawPage]
    source: str | None = None  # filled in by loaders.load() if the loader does not set it
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    on_close: Callable[[], object] | None = None
    budget: OCRBudget | None = None  # filled in by loaders.load(): pages left without OCR

    def close(self) -> None:
        close = getattr(self.pages, "close", None)
        if close is not None:
            close()
        if self.on_close is not None:
            self.on_close()

    def __enter__(self) -> RawDocument:  # noqa: PYI034 - typing.Self needs Python 3.11
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def native_block(text: str, box: tuple[int, int, int, int]) -> TextBlock:
    """Text that comes from the file itself: confidence 1.0, engine "native"."""
    return TextBlock(text=text, box=box, confidence=1.0, engine="native")


def grid_rows(cells: list[list[str]]) -> tuple[list[list[TextBlock]], tuple[int, int]]:
    """Text without geometry (docx, xlsx, html, txt, email) -> rows of blocks, one column per cell. The
    file gives the rows; the boxes are an approximate grid so the output has the same shape as the OCR's."""
    rows: list[list[TextBlock]] = []
    width = COL_W
    for r, row in enumerate(cells):
        y0, y1 = r * ROW_H, (r + 1) * ROW_H
        if len(row) == 1:
            spans = [(0, max(COL_W, min(len(row[0]) * CHAR_W, 4000)))]
        else:
            spans = [(c * COL_W, (c + 1) * COL_W) for c in range(len(row))]
        rows.append([native_block(text, (x0, y0, x1, y1)) for text, (x0, x1) in zip(row, spans) if text.strip()])
        if spans:
            width = max(width, spans[-1][1])
    return rows, (width, max(1, len(cells)) * ROW_H)


def grid_tables(rows: list[list[TextBlock]], spans: list[tuple[int, int]]) -> list:
    """Tables of a file without geometry (docx, xlsx, html...): the rows [start, end) of each one, with the
    column of each cell according to the grid of grid_rows()."""
    from ..core.layout import TableCell, TableGrid

    tables = []
    for start, end in spans:
        cells = [
            TableCell(r - start, b.box[0] // COL_W, 1, 1, b.box, b.text) for r in range(start, end) for b in rows[r]
        ]
        if cells:
            box = (
                min(c.box[0] for c in cells),
                min(c.box[1] for c in cells),
                max(c.box[2] for c in cells),
                max(c.box[3] for c in cells),
            )
            tables.append(TableGrid(None, box, end - start, max(c.col for c in cells) + 1, cells))
    return tables


def single_page(
    kind: str,
    cells: list[list[str]],
    metadata: dict | None = None,
    warnings: list[str] | None = None,
    tables: list[tuple[int, int]] | None = None,
) -> RawDocument:
    rows, size = grid_rows(cells)
    page = RawPage(1, rows=rows, size=size, tables=grid_tables(rows, tables or []))
    return RawDocument(kind, iter([page]), metadata=metadata or {}, warnings=warnings or [])


def nested_pages(
    parent: Source, name: str, data: bytes, config: OCRConfig, prefix: str, warnings: list[str]
) -> Iterator[RawPage]:
    """Pages of a file inside another one (attachment of an email, image of a docx), each one with its
    provenance in `origin`. If it cannot be read, a warning and the rest goes on. The caller numbers them.
    Without icons or bullets, and spending the same OCR budget as the document."""
    from . import load  # deferred: loaders/__init__ imports this module

    if parent.depth >= MAX_DEPTH:
        warnings.append(f"{prefix}{name}: nested too deep, skipped")
        return
    inner = Source(name=name, data=data, depth=parent.depth + 1, budget=parent.budget, min_image_side=MIN_IMAGE_SIDE)
    try:
        with load(inner, config) as sub:
            for page in sub.pages:
                page.origin = f"{prefix}{name}" + (f", {page.origin}" if page.origin else "")
                yield page
            warnings.extend(f"{prefix}{name}: {w}" for w in sub.warnings)
    except OmnilectorError as e:  # unsupported format, missing extra...: the rest of the file is fine
        warnings.append(f"{prefix}{name}: {e}")
