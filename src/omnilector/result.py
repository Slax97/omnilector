# SPDX-License-Identifier: AGPL-3.0-or-later
"""Common output: Document -> Page -> Line (+ Page.blocks and Page.tables).

The same format whatever the engine (RapidOCR, Tesseract, EasyOCR, TrOCR), the native text of the file
(PDF, docx, xlsx...) or a combination of several. TextBlock and RecognitionCandidate are the engines'
internal format (pixels of the image read); make_page() turns them into the output.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .utils.text import nfc, text_key

if TYPE_CHECKING:
    from .core.layout import TableGrid, Zone

Rel = tuple[float, float, float, float]


def _plain(value: object) -> object:
    """Basic JSON types: tuples -> lists, dates -> ISO, numpy numbers -> Python ones, and anything else,
    as text."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    return str(value)


@dataclass
class Line:
    """A line of text (or a table cell)."""

    text: str
    # x0, y0, x1, y1 relative to the page (0-1). In formats without geometry (docx, xlsx, html, txt,
    # email) it is an approximate row/column grid.
    box: tuple[float, float, float, float]
    confidence: float  # 0-1; 1.0 = native text of the file
    engines: list[str]  # who read this text: ["rapidocr"], ["rapidocr", "easyocr"], ["native"]
    alternatives: list[str] = field(default_factory=list)  # other different readings of the same zone
    handwritten: bool = False
    row: int = 0  # visual row: lines of the same row are at the same height ("TOTAL" | "48,22 EUR")
    block: int | None = None  # index in Page.blocks (layout zone); None without layout analysis


@dataclass
class Block:
    """Zone of the page, in reading order: title | heading | text | list | table | caption | figure |
    stamp | header | footer | page_number | footnote | formula."""

    kind: str
    box: Rel  # relative to the page, 0-1
    lines: list[int] = field(default_factory=list)  # indices in Page.lines

    def text(self, page: Page) -> str:
        return "\n".join(page.lines[i].text for i in self.lines)


@dataclass
class Cell:
    """Table cell; a merged cell spans rowspan rows and colspan columns."""

    row: int
    col: int
    text: str
    box: Rel
    rowspan: int = 1
    colspan: int = 1


@dataclass
class Table:
    """Table with its structure: from a PDF with text, from a scan (table model) or from a photo."""

    box: Rel
    rows: int
    cols: int
    cells: list[Cell]
    block: int | None = None  # index in Page.blocks, if the page has layout analysis

    def grid(self) -> list[list[str]]:
        """Text by row and column (a merged cell, in its first position)."""
        out = [[""] * self.cols for _ in range(self.rows)]
        for c in self.cells:
            if c.row < self.rows and c.col < self.cols:
                out[c.row][c.col] = f"{out[c.row][c.col]} {c.text}".strip()
        return out

    def to_markdown(self) -> str:
        rows = [[t.replace("|", "\\|").replace("\n", " ") for t in r] for r in self.grid()]
        if not rows:
            return ""
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * self.cols]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(lines)

    def to_html(self) -> str:
        body = []
        for r in range(self.rows):
            cells = sorted((c for c in self.cells if c.row == r), key=lambda c: c.col)
            tds = "".join(
                "<td"
                + (f' rowspan="{c.rowspan}"' if c.rowspan > 1 else "")
                + (f' colspan="{c.colspan}"' if c.colspan > 1 else "")
                + f">{html.escape(c.text)}</td>"
                for c in cells
            )
            body.append(f"<tr>{tds}</tr>")
        return "<table>" + "".join(body) + "</table>"


@dataclass
class Page:
    """A page (or image, spreadsheet, slide, attachment...)."""

    number: int = 1
    lines: list[Line] = field(default_factory=list)  # in reading order
    method: str = "ocr"  # native (text of the file) | ocr | hybrid (both)
    origin: str | None = None  # "sheet Sales", "attachment invoice.pdf", "image image1.png"...
    # Degrees (counterclockwise) the image was rotated to read it upright: skewed photo, sideways or
    # upside-down scan. Boxes are those of the rotated page (whole, not cropped). 0 = as it was.
    rotation: float = 0.0
    script: str | None = None  # script the OCR read with (latin, cyrillic, arabic...); None = native text
    blocks: list[Block] = field(default_factory=list)  # zones in reading order (with layout analysis)
    tables: list[Table] = field(default_factory=list)  # tables with rows, columns and cells

    @property
    def text(self) -> str:
        """One line per row; the pieces of one row, separated by a space. That way the text is the same
        whether an engine splits the row into pieces or reads it whole."""
        rows: list[list[str]] = []
        for i, line in enumerate(self.lines):
            if i and line.row == self.lines[i - 1].row:
                rows[-1].append(line.text)
            else:
                rows.append([line.text])
        return "\n".join(" ".join(r) for r in rows)

    @property
    def engines(self) -> list[str]:
        return list(dict.fromkeys(e for line in self.lines for e in line.engines))

    def _markdown_rows(self) -> str:
        """Without layout analysis: one line per row; each table, in the place of its lines."""

        def table_of(line: Line) -> int | None:
            cx, cy = (line.box[0] + line.box[2]) / 2, (line.box[1] + line.box[3]) / 2
            return next(
                (i for i, t in enumerate(self.tables) if t.box[0] <= cx <= t.box[2] and t.box[1] <= cy <= t.box[3]),
                None,
            )

        parts: list[str] = []
        done: set[int] = set()
        for i, line in enumerate(self.lines):
            table = table_of(line)
            if table is not None:
                if table not in done:
                    done.add(table)
                    parts.append("\n" + self.tables[table].to_markdown() + "\n")
                continue
            same_row = i and line.row == self.lines[i - 1].row and table_of(self.lines[i - 1]) is None
            if same_row:
                parts[-1] += " " + line.text
            else:
                parts.append(line.text)
        parts += ["\n" + t.to_markdown() for i, t in enumerate(self.tables) if i not in done]
        return "\n".join(parts).strip()

    def to_markdown(self) -> str:
        """Titles as headings, tables as tables and each zone as a paragraph; without layout analysis, one line
        per row with each table in its place."""
        if not self.blocks:
            return self._markdown_rows()
        tables = {t.block: t for t in self.tables if t.block is not None}
        parts = []
        for i, block in enumerate(self.blocks):
            text = " ".join(self.lines[j].text for j in block.lines)
            if block.kind == "table" and i in tables:
                parts.append(tables[i].to_markdown())
            elif block.kind == "page_number" or not text:
                continue
            elif block.kind == "title":
                parts.append(f"### {text}")
            elif block.kind == "heading":
                parts.append(f"#### {text}")
            else:
                parts.append(text)
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        """The page in basic types (for JSON): its fields, with the text before the lines."""
        data = _plain(asdict(self))
        nested = ("lines", "blocks", "tables")
        return {k: v for k, v in data.items() if k not in nested} | {"text": self.text} | {k: data[k] for k in nested}


@dataclass
class Document:
    """Result of reading any input."""

    pages: list[Page] = field(default_factory=list)
    source: str | None = None
    kind: str = "unknown"  # image | pdf | docx | xlsx | pptx | html | text | email
    metadata: dict = field(default_factory=dict)  # title, author, sheets, email subject...
    warnings: list[str] = field(default_factory=list)  # pages that failed, limits applied...
    # Why it could not be read (set by OCR.read_many; read() raises the exception): missing, corrupt,
    # unsupported format, missing extra, or all its pages failed. None = it was read
    error: Exception | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.lines)

    @property
    def lines(self) -> list[Line]:
        return [line for p in self.pages for line in p.lines]

    @property
    def engines(self) -> list[str]:
        return list(dict.fromkeys(e for p in self.pages for e in p.engines))

    @property
    def tables(self) -> list[Table]:
        return [t for p in self.pages for t in p.tables]

    def to_markdown(self) -> str:
        parts: list[str] = []
        if self.source and not self.source.startswith("<"):
            parts.append(f"# {Path(self.source).name}")
        for p in self.pages:
            if len(self.pages) > 1:
                parts.append(f"## Page {p.number}" + (f" ({p.origin})" if p.origin else ""))
            text = p.to_markdown()
            if text:
                parts.append(text)
        return "\n\n".join(parts) + "\n"

    def to_dict(self) -> dict:
        """The document in basic types (for JSON): its fields, the text and the error as
        {"type": "DocumentLoadError", "message": "..."}."""
        error = None if self.error is None else {"type": type(self.error).__name__, "message": str(self.error)}
        return {
            "source": self.source,
            "kind": self.kind,
            "text": self.text,
            "error": error,
            "warnings": list(self.warnings),
            "metadata": _plain(self.metadata),
            "pages": [p.to_dict() for p in self.pages],
        }

    def to_json(self, indent: int | None = None) -> str:
        """to_dict() as JSON, with the text as it is (accents, Cyrillic... not escaped)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# --------------------------------------------------------------------------- internal format


@dataclass
class RecognitionCandidate:
    """A reading of a zone by one engine."""

    text: str
    confidence: float
    engine: str


@dataclass
class TextBlock:
    """What an engine returns (engines/): box in pixels of the image read and every reading of that zone
    in candidates (several engines, variants, scales). If none are given, the reading itself is the
    first one: that way it is kept even if another engine replaces it later."""

    text: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    confidence: float = 0.0
    engine: str = "unknown"  # "native" if it is text of the file
    is_handwritten: bool = False
    candidates: list[RecognitionCandidate] = field(default_factory=list)
    # The 4 corners of the line if the engine gives them (RapidOCR, EasyOCR) or the text is rotated: with
    # them we know whether the page is skewed, and which text is not horizontal (margins, stamps...)
    quad: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if not self.candidates:
            self.candidates = [RecognitionCandidate(self.text, self.confidence, self.engine)]


def _rel(value: float, total: int) -> float:
    return round(min(1.0, max(0.0, value / total)), 4)


def _line(block: TextBlock, w: int, h: int, row: int, zone: int | None = None) -> Line:
    text = nfc(block.text.strip())
    if len(block.candidates) == 1:  # a single reading (native text, one engine): nothing to compare
        engines, alternatives = [block.engine], []
    else:
        final = text_key(text)
        engines = list(
            dict.fromkeys([block.engine, *(c.engine for c in block.candidates if text_key(c.text) == final)])
        )
        alternatives = list(
            dict.fromkeys(nfc(c.text.strip()) for c in block.candidates if c.text.strip() and text_key(c.text) != final)
        )
    x0, y0, x1, y1 = block.box
    return Line(
        text=text,
        box=(_rel(x0, w), _rel(y0, h), _rel(x1, w), _rel(y1, h)),
        confidence=round(float(block.confidence), 4),
        engines=engines,
        alternatives=alternatives,
        handwritten=block.is_handwritten,
        row=row,
        block=zone,
    )


def _box(box: tuple[float, float, float, float], w: int, h: int) -> Rel:
    x0, y0, x1, y1 = box
    return (_rel(x0, w), _rel(y0, h), _rel(x1, w), _rel(y1, h))


def make_page(
    rows: list[list[TextBlock]],
    size: tuple[int, int],
    number: int = 1,
    method: str = "ocr",
    origin: str | None = None,
    rotation: float = 0.0,
    script: str | None = None,
    zones: list[Zone] | None = None,
    tables: list[TableGrid] | None = None,
) -> Page:
    """Rows of internal blocks in reading order (pixels of an image of `size`) -> Page.

    engines = engines whose reading matches the final text; alternatives = different readings.
    With `zones` (layout), the rows come from the zones, in their order, and each line knows its zone.
    """
    w, h = max(1, size[0]), max(1, size[1])
    lines: list[Line] = []
    blocks: list[Block] = []
    groups = [(None, rows)] if zones is None else [(i, z.rows) for i, z in enumerate(zones)]
    for zone, zone_rows in groups:
        first = len(lines)
        for row in zone_rows:
            kept = [b for b in row if b.text.strip()]
            index = lines[-1].row + 1 if lines else 0
            lines.extend(_line(b, w, h, index, zone) for b in kept)
        if zones is not None:
            blocks.append(Block(zones[zone].kind, _box(zones[zone].box, w, h), list(range(first, len(lines)))))
    found = [
        Table(
            _box(t.box, w, h),
            t.rows,
            t.cols,
            [Cell(c.row, c.col, c.text, _box(c.box, w, h), c.rowspan, c.colspan) for c in t.cells],
            t.zone,
        )
        for t in tables or []
    ]
    return Page(
        number=number,
        lines=lines,
        method=method,
        origin=origin,
        rotation=round(rotation, 2),
        script=script,
        blocks=blocks,
        tables=found,
    )
