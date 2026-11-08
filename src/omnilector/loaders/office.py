# SPDX-License-Identifier: AGPL-3.0-or-later
"""Office: docx (python-docx), xlsx (openpyxl), pptx (python-pptx). Extra [office], all MIT.

Native text, no OCR. Embedded images (the photo of a ticket pasted in a Word file...) are read with
the image loader, as separate pages, if config.ocr_embedded_images.

The docx is read from its XML, in order: python-docx's paragraph.text skips content controls (the
values of a form), text boxes, inserted changes, fields and nested tables.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import TYPE_CHECKING

from ..exceptions import DocumentLoadError
from ..utils.deps import require
from ..utils.geometry import reading_rows
from ..utils.text import normalize_space
from .base import (
    RawDocument,
    RawPage,
    Source,
    grid_rows,
    grid_tables,
    native_block,
    nested_pages,
)

if TYPE_CHECKING:
    from ..config import OCRConfig

EMU_PER_PX = 9525  # 914400 EMU per inch / 96 px per inch
MAX_SHEET_ROWS = 100_000


def _image_pages(
    images: list[tuple[str, bytes]], start: int, prefix: str, src: Source, config: OCRConfig, warnings: list[str]
) -> Iterator[RawPage]:
    """Embedded images -> pages for OCR numbered from `start` (no icons or bullets)."""
    if not config.ocr_embedded_images:
        return
    number = start
    for name, blob in images:
        for page in nested_pages(src, name, blob, config, prefix, warnings):
            page.number = number
            number += 1
            yield page


# --------------------------------------------------------------------------- docx

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_P, _TBL, _TR, _TC, _T, _TXBX = (_W + t for t in ("p", "tbl", "tr", "tc", "t", "txbxContent"))
# Wrappers that do not change the text: content controls (forms) and custom XML
_TRANSPARENT = frozenset(_W + t for t in ("sdt", "sdtContent", "customXml"))
# What is not visible: the alternative copy (VML) of a drawing, what was deleted and what was moved elsewhere
_HIDDEN = frozenset(
    {"{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback", _W + "del", _W + "moveFrom"}
)
_CHARS = {_W + "tab": "\t", _W + "ptab": "\t", _W + "br": "\n", _W + "cr": "\n", _W + "noBreakHyphen": "-"}


def _runs(el, text: list[str], boxes: list) -> None:
    """Visible text of a paragraph, including that of content controls, fields, hyperlinks and inserted
    changes. Text boxes are stored separately: they are paragraphs of their own."""
    for child in el:
        tag = child.tag
        if tag == _T:
            text.append(child.text or "")
        elif tag in _CHARS:
            text.append(_CHARS[tag])
        elif tag == _TXBX:
            boxes.append(child)
        elif tag not in _HIDDEN:
            _runs(child, text, boxes)


def _items(el) -> Iterator[tuple[str, object]]:
    """("p", text) and ("tbl", table) of a container (body, cell, header, note...) in order; the paragraphs
    of a text box, right after the paragraph that contains it."""
    for child in el:
        tag = child.tag
        if tag == _P:
            text: list[str] = []
            boxes: list = []
            _runs(child, text, boxes)
            yield "p", "".join(text)
            for box in boxes:
                yield from _items(box)
        elif tag == _TBL:
            yield "tbl", child
        elif tag in _TRANSPARENT:
            yield from _items(child)


def _children(el, tag: str) -> Iterator[object]:
    """Children with that tag, including those wrapped in a content control."""
    for child in el:
        if child.tag == tag:
            yield child
        elif child.tag in _TRANSPARENT:
            yield from _children(child, tag)


def _table_rows(table) -> list[list[str]]:
    """One row per table row, one cell per column; nested tables, right after."""
    rows: list[list[str]] = []
    for tr in _children(table, _TR):
        texts, nested = [], []
        for tc in _children(tr, _TC):
            parts = []
            for kind, value in _items(tc):
                if kind == "p":
                    parts.append(value)
                else:
                    nested.append(value)
            texts.append(normalize_space(" ".join(parts)))
        if any(texts):
            rows.append(texts)
        for inner in nested:
            rows += _table_rows(inner)
    return rows


def _story_rows(el, spans: list[tuple[int, int]] | None = None) -> list[list[str]]:
    """Rows of text of a container: one per paragraph line and one per table row. In `spans`, the rows
    [start, end) of each table."""
    rows: list[list[str]] = []
    for kind, value in _items(el):
        if kind == "p":
            rows += [[line.strip()] for line in str(value).split("\n")]
        else:
            table = _table_rows(value)
            if spans is not None and table:
                spans.append((len(rows), len(rows) + len(table)))
            rows += table
    return rows


def _part_roots(doc, kind: str) -> Iterator[object]:
    """XML of the document parts of that type (header, footer, footnotes, endnotes)."""
    from docx.oxml import parse_xml

    for rel in doc.part.rels.values():
        if not rel.is_external and rel.reltype.endswith("/" + kind):
            part = rel.target_part
            yield part.element if hasattr(part, "element") else parse_xml(part.blob)


def _repeated_rows(doc, kind: str) -> list[list[str]]:
    """Headers or footers of every section, without repeating rows or leaving empty rows."""
    seen: set[tuple[str, ...]] = set()
    rows: list[list[str]] = []
    for root in _part_roots(doc, kind):
        for row in _story_rows(root):
            if any(row) and tuple(row) not in seen:
                seen.add(tuple(row))
                rows.append(row)
    return rows


def _note_rows(doc) -> list[list[str]]:
    """Footnotes and endnotes (without the separators, which have no text)."""
    rows: list[list[str]] = []
    for kind in ("footnotes", "endnotes"):
        for root in _part_roots(doc, kind):
            for note in root:
                if note.get(_W + "type") is None:
                    rows += [row for row in _story_rows(note) if any(row)]
    return rows


def load_docx(src: Source, config: OCRConfig) -> RawDocument:
    docx = require("docx", "python-docx", "office")
    try:
        doc = docx.Document(src.stream())
    except Exception as e:
        raise DocumentLoadError(f"docx corrupt or unreadable: {src.name}: {e}") from e
    spans: list[tuple[int, int]] = []
    body = _story_rows(doc.element.body, spans)
    header = _repeated_rows(doc, "header")
    cells = header + body + _note_rows(doc) + _repeated_rows(doc, "footer")
    images = [
        (rel.target_part.partname.rsplit("/", 1)[-1], rel.target_part.blob)
        for rel in doc.part.rels.values()
        if rel.reltype.endswith("/image") and not rel.is_external
    ]
    props = doc.core_properties
    metadata = {k: v for k, v in {"title": props.title, "author": props.author, "created": props.created}.items() if v}
    warnings: list[str] = []
    rows, size = grid_rows(cells)
    tables = grid_tables(rows, [(a + len(header), b + len(header)) for a, b in spans])

    def pages():
        yield RawPage(1, rows=rows, size=size, tables=tables)
        yield from _image_pages(images, 2, "image ", src, config, warnings)

    return RawDocument("docx", pages(), metadata=metadata, warnings=warnings)


# --------------------------------------------------------------------------- xlsx


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time() else value.isoformat(sep=" ")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_xlsx(src: Source, config: OCRConfig) -> RawDocument:
    openpyxl = require("openpyxl", extra="office")
    try:
        wb = openpyxl.load_workbook(src.stream(), read_only=True, data_only=True)
    except Exception as e:
        raise DocumentLoadError(f"xlsx corrupt or unreadable: {src.name}: {e}") from e
    sheets = wb.worksheets
    warnings: list[str] = []
    count = len(sheets) if config.max_pages is None else min(len(sheets), config.max_pages)
    if count < len(sheets):
        warnings.append(f"{len(sheets)} sheets: only the first {count} are processed (max_pages)")

    def pages():
        for n, ws in enumerate(sheets[:count], 1):
            cells: list[list[str]] = []
            for values in ws.iter_rows(values_only=True):
                row = [_cell_text(v) for v in values]
                while row and not row[-1]:
                    row.pop()
                if row:
                    cells.append(row)
                if len(cells) >= MAX_SHEET_ROWS:
                    warnings.append(f"sheet {ws.title}: only the first {MAX_SHEET_ROWS} rows with data")
                    break
            rows, size = grid_rows(cells)
            yield RawPage(
                n, rows=rows, size=size, origin=f"sheet {ws.title}", tables=grid_tables(rows, [(0, len(rows))])
            )

    metadata = {"sheets": [ws.title for ws in sheets]}
    return RawDocument("xlsx", pages(), metadata=metadata, warnings=warnings, on_close=wb.close)


# --------------------------------------------------------------------------- pptx


def _box(shape) -> tuple[int, int, int, int]:
    x, y = int((shape.left or 0) / EMU_PER_PX), int((shape.top or 0) / EMU_PER_PX)
    w, h = int((shape.width or 0) / EMU_PER_PX), int((shape.height or 0) / EMU_PER_PX)
    return x, y, x + max(w, 1), y + max(h, 1)


def _slide_content(
    shapes, group_type: object, picture_type: object, blocks: list, pictures: list, tables: list
) -> None:
    """Texts (with their real box in px), tables by cell and images; goes into groups."""
    from ..core.layout import TableCell, TableGrid

    for shape in shapes:
        if shape.shape_type == group_type:
            _slide_content(shape.shapes, group_type, picture_type, blocks, pictures, tables)
            continue
        x0, y0, x1, y1 = _box(shape)
        if getattr(shape, "has_table", False):
            table, ys, cells = shape.table, y0, []
            for r, row in enumerate(table.rows):
                rh, xs = int(row.height / EMU_PER_PX), x0
                for c, cell in enumerate(row.cells):
                    cw = int(table.columns[c].width / EMU_PER_PX)
                    text = normalize_space(cell.text)
                    if text:
                        blocks.append(native_block(text, (xs, ys, xs + cw, ys + rh)))
                    if not cell.is_spanned:
                        span = (cell.span_height, cell.span_width) if cell.is_merge_origin else (1, 1)
                        cells.append(TableCell(r, c, span[0], span[1], (xs, ys, xs + cw, ys + rh), text))
                    xs += cw
                ys += rh
            if cells:
                tables.append(
                    TableGrid(
                        None, (x0, y0, max(c.box[2] for c in cells), ys), len(table.rows), len(table.columns), cells
                    )
                )
        elif getattr(shape, "has_text_frame", False):
            paras = [t for t in (normalize_space(p.text) for p in shape.text_frame.paragraphs) if t]
            step = max(1, (y1 - y0) // max(1, len(paras)))
            for i, text in enumerate(paras):
                blocks.append(native_block(text, (x0, y0 + i * step, x1, y0 + (i + 1) * step)))
        if shape.shape_type == picture_type:
            try:
                pictures.append((shape.image.filename or f"image{len(pictures) + 1}", shape.image.blob))
            except Exception:  # noqa: BLE001, S112 - linked (external) or damaged image: the text is still fine
                continue


def _notes(slide) -> list[str]:
    """Speaker notes, one line per paragraph."""
    if not slide.has_notes_slide or slide.notes_slide.notes_text_frame is None:
        return []
    return [t for t in (normalize_space(x) for x in slide.notes_slide.notes_text_frame.text.splitlines()) if t]


def load_pptx(src: Source, config: OCRConfig) -> RawDocument:
    pptx = require("pptx", "python-pptx", "office")
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    try:
        prs = pptx.Presentation(src.stream())
    except Exception as e:
        raise DocumentLoadError(f"pptx corrupt or unreadable: {src.name}: {e}") from e
    size = (int((prs.slide_width or 9144000) / EMU_PER_PX), int((prs.slide_height or 6858000) / EMU_PER_PX))
    slides = list(prs.slides)
    warnings: list[str] = []
    count = len(slides) if config.max_pages is None else min(len(slides), config.max_pages)
    if count < len(slides):
        warnings.append(f"{len(slides)} slides: only the first {count} are processed (max_pages)")

    def pages():
        number = 0
        for s, slide in enumerate(slides[:count], 1):
            blocks: list = []
            pictures: list[tuple[str, bytes]] = []
            tables: list = []
            _slide_content(slide.shapes, MSO_SHAPE_TYPE.GROUP, MSO_SHAPE_TYPE.PICTURE, blocks, pictures, tables)
            number += 1
            yield RawPage(number, rows=reading_rows(blocks), size=size, origin=f"slide {s}", tables=tables)
            notes = _notes(slide)
            if notes:  # the notes are not on the slide: a separate page
                number += 1
                rows, notes_size = grid_rows([[t] for t in notes])
                yield RawPage(number, rows=rows, size=notes_size, origin=f"slide {s}, notes")
            for page in _image_pages(pictures, number + 1, f"slide {s}, ", src, config, warnings):
                number = page.number
                yield page

    return RawDocument("pptx", pages(), metadata={"slides": len(slides)}, warnings=warnings)
