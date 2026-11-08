# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenDocument (LibreOffice): odt, ods, odp. No dependencies: the XML inside the ZIP with the standard library.

As in Office: native text (tables by cell; one page per sheet, slide and notes) and the embedded
images as pages for OCR, if config.ocr_embedded_images.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from typing import TYPE_CHECKING
from xml.etree import ElementTree

from ..exceptions import DocumentLoadError
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

_NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
    "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
    "presentation": "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0",
    "xlink": "http://www.w3.org/1999/xlink",
    "dc": "http://purl.org/dc/elements/1.1/",
    "meta": "urn:oasis:names:tc:opendocument:xmlns:meta:1.0",
}


def _q(name: str) -> str:
    prefix, local = name.split(":")
    return f"{{{_NS[prefix]}}}{local}"


_PARAGRAPHS = {_q("text:p"), _q("text:h")}
_TABLE, _ROW, _CELL, _COVERED = (
    _q(t) for t in ("table:table", "table:table-row", "table:table-cell", "table:covered-table-cell")
)
_FRAME, _IMAGE, _TEXT_BOX, _GROUP = (_q(t) for t in ("draw:frame", "draw:image", "draw:text-box", "draw:g"))
# Paragraph containers: lists, sections, text boxes, footnotes
_CONTAINERS = {
    _q(t)
    for t in (
        "text:list",
        "text:list-item",
        "text:list-header",
        "text:section",
        "text:note-body",
        "text:index-body",
        "draw:frame",
        "draw:text-box",
        "draw:custom-shape",
        "draw:g",
        "office:text",
        "office:annotation-end",
    )
}
# Inside a paragraph, what is not text of the line: comments, the note mark (its text goes separately)
# and deleted changes
_NOT_INLINE = {_q(t) for t in ("office:annotation", "text:note-citation", "text:tracked-changes")}
_ROW_GROUPS = {_q(t) for t in ("table:table-header-rows", "table:table-rows", "table:table-row-group")}
MAX_REPEAT = 256  # repeated cells or rows (sheets declare thousands of empty ones at the end)
CM_PER_UNIT = {"cm": 1.0, "mm": 0.1, "in": 2.54, "pt": 2.54 / 72, "pc": 2.54 / 6, "px": 2.54 / 96}
PX_PER_CM = 96 / 2.54


def _inline(el, out: list[str], blocks: list) -> None:
    """Text of a paragraph (ODF spaces, tabs and line breaks included); the text boxes and footnotes it
    contains are stored separately: they are paragraphs of their own."""
    for child in el:
        tag = child.tag
        if tag == _q("text:s"):
            out.append(" " * int(child.get(_q("text:c"), "1")))
        elif tag == _q("text:tab"):
            out.append("\t")
        elif tag == _q("text:line-break"):
            out.append("\n")
        elif tag in (_q("text:note"), _FRAME):
            blocks.append(child)
        elif tag not in _NOT_INLINE:
            if child.text:
                out.append(child.text)
            _inline(child, out, blocks)
        if child.tail:
            out.append(child.tail)


def _items(el) -> Iterator[tuple[str, object]]:
    """("p", text) and ("tbl", table) in order; what is inside a paragraph (text boxes, notes), right after it."""
    for child in el:
        tag = child.tag
        if tag in _PARAGRAPHS:
            out = [child.text or ""]
            inside: list = []
            _inline(child, out, inside)
            yield "p", "".join(out)
            for block in inside:
                yield from _items(block)
        elif tag == _TABLE:
            yield "tbl", child
        elif tag in _CONTAINERS or tag == _q("text:note"):
            yield from _items(child)


def _rows_of(table) -> Iterator[object]:
    for child in table:
        if child.tag == _ROW:
            yield child
        elif child.tag in _ROW_GROUPS:
            yield from _rows_of(child)


def _cell_text(cell) -> tuple[str, list]:
    parts, nested = [], []
    for kind, value in _items(cell):
        if kind == "p":
            parts.append(value)
        else:
            nested.append(value)
    return normalize_space(" ".join(parts)), nested


def _table_rows(table) -> list[list[str]]:
    """Rows of a table (or sheet), one cell per column; repeated ones are expanded (with a limit) and nested
    tables go after their row."""
    rows: list[list[str]] = []
    for tr in _rows_of(table):
        cells: list[str] = []
        nested: list = []
        for tc in tr:
            if tc.tag not in (_CELL, _COVERED):
                continue
            text, inner = _cell_text(tc)
            nested += inner
            cells += [text] * min(int(tc.get(_q("table:number-columns-repeated"), "1")), MAX_REPEAT)
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            rows += [cells] * min(int(tr.get(_q("table:number-rows-repeated"), "1")), MAX_REPEAT)
        for inner in nested:
            rows += _table_rows(inner)
    return rows


def _story_rows(el, spans: list[tuple[int, int]] | None = None) -> list[list[str]]:
    """Rows of text; in `spans`, the rows [start, end) of each table."""
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


def _xml(z: zipfile.ZipFile, name: str, src: Source):
    try:
        return ElementTree.fromstring(z.read(name))
    except KeyError:
        return None
    except ElementTree.ParseError as e:  # encrypted (with a password) or damaged
        raise DocumentLoadError(f"{src.name}: {name} unreadable (password protected?): {e}") from e


def _metadata(z: zipfile.ZipFile, src: Source) -> dict:
    meta = _xml(z, "meta.xml", src)
    if meta is None:
        return {}
    found = {
        "title": meta.findtext(".//dc:title", namespaces=_NS),
        "author": meta.findtext(".//meta:initial-creator", namespaces=_NS)
        or meta.findtext(".//dc:creator", namespaces=_NS),
        "created": meta.findtext(".//meta:creation-date", namespaces=_NS),
    }
    return {k: v for k, v in found.items() if v}


def _images(z: zipfile.ZipFile, root) -> list[tuple[str, bytes]]:
    names = dict.fromkeys(img.get(_q("xlink:href"), "") for img in root.iter(_IMAGE))
    return [(name.rsplit("/", 1)[-1], z.read(name)) for name in names if name in z.namelist()]


def _image_pages(
    images, start: int, prefix: str, src: Source, config: OCRConfig, warnings: list[str]
) -> Iterator[RawPage]:
    if not config.ocr_embedded_images:
        return
    number = start
    for name, blob in images:
        for page in nested_pages(src, name, blob, config, prefix, warnings):
            page.number = number
            number += 1
            yield page


def _open(src: Source) -> tuple[zipfile.ZipFile, object]:
    try:
        z = zipfile.ZipFile(src.stream())
    except (zipfile.BadZipFile, ValueError, OSError) as e:
        raise DocumentLoadError(f"Corrupt OpenDocument: {src.name}: {e}") from e
    content = _xml(z, "content.xml", src)
    if content is None:
        z.close()
        raise DocumentLoadError(f"OpenDocument without content.xml: {src.name}")
    return z, content


# --------------------------------------------------------------------------- odt


def _master_rows(styles, kinds: tuple[str, ...]) -> list[list[str]]:
    """Headers or footers of the master pages (styles.xml), without repetitions or empty rows."""
    seen: set[tuple[str, ...]] = set()
    rows: list[list[str]] = []
    if styles is None:
        return rows
    for kind in kinds:
        for part in styles.iter(_q(kind)):
            for row in _story_rows(part):
                if any(row) and tuple(row) not in seen:
                    seen.add(tuple(row))
                    rows.append(row)
    return rows


def load_odt(src: Source, config: OCRConfig) -> RawDocument:
    z, content = _open(src)
    with z:
        styles = _xml(z, "styles.xml", src)
        body = content.find("office:body/office:text", _NS)
        cells = _master_rows(styles, ("style:header", "style:header-first", "style:header-left"))
        spans: list[tuple[int, int]] = []
        offset = len(cells)
        cells += _story_rows(body, spans) if body is not None else []
        cells += _master_rows(styles, ("style:footer", "style:footer-first", "style:footer-left"))
        images = _images(z, content)
        metadata = _metadata(z, src)
    rows, size = grid_rows(cells)
    tables = grid_tables(rows, [(a + offset, b + offset) for a, b in spans])
    warnings: list[str] = []

    def pages():
        yield RawPage(1, rows=rows, size=size, tables=tables)
        yield from _image_pages(images, 2, "image ", src, config, warnings)

    return RawDocument("odt", pages(), metadata=metadata, warnings=warnings)


# --------------------------------------------------------------------------- ods


def load_ods(src: Source, config: OCRConfig) -> RawDocument:
    z, content = _open(src)
    with z:
        sheets = content.findall("office:body/office:spreadsheet/table:table", _NS)
        names = [sheet.get(_q("table:name"), f"sheet {i}") for i, sheet in enumerate(sheets, 1)]
        tables = [_table_rows(sheet) for sheet in sheets]
        metadata = _metadata(z, src) | {"sheets": names}
    warnings: list[str] = []
    count = len(sheets) if config.max_pages is None else min(len(sheets), config.max_pages)
    if count < len(sheets):
        warnings.append(f"{len(sheets)} sheets: only the first {count} are processed (max_pages)")

    def pages():
        for n, (name, cells) in enumerate(zip(names[:count], tables), 1):
            rows, size = grid_rows(cells)
            yield RawPage(n, rows=rows, size=size, origin=f"sheet {name}", tables=grid_tables(rows, [(0, len(rows))]))

    return RawDocument("ods", pages(), metadata=metadata, warnings=warnings)


# --------------------------------------------------------------------------- odp


def _px(value: str | None) -> int:
    """ODF length ("2.5cm", "1in", "12pt") -> pixels at 96 dpi."""
    found = re.fullmatch(r"\s*(-?[\d.]+)\s*([a-z]*)\s*", value or "")
    if not found:
        return 0
    number, unit = float(found.group(1)), found.group(2) or "cm"
    return round(number * CM_PER_UNIT.get(unit, 1.0) * PX_PER_CM)


def _shapes(el, blocks: list, ox: int = 0, oy: int = 0) -> None:
    """Text of the frames of a slide with their real box; goes into groups."""
    for shape in el:
        if shape.tag == _GROUP:
            _shapes(shape, blocks, ox, oy)
            continue
        x0, y0 = ox + _px(shape.get(_q("svg:x"))), oy + _px(shape.get(_q("svg:y")))
        x1, y1 = x0 + max(1, _px(shape.get(_q("svg:width")))), y0 + max(1, _px(shape.get(_q("svg:height"))))
        rows = _story_rows(shape)
        lines = [" ".join(t for t in row if t) for row in rows if any(row)]
        step = max(1, (y1 - y0) // max(1, len(lines)))
        blocks.extend(native_block(text, (x0, y0 + i * step, x1, y0 + (i + 1) * step)) for i, text in enumerate(lines))


def _slide_size(styles) -> tuple[int, int]:
    props = styles.find(".//style:page-layout-properties", _NS) if styles is not None else None
    if props is None or props.get(_q("fo:page-width")) is None:
        return _px("28cm"), _px("15.75cm")
    return _px(props.get(_q("fo:page-width"))), _px(props.get(_q("fo:page-height")))


def load_odp(src: Source, config: OCRConfig) -> RawDocument:
    z, content = _open(src)
    with z:
        size = _slide_size(_xml(z, "styles.xml", src))
        slides = content.findall("office:body/office:presentation/draw:page", _NS)
        prepared = []
        for slide in slides:
            notes = slide.find("presentation:notes", _NS)
            blocks: list = []
            _shapes([s for s in slide if s.tag != _q("presentation:notes")], blocks)
            note_rows = [row for row in _story_rows(notes) if any(row)] if notes is not None else []
            prepared.append((blocks, note_rows, _images(z, slide)))
        metadata = _metadata(z, src) | {"slides": len(slides)}
    warnings: list[str] = []
    count = len(slides) if config.max_pages is None else min(len(slides), config.max_pages)
    if count < len(slides):
        warnings.append(f"{len(slides)} slides: only the first {count} are processed (max_pages)")

    def pages():
        number = 0
        for s, (blocks, note_rows, images) in enumerate(prepared[:count], 1):
            number += 1
            yield RawPage(number, rows=reading_rows(blocks), size=size, origin=f"slide {s}")
            if note_rows:  # the notes are not on the slide: a separate page
                number += 1
                rows, notes_size = grid_rows(note_rows)
                yield RawPage(number, rows=rows, size=notes_size, origin=f"slide {s}, notes")
            for page in _image_pages(images, number + 1, f"slide {s}, ", src, config, warnings):
                number = page.number
                yield page

    return RawDocument("odp", pages(), metadata=metadata, warnings=warnings)
