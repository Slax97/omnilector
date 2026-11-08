# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plain text (txt, md, csv/tsv, json, log...) and HTML. No extra dependencies."""

from __future__ import annotations

import csv
import io
from html.parser import HTMLParser
from itertools import islice
from typing import TYPE_CHECKING

from ..routing.languages import legacy_encodings
from ..utils.text import normalize_space
from .base import RawDocument, Source, single_page

if TYPE_CHECKING:
    from ..config import OCRConfig

MAX_TEXT_ROWS = 100_000  # huge texts and CSVs: the rest is reported and not read

_BOMS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)


def decode_text(data: bytes, languages: tuple[str, ...] = ()) -> tuple[str, str]:
    """bytes -> (text, encoding): BOM, UTF-8 and, otherwise, the most likely legacy encoding. Among those
    charset-normalizer finds equally good, the one of the given languages wins or, without a hint, the most
    common one (in short texts cp1250 and cp1252 tie: "ñ" or "ń")."""
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return data.decode(encoding, errors="replace"), encoding
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    preference = legacy_encodings(languages)
    try:
        from charset_normalizer import from_bytes
    except ImportError:
        candidates = preference
    else:
        matches = from_bytes(data)
        best = matches.best()
        if best is None:
            candidates = preference
        else:
            tied = {
                enc
                for m in matches
                if m.chaos <= best.chaos and m.coherence >= best.coherence
                for enc in (m.encoding, *m.could_be_from_charset)
            }
            choice = next((enc for enc in preference if enc in tied), None)
            if choice is None:
                return str(best), best.encoding
            candidates = [choice]
    for encoding in candidates:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("cp1252", errors="replace"), "cp1252"


def text_cells(text: str) -> list[list[str]]:
    """One row per line."""
    return [[line.rstrip()] for line in islice(text.splitlines(), MAX_TEXT_ROWS + 1)]


def _csv_cells(text: str, suffix: str) -> list[list[str]]:
    """Rows of a CSV/TSV (delimiter detected)."""
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel_tab if suffix == ".tsv" else csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    return [[c.strip() for c in row] for row in islice(reader, MAX_TEXT_ROWS + 1)]


def load_text(src: Source, config: OCRConfig) -> RawDocument:
    text, encoding = decode_text(src.read_bytes(), config.language_hints)
    cells = _csv_cells(text, src.suffix) if src.suffix in {".csv", ".tsv"} else text_cells(text)
    warnings = []
    if len(cells) > MAX_TEXT_ROWS:
        warnings.append(f"only the first {MAX_TEXT_ROWS} rows are read")
        cells = cells[:MAX_TEXT_ROWS]
    return single_page("text", cells, {"encoding": encoding}, warnings)


# --------------------------------------------------------------------------- html

_SKIP = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK = {
    "p",
    "div",
    "br",
    "li",
    "dt",
    "dd",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "section",
    "article",
    "header",
    "footer",
    "nav",
    "aside",
    "main",
    "blockquote",
    "pre",
    "ul",
    "ol",
    "dl",
    "form",
    "hr",
    "figure",
    "figcaption",
    "address",
    "caption",
}


class HTMLText(HTMLParser):
    """HTML -> rows of visible text: one per line, one cell per <td>/<th>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.title = ""
        self._buf: list[str] = []
        self._cells: list[str] | None = None  # table row in progress
        self._skip = 0
        self._in_title = False
        self._open: list[list] = []  # open tables: [start row, has other tables inside]
        self.tables: list[tuple[int, int]] = []  # rows [start, end) of the data tables

    def _line(self) -> None:
        if self._cells is not None:  # line break inside a cell: the same cell goes on
            self._buf.append(" ")
            return
        text = normalize_space("".join(self._buf))
        self._buf = []
        if text:
            self.rows.append([text])

    def _end_row(self) -> None:
        if self._cells is None:
            self._line()
            return
        if any(self._cells):
            self.rows.append(self._cells)
        self._cells = None
        self._buf = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in _SKIP:
            self._skip += 1
        elif tag == "table":
            self._end_row()
            if self._open:
                self._open[-1][1] = True
            self._open.append([len(self.rows), False])
        elif tag == "tr":
            self._end_row()
            self._cells = []
        elif tag in ("td", "th"):
            if self._cells is None:
                self._line()
                self._cells = []
            self._buf = []
        elif tag in _BLOCK:
            self._line()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in ("td", "th"):
            if self._cells is not None:
                self._cells.append(normalize_space("".join(self._buf)))
                self._buf = []
        elif tag == "tr":
            self._end_row()
        elif tag == "table":
            self._end_row()
            if self._open:
                start, nested = self._open.pop()
                rows = self.rows[start:]
                # data table: 2x2 or more and without tables inside (those used to lay out websites and emails do not count)
                if not nested and len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    self.tables.append((start, len(self.rows)))
        elif tag in _BLOCK:
            self._line()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self._buf.append(data)

    def close(self) -> None:
        super().close()
        self._end_row()
        self._line()


def html_cells(html: str, tables: list[tuple[int, int]] | None = None) -> tuple[list[list[str]], str]:
    """HTML -> (rows of cells, title); in `tables`, the rows of each data table."""
    parser = HTMLText()
    parser.feed(html)
    parser.close()
    if tables is not None:
        tables.extend(parser.tables)
    return parser.rows, normalize_space(parser.title)


def load_html(src: Source, config: OCRConfig) -> RawDocument:
    text, encoding = decode_text(src.read_bytes(), config.language_hints)
    tables: list[tuple[int, int]] = []
    cells, title = html_cells(text, tables)
    return single_page("html", cells, {"encoding": encoding} | ({"title": title} if title else {}), tables=tables)
