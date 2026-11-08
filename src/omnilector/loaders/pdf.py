# SPDX-License-Identifier: AGPL-3.0-or-later
"""PDF with PyMuPDF (AGPL-3.0, see the README). Decided PER PAGE:

- native: it has useful text -> extracted with its boxes, without OCR.
- ocr: scanned or with unreadable text (broken fonts) -> rasterized and sent to the OCR.
- hybrid: text + images (photos, stamps, signatures, logos) -> the OCR only reads the crop of each
  image, with the native text covered. Those that leave no ink (a scanned PDF with a text layer) do
  not go through the OCR; the same image repeated on several pages (the letterhead logo), only once.

Also: form checkboxes (checked or not), file attachments (a PDF/A-3 invoice with its XML, a PDF
portfolio), each one with its loader, and password-protected PDFs (config.passwords).

Coordinates in pixels of the render (pdf_dpi) so that native text and OCR share a system. The reading
order (rows, text columns) is the same as the OCR's: utils.geometry.reading_rows.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..exceptions import DocumentLoadError
from ..result import TextBlock
from ..routing.analyzer import BLANK_INK, image_signals
from ..utils.deps import require
from ..utils.geometry import reading_rows
from .base import ImageRegion, RawDocument, RawPage, Source, native_block, nested_pages

if TYPE_CHECKING:
    from ..config import OCRConfig

MAX_RENDER_SIDE = 6000  # huge pages (drawings, A0): the DPI is lowered so memory does not run out
MIN_IMAGE_PT = 16  # images with a shorter side (bullets, icons; 72 pt = 1 inch): no OCR
MIN_IMAGE_AREA_PT = 1200  # nor those with a smaller area (about 1.5 cm2)
GARBAGE_RATIO = 0.3  # share of garbage characters above which the native text is not valid
STRAIGHT = 0.02  # vertical component of the text direction below which it is horizontal
# Form checkboxes and radio buttons: checked or not (the page text would only show the mark of the
# checked ones, and in ZapfDingbats: a "3")
CHECKBOX = ("☑", "☐")
RADIO = ("◉", "○")
# ZapfDingbats follows the Unicode Dingbats block ("3" -> U+2713 ✓) except the symbols Unicode already
# had elsewhere (their gaps in the block were filled later with others: ✨, ❌...)
_DINGBATS = {"%": "☎", "*": "☛", "+": "☞", "H": "★", "l": "●", "n": "■", "s": "▲", "t": "▼", "u": "◆", "w": "◗"}


def dingbats(text: str) -> str:
    """Text in the ZapfDingbats font -> its symbols ("3" -> "✓", "l" -> "●")."""
    return "".join(_DINGBATS.get(c) or (chr(0x2700 + ord(c) - 0x20) if "!" <= c <= "~" else c) for c in text)


def _is_dingbats(font: str) -> bool:
    return "dingbat" in font.lower() or font.endswith("ZaDb")


def _is_garbage(ch: str) -> bool:
    return ch == "�" or 0xE000 <= ord(ch) <= 0xF8FF or (not ch.isprintable() and not ch.isspace())


def useful_chars(blocks: list[TextBlock]) -> int:
    """Readable characters; 0 if the text is mostly garbage (fonts without a Unicode map)."""
    chars = [ch for b in blocks for ch in b.text if not ch.isspace()]
    if not chars:
        return 0
    bad = sum(1 for ch in chars if _is_garbage(ch))
    return 0 if bad / len(chars) > GARBAGE_RATIO else len(chars) - bad


def _form_marks(page, pymupdf) -> list[tuple[str, object]]:
    """Checkboxes and radio buttons of a fillable form: (☑ or ☐, ◉ or ○ for radio buttons; their rectangle
    on the page)."""
    if not page.parent.is_pdf:
        return []
    marks = []
    for widget in page.widgets(types=(pymupdf.PDF_WIDGET_TYPE_CHECKBOX, pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON)):
        value = widget.field_value
        checked = value is True or (value not in (None, False, "", "Off") and value == widget.on_state())
        symbols = RADIO if widget.field_type == pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON else CHECKBOX
        marks.append((symbols[0] if checked else symbols[1], widget.rect))
    return marks


def _native_blocks(page, scale: float, pymupdf) -> list[TextBlock]:
    """One line of native text = one TextBlock, with its box in pixels of the render. Rotated text (the
    vertical CSV of the margin, a watermark) carries its corners: it does not join the rows. Symbols in
    ZapfDingbats, as symbols; the checkboxes of a form, with their state."""
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
    rotate = page.rotation_matrix
    marks = _form_marks(page, pymupdf)
    blocks: list[TextBlock] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(
                dingbats(s.get("text", "")) if _is_dingbats(s.get("font", "")) else s.get("text", "") for s in spans
            ).strip()
            if not text:
                continue
            if (
                spans
                and all(_is_dingbats(s.get("font", "")) for s in spans)
                and any(pymupdf.Rect(line["bbox"]).intersects(rect) for _, rect in marks)
            ):
                continue  # the drawn mark of a checkbox: it already comes out with its state
            rect = pymupdf.Rect(line["bbox"]) * rotate
            box = (int(rect.x0 * scale), int(rect.y0 * scale), math.ceil(rect.x1 * scale), math.ceil(rect.y1 * scale))
            item = native_block(text, box)
            dx, dy = line.get("dir", (1.0, 0.0))
            if abs(dx * rotate.b + dy * rotate.d) > STRAIGHT:  # direction already on the page as it is displayed
                quad = pymupdf.recover_line_quad(line) * rotate
                item.quad = tuple(
                    (round(p.x * scale), round(p.y * scale)) for p in (quad.ul, quad.ur, quad.lr, quad.ll)
                )
            blocks.append(item)
    for symbol, rect in marks:
        r = rect * rotate
        blocks.append(
            native_block(
                symbol, (int(r.x0 * scale), int(r.y0 * scale), math.ceil(r.x1 * scale), math.ceil(r.y1 * scale))
            )
        )
    return blocks


def _pixels(pix) -> np.ndarray:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr[:, :, :3].copy()  # own (writable) copy: the pixmap buffer is freed


def _cover(image: np.ndarray, boxes: list[tuple[int, int, int, int]], pad: int) -> int:
    """Covers the boxes (coordinates of `image`) with white: the OCR will only see what is not native text.
    Returns how many of them touched the image."""
    h, w = image.shape[:2]
    touched = 0
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        if x0 < x1 and y0 < y1:
            image[y0:y1, x0:x1] = 255
            touched += 1
    return touched


def _image_rects(page, pymupdf) -> list[tuple[object, int]]:
    """Visible images of the page (rectangle on the page as displayed, xref), without icons or bullets;
    overlapping ones, together (a stamp over a signature is read at once)."""
    rects: list[tuple[object, int]] = []
    for info in page.get_image_info(xrefs=True):
        rect = (pymupdf.Rect(info["bbox"]) * page.rotation_matrix) & page.rect
        if rect.is_empty or min(rect.width, rect.height) < MIN_IMAGE_PT or rect.width * rect.height < MIN_IMAGE_AREA_PT:
            continue
        rects.append((rect, info.get("xref", 0)))
    merged = True
    while merged:
        merged = False
        for i, (a, _) in enumerate(rects):
            j = next((j for j in range(i + 1, len(rects)) if a.intersects(rects[j][0])), None)
            if j is not None:
                rects[i] = (a | rects[j][0], 0)  # several images together: no longer a single xref
                del rects[j]
                merged = True
                break
    return rects


def _regions(page, matrix, blocks: list[TextBlock], pad: int, pymupdf) -> list[ImageRegion]:
    """Crop of each image at the page's scale, with the native text on top of it covered. Without ink
    after covering it (a scan with a text layer) there is nothing to read."""
    out: list[ImageRegion] = []
    for rect, xref in _image_rects(page, pymupdf):
        pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
        image = _pixels(pix)
        native = [(b.box[0] - pix.x, b.box[1] - pix.y, b.box[2] - pix.x, b.box[3] - pix.y) for b in blocks]
        covered = _cover(image, native, pad)
        if image_signals(image)["ink"] < BLANK_INK:
            continue
        key = ("xref", xref, image.shape) if xref and not covered else None
        out.append(ImageRegion(image=image, origin=(pix.x, pix.y), key=key))
    return out


def _tables(page, scale: float, pymupdf) -> list:
    """Tables of the PDF itself (lines and text alignment, no model), in pixels of the render."""
    from ..core.layout import TableCell, TableGrid

    rotate = page.rotation_matrix
    grids = []
    getattr(pymupdf, "no_recommend_layout", lambda: None)()  # without the pymupdf_layout notice in the output
    try:
        found = page.find_tables().tables
    except Exception:  # noqa: BLE001 - odd PDF: the text is already there, only its structure is missing
        return []

    def px(rect) -> tuple[int, int, int, int]:
        r = pymupdf.Rect(rect) * rotate
        return int(r.x0 * scale), int(r.y0 * scale), math.ceil(r.x1 * scale), math.ceil(r.y1 * scale)

    for table in found:
        cells = []
        for r, row in enumerate(table.rows):
            for c, cell in enumerate(row.cells):
                if cell is None:  # part of a merged cell
                    continue
                # the text of each cell from its rectangle: table.extract() in PyMuPDF 2.0 glued words together and
                # moved commas ("32,50 L" -> "3250L ,")
                text = page.get_textbox(pymupdf.Rect(cell))
                cells.append(TableCell(r, c, 1, 1, px(cell), " ".join(text.split())))
        if cells:
            grids.append(TableGrid(None, px(table.bbox), table.row_count, table.col_count, cells))
    return grids


def _embedded(doc) -> list[tuple[str, object]]:
    """Files attached to the document (name, function that returns its content): a PDF/A-3 invoice with its
    XML (Factur-X, ZUGFeRD), the documents of a PDF portfolio."""
    if not doc.is_pdf:
        return []
    return [
        (doc.embfile_info(name).get("filename") or name, lambda name=name: doc.embfile_get(name))
        for name in doc.embfile_names()
    ]


def _annotation_files(page, pymupdf) -> list[tuple[str, object]]:
    """Files attached in annotations of the page (the paper clip of a comment). Taken out right away: the
    annotation stops being valid when its page is released."""
    if not page.parent.is_pdf:
        return []
    found: list[tuple[str, object]] = []
    for annot in page.annots(types=(pymupdf.PDF_ANNOT_FILE_ATTACHMENT,)):
        name = annot.file_info.get("filename") or "attachment"
        try:
            data = annot.get_file()
        except Exception as e:  # noqa: BLE001 - broken attachment: a warning comes when reading the attachments
            data = e
        found.append((name, lambda data=data: _raise_or(data)))
    return found


def _raise_or(value: object) -> object:
    if isinstance(value, Exception):
        raise value
    return value


def open_document(src: Source, config: OCRConfig, pymupdf, filetype: str = "pdf", label: str = "PDF"):
    """The open document; with a password, the first one in config.passwords that works."""
    try:
        if src.path is not None:
            doc = pymupdf.open(src.path, filetype=filetype)
        else:
            doc = pymupdf.open(stream=src.read_bytes(), filetype=filetype)
    except Exception as e:
        raise DocumentLoadError(f"{label} corrupt or unreadable: {src.name}: {e}") from e
    if doc.needs_pass and not any(doc.authenticate(password) for password in config.passwords):
        doc.close()
        hint = "none of the passwords works" if config.passwords else "pass the password in passwords"
        raise DocumentLoadError(f"{label} protected with a password: {src.name} ({hint})")
    return doc


def _page(page, number: int, config: OCRConfig, src: Source, min_chars: int, pymupdf) -> RawPage:
    w_pt, h_pt = page.rect.width, page.rect.height
    dpi = max(36, min(config.pdf_dpi, int(72 * MAX_RENDER_SIDE / max(w_pt, h_pt, 1))))
    scale = dpi / 72
    matrix = pymupdf.Matrix(scale, scale)
    irect = (page.rect * matrix).irect  # same rounding as the pixmap
    size = (irect.width, irect.height)
    blocks = _native_blocks(page, scale, pymupdf)
    if useful_chars(blocks) < min_chars:
        if not src.budget or src.budget.take():
            return RawPage(number, image=_pixels(page.get_pixmap(matrix=matrix, alpha=False)), size=size)
        return RawPage(number, size=size)  # no OCR budget left (max_ocr_pages): empty page
    rows = reading_rows(blocks)
    regions = []
    if config.ocr_embedded_images and src.budget and src.budget.expired():
        if _image_rects(page, pymupdf):  # out of time (time_limit): its images are not even cropped
            src.budget.late += 1
    elif config.ocr_embedded_images:
        regions = _regions(page, matrix, blocks, max(2, round(scale)), pymupdf)
    if regions and src.budget and not src.budget.take():
        regions = []  # no OCR budget left: the native text stays
    tables = _tables(page, scale, pymupdf) if config.layout else []
    return RawPage(number, rows=rows, size=size, regions=regions, tables=tables)


def load_document(src: Source, config: OCRConfig, filetype: str = "pdf", kind: str = "pdf") -> RawDocument:
    """PDF and everything else PyMuPDF opens with pages (EPUB, XPS, MOBI, FB2), with the same decision per
    page: native text, OCR or both. Only PDF and XPS (where scans happen) treat a page with little text as
    scanned; in an ebook, a page with some text is native."""
    pymupdf = require("pymupdf")
    label = kind.upper()
    doc = open_document(src, config, pymupdf, filetype, label)
    if doc.page_count == 0:
        doc.close()
        raise DocumentLoadError(f"empty {label}: {src.name}")
    metadata = {"pages": doc.page_count} | {k: v for k, v in (doc.metadata or {}).items() if v}
    warnings = []
    count = doc.page_count if config.max_pages is None else min(doc.page_count, config.max_pages)
    if count < doc.page_count:
        warnings.append(f"{label} with {doc.page_count} pages: only the first {count} are processed (max_pages)")
    min_chars = config.pdf_min_chars if kind in ("pdf", "xps") else 1

    def pages():
        attachments = _embedded(doc)
        for i in range(count):
            page = doc[i]
            attachments += _annotation_files(page, pymupdf)
            yield _page(page, i + 1, config, src, min_chars, pymupdf)
        if attachments:
            metadata["attachments"] = [name for name, _ in attachments]
        number = count
        for name, content in attachments:  # after the pages, each attachment with its loader
            try:
                data = content()
            except Exception as e:  # noqa: BLE001 - broken attachment: the rest of the PDF is fine
                warnings.append(f"attachment {name}: could not be extracted ({e})")
                continue
            for page in nested_pages(src, name, data, config, "attachment ", warnings):
                number += 1
                page.number = number
                yield page

    return RawDocument(kind, pages(), metadata=metadata, warnings=warnings, on_close=doc.close)


def load(src: Source, config: OCRConfig) -> RawDocument:
    return load_document(src, config)


def load_epub(src: Source, config: OCRConfig) -> RawDocument:
    return load_document(src, config, "epub", "epub")


def load_xps(src: Source, config: OCRConfig) -> RawDocument:
    return load_document(src, config, "xps", "xps")


def load_ebook(src: Source, config: OCRConfig) -> RawDocument:
    """MOBI or FB2 (ebooks)."""
    filetype = "mobi" if src.head(68)[60:68] == b"BOOKMOBI" else "fb2"
    return load_document(src, config, filetype, "ebook")
