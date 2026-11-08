# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layout: zones in reading order, tables with cells (fake models) and tables of native PDFs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest

from omnilector import OCR
from omnilector.core.layout import LayoutAnalyzer
from omnilector.result import Cell, Table

from .conftest import make_engine, reader, white

# 1000 x 800 page in two columns: the layout model knows it is read column by column
LINES = [
    ("Annual report 2026", (100, 40, 900, 80), 0.99),
    ("The left column starts here", (100, 120, 480, 150), 0.98),  # same height as the right one
    ("Right column, first line", (520, 120, 900, 150), 0.98),
    ("and the left one goes on in this line", (100, 160, 480, 190), 0.98),
    ("Item", (100, 400, 300, 430), 0.97),
    ("Amount", (600, 400, 800, 430), 0.97),
    ("Diesel", (100, 450, 300, 480), 0.97),
    ("45,67", (600, 450, 800, 480), 0.97),
]
ZONES = [  # (PP-DocLayout class, box) in the reading order the model gives
    ("doc_title", (90, 30, 910, 90)),
    ("text", (90, 110, 490, 200)),
    ("text", (510, 110, 910, 160)),
    ("table", (90, 390, 910, 490)),
    ("image", (90, 600, 400, 750)),  # figure without text: still a zone
]
CELLS = [  # table cells in coordinates of the crop (the table starts at x=82, y=382 with the margin)
    ((10, 10, 220, 50), (0, 0, 0, 0)),
    ((500, 10, 720, 50), (0, 0, 1, 1)),
    ((10, 60, 220, 100), (1, 1, 0, 0)),
    ((500, 60, 720, 100), (1, 1, 1, 1)),
]


@pytest.fixture
def fake_layout(monkeypatch):
    def corners(box):
        x0, y0, x1, y1 = box
        return [x0, y0, x1, y0, x1, y1, x0, y1]

    def layout(image):
        return SimpleNamespace(class_names=[k for k, _ in ZONES], boxes=[np.array(b) for _, b in ZONES])

    def table(image):
        return SimpleNamespace(
            cell_bboxes=[[corners(b) for b, _ in CELLS]], logic_points=[[logic for _, logic in CELLS]]
        )

    monkeypatch.setattr(LayoutAnalyzer, "unavailable_reason", staticmethod(lambda: None))
    monkeypatch.setattr(LayoutAnalyzer, "load", lambda self: (layout, table))


def read_page(use_engines, **options):
    use_engines(make_engine("rapidocr", reader(*((t, b, c, "rapidocr") for t, b, c in LINES), width=1000)))
    return OCR(strategy="balanced", handwritten=False, **options).read(white(1000, 800)).pages[0]


def test_zones_give_reading_order_blocks_and_tables(use_engines, fake_layout):
    page = read_page(use_engines)
    assert page.text.splitlines()[:4] == [
        "Annual report 2026",
        "The left column starts here",
        "and the left one goes on in this line",  # the whole left one before the right one
        "Right column, first line",
    ]
    assert [b.kind for b in page.blocks] == ["title", "text", "text", "table", "figure"]
    assert page.lines[0].block == 0 and page.lines[3].block == 2 and page.blocks[1].lines == [1, 2]
    (table,) = page.tables
    assert table.block == 3 and (table.rows, table.cols) == (2, 2)
    assert table.grid() == [["Item", "Amount"], ["Diesel", "45,67"]]
    markdown = page.to_markdown()
    assert markdown.startswith("### Annual report 2026\n\nThe left column starts here and the left one goes on")
    assert "| Item | Amount |\n|---|---|\n| Diesel | 45,67 |" in markdown


def test_without_layout_reading_order_is_by_rows(use_engines, fake_layout):
    page = read_page(use_engines, layout=False)
    assert page.blocks == [] and page.tables == [] and all(line.block is None for line in page.lines)
    assert page.text.splitlines()[1] == "The left column starts here Right column, first line"


def test_table_spans_and_html():
    table = Table(
        (0, 0, 1, 1),
        2,
        3,
        [
            Cell(0, 0, "Group", (0, 0, 0.3, 0.5)),
            Cell(0, 1, "Result", (0.3, 0, 1, 0.5), colspan=2),
            Cell(1, 0, "A", (0, 0.5, 0.3, 1)),
            Cell(1, 1, "1", (0.3, 0.5, 0.6, 1)),
            Cell(1, 2, "2", (0.6, 0.5, 1, 1)),
        ],
    )
    assert table.grid() == [["Group", "Result", ""], ["A", "1", "2"]]
    assert '<td colspan="2">Result</td>' in table.to_html()


def test_native_pdf_tables_come_from_the_pdf(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Invoice 7 - detail", fontsize=14)
    rows = [("Item", "Quantity", "Amount"), ("Diesel A", "32,50 L", "45,67"), ("Total", "", "45,67")]
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            x, y = 72 + c * 150, 90 + r * 20
            page.draw_rect(pymupdf.Rect(x, y, x + 150, y + 20), color=(0, 0, 0))
            page.insert_text((x + 4, y + 14), text, fontsize=10)
    page.insert_text((72, 200), "Thank you for your visit", fontsize=10)
    path = tmp_path / "invoice.pdf"
    doc.save(path)
    result = OCR(strategy="balanced").read(path)
    (table,) = result.tables
    assert table.grid() == [list(r) for r in rows] and result.pages[0].method == "native"
    assert result.to_markdown().splitlines()[2:7] == [
        "Invoice 7 - detail",
        "",
        "| Item | Quantity | Amount |",
        "|---|---|---|",
        "| Diesel A | 32,50 L | 45,67 |",
    ]
    assert OCR(strategy="fast").read(path).tables == []  # fast: no structure


def test_native_formats_expose_their_tables():
    import io

    docx = pytest.importorskip("docx")
    openpyxl = pytest.importorskip("openpyxl")
    pptx = pytest.importorskip("pptx")
    d = docx.Document()
    d.add_paragraph("Invoice 7")
    t = d.add_table(rows=2, cols=2)
    for (r, c), v in {(0, 0): "Item", (0, 1): "Amount", (1, 0): "Diesel", (1, 1): "45,67"}.items():
        t.cell(r, c).text = v
    d.add_paragraph("Thanks")
    buf = io.BytesIO()
    d.save(buf)
    doc = OCR(strategy="fast").read(buf.getvalue())
    assert [t.grid() for t in doc.tables] == [[["Item", "Amount"], ["Diesel", "45,67"]]]
    assert doc.to_markdown().strip().splitlines() == [
        "Invoice 7",
        "",
        "| Item | Amount |",
        "|---|---|",
        "| Diesel | 45,67 |",
        "",
        "Thanks",
    ]
    # html: the data table yes; the layout one (with another table inside) and the single-cell one, no
    html = (
        b"<p>Hello</p><table><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></table>"
        b"<table><tr><td><table><tr><td>x</td></tr></table></td><td>y</td></tr><tr><td>z</td><td>w</td></tr></table>"
    )
    assert [t.grid() for t in OCR(strategy="fast").read(b"<html><body>" + html + b"</body></html>").tables] == [
        [["A", "1"], ["B", "2"]]
    ]
    wb = openpyxl.Workbook()
    wb.active.append(["Product", "Amount"])
    wb.active.append(["Diesel", 45.5])
    buf = io.BytesIO()
    wb.save(buf)
    assert OCR(strategy="fast").read(buf.getvalue()).tables[0].grid() == [["Product", "Amount"], ["Diesel", "45.5"]]
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    shape = slide.shapes.add_table(2, 2, 0, 0, 3_000_000, 1_000_000)
    for (r, c), v in {(0, 0): "Mes", (0, 1): "Sales", (1, 0): "Enero", (1, 1): "12"}.items():
        shape.table.cell(r, c).text = v
    buf = io.BytesIO()
    prs.save(buf)
    assert OCR(strategy="fast").read(buf.getvalue()).tables[0].grid() == [["Mes", "Sales"], ["Enero", "12"]]
