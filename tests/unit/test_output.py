# SPDX-License-Identifier: AGPL-3.0-or-later
"""Common output: the same format wherever the text comes from (one engine, another, several or the file)."""

from __future__ import annotations

import dataclasses

from omnilector import OCR, Line, OCRConfig
from omnilector.core.fusion import fuse_blocks
from omnilector.result import make_page
from omnilector.utils.geometry import group_rows, reading_rows

from .conftest import make_engine, reader, tb, white

LINE_FIELDS = {"text", "box", "confidence", "engines", "alternatives", "handwritten", "row", "block"}


def test_same_output_whatever_engine_and_how_it_splits_lines(use_engines):
    # one engine reads the whole row; another splits it in two pieces (like RapidOCR with large gaps)
    whole = reader(
        ("TOTAL 48,22 EUR", (10, 50, 290, 70), 0.9, "tesseract"), ("Thanks", (10, 90, 120, 110), 0.8, "tesseract")
    )
    split = reader(
        ("TOTAL", (10, 50, 80, 70), 0.9, "easyocr"),
        ("48,22 EUR", (200, 51, 290, 71), 0.9, "easyocr"),
        ("Thanks", (10, 90, 120, 110), 0.8, "easyocr"),
    )
    texts = []
    for name, read in (("tesseract", whole), ("easyocr", split)):
        use_engines(make_engine(name, read))
        doc = OCR(strategy="fast", handwritten=False).read(white(300, 150))
        assert all({f.name for f in dataclasses.fields(line)} == LINE_FIELDS for line in doc.lines)
        assert doc.engines == [name]
        texts.append(doc.text)
    assert texts[0] == texts[1] == "TOTAL 48,22 EUR\nThanks"


def test_combined_engines_report_agreement_and_alternatives():
    fused = fuse_blocks(
        [
            tb("2124 CZT", (10, 10, 110, 30), 0.90, "rapidocr"),
            tb("2124 CZT", (11, 9, 111, 31), 0.60, "tesseract"),
            tb("2124 CZI", (9, 10, 109, 30), 0.95, "easyocr"),
        ]
    )
    (line,) = make_page(group_rows(fused), (200, 100)).lines
    assert line == Line(
        text="2124 CZT",
        box=(0.05, 0.1, 0.55, 0.3),  # relative to the page (200x100 px)
        confidence=0.9,
        engines=["rapidocr", "tesseract"],  # the two that agree
        alternatives=["2124 CZI"],  # what the third one read
        handwritten=False,
        row=0,
    )


def test_native_text_uses_the_same_output(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("Total 45,67\nThanks", encoding="utf-8")
    doc = OCR(strategy="fast").read(path)
    assert [(line.text, line.engines, line.confidence, line.row) for line in doc.lines] == [
        ("Total 45,67", ["native"], 1.0, 0),
        ("Thanks", ["native"], 1.0, 1),
    ]
    assert all(0.0 <= v <= 1.0 for line in doc.lines for v in line.box)
    assert doc.pages[0].method == "native" and doc.engines == ["native"]


def test_rows_join_pieces_of_the_same_line():
    form = [
        tb("Total", (0, 0, 50, 20), 1.0, "native"),
        tb("45,67", (300, 2, 360, 22), 1.0, "native"),
        tb("Thanks", (0, 40, 80, 60), 1.0, "native"),
    ]
    page = make_page(reading_rows(form), (400, 100))
    assert [line.row for line in page.lines] == [0, 0, 1] and page.text == "Total 45,67\nThanks"
    # a large amount next to its small label: same row (they overlap vertically)
    receipt = [
        tb("TOTAL", (0, 30, 80, 50), 0.9),
        tb("48,22", (200, 10, 400, 70), 0.9),
        tb("Thanks", (0, 90, 90, 110), 0.9),
    ]
    assert make_page(reading_rows(receipt), (400, 120)).text == "TOTAL 48,22\nThanks"


def prose(side: str, i: int, x: int, y: int) -> object:
    return tb(f"Column {side}, line {i} of running text", (x, y, x + 360, y + 20), 1.0, "native")


def test_text_columns_are_read_one_after_the_other():
    title = tb("Title that spans both columns of the page", (0, 0, 800, 24), 1.0, "native")
    left = [prose("left", i, 0, 50 + 30 * i) for i in range(4)]
    right = [prose("right", i, 440, 50 + 30 * i) for i in range(4)]
    footer = tb("Footer centered under both columns", (100, 200, 700, 220), 1.0, "native")
    lines = make_page(reading_rows([*right, footer, *left, title]), (800, 240)).text.splitlines()
    assert lines == [title.text] + [b.text for b in left] + [b.text for b in right] + [footer.text]


def test_right_to_left_rows_and_columns():
    # a sign in Arabic that the engine splits in two pieces: read from right to left
    row = [tb("الملك فهد", (0, 0, 200, 30), 0.9), tb("شارع", (300, 0, 400, 30), 0.9)]
    assert make_page(reading_rows(row), (400, 30)).text == "شارع الملك فهد"
    mixed = [tb("Restaurant", (0, 0, 120, 30), 0.9), tb("مطعم", (300, 0, 400, 30), 0.9)]
    assert make_page(reading_rows(mixed), (400, 30)).text == "Restaurant مطعم"  # the majority of the letters decides
    # two columns of running text in Arabic: the right one first
    right = [
        tb(f"سطر عربي طويل من العمود الأيمن {i}", (440, 50 + 30 * i, 800, 70 + 30 * i), 1.0, "native") for i in range(4)
    ]
    left = [
        tb(f"سطر عربي طويل من العمود الأيسر {i}", (0, 50 + 30 * i, 360, 70 + 30 * i), 1.0, "native") for i in range(4)
    ]
    lines = make_page(reading_rows([*left, *right]), (800, 200)).text.splitlines()
    assert lines == [b.text for b in right] + [b.text for b in left]


def test_tables_and_tickets_stay_in_rows():
    # column of amounts (short text): still a table, each item with its amount
    rows = [
        (
            tb(f"Item number {i} of the invoice", (0, 30 * i, 300, 30 * i + 20), 1.0),
            tb(f"{i},50", (600, 30 * i, 660, 30 * i + 20), 1.0),
        )
        for i in range(5)
    ]
    text = make_page(reading_rows([b for row in rows for b in row]), (700, 160)).text
    assert text.splitlines()[0] == "Item number 0 of the invoice 0,50"


def test_rotated_text_does_not_swallow_the_rows_it_crosses():
    # the vertical CSV of the margin of a signed PDF: its box takes the height of all the lines
    lines = [tb(f"Line {i} of the letter", (100, 40 * i, 500, 40 * i + 20), 1.0, "native") for i in range(5)]
    csv = tb("CSV: ABC123 verificable en sede electronica", (10, 0, 30, 400), 1.0, "native")
    csv.quad = ((10, 400), (10, 0), (30, 0), (30, 400))
    text = make_page(reading_rows([csv, *lines]), (600, 400)).text
    assert text.splitlines() == [b.text for b in lines] + [csv.text]
    # no corners (an engine that only gives boxes): a tall and narrow box with text is vertical
    csv.quad = None
    assert make_page(reading_rows([csv, *lines]), (600, 400)).text.splitlines()[-1] == csv.text


def test_empty_readings_are_not_lines():
    page = make_page([[tb("  ", (0, 0, 10, 10), 0.1)], [tb("hello", (0, 20, 40, 30), 0.9)]], (100, 100))
    assert [(line.text, line.row) for line in page.lines] == [("hello", 0)]


def test_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OMNILECTOR_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("OMNILECTOR_THREADS", "2")
    cfg = OCRConfig()
    assert cfg.model_cache_dir == str(tmp_path / "models") and (tmp_path / "models").is_dir()
    assert cfg.threads == 2
    assert OCRConfig(model_cache_dir=str(tmp_path / "other"), threads=1).threads == 1  # explicit settings win
