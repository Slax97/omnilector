# SPDX-License-Identifier: AGPL-3.0-or-later
"""read(): any input -> Document; native/OCR/hybrid pages, errors and CLI."""

from __future__ import annotations

import os

import numpy as np
import pymupdf
import pytest

import omnilector
from omnilector import OCR, Document
from omnilector.config import OCRConfig
from omnilector.exceptions import InferenceError, ModelNotAvailableError

from .conftest import make_engine, make_hybrid_pdf, make_mixed_pdf, png_bytes, reader
from .test_loaders import stamp_pdf


def test_read_pdf_native_and_scanned_pages(use_engines, tmp_path):
    engine = make_engine("rapidocr", reader(("SCANNED", (100, 100, 900, 200), 0.9, "rapidocr"), width=2480))
    use_engines(engine)
    path = tmp_path / "mixed.pdf"
    make_mixed_pdf(path)
    doc = OCR(strategy="fast", handwritten=False).read(path)
    assert isinstance(doc, Document) and doc.kind == "pdf" and doc.source == str(path)
    assert [p.method for p in doc.pages] == ["native", "ocr"]
    assert doc.pages[0].engines == ["native"] and doc.pages[1].engines == ["rapidocr"]
    assert len(engine.calls) == 1  # the page with text does not go through the OCR
    assert doc.pages[0].text.startswith("Native line number 0") and doc.pages[1].text == "SCANNED"
    assert doc.text == doc.pages[0].text + "\n\n" + doc.pages[1].text  # pages in order
    markdown = doc.to_markdown()
    assert markdown.startswith("# mixed.pdf") and "## Page 2" in markdown


def test_hybrid_page_ocr_only_sees_the_image(use_engines, tmp_path):
    path = tmp_path / "hybrid.pdf"
    make_hybrid_pdf(path)
    engine = make_engine("rapidocr", reader(("IN THE PHOTO", (300, 1500, 1500, 1600), 0.9, "rapidocr"), width=2480))
    use_engines(engine)
    page = OCR(strategy="balanced").read(path).pages[0]
    (image,) = engine.calls  # a single read (no variants or consensus) and only of the image crop
    assert image.shape[:2] == (2084, 2084)  # Rect(50, 200, 550, 700) at 300 DPI, not the 2480 x 3508 page
    texts = [line.text for line in page.lines]
    assert page.method == "hybrid" and page.engines == ["native", "rapidocr"]
    assert texts[0] == "Native text of the hybrid page 0" and texts[-1] == "IN THE PHOTO"
    foto = page.lines[-1].box
    assert foto[1] > 200 / 842  # in its place on the page (inside the image), not at the very top


def test_hybrid_page_keeps_its_text_when_the_ocr_fails(use_engines, tmp_path):
    def offline(image):
        raise ModelNotAvailableError("no network")

    use_engines(make_engine("rapidocr", offline))
    path = tmp_path / "hybrid.pdf"
    make_hybrid_pdf(path)
    doc = OCR(strategy="balanced").read(path)  # before: an exception, and the native text was lost too
    assert doc.pages[0].method == "native" and doc.text.startswith("Native text of the hybrid page 0")
    assert any("no network" in w for w in doc.warnings)


def test_same_image_on_every_page_is_read_once(use_engines, tmp_path):
    engine = make_engine("rapidocr", reader(("STAMP 2026", (40, 100, 660, 200), 0.9, "rapidocr"), width=1042))
    use_engines(engine)
    path = tmp_path / "letterhead.pdf"
    stamp_pdf(path, pages_with_logo=3)
    doc = OCR(strategy="balanced").read(path)
    assert len(engine.calls) == 1 and all(p.method == "hybrid" and "STAMP 2026" in p.text for p in doc.pages)


def test_ocr_page_limit_is_a_warning(use_engines, tmp_path):
    engine = make_engine("rapidocr", reader(("SCANNED", (100, 100, 900, 200), 0.9, "rapidocr"), width=2480))
    use_engines(engine)
    pdf = pymupdf.open()
    for _ in range(3):
        pdf.new_page().insert_image(pymupdf.Rect(50, 50, 550, 300), stream=png_bytes(800, 300, "SCANNED"))
    path = tmp_path / "scanned.pdf"
    pdf.save(path)
    doc = OCR(strategy="fast", handwritten=False, max_ocr_pages=2).read(path)
    assert len(engine.calls) == 2 and [p.text for p in doc.pages] == ["SCANNED", "SCANNED", ""]
    assert any("max_ocr_pages" in w for w in doc.warnings)


def test_hybrid_page_without_anything_else_skips_ocr(use_engines, tmp_path):
    """Scanned PDF with a text layer: outside the native text there is nothing left to read."""
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_image(pymupdf.Rect(0, 0, 595, 842), stream=png_bytes(600, 800, ""))  # blank "scan"
    for i in range(5):
        page.insert_text((72, 100 + 20 * i), f"Text layer of the scan, line {i}", fontsize=11)
    path = tmp_path / "searchable.pdf"
    pdf.save(path)
    engine = make_engine("rapidocr", lambda image: pytest.fail("no deberia hacer OCR"))
    use_engines(engine)
    doc = OCR().read(path)
    assert doc.pages[0].method == "native" and doc.text.startswith("Text layer of the scan, line 0")


def test_native_formats_never_touch_ocr(use_engines, tmp_path):
    use_engines(make_engine("rapidocr", lambda image: pytest.fail("no deberia hacer OCR")))
    path = tmp_path / "note.txt"
    path.write_text("Total: 45,67 EUR\nThanks", encoding="utf-8")
    doc = omnilector.read(path, strategy="fast")
    assert doc.kind == "text" and doc.text == "Total: 45,67 EUR\nThanks" and doc.pages[0].method == "native"


def test_read_bytes_lists_and_shared_instances(use_engines, tmp_path):
    import omnilector.ocr as ocr_mod

    use_engines(make_engine("rapidocr", reader(("Hello 123", (10, 30, 120, 60), 0.9, "rapidocr"), width=320)))
    doc = omnilector.read(png_bytes(), strategy="fast", handwritten=False)
    assert doc.kind == "image" and doc.text == "Hello 123" and doc.source == "<bytes>"
    docs = OCR(strategy="fast", handwritten=False).read_many([png_bytes(), png_bytes()])
    assert [d.text for d in docs] == ["Hello 123", "Hello 123"]
    path = tmp_path / "a.txt"
    path.write_text("hello", encoding="utf-8")
    omnilector.read(path, strategy="fast", handwritten=False)  # same configuration: same instance
    omnilector.read(path, languages="es")
    omnilector.read(path, languages=["es"])  # "es" and ["es"] are the same configuration
    assert len(ocr_mod._SHARED) == 2


def test_page_failure_is_a_warning_total_failure_is_an_error(use_engines, tmp_path):
    def broken(image):
        raise InferenceError("it broke")

    use_engines(make_engine("rapidocr", broken))
    path = tmp_path / "mixed.pdf"
    make_mixed_pdf(path)
    doc = OCR(strategy="fast", handwritten=False).read(path)
    assert [p.method for p in doc.pages] == ["native", "ocr"] and doc.pages[1].lines == []
    assert any("page 2" in w and "it broke" in w for w in doc.warnings)
    with pytest.raises(InferenceError):  # nothing readable: an error, not an empty document
        OCR(strategy="fast", handwritten=False).read(np.full((50, 50, 3), 255, np.uint8))


def test_searchable_pdf(use_engines, tmp_path):
    from typer.testing import CliRunner

    from omnilector.cli import app
    from omnilector.exceptions import UnsupportedFormatError

    use_engines(make_engine("rapidocr", reader(("SCANNED 2026", (100, 100, 900, 200), 0.9, "rapidocr"), width=2480)))
    mixed = tmp_path / "mixed.pdf"
    make_mixed_pdf(mixed)
    ocr = OCR(strategy="fast", handwritten=False)
    doc = ocr.searchable_pdf(mixed, tmp_path / "searchable.pdf")
    assert [p.method for p in doc.pages] == ["native", "ocr"]
    with pymupdf.open(tmp_path / "searchable.pdf") as out, pymupdf.open(mixed) as original:
        assert out[0].get_text() == original[0].get_text()  # the page with text, unchanged
        assert out[1].get_text().strip() == "SCANNED 2026" and original[1].get_text().strip() == ""
        assert len(out[1].get_images()) == 1  # the scan is still there, as it was
        words = out[1].get_text("words")
        assert (
            words[0][0] < 0.1 * out[1].rect.width and words[0][1] < 0.1 * out[1].rect.height
        )  # top left, where it was read
    ocr.searchable_pdf(png_bytes(2480, 600), tmp_path / "image.pdf")
    with pymupdf.open(tmp_path / "image.pdf") as out:
        assert out.page_count == 1 and out[0].get_text().strip() == "SCANNED 2026" and len(out[0].get_images()) == 1
    note = tmp_path / "note.txt"
    note.write_text("text", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):  # native text can already be searched
        ocr.searchable_pdf(note, tmp_path / "note.pdf")
    result = CliRunner().invoke(app, ["pdf", str(mixed), str(tmp_path / "cli.pdf"), "--strategy", "fast"])
    assert (
        result.exit_code == 0 and (tmp_path / "cli.pdf").exists() and "2 pages, 1 lines read with OCR" in result.stderr
    )


def test_time_limit_per_document(use_engines, monkeypatch):
    import io
    import time

    from PIL import Image

    now = [0.0]  # fake clock: each read of the engine "takes" 10 s
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    def slow(image):
        now[0] += 10
        return [omnilector.result.TextBlock("Hello 123", (10, 10, 100, 30), 0.9, "rapidocr")]

    engine = make_engine("rapidocr", slow)
    use_engines(engine)
    buf = io.BytesIO()
    pages = [Image.new("RGB", (200, 100), "white") for _ in range(3)]
    pages[0].save(buf, "TIFF", save_all=True, append_images=pages[1:])
    doc = OCR(strategy="balanced", handwritten=False, time_limit=5).read(buf.getvalue())
    # the first page, without the pass 2 it asked for (it was already at 10 s); the other two, without OCR
    assert len(engine.calls) == 1 and doc.text == "Hello 123"
    assert doc.warnings == [
        "time limit of 5 s (time_limit): 2 pages or images without OCR and 1 read without the extra passes"
    ]
    now[0] = 0.0
    assert OCR(strategy="balanced", handwritten=False).read(buf.getvalue()).warnings == []  # without a limit, nothing
    with pytest.raises(ValueError):
        OCRConfig(time_limit=0)
    monkeypatch.setenv("OMNILECTOR_TIME_LIMIT", "30")
    assert OCRConfig().time_limit == 30.0


def test_load_models_prepares_installed_engines(use_engines):
    page = make_engine("rapidocr", scripts=frozenset({"latin", "east_slavic"}))
    use_engines(
        page,
        make_engine("tesseract", reason="binary missing"),
        make_engine("trocr", level="line", scripts=frozenset({"latin"})),
    )
    assert OCR(languages=["es", "ru"]).load_models() == ["rapidocr", "rapidocr:east_slavic", "trocr"]
    assert OCR().load_models(["trocr"]) == ["trocr"]
    # without languages: first the model that reads the pages ("rapidocr", the Latin one) and then those tried
    page = make_engine("rapidocr", scripts=frozenset({"latin", "east_slavic", "cyrillic", "arabic", "greek"}))
    use_engines(page, make_engine("trocr", level="line", scripts=frozenset({"latin"})))
    assert OCR().load_models() == [
        "rapidocr",
        "rapidocr:arabic",
        "rapidocr:cyrillic",
        "rapidocr:east_slavic",
        "rapidocr:greek",
        "trocr",
    ]
    # detect_languages: only those of these scripts (Russian brings both Cyrillic models); without Latin
    # languages it starts with Russian and TrOCR is not needed
    assert OCR(detect_languages=["ru", "ar"]).load_models() == [
        "rapidocr:east_slavic",
        "rapidocr:arabic",
        "rapidocr:cyrillic",
    ]


def test_config_validation():
    assert OCRConfig().max_pages is None and OCRConfig().max_ocr_pages == 50  # native pages have no limit
    with pytest.raises(ValueError):
        OCRConfig(max_ocr_pages=-1)
    assert OCRConfig(device="auto").device == "auto"
    assert OCRConfig(model_size="large").model_size == "large"
    assert OCRConfig(strategy="fast").ocr_embedded_images is False
    assert OCRConfig(strategy="fast", ocr_embedded_images=True).ocr_embedded_images is True  # explicit settings win
    with pytest.raises(ValueError):
        OCRConfig(threads=0)
    with pytest.raises(ValueError):
        OCRConfig(model_size="xl")


def test_cli(tmp_path, use_engines):
    from typer.testing import CliRunner

    from omnilector.cli import app

    path = tmp_path / "a.txt"
    path.write_text("Hello CLI", encoding="utf-8")
    runner = CliRunner()
    out = runner.invoke(app, ["read", str(path), "--strategy", "fast"])
    assert out.exit_code == 0, out.output
    assert out.stdout.strip() == "Hello CLI"
    markdown = runner.invoke(app, ["read", str(path), "--format", "md", "--strategy", "fast"])
    assert markdown.stdout.startswith("# a.txt")
    assert runner.invoke(app, ["read", str(path), "--format", "xml"]).exit_code != 0  # unknown format: error
    assert runner.invoke(app, ["read", str(path), "-v", "--strategy", "fast"]).exit_code == 0
    use_engines(make_engine("rapidocr"))
    engines = runner.invoke(app, ["engines"])
    assert engines.exit_code == 0 and "rapidocr" in engines.stdout
    download = runner.invoke(app, ["download", "--lang", "es"])
    assert download.exit_code == 0 and download.stdout.strip() == "ready: rapidocr"


def test_cli_redirected_output_is_utf8(tmp_path):
    """On Windows, with the output redirected, Cyrillic text used to make the CLI fail (cp1252)."""
    import subprocess
    import sys

    path = tmp_path / "cyrillic.txt"
    path.write_text("Привет мир\nSeñor Muñoz", encoding="utf-8")
    cmd = [sys.executable, "-c", "from omnilector.cli import main; main()", "read", str(path), "--strategy", "fast"]
    done = subprocess.run(
        cmd, capture_output=True, timeout=120, check=False, env={**os.environ, "PYTHONIOENCODING": ""}
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert done.stdout.decode("utf-8").splitlines() == ["Привет мир", "Señor Muñoz"]
