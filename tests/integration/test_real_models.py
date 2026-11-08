# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integration: real models needed. Skipped if dependencies or network are missing."""

from __future__ import annotations

import io

import numpy as np
import pytest

from omnilector.exceptions import OmnilectorError

from .. import quality
from ..helpers import make_mixed_pdf

pytestmark = pytest.mark.integration


def render(lines: list[str], size: int = 40, width: int = 1000):
    if quality.font_path() is None:
        pytest.skip("no TrueType font with accents")
    return quality.render(lines, size=size, width=width)


def test_real_rapidocr():
    pytest.importorskip("rapidocr")
    from omnilector import OCR

    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    ocr = OCR(strategy="fast", handwritten=False)
    doc = ocr.read(img)
    assert len(doc.pages) == 1 and doc.text == ""  # blank image: page without lines, no error


def test_real_rapidocr_spanish():
    pytest.importorskip("rapidocr")
    from omnilector import OCR

    img = np.array(render(["Invoice A-2026-0007", "Año 2026: señor Muñoz", "Total 45,67 €"]))
    text = OCR(strategy="fast", handwritten=False).read(img).text
    assert "Muñoz" in text and "Año" in text and "45,67" in text


def test_real_mixed_pdf(tmp_path):
    pytest.importorskip("rapidocr")
    from omnilector import read

    buf = io.BytesIO()
    render(["SCANNED 2026", "Total 45,67 EUR"]).save(buf, "PNG")
    path = tmp_path / "mixed.pdf"
    make_mixed_pdf(path, image_png=buf.getvalue())
    result = read(path, strategy="fast", handwritten=False)
    assert [p.method for p in result.pages] == ["native", "ocr"]
    assert "SCANNED 2026" in result.pages[1].text


def test_real_rapidocr_other_script():
    """Non-Latin script: RapidOCR uses the PP-OCRv5 model of that script (downloaded the 1st time)."""
    pytest.importorskip("rapidocr")
    from omnilector import OCR

    img = np.array(render(["Привет мир 2026"]))
    try:
        doc = OCR(strategy="fast", handwritten=False, languages=["ru"]).read(img)
    except OmnilectorError as e:
        pytest.skip(f"Cyrillic model not available (offline?): {e}")
    assert doc.engines == ["rapidocr"] and "Привет" in doc.text  # the Latin model does not read Cyrillic


def test_real_trocr_confidence():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from omnilector.config import OCRConfig
    from omnilector.engines import TrOCREngine

    img = np.array(render(["hello world"], size=48, width=600))
    try:
        (cand,) = TrOCREngine(OCRConfig()).read_lines(img, [(20, 25, 580, 125)])
    except OmnilectorError as e:
        pytest.skip(f"TrOCR no disponible: {e}")
    assert cand is not None and "hello" in cand.text.lower()
    assert 0.0 < cand.confidence <= 1.0  # real confidence, not a fixed value


def test_real_tesseract():
    pytest.importorskip("pytesseract")
    from omnilector import OCR
    from omnilector.engines.tesseract import find_tesseract

    if find_tesseract() is None:
        pytest.skip("no tesseract binary")
    img = np.array(render(["Invoice A-2026-0007", "Total 45,67 EUR"]))
    doc = OCR(engine="tesseract", strategy="fast", handwritten=False, languages=["es"]).read(img)
    assert "45,67" in doc.text and "Invoice" in doc.text and doc.engines == ["tesseract"]


def test_real_easyocr():
    pytest.importorskip("easyocr")
    from omnilector import OCR

    img = np.array(render(["Total 45,67 EUR"]))
    try:
        doc = OCR(engine="easyocr", strategy="fast", handwritten=False, languages=["es"]).read(img)
    except OmnilectorError as e:
        pytest.skip(f"EasyOCR without models (offline?): {e}")
    assert "45,67" in doc.text and doc.engines == ["easyocr"]


def test_real_kraken_spanish_handwriting():
    """Spanish handwriting with kraken (PP-OCRv6): keeps accents and ñ, which TrOCR (English) loses."""
    pytest.importorskip("kraken")
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    from omnilector.config import OCRConfig
    from omnilector.engines import KrakenEngine

    font = next((p for p in ("C:/Windows/Fonts/Inkfree.ttf", "C:/Windows/Fonts/segoepr.ttf") if Path(p).exists()), None)
    if font is None:
        pytest.skip("no handwriting font")
    text = "Recibí de Juan Martínez la cantidad de 45,67 euros"
    img = Image.new("RGB", (1200, 110), "white")
    ImageDraw.Draw(img).text((30, 20), text, fill=(20, 20, 60), font=ImageFont.truetype(font, 44))
    try:
        (cand,) = KrakenEngine(OCRConfig()).read_lines(np.array(img), [(0, 0, 1200, 110)])
    except OmnilectorError as e:
        pytest.skip(f"kraken without a model (offline?): {e}")
    assert (
        cand is not None and "martínez" in cand.text.lower() and "45,67" in cand.text
    )  # with the composed accent (NFC)
    assert 0.5 < cand.confidence <= 1.0
