# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests: no heavy models needed (RapidOCR is mocked)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from omnilector import OCR, OCRConfig
from omnilector.core.fusion import fuse_blocks
from omnilector.exceptions import DocumentLoadError
from omnilector.result import Document, TextBlock
from omnilector.utils.geometry import box_iou, reading_order, scale_box
from omnilector.utils.image import load_image

from .conftest import make_engine


def white_image(w=200, h=100):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def test_config_strategies():
    assert OCRConfig(strategy="fast").scales == (1.0,)
    assert OCRConfig(strategy="balanced").scales == (1.0, 1.5)
    assert OCRConfig(strategy="maximum_recall").scales == (1.0, 1.5, 2.0)
    # explicit settings win: fast and balanced used to overwrite preprocessing, multi_scale and scales
    assert OCRConfig(strategy="balanced", scales=(1.0, 3.0)).scales == (1.0, 3.0)
    assert OCRConfig(strategy="fast", preprocessing=True).preprocessing is True
    with pytest.raises(ValueError):
        OCRConfig(strategy="nope")


def test_logging_is_left_to_the_application(monkeypatch):
    import logging

    logger = logging.getLogger("omnilector")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [logging.NullHandler()])  # the application already configured logging
    OCR(strategy="fast")
    assert logger.handlers == [] and logger.propagate  # no handler of its own: each warning comes out once
    OCR(strategy="fast", log_level="DEBUG")
    assert logger.level == logging.DEBUG and logger.handlers == []
    OCR(strategy="fast")  # another instance without log_level does not change the level of the first one
    assert logger.level == logging.DEBUG
    monkeypatch.setattr(root, "handlers", [])
    OCR(strategy="fast", log_level="INFO")  # logging not configured: a handler to show what was asked for
    assert len(logger.handlers) == 1 and not logger.propagate
    with pytest.raises(ValueError):
        OCR(strategy="fast", log_level="CHARLA")


def test_load_inputs(tmp_path):
    _, size = load_image(white_image())
    assert size == (200, 100)
    pil = Image.fromarray(white_image())
    _, s2 = load_image(pil)
    assert s2 == (200, 100)
    import io

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    _, s3 = load_image(buf.getvalue())
    assert s3 == (200, 100)
    p = tmp_path / "a.png"
    pil.save(p)
    _, s4 = load_image(str(p))
    assert s4 == (200, 100)
    with pytest.raises(DocumentLoadError):
        load_image(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(DocumentLoadError):
        load_image(tmp_path / "nope.png")
    with pytest.raises(DocumentLoadError):
        load_image(12345)


def test_geometry():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert scale_box((100, 0, 200, 50), 2.0) == (50, 0, 100, 25)
    # fusion: same text + overlap = 1; different place = 2
    b1 = TextBlock(text="45", box=(0, 0, 20, 10), confidence=0.9, engine="rapidocr")
    b2 = TextBlock(text="45", box=(1, 1, 21, 11), confidence=0.5, engine="rapidocr")
    b3 = TextBlock(text="45", box=(200, 200, 220, 210), confidence=0.9, engine="rapidocr")
    assert len(fuse_blocks([b1, b2, b3])) == 2
    # order: top row first even if its x is larger
    a = TextBlock(text="a", box=(200, 10, 250, 30), confidence=1.0, engine="t")
    b = TextBlock(text="b", box=(10, 100, 60, 120), confidence=1.0, engine="t")
    assert [x.text for x in reading_order([b, a])] == ["a", "b"]


def test_cut_glyph_at_overlapping_word_boxes():
    from omnilector.core.pipeline import _cut_glyphs

    def blocks(*items):
        return [TextBlock(text=t, box=box, confidence=0.97, engine="rapidocr") for t, box in items]

    # the box of "Firmado:" takes the stroke of the L of the name (it happened on Linux, with DejaVu)
    found = blocks(("Firmado: I", (11, 24, 235, 84)), ("Luis", (208, 26, 360, 84)))
    _cut_glyphs(found)
    assert found[0].text == "Firmado:" and found[0].candidates[0].text == "Firmado: I"
    # a real single character, separated by a space, does not fall inside the next box
    real = blocks(("Diesel y", (10, 20, 200, 60)), ("aceite", (190, 20, 320, 60)), ("Chapter I", (10, 100, 220, 140)))
    _cut_glyphs(real)
    assert [b.text for b in real] == ["Diesel y", "aceite", "Chapter I"]


def test_preprocessing_keeps_original():
    from omnilector.preprocessing.pipeline import build_variants

    img = white_image()
    variants = build_variants(img, ["clahe", "sharpen"])
    assert list(variants) == ["original", "clahe", "sharpen"]
    assert variants["original"] is img  # no unnecessary copy of the original
    assert list(build_variants(img, [])) == ["original"]


def test_ocr_mocked(use_engines):
    """Pipeline with a fake RapidOCR: checks fusion/order without downloading models."""

    def fake_read(image):
        return [TextBlock(text="2124 CZT", box=(10, 10, 100, 30), confidence=0.4, engine="rapidocr")]

    use_engines(make_engine("rapidocr", fake_read))
    ocr = OCR(strategy="fast", handwritten=False)
    res = ocr.read(white_image())
    assert isinstance(res, Document)
    assert res.text == "2124 CZT"
    assert res.lines[0].confidence == pytest.approx(0.4)  # low confidence is kept
    assert res.lines[0].box == (0.05, 0.1, 0.5, 0.3)  # box relative to the 200x100 image
    # several images reusing the models
    rss = list(ocr.read_many([white_image(), white_image()]))
    assert len(rss) == 2


def test_low_confidence_not_dropped(use_engines):
    from omnilector.result import RecognitionCandidate

    def fake_read(image):
        return [
            TextBlock(
                text="handwriting",
                box=(0, 0, 50, 20),
                confidence=0.32,
                engine="rapidocr",
                candidates=[RecognitionCandidate("handwriting", 0.32, "rapidocr")],
            )
        ]

    use_engines(make_engine("rapidocr", fake_read))
    ocr = OCR(strategy="maximum_recall", handwritten=False)
    # maximum_recall with 1 block triggers PASS2; variants are mocked to keep it light
    res = ocr.read(white_image())
    assert any(line.confidence == pytest.approx(0.32) for line in res.lines)
