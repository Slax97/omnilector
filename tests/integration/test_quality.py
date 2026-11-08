# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quality with real models: each synthetic case (tests/quality.py) below its maximum error.

Protects what was hard to fix: skewed, sideways or upside-down pages, transparency, columns, stamps on
PDFs with text, docx forms and screenshots pasted in an email. To measure by hand, with times and per
strategy: python benchmark.py
"""

from __future__ import annotations

import pytest

from ..quality import CASES, font_path, score

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ocr():
    pytest.importorskip("rapidocr")
    if font_path() is None:
        pytest.skip("no TrueType font with accents")
    from omnilector import OCR

    return OCR(strategy="balanced", handwritten=False)  # pass 2 and PDF images, no consensus


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_quality(ocr, case):
    for module in case.requires:
        pytest.importorskip(module)
    result = score(ocr.read(case.make()).text, case.truth)
    assert result.cer <= case.max_cer, f"{case.name}: CER {result.cer:.2%}"
    assert result.lines >= 0.85, f"{case.name}: only {result.lines:.0%} of the lines come out whole"
