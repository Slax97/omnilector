# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real images from the internet (tests/realdata.py): other scripts without giving the language, a scanned
table and tickets. Skipped if they cannot be downloaded."""

from __future__ import annotations

import urllib.error

import pytest

from .. import realdata

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ocr():
    pytest.importorskip("rapidocr")
    from omnilector import OCR

    return OCR(strategy="balanced", handwritten=False)


def _get(fetch, *args):
    try:
        return fetch(*args)
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"no network or no sample: {e}")


@pytest.mark.parametrize("sample", realdata.SAMPLES, ids=[s.name for s in realdata.SAMPLES])
def test_real_image(ocr, sample):
    for module in sample.requires:
        pytest.importorskip(module)
    doc = ocr.read(_get(realdata.fetch, sample))
    for text in sample.expect:
        assert text in doc.text, f"{sample.name}: {text!r} missing in {doc.text!r}"
    if sample.scripts:
        assert doc.pages[0].script in sample.scripts
    if sample.table:
        assert (doc.tables[0].rows, doc.tables[0].cols) == sample.table


# With the language fixed or a list of languages: the same texts as without them (they used to be lost) and more
LANGUAGE_CASES = [
    ("kyiv.jpg", {"languages": ["uk"]}, []),  # only with the East Slavic model it came out "вYлиця"
    ("moscow.jpg", {"languages": ["ru"]}, ["УЛИЦА ГРИНА"]),  # it came out "YЦА ФPNНА"
    ("hurghada.jpg", {"languages": ["ru", "en", "ar"]}, []),  # the whole page with Russian: the Arabic was lost
    ("moscow.jpg", {"detect_languages": ["es", "ru"]}, []),  # only the East Slavic one: "АлександреГрине"
    ("kyiv.jpg", {"detect_languages": ["uk", "en"]}, []),
]


@pytest.mark.parametrize(
    ("name", "options", "extra"), LANGUAGE_CASES, ids=[f"{n}-{next(iter(o))}" for n, o, _ in LANGUAGE_CASES]
)
def test_real_image_with_languages(name, options, extra):
    pytest.importorskip("rapidocr")
    from omnilector import OCR

    sample = next(s for s in realdata.SAMPLES if s.name == name)
    doc = OCR(strategy="balanced", handwritten=False, **options).read(_get(realdata.fetch, sample))
    for text in [*sample.expect, *extra]:
        assert text in doc.text, f"{name} {options}: {text!r} missing in {doc.text!r}"
