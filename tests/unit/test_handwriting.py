# SPDX-License-Identifier: AGPL-3.0-or-later
"""Handwriting: line engine by language (TrOCR for English, kraken for the rest)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from omnilector import OCR
from omnilector.config import OCRConfig
from omnilector.core.strategies import get_strategy
from omnilector.engines import KrakenEngine
from omnilector.result import RecognitionCandidate
from omnilector.routing import PageProfile, Router
from omnilector.routing.languages import guess_language
from omnilector.utils.text import without_marks

from .conftest import make_engine, reader, white

LATIN = frozenset({"latin"})


def line_engines(
    seen: dict,
    english: str = "The cat and the dog are in the garden",
    other: str = "the cat amd the dog are im the gardem",
):
    def trocr_lines(boxes):
        seen.setdefault("trocr", []).append(len(boxes))
        return [RecognitionCandidate(english, 0.9, "trocr") for _ in boxes]

    def kraken_lines(boxes):
        seen.setdefault("kraken", []).append(len(boxes))
        return [RecognitionCandidate(other, 0.97, "kraken") for _ in boxes]

    trocr = make_engine("trocr", level="line", handwriting=True, scripts=LATIN, cost=3, read_lines=trocr_lines)
    kraken = make_engine(
        "kraken",
        level="line",
        handwriting=True,
        scripts=frozenset({"latin", "cyrillic"}),
        cost=2,
        read_lines=kraken_lines,
    )
    trocr.info = type(trocr.info)(**{**trocr.info.__dict__, "languages": frozenset({"en"})})
    return trocr, kraken


def test_guess_language():
    assert guess_language(["Become a success with a dise and hey presto. You are a star"]) == "en"
    assert guess_language(["el expediente esta pendiente y los documentos no estan en la oficina"]) == "es"
    assert guess_language(["de la villa"]) is None  # "de", "la": Spanish, Catalan, French... and little text
    assert guess_language(["GASOLINERA LUIS SL CIF B12345678"]) is None  # no words of any language
    assert guess_language(["hola"]) is None
    # nearly ties with Catalan ("de", "la", "el", "en"): Spanish goes first in the preference (before, None)
    assert guess_language(["de la casa y el perro en la calle"]) == "es"
    assert guess_language(["le chat est dans la maison avec les enfants et le chien"]) == "fr"  # clearly French
    spanish = ["el expediente esta pendiente y los documentos no estan en la oficina"]
    assert guess_language(spanish, among=["es", "en"]) == "es"  # only among those of detect_languages
    assert guess_language(spanish, among=["en", "de"]) is None and guess_language(spanish, among=["ru"]) is None


def test_router_picks_the_line_engine_by_language(use_engines):
    trocr, kraken = line_engines({})
    use_engines(make_engine("rapidocr"), trocr, kraken)

    def line(lang: str | None, script: str = "latin") -> str | None:
        router = Router(OCRConfig(languages=[lang] if lang else ()), get_strategy("accurate"))
        return router.line_engine(PageProfile(script=script), lang)

    assert line("en") == "trocr"  # the language's specialist
    assert (
        line("es") == "kraken" and line(None) == "kraken"
    )  # the general one, if there is no specialist or the language is unknown
    assert line("ru", "east_slavic") is None or line("ru", "cyrillic") == "kraken"
    use_engines(make_engine("rapidocr"), trocr)  # without kraken installed: TrOCR for everything, as before
    assert line("es") == "trocr"


def test_unknown_language_reads_with_kraken_then_english_specialist(use_engines):
    seen: dict = {}
    trocr, kraken = line_engines(seen)
    scrawl = reader(("tlie cat amd tke dog", (10, 10, 290, 40), 0.4, "rapidocr"), width=300)
    use_engines(make_engine("rapidocr", scrawl), trocr, kraken)
    line = OCR(strategy="accurate").read(white(300, 60)).lines[0]
    assert seen == {"kraken": [1], "trocr": [1]}  # kraken first; its reading says "English": TrOCR reads it again
    assert line.text == "The cat and the dog are in the garden" and line.engines == ["trocr"] and line.handwritten
    assert "the cat amd the dog are im the gardem" in line.alternatives


@pytest.mark.parametrize(("languages", "expected"), [(["es"], {"kraken": [1]}), (["en"], {"trocr": [1]})])
def test_known_language_goes_straight_to_its_engine(use_engines, languages, expected):
    seen: dict = {}
    trocr, kraken = line_engines(seen, other="el gato y el perro estan en el jardin")
    use_engines(
        make_engine("rapidocr", reader(("el gato y el perro", (10, 10, 290, 40), 0.4, "rapidocr"), width=300)),
        trocr,
        kraken,
    )
    OCR(strategy="accurate", languages=languages).read(white(300, 60))
    assert seen == expected


@pytest.mark.parametrize(("detect", "expected"), [(["en", "ru"], {"trocr": [1]}), (["es", "ru"], {"kraken": [1]})])
def test_detect_languages_decide_the_handwriting_language(use_engines, detect, expected):
    # the only Latin language of the list is the page's: with "en", TrOCR directly (two words are not enough
    # to guess it); without "en", kraken's reading, which sounds English, is not read again by TrOCR
    seen: dict = {}
    trocr, kraken = line_engines(seen)
    use_engines(
        make_engine("rapidocr", reader(("tlie cat", (10, 10, 290, 40), 0.4, "rapidocr"), width=300)), trocr, kraken
    )
    OCR(strategy="accurate", detect_languages=detect).read(white(300, 60))
    assert seen == expected


def test_kraken_adapter_maps_lines_and_confidence(monkeypatch):
    class FakeNet:
        def predict(self, image, segmentation, config):
            assert [line.bbox for line in segmentation.lines] == [
                (0, 0, 50, 20),
                (0, 30, 50, 50),
            ]  # without the tiny box
            return [
                SimpleNamespace(id="2", prediction=" Hello ", confidences=[0.9, 0.7]),
                SimpleNamespace(id="0", prediction="Bye", confidences=[1.0]),
            ]

    pytest.importorskip("kraken")
    engine = KrakenEngine(OCRConfig())
    monkeypatch.setattr(engine, "load", lambda: (FakeNet(), None))
    found = engine.read_lines(white(60, 60), [(0, 0, 50, 20), (0, 0, 2, 2), (0, 30, 50, 50)])
    assert [(c.text, round(c.confidence, 2)) if c else None for c in found] == [("Bye", 1.0), None, ("Hello", 0.8)]


def test_kraken_reads_boxes_that_go_over_the_edge(monkeypatch):
    """kraken does not read a box outside the image, and EasyOCR gives them: they are clipped."""
    seen = []

    class FakeNet:
        def predict(self, image, segmentation, config):
            seen.extend(line.bbox for line in segmentation.lines)
            return []

    pytest.importorskip("kraken")
    engine = KrakenEngine(OCRConfig())
    monkeypatch.setattr(engine, "load", lambda: (FakeNet(), None))
    engine.read_lines(white(60, 60), [(-15, 2, 40, 80), (70, 0, 90, 20)])
    assert seen == [(0, 2, 40, 60)]  # the second one is outside altogether


def test_kraken_decides_the_direction_of_right_to_left_lines(monkeypatch):
    """kraken goes by the first letter of the line: on right-to-left scripts, by most of its letters."""

    class Record(SimpleNamespace):
        def logical_order(self, base_dir):
            return Record(id=self.id, prediction=f"{base_dir}: {self.prediction}", confidences=self.confidences)

    class FakeNet:
        def predict(self, image, segmentation, config):
            return [
                Record(id="0", prediction="m مرحبا 3", confidences=[0.9]),  # a stray Latin letter on the left
                Record(id="1", prediction="Hello World", confidences=[0.9]),  # a Latin line on an Arabic page
            ]

    pytest.importorskip("kraken")
    boxes = [(0, 0, 50, 20), (0, 30, 50, 50)]
    engine = KrakenEngine(OCRConfig(), "arabic")
    monkeypatch.setattr(engine, "load", lambda: (FakeNet(), None))
    assert [c.text for c in engine.read_lines(white(60, 60), boxes)] == ["R: m مرحبا 3", "L: Hello World"]
    # in the other scripts, it stays in kraken's hands
    engine = KrakenEngine(OCRConfig(), "latin")
    monkeypatch.setattr(engine, "load", lambda: (FakeNet(), None))
    assert [c.text for c in engine.read_lines(white(60, 60), boxes)] == ["m مرحبا 3", "Hello World"]


def test_without_marks():
    assert without_marks("quyền và tự do đó") == ("quyen va tu do do", 7)  # ề and ự have two, đ its stroke
    assert without_marks("Señor") == ("Senor", 1)


@pytest.mark.parametrize(
    ("page", "line", "kept"),
    [
        ("quyền và tự do đó của", "quyên và tu do dó cúa", "quyền và tự do đó của"),  # the same, without its marks
        ("Senor Garcia", "Señor García", "Señor García"),  # with more marks: the line engine sees them
        ("ql gaito y el pcrro", "el gato y el perro", "el gato y el perro"),  # other letters: it reads them better
        ("que epúeda dap", "que pueda dis", "que pueda dis"),  # fewer marks, but also other letters
    ],
)
def test_line_engine_does_not_replace_a_reading_only_by_dropping_its_marks(use_engines, page, line, kept):
    use_engines(
        make_engine("rapidocr", reader((page, (10, 10, 280, 40), 0.6, "rapidocr")), scripts=LATIN),
        make_engine(
            "kraken",
            level="line",
            handwriting=True,
            scripts=LATIN,
            read_lines=lambda boxes: [RecognitionCandidate(line, 0.97, "kraken") for _ in boxes],
        ),
    )
    doc = OCR(languages=["es"], strategy="accurate").read(white(300, 200))
    assert doc.text == kept and line in [*doc.lines[0].alternatives, doc.text]  # the other reading is kept
