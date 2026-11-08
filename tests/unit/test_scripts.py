# SPDX-License-Identifier: AGPL-3.0-or-later
"""Automatic script (without languages): a fake engine that only reads well with the model of the script."""

from __future__ import annotations

from typing import ClassVar

import pytest

from omnilector import OCR
from omnilector.engines.base import PAGE_KINDS, EngineInfo, OCREngine
from omnilector.result import RecognitionCandidate, TextBlock
from omnilector.routing.languages import SCRIPTS, mixed_scripts, script_share

from .conftest import white

LINES = {  # box -> reading of each model (the "latin" one is the multi model: Latin + Chinese + Japanese)
    (100, 100, 700, 140): {"latin": ("", 0.0), "cyrillic": ("ул. Медникарска", 0.85), "greek": ("yπ. MeAHμKApCK", 0.5)},
    (100, 200, 700, 240): {"latin": ("Mednikarska Str.", 0.99), "cyrillic": ("Mednikarska Str.", 0.97)},
}


def page_engine(lines: dict, name: str = "rapidocr", flipped: bool | None = None):
    class ByScript(OCREngine):
        """Reads like a model of that script would: it only understands its script and Latin."""

        info = EngineInfo(name=name, scripts=SCRIPTS, kinds=PAGE_KINDS)
        calls: ClassVar[list[str]] = []  # script of each page read whole
        tried: ClassVar[list[str]] = []  # script of each try on a few lines

        def _reading(self, box):
            options = lines[box]
            return options.get(self.script) or (next(iter(options.values()))[0] if self.script != "latin" else "", 0.3)

        def read(self, image):
            type(self).calls.append(self.script)
            return [
                TextBlock(text=t, box=box, confidence=c, engine=name) for box in lines for t, c in [self._reading(box)]
            ]

        def read_lines(self, image, boxes):
            type(self).tried.append(self.script)
            return [
                RecognitionCandidate(t, c, name) if t else None for t, c in (self._reading(tuple(b)) for b in boxes)
            ]

        def upside_down(self, image, boxes):
            return flipped

    ByScript.calls, ByScript.tried = [], []
    return ByScript


def test_script_helpers():
    assert script_share("ул. Медникарска", ["cyrillic"]) == 1.0 and script_share("Mednikarska", ["cyrillic"]) == 0.0
    assert script_share("123", ["latin"]) == 0.0  # without letters it cannot be told
    assert mixed_scripts("中YAAΣION") and not mixed_scripts("315 西 愚园路 东 309") and not mixed_scripts("Señor Muñoz")


def test_other_script_is_detected_and_page_reread(use_engines):
    engine = page_engine(LINES)
    use_engines(engine)
    page = OCR(strategy="fast", handwritten=False).read(white(800, 300)).pages[0]
    assert page.script == "cyrillic" and page.text.splitlines() == ["ул. Медникарска", "Mednikarska Str."]
    assert engine.calls == ["latin", "cyrillic"]  # pass 1 with the multi model and a re-read with the Cyrillic one


def test_known_language_or_disabled_detection_reads_as_is(use_engines):
    for options in ({"languages": ["es"]}, {"auto_script": False}):
        engine = page_engine(LINES)
        use_engines(engine)
        page = OCR(strategy="fast", handwritten=False, **options).read(white(800, 300)).pages[0]
        assert page.script == "latin" and page.text == "Mednikarska Str."
        assert engine.tried == []  # with ["es"] there is no other script to try: no extra time either


def test_known_languages_read_each_line_with_their_own_model(use_engines):
    # languages=["ru", "ar"]: the page is read with the Russian model (the first non-Latin one, as always) and
    # the Arabic line, which that model does not understand, is won by the Arabic one (it used to be lost).
    # Russian also brings the other Cyrillic model
    sign = {
        (100, 100, 700, 140): {"east_slavic": ("Русские пельмени", 0.95)},
        (100, 200, 700, 240): {"east_slavic": ("", 0.0), "arabic": ("كوكتيل كافيه", 0.9)},
        **{(100, 300 + 60 * i, 700, 340 + 60 * i): {"east_slavic": (f"Строка номер {i}", 0.95)} for i in range(3)},
    }
    engine = page_engine(sign)
    use_engines(engine)
    page = OCR(strategy="fast", handwritten=False, languages=["ru", "ar"]).read(white(800, 600)).pages[0]
    assert page.script == "east_slavic" and page.text.splitlines()[:2] == ["Русские пельмени", "كوكتيل كافيه"]
    assert engine.calls == ["east_slavic"] and engine.tried == ["arabic", "cyrillic"]


def test_detection_only_among_the_chosen_languages(use_engines):
    # detect_languages: only the models of its scripts are tried (here, the Cyrillic one of Bulgarian)
    engine = page_engine(LINES)
    use_engines(engine)
    page = OCR(strategy="fast", handwritten=False, detect_languages=["es", "bg"]).read(white(800, 300)).pages[0]
    assert page.script == "cyrillic" and page.text.splitlines() == ["ул. Медникарска", "Mednikarska Str."]
    assert engine.tried == ["cyrillic", "east_slavic"]  # both Cyrillic ones; without a limit, the 12 that are not Latin
    # without the sign's script among the candidates there is nothing to detect: it stays as the Latin model reads it
    engine = page_engine(LINES)
    use_engines(engine)
    page = OCR(strategy="fast", handwritten=False, detect_languages="es,el").read(white(800, 300)).pages[0]
    assert page.script == "latin" and page.text == "Mednikarska Str." and engine.tried == ["greek"]
    engine = page_engine(LINES)
    use_engines(engine)
    page = OCR(strategy="fast", handwritten=False, detect_languages=["es", "en"]).read(white(800, 300)).pages[0]
    assert page.script == "latin" and engine.tried == []  # English goes with the same model: nothing to try


def test_detection_without_latin_languages_starts_with_the_first(use_engines):
    # ["bg", "el"]: it starts with the Cyrillic model (not with the Latin one, which is not in the list)
    engine = page_engine(LINES)
    use_engines(engine)
    page = OCR(strategy="fast", handwritten=False, detect_languages=["bg", "el"]).read(white(800, 300)).pages[0]
    assert page.script == "cyrillic" and page.text.splitlines() == ["ул. Медникарска", "Mednikarska Str."]
    assert engine.calls == ["cyrillic"]  # a single read of the page


def test_a_latin_page_after_another_script_goes_back_to_latin(use_engines):
    # the previous page was Russian and this one starts with its model, which reads Latin text badly. The lines
    # are won by the multi model (Latin, Chinese and Japanese), which used to be tried as "Chinese": a Latin
    # reading measured with Chinese letters never won and the page stayed with the Russian model
    multi = {"latin", "chinese", "japanese"}
    latin_page = {
        (100, 100 + 60 * i, 700, 140 + 60 * i): {
            "east_slavic": ("Clоsеd еvеry", 0.5),
            "latin": ("Closed every Sunday", 0.97),
        }
        for i in range(3)
    }

    class Engine(page_engine(latin_page)):
        @classmethod
        def recognizer_for(cls, script):
            return "multi" if script in multi else script

        def _reading(self, box):
            return latin_page[box].get("latin" if self.script in multi else self.script, ("", 0.0))

    use_engines(Engine)
    read = OCR(strategy="fast", handwritten=False).pipeline.run(white(800, 300), script="east_slavic")
    assert read.script == "latin" and [b.text for row in read.rows for b in row] == ["Closed every Sunday"] * 3
    assert Engine.calls == ["east_slavic", "latin"] and Engine.tried[0] == "latin"  # the multi model, loaded as Latin


def test_latin_handwriting_does_not_switch_script(use_engines):
    # doubtful lines in Latin: the other models also read them with Latin letters (they are not of their script)
    scrawl = {
        (100, 100, 700, 140): {"latin": ("tonalero al degaxse Rimoy", 0.6), "cyrillic": ("tonaIero aI degaxse", 0.7)}
    }
    engine = page_engine(scrawl)
    use_engines(engine)
    page = OCR(strategy="fast", handwritten=False).read(white(800, 300)).pages[0]
    assert page.script == "latin" and page.text == "tonalero al degaxse Rimoy" and engine.calls == ["latin"]


def test_lines_of_a_distinct_script_keep_their_reading(use_engines):
    # bilingual sign: one line in Korean (few of the doubtful ones: the page is not re-read) and another in Latin
    sign = {
        (100, 100, 700, 140): {"latin": ("川量", 0.81), "korean": ("서울 평양", 0.98)},
        (100, 200, 700, 240): {"latin": ("Seoul Pyeongyang", 0.99), "korean": ("Seoul Pyeongyang", 0.95)},
        **{(100, 300 + 60 * i, 700, 340 + 60 * i): {"latin": ("blurry", 0.5)} for i in range(4)},
    }
    use_engines(page_engine(sign))
    page = OCR(strategy="fast", handwritten=False).read(white(800, 600)).pages[0]
    assert page.text.splitlines()[:2] == ["서울 평양", "Seoul Pyeongyang"]
    assert page.lines[0].alternatives == ["川量"]  # what the multi model read is not lost


def test_lookalike_scripts_decided_by_their_own_letters(use_engines):
    # the page becomes Cyrillic; "МЕНЮ" (thanks to its Ю, which does not exist in Latin) beats the Latin
    # reading "MEHIO" even though that one was more confident. "PECTOPAH" / "РЕСТОРАН" cannot be decided: it stays
    menu = {
        (100, 100, 700, 140): {"latin": ("", 0.0), "cyrillic": ("ул. Медникарска", 0.85)},
        (100, 200, 300, 240): {"latin": ("MEHIO", 0.99), "cyrillic": ("МЕНЮ", 0.9)},
        (100, 300, 700, 340): {"latin": ("Cocktail Cafe", 0.99), "cyrillic": ("Cocktail Cafe", 0.9)},
        (100, 400, 400, 440): {"latin": ("PECTOPAH", 0.99), "cyrillic": ("РЕСТОРАН", 0.9)},
    }
    use_engines(page_engine(menu))
    page = OCR(strategy="fast", handwritten=False).read(white(800, 500)).pages[0]
    assert page.script == "cyrillic" and page.text.splitlines() == [
        "ул. Медникарска",
        "МЕНЮ",
        "Cocktail Cafe",
        "PECTOPAH",
    ]
    # Latin page with a single Cyrillic line (not enough to re-read the page): the line keeps its Cyrillic
    # reading, which has letters of its own (before, between Latin and Cyrillic, never)
    faded = {(100, 100 + 60 * i, 700, 140 + 60 * i): {"latin": (f"blurry line number {i}", 0.6)} for i in range(5)}
    welcome = {(100, 500, 700, 540): {"latin": ("Ao6po noxanoBaTb", 0.7), "cyrillic": ("Добро пожаловать", 0.95)}}
    use_engines(page_engine(faded | welcome))
    page = OCR(strategy="fast", handwritten=False).read(white(800, 600)).pages[0]
    assert page.script == "latin" and page.text.splitlines()[-1] == "Добро пожаловать"
    assert page.lines[-1].alternatives == ["Ao6po noxanoBaTb"]


def test_a_few_odd_lines_do_not_reread_a_whole_page(use_engines):
    # Spanish form (XFUND): a line of dots and a real Kannada one among 8 Latin lines. Before, 1 of 2
    # doubtful lines was enough to re-read the whole page with the Kannada model ("೦9-08-2020")
    form = {(100, 100 + 50 * i, 700, 130 + 50 * i): {"latin": (f"Field of the form no. {i}", 0.98)} for i in range(8)}
    form[(100, 520, 700, 550)] = {"latin": ("09-08-2020", 0.97), "kannada": ("೦9-08-2020", 0.99)}
    form[(100, 570, 700, 600)] = {
        "latin": ("•", 0.4),
        "kannada": ("ಪ.....", 0.9),
    }  # dots: one letter is not a reading
    form[(100, 620, 700, 650)] = {"latin": ("", 0.0), "kannada": ("ಕನ್ನಡ ಪದ", 0.95)}
    engine = page_engine(form)
    use_engines(engine)
    page = OCR(strategy="fast", handwritten=False).read(white(800, 700)).pages[0]
    assert page.script == "latin" and engine.calls == ["latin"]  # the page is not re-read
    lines = page.text.splitlines()
    assert lines[8:] == ["09-08-2020", "•", "ಕನ್ನಡ ಪದ"]  # the Kannada line is: one by one


def test_document_pages_start_with_the_detected_script(use_engines, tmp_path):
    import pymupdf

    engine = page_engine(LINES)
    use_engines(engine)
    pdf = pymupdf.open()
    for _ in range(2):
        pdf.new_page().insert_image(pymupdf.Rect(50, 50, 550, 300), stream=_png())
    path = tmp_path / "ruso.pdf"
    pdf.save(path)
    doc = OCR(strategy="fast", handwritten=False).read(path)
    assert [p.script for p in doc.pages] == ["cyrillic", "cyrillic"]
    assert engine.calls == ["latin", "cyrillic", "cyrillic"]  # the second page starts with Cyrillic


@pytest.mark.parametrize("readable_upright", [True, False])
def test_upside_down_is_confirmed_by_the_recognizer(use_engines, readable_upright):
    """The orientation classifier says "upside down"; it is only rotated if reading it inverted is better."""
    box = (100, 100, 700, 140)

    class Engine(page_engine({box: {"latin": ("Invoice 2026", 0.95)}}, flipped=True)):
        def read_lines(self, image, boxes):
            upright = image[0, 0, 0] == 255  # the rotated image carries a mark in the corner
            good = upright == readable_upright
            return [RecognitionCandidate("Invoice 2026" if good else "9202 ǝɔᴉoʌuI", 0.95 if good else 0.4, "rapidocr")]

    use_engines(Engine)
    image = white(800, 300)
    image[-1, -1] = 0  # when rotated 180 degrees, the mark moves to the corner (0, 0)
    page = OCR(strategy="fast", handwritten=False, auto_script=False).read(image).pages[0]
    assert page.rotation == (0.0 if readable_upright else 180.0)


def _png() -> bytes:
    from ..helpers import png_bytes

    return png_bytes(800, 300, "")
