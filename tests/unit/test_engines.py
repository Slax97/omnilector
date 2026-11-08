# SPDX-License-Identifier: AGPL-3.0-or-later
"""Engines (without loading models), fusion by consensus and pipeline with fake engines."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest

from omnilector import OCR, list_engines
from omnilector.config import OCRConfig
from omnilector.core.fusion import drop_repeats, fuse_blocks, join_pieces
from omnilector.engines import EasyOCREngine, OCREngine, RapidOCREngine
from omnilector.engines.base import EngineInfo
from omnilector.engines.tesseract import lines_from_data
from omnilector.engines.trocr import sequence_confidence, token_logprobs
from omnilector.exceptions import ModelNotAvailableError
from omnilector.result import RecognitionCandidate

from .conftest import make_engine, reader, tb, white

# --------------------------------------------------------------------------- engines


def test_builtin_engines_registered():
    info = {e["name"]: e for e in list_engines()}
    assert {"rapidocr", "trocr", "tesseract", "easyocr"} <= set(info)
    assert info["rapidocr"]["available"] is True  # base dependency
    assert info["trocr"]["level"] == "line" and info["trocr"]["handwriting"] is True
    assert info["easyocr"]["kinds"] == ["photo"] and info["tesseract"]["kinds"] == ["scan"]
    assert all(e["reason"] for e in info.values() if not e["available"])  # if something is missing, it says what


def test_rapidocr_params(tmp_path):
    from rapidocr import LangRec, ModelType, OCRVersion

    cfg = OCRConfig(model_size="large", model_cache_dir=str(tmp_path), threads=2, det_box_thresh=0.25)
    p = RapidOCREngine(cfg).build_params()
    assert p["Global.text_score"] == 0.0  # RapidOCR no longer drops text with confidence < 0.5
    assert p["Det.box_thresh"] == 0.25  # the config threshold reaches the engine
    assert p["Det.model_type"] == ModelType.MEDIUM and p["Rec.model_type"] == ModelType.MEDIUM  # a real "large"
    assert p["Global.model_root_dir"] == str(tmp_path)
    assert p["EngineConfig.onnxruntime.intra_op_num_threads"] == 2
    assert p["EngineConfig.onnxruntime.inter_op_num_threads"] == 1
    assert p["Global.max_side_len"] == 4000
    korean = RapidOCREngine(OCRConfig(), script="korean").build_params()
    assert korean["Rec.lang_type"] == LangRec.KOREAN
    assert korean["Rec.ocr_version"] == OCRVersion("PP-OCRv5") and korean["Rec.model_type"] == ModelType.MOBILE
    assert korean["Global.use_cls"] is False and p["Global.use_cls"] is False  # without rotating upright lines
    assert RapidOCREngine(OCRConfig(strategy="fast")).build_params()["Global.max_side_len"] == 2000
    assert RapidOCREngine(OCRConfig(strategy="fast", max_image_side=3000)).max_side == 3000  # explicit settings win


def test_gpu_requested_without_gpu_falls_back_to_cpu(monkeypatch, caplog):
    import onnxruntime

    from omnilector.engines import base

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(base, "_warned_no_gpu", False)
    assert RapidOCREngine(OCRConfig(device="cuda")).build_params()["EngineConfig.onnxruntime.use_cuda"] is False
    assert "no GPU" in caplog.text
    assert RapidOCREngine(OCRConfig(device="cpu")).build_params()["EngineConfig.onnxruntime.use_cuda"] is False
    monkeypatch.setattr(
        onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert RapidOCREngine(OCRConfig(device="cuda")).build_params()["EngineConfig.onnxruntime.use_cuda"] is True


def test_rapidocr_sends_bgr_and_keeps_low_confidence(monkeypatch):
    seen = {}

    class Out:
        txts = ("hello", "debil")
        scores = (0.9, 0.12)
        boxes = np.array([[[0, 0], [10, 0], [10, 5], [0, 5]], [[0, 10], [10, 10], [10, 15], [0, 15]]], np.float32)

    def fake_rapidocr(img):
        seen["img"] = img
        return Out()

    engine = RapidOCREngine(OCRConfig())
    monkeypatch.setattr(engine, "load", lambda: fake_rapidocr)
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[..., 0] = 255  # pure red
    blocks = engine.read(rgb)
    assert seen["img"][0, 0].tolist() == [0, 0, 255]  # RapidOCR gets BGR
    assert [(b.text, b.confidence, b.box) for b in blocks] == [
        ("hello", 0.9, (0, 0, 10, 5)),
        ("debil", 0.12, (0, 10, 10, 15)),
    ]


def test_failed_model_load_is_not_retried_per_page():
    class Broken(OCREngine):
        info = EngineInfo(name="broken")
        builds = 0

        def _build(self):
            type(self).builds += 1
            raise OSError("no network")

    engine = Broken(OCRConfig())
    for _ in range(3):
        with pytest.raises(ModelNotAvailableError, match="no network"):
            engine.load()
    assert Broken.builds == 1


def _ctc(columns: list[tuple[str | None, float]], letters: str) -> tuple[np.ndarray, object]:
    """Recognizer output for a line: per column, the winning letter (None = the CTC blank) and the
    probability of the space. And RapidOCR's decoding with those letters."""
    ctc = pytest.importorskip("rapidocr.ch_ppocr_rec.utils").CTCLabelDecode(character=list(letters))
    preds = np.zeros((1, len(columns), len(ctc.character)), np.float32)
    for t, (letter, space) in enumerate(columns):
        preds[0, t, ctc.character.index(letter) if letter else 0] = 0.9
        preds[0, t, -1] = space
    return preds, ctc


def test_rapidocr_decoding_keeps_the_spaces_the_model_sees():
    from omnilector.engines.rapidocr import _SpacedDecoder

    def spaced(letters, columns):
        preds, ctc = _ctc(columns, letters)
        return _SpacedDecoder(ctc)(preds)[0][0][0], ctc(preds)[0][0][0]

    def word(letters, at, space=None):
        """The letters in those columns; space = (from, to, space probability) in between."""
        columns = [(None, 0.0)] * (at[-1] + 1)
        for letter, t in zip(letters, at):
            columns[t] = (letter, 0.0)
        if space:
            for t in range(space[0], space[1]):
                columns[t] = (None, space[2])
        return spaced(letters, columns)

    # "abc" and "123" farther apart than the other letters, with the space seen (0.1) but the blank winning
    assert word("abc123", [0, 2, 4, 8, 10, 12], (5, 8, 0.1)) == ("abc 123", "abc123")  # RapidOCR: "41001Sevilla"
    # no trace of the space, same as RapidOCR
    assert word("abc123", [0, 2, 4, 8, 10, 12], (5, 8, 0.01)) == ("abc123", "abc123")
    # letters at their usual pitch: even if the model hesitates, the word is not split ("AL IMENTOS")
    assert word("abcdef", [0, 2, 4, 6, 8, 10], (1, 2, 0.3)) == ("abcdef", "abcdef")
    # a short word is not enough to measure the pitch: "TIPO" does not come out "TI PO"
    assert word("tipo", [0, 2, 6, 8], (3, 6, 0.08)) == ("tipo", "tipo")
    # Chinese and Japanese do not separate words with spaces
    assert word("中文字句子", [0, 4, 6, 8, 10], (1, 4, 0.1)) == ("中文字句子", "中文字句子")


def test_rapidocr_decoding_is_rapidocr_plus_spaces():
    from omnilector.engines.rapidocr import _SpacedDecoder

    letters = "abcde1"
    _, ctc = _ctc([], letters)
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(5, 40, len(ctc.character))) * 3
    preds = (np.exp(logits) / np.exp(logits).sum(axis=2, keepdims=True)).astype(np.float32)
    theirs, ours = ctc(preds)[0], _SpacedDecoder(ctc)(preds)[0]
    for (text, conf), (spaced, spaced_conf) in zip(theirs, ours):
        assert spaced.replace(" ", "") == text.replace(" ", "")  # the same letters, in the same order
        assert spaced_conf == pytest.approx(conf, abs=1e-5)  # and the same confidence

    class Words:  # the boxes per word, as RapidOCR computes them
        character: ClassVar[list[str]] = list(ctc.character)

        def __call__(self, preds, return_word_box=False, **kwargs):
            return "rapidocr", return_word_box

    assert _SpacedDecoder(Words())(preds, True, max_wh_ratio=1.0) == ("rapidocr", True)


def test_rapidocr_recognizers_get_the_spaced_decoding_once():
    from omnilector.engines.rapidocr import _SpacedDecoder, _with_spaces

    class Recognizer:
        def __init__(self, decoder):
            self.postprocess_op = decoder

    _, ctc = _ctc([], "ab")
    rec = _with_spaces(_with_spaces(Recognizer(ctc)))
    assert isinstance(rec.postprocess_op, _SpacedDecoder) and rec.postprocess_op.decoder is ctc
    other = object()  # another decoding (another RapidOCR version): left alone
    assert _with_spaces(Recognizer(other)).postprocess_op is other


def _latin_reader(monkeypatch, engine, text, score):
    calls = []

    class Latin:
        def __call__(self, inputs):
            calls.append(len(inputs.img))
            return type("Out", (), {"txts": (text,) * len(inputs.img), "scores": (score,) * len(inputs.img)})

    monkeypatch.setattr(engine, "_latin_recognizer", lambda: Latin())
    return calls


def test_rapidocr_rereads_invented_ideograms_with_the_latin_model(monkeypatch):
    engine = RapidOCREngine(OCRConfig())
    calls = _latin_reader(monkeypatch, engine, "**** **** ****", 0.84)
    ticket = ["SUPERMERCADOS VALLE VERDE", "41001 Sevilla", "培塔培林 林塔培塔", "4821", "TOTAL 43,66", "THANK YOU"]
    scores = [0.98, 0.97, 0.43, 0.99, 0.99, 0.98]
    crops = [np.zeros((10, 40, 3), np.uint8)] * len(ticket)
    txts = list(ticket)
    engine._latin_verified(crops, txts, scores)
    assert txts[2] == "**** **** ****" and scores[2] == 0.84 and calls == [1]  # only that line
    # the Latin reading less confident than the multi model's: the multi one stays
    txts, low = list(ticket), [0.98, 0.97, 0.43, 0.99, 0.99, 0.98]
    _latin_reader(monkeypatch, engine, "trt", 0.3)
    engine._latin_verified(crops, txts, low)
    assert txts[2] == "培塔培林 林塔培塔"


def test_rapidocr_keeps_real_chinese_and_japanese(monkeypatch):
    engine = RapidOCREngine(OCRConfig())
    calls = _latin_reader(monkeypatch, engine, "MINATO", 0.9)
    # a sign in Japanese: many lines with ideograms, even if some are doubtful
    sign = ["ポイ捨て禁止", "MINATO CITY", "港区", "条例"]
    engine._latin_verified([np.zeros((10, 40, 3), np.uint8)] * 4, sign, [0.95, 0.97, 0.5, 0.92])
    # a confident Chinese name on a Latin page
    menu = ["Restaurante Gran Muralla", "龍", "Menu of the day", "Postre", "Cafe", "Total"]
    engine._latin_verified([np.zeros((10, 40, 3), np.uint8)] * 6, menu, [0.98, 0.93, 0.97, 0.98, 0.99, 0.99])
    assert sign[2] == "港区" and menu[1] == "龍" and calls == []
    # with Chinese or Japanese among the languages, it is not even checked; nor on pages of another script
    assert RapidOCREngine(OCRConfig(languages=["es"]))._ideograms_unexpected() is True
    assert RapidOCREngine(OCRConfig(languages=["es", "ja"]))._ideograms_unexpected() is False
    assert RapidOCREngine(OCRConfig(detect_languages=["zh", "en"]))._ideograms_unexpected() is False
    assert RapidOCREngine(OCRConfig(), script="cyrillic")._ideograms_unexpected() is False


def test_rapidocr_prepares_the_latin_model_for_offline_use(monkeypatch):
    from omnilector.engines.rapidocr import REC_ONLY_MB

    latin, cyrillic = RapidOCREngine(OCRConfig()), RapidOCREngine(OCRConfig(), script="cyrillic")
    prepared = []
    for engine in (latin, cyrillic):
        monkeypatch.setattr(engine, "load", lambda: "engine")
        monkeypatch.setattr(engine, "_latin_recognizer", lambda e=engine: prepared.append(e.script))
    assert latin.prepare() == "engine" and cyrillic.prepare() == "engine"
    assert prepared == ["latin"]  # load_models() / omnilector download leave it downloaded
    latin._latin = object()
    assert latin.resident_mb() == REC_ONLY_MB  # its memory counts


def test_tesseract_lines_from_data():
    data = {
        "text": ["", "Total", "45,67", "EUR", "Thanks"],
        "conf": [-1, 96, 90, 93, 80],
        "block_num": [1, 1, 1, 1, 2],
        "par_num": [1, 1, 1, 1, 1],
        "line_num": [1, 1, 1, 1, 1],
        "left": [0, 10, 80, 150, 10],
        "top": [0, 10, 12, 11, 60],
        "width": [0, 60, 60, 40, 90],
        "height": [0, 20, 20, 20, 22],
    }
    blocks = lines_from_data(data)
    assert [b.text for b in blocks] == ["Total 45,67 EUR", "Thanks"]
    assert blocks[0].box == (10, 10, 190, 32) and blocks[0].confidence == pytest.approx(0.93)


def test_tesseract_needs_binary(monkeypatch):
    from omnilector.engines import tesseract

    monkeypatch.setattr(tesseract, "missing_modules", lambda *a, **k: None)  # as if pytesseract were installed
    monkeypatch.setattr(tesseract, "find_tesseract", lambda: None)
    assert "binary" in tesseract.TesseractEngine.unavailable_reason()


def test_easyocr_adapter(monkeypatch):
    class FakeReader:
        def readtext(self, image, detail, paragraph):
            return [
                ([[0, 0], [50, 0], [50, 10], [0, 10]], "Hello", 0.8),
                ([[0, 20], [5, 20], [5, 25], [0, 25]], " ", 0.1),
            ]

    engine = EasyOCREngine(OCRConfig(languages=["es"]))
    monkeypatch.setattr(engine, "load", lambda: FakeReader())
    assert [(b.text, b.box, b.engine) for b in engine.read(white(60, 30))] == [("Hello", (0, 0, 50, 10), "easyocr")]


def test_trocr_confidence_helpers():
    assert sequence_confidence([math.log(0.9), math.log(0.8)]) == pytest.approx(math.sqrt(0.72))
    assert sequence_confidence([]) == 0.0
    # cut at the end of sequence: what comes after is padding
    assert token_logprobs([5, 7, 2, 1, 1], [-0.1, -0.2, -0.05, -9.0, -9.0], eos_id=2) == [-0.1, -0.2, -0.05]


# --------------------------------------------------------------------------- fusion


def test_fusion_consensus_between_engines():
    fused = fuse_blocks(
        [
            tb("2124 CZI", (10, 10, 110, 30), 0.95, "easyocr"),
            tb("2124 CZT", (11, 10, 111, 31), 0.90, "rapidocr"),
            tb("2124 CZT", (9, 9, 109, 30), 0.60, "tesseract"),
        ]
    )
    assert len(fused) == 1 and fused[0].text == "2124 CZT"  # 2 engines agreeing > 1 more confident
    assert fused[0].confidence == pytest.approx(0.90)
    assert {(c.text, c.engine) for c in fused[0].candidates} == {
        ("2124 CZI", "easyocr"),
        ("2124 CZT", "rapidocr"),
        ("2124 CZT", "tesseract"),
    }


def test_fusion_keeps_neighbours_and_absorbs_contained_reads():
    assert sorted(b.text for b in fuse_blocks([tb("123", (0, 0, 30, 10), 0.9), tb("456", (34, 0, 64, 10), 0.9)])) == [
        "123",
        "456",
    ]
    # what is contained in another reading goes to its candidates: neither a separate line nor lost
    (line,) = fuse_blocks([tb("Total 45,67", (0, 0, 200, 20), 0.8), tb("Total", (2, 1, 60, 19), 0.99)])
    assert line.text == "Total 45,67" and "Total" in {c.text for c in line.candidates}
    (line,) = fuse_blocks([tb("SCANNED TICKET", (270, 310, 1268, 483), 0.99), tb("1", (805, 363, 827, 369), 0.58)])
    assert line.text == "SCANNED TICKET"  # piece of a letter read separately (seen in a real PDF)
    # the tight box of another engine inside RapidOCR's (IoU < 0.5): a single line
    (line,) = fuse_blocks([tb("2124 CZT", (0, 0, 400, 100), 0.9), tb("2124 CZI", (20, 30, 380, 75), 0.8, "tesseract")])
    assert line.text == "2124 CZT" and ("2124 CZI", "tesseract") in {(c.text, c.engine) for c in line.candidates}
    # small text next to it (not inside) is kept
    fused = fuse_blocks([tb("SCANNED TICKET", (270, 310, 1268, 483), 0.99), tb("1", (1300, 450, 1320, 480), 0.58)])
    assert sorted(b.text for b in fused) == ["1", "SCANNED TICKET"]


def test_drop_repeats_keeps_each_stretch_of_a_line_once():
    # the same line cut in different places by two engines: real case, a Catalan register of 1762 px
    pieces = [
        tb("a de Antoni fabrer", (280, 20, 780, 160), 0.93),
        tb("fabrer guanter de", (570, 40, 1040, 150), 0.99, "easyocr"),
        tb("er guanter de Bag.a 9 de Mathiana defucts", (700, 0, 1760, 170), 0.94),
    ]
    kept = drop_repeats(pieces)
    assert sorted(b.text for b in kept) == ["a de Antoni fabrer", "er guanter de Bag.a 9 de Mathiana defucts"]
    assert "fabrer guanter de" in {c.text for b in kept for c in b.candidates}  # nothing is lost
    # a piece inside a line that only partly fills it vertically (too little to be "contained")
    line, end = (
        tb("dit dia rebere de Viala pages del regne de frança habi-", (40, 22, 1650, 163), 0.98),
        tb("pages del regne de frança habi=", (970, 0, 1675, 174), 0.98),
    )
    assert [b.text for b in drop_repeats([line, end])] == [line.text]


def test_drop_repeats_does_not_touch_what_is_really_different():
    # neighbours in a row
    row = [tb("TOTAL", (0, 0, 100, 20), 0.9), tb("48,22", (110, 0, 200, 20), 0.9)]
    assert len(drop_repeats(row)) == 2
    # slightly overlapping pieces: each one still has its own stretch
    row = [tb("Queda de pie", (0, 0, 200, 20), 0.9), tb("de pie la casa", (120, 0, 340, 20), 0.9)]
    assert len(drop_repeats(row)) == 2
    # another line whose box overlaps it (a skewed page, boxes that are too tall): different text stays
    lines = [tb("Primera linea de la carta", (0, 0, 600, 40), 0.9), tb("xyz 123 otro renglon", (20, 8, 580, 44), 0.9)]
    assert len(drop_repeats(lines)) == 2
    # a line inside the box of a whole paragraph is not its row
    para = [tb("ilegible", (0, 0, 600, 300), 0.5), tb("ilegible", (10, 100, 590, 130), 0.9)]
    assert len(drop_repeats(para)) == 2
    # something nobody could read over what another one reads: it does not go to the line engine to repeat it
    empty = [tb("Texto que ya se ha leido", (0, 0, 600, 40), 0.9), tb("", (100, 2, 300, 38), 0.0)]
    assert [b.text for b in drop_repeats(empty)] == ["Texto que ya se ha leido"]
    # ...and if it is the largest one, it stays: the line engine may read it whole
    empty = [tb("", (0, 0, 600, 40), 0.0), tb("una parte", (100, 2, 300, 38), 0.9)]
    assert len(drop_repeats(empty)) == 2


def test_join_pieces_gives_the_line_engine_the_whole_line():
    pieces = [
        tb("d' assmame", (10, 10, 400, 90), 0.5),
        tb("habitation concerrant", (350, 5, 800, 95), 0.7),
        tb("concerant ce logement", (700, 10, 1200, 90), 0.6),
    ]
    (line,) = join_pieces(pieces, 0.85)
    assert line.box == (10, 5, 1200, 95) and line.text == "d' assmame habitation concerrant concerant ce logement"
    assert {c.text for c in line.candidates} == {b.text for b in pieces}  # nothing is lost
    # an empty zone that spans the pieces goes with them
    (line,) = join_pieces([tb("", (0, 0, 1000, 100), 0.0), tb("dit dia", (10, 10, 340, 90), 0.6)], 0.85)
    assert line.box == (0, 0, 1000, 100) and line.text == "dit dia"
    # right to left, in its order
    (line,) = join_pieces([tb("عالم", (0, 0, 200, 40), 0.5), tb("مرحبا", (150, 0, 400, 40), 0.5)], 0.85)
    assert line.text == "مرحبا عالم"
    # a sure piece of a few letters does not make a doubtful line look sure: the line engine still has to beat it
    (line,) = join_pieces([tb("kay", (0, 20, 135, 160), 0.8), tb("fay Comp toy stows", (20, 20, 1700, 155), 0.5)], 0.85)
    assert round(line.confidence, 2) == 0.54


def test_join_pieces_only_joins_doubtful_pieces_of_one_line():
    # what is read with confidence is not touched
    confident = [tb("Total", (0, 0, 120, 20), 0.95), tb("45,67", (100, 0, 200, 20), 0.5)]
    assert sorted(b.text for b in join_pieces(confident, 0.85)) == ["45,67", "Total"]
    # separate pieces of a row, and different lines, stay as they are
    apart = [tb("uno", (0, 0, 100, 20), 0.5), tb("dos", (120, 0, 200, 20), 0.5), tb("tres", (0, 40, 200, 60), 0.5)]
    assert sorted(b.text for b in join_pieces(apart, 0.85)) == ["dos", "tres", "uno"]
    # a line inside the box of a whole paragraph is not its row
    para = [tb("", (0, 0, 600, 300), 0.0), tb("una linea", (10, 100, 590, 130), 0.5)]
    assert len(join_pieces(para, 0.85)) == 2
    # two slanted lines, one above the other: as wide as each other, but they only share part of their height
    slanted = [tb("la de abajo", (114, 1595, 1190, 1689), 0.49), tb("la de arriba", (122, 1561, 1181, 1643), 0.47)]
    assert sorted(b.text for b in join_pieces(slanted, 0.85)) == ["la de abajo", "la de arriba"]


# --------------------------------------------------------------------------- pipeline


def test_pipeline_falls_back_when_engine_fails(use_engines):
    def broken(image):
        raise RuntimeError("broken model")

    use_engines(
        make_engine("rapidocr", broken), make_engine("tesseract", reader(("hello", (0, 0, 50, 10), 0.9, "tesseract")))
    )
    doc = OCR(strategy="fast", handwritten=False).read(white())
    assert doc.text == "hello" and doc.engines == ["tesseract"]


def test_pipeline_ensemble_only_when_text_is_missing(use_engines):
    lines = [(f"line {i}", (10, 10 + 30 * i, 300, 35 + 30 * i), 0.95, "rapidocr") for i in range(8)]
    other = make_engine("easyocr", reader(("hidden", (100, 300, 300, 330), 0.7, "easyocr"), width=400))
    use_engines(make_engine("rapidocr", reader(*lines, width=400)), other)
    doc = OCR(handwritten=False).read(white())
    assert other.calls == [] and doc.engines == ["rapidocr"]  # nothing lost: no CPU spent on another engine

    weak = make_engine("rapidocr", reader(("x", (0, 0, 10, 10), 0.3, "rapidocr"), width=400))
    use_engines(weak, other)
    doc = OCR(handwritten=False).read(white())
    assert "hidden" in doc.text and doc.engines == ["rapidocr", "easyocr"]


def inked(w: int, h: int) -> np.ndarray:
    """A page that is not blank: thin strokes, like letters."""
    page = white(w, h)
    page[20 : h - 20 : 9, 20 : w - 20] = 0
    return page


def test_extra_passes_only_add_new_zones_that_are_text(use_engines):
    # a sign: few lines -> second pass. The variants find noise around, and also text pass 1 did not see
    first = reader(("FYLASION", (100, 100, 300, 130), 0.99, "rapidocr"), width=800)
    others = reader(
        ("FYLASION", (100, 100, 300, 130), 0.98, "rapidocr"),  # again the same zone
        ("ASION", (400, 100, 600, 130), 0.89, "rapidocr"),  # text nobody had seen
        ("2X7", (100, 300, 300, 380), 0.46, "rapidocr"),  # noise: a drawing, read as something else in each variant
        ("©", (500, 300, 540, 340), 0.19, "rapidocr"),
        ("L", (0, 300, 40, 340), 0.77, "rapidocr"),
        ("", (300, 300, 380, 380), 0.0, "rapidocr"),
        width=800,
    )
    engine = make_engine("rapidocr", lambda image: others(image) if engine.calls[1:] else first(image))
    use_engines(engine)
    doc = OCR(handwritten=False).read(white(800, 500))
    assert sorted(line.text for line in doc.lines) == ["ASION", "FYLASION"]


def test_extra_passes_keep_whatever_there_is_if_pass_one_read_nothing(use_engines):
    blurred = reader(("borroso", (100, 100, 300, 130), 0.4, "rapidocr"), width=800)
    engine = make_engine("rapidocr", lambda image: blurred(image) if engine.calls[1:] else [])
    use_engines(engine)
    assert OCR(handwritten=False).read(inked(800, 500)).text == "borroso"


def test_line_engine_reads_every_line_in_its_print_languages(use_engines):
    seen = []

    def lines(boxes):
        seen.extend(boxes)
        return [RecognitionCandidate("سطر صحيح", 0.95, "kraken") for _ in boxes]

    page_lines = [(f"خطأ {i}", (10, 10 + 40 * i, 400, 40 + 40 * i), 0.98, "rapidocr") for i in range(4)]
    use_engines(
        make_engine("rapidocr", reader(*page_lines, width=500), scripts=frozenset({"arabic"})),
        make_engine(
            "kraken",
            level="line",
            handwriting=True,
            scripts=frozenset({"arabic"}),
            print_languages=frozenset({"ar"}),
            read_lines=lines,
        ),
    )
    doc = OCR(languages=["ar"]).read(white(500, 300))
    assert len(seen) == 4 and {line.text for line in doc.lines} == {"سطر صحيح"}  # confident lines too
    assert not any(line.handwritten for line in doc.lines)  # re-reading print does not make it handwriting
    assert doc.engines == ["kraken"]
    assert "خطأ 0" in {alt for line in doc.lines for alt in line.alternatives}  # the other reading is kept
    # the same engine in another language: only what is doubtful
    seen.clear()
    assert OCR(languages=["fa"]).read(white(500, 300)).text.count("خطأ") == 4 and seen == []


def test_line_engine_reads_print_without_handwriting_but_not_in_a_quick_pass(use_engines):
    seen = []

    def lines(boxes):
        seen.extend(boxes)
        return [RecognitionCandidate("سطر صحيح", 0.95, "kraken") for _ in boxes]

    use_engines(
        make_engine(
            "rapidocr", reader(("خطأ", (10, 10, 400, 40), 0.5, "rapidocr"), width=500), scripts=frozenset({"arabic"})
        ),
        make_engine(
            "kraken",
            level="line",
            handwriting=True,
            scripts=frozenset({"arabic"}),
            print_languages=frozenset({"ar"}),
            read_lines=lines,
        ),
    )
    # a doubtful line, with handwriting looked for: it may be handwriting
    assert OCR(languages=["ar"]).read(white(500, 300)).lines[0].handwritten
    for options in ({"handwritten": False}, {"strategy": "balanced"}):
        doc = OCR(languages=["ar"], **options).read(white(500, 300))
        assert doc.text == "سطر صحيح" and not doc.lines[0].handwritten  # print, read by the best engine for it
    seen.clear()
    assert OCR(languages=["ar"], strategy="fast").read(white(500, 300)).text == "خطأ" and seen == []


def test_engine_weak_in_the_language_does_not_join_the_consensus(use_engines):
    # Vietnamese, a few lines: maximum_recall asks the other engines too
    tess = make_engine("tesseract", reader(("những quyền", (10, 10, 280, 40), 0.9, "tesseract")), cost=2)
    easy = make_engine("easyocr", reader(("những quyền", (10, 10, 280, 40), 0.85, "easyocr")), cost=3)
    rapid = make_engine(
        "rapidocr", reader(("nhung quyen", (10, 10, 280, 40), 0.99, "rapidocr")), weak_languages={"vi": 0.8}
    )
    use_engines(tess, easy, rapid)
    doc = OCR(languages=["vi"]).read(white(300, 200))
    assert doc.text == "những quyền" and easy.calls and not rapid.calls  # its sure reading without diacritics won


def test_line_engine_reads_the_print_of_its_languages_even_when_it_doubts(use_engines):
    # on two old Persian books, kraken doubting still read better than RapidOCR sure: 51-52 % against 74-79 %
    use_engines(
        make_engine(
            "rapidocr",
            reader(("الح ل ل لعلمين", (10, 10, 400, 40), 0.7, "rapidocr"), width=500),
            scripts=frozenset({"arabic"}),
        ),
        make_engine(
            "kraken",
            level="line",
            handwriting=True,
            scripts=frozenset({"arabic"}),
            print_languages=frozenset({"fa"}),
            read_lines=lambda boxes: [RecognitionCandidate("الحمد لله رب العالمين", 0.3, "kraken") for _ in boxes],
        ),
    )
    doc = OCR(languages=["fa"], strategy="accurate").read(white(500, 300))
    assert doc.text == "الحمد لله رب العالمين" and "الح ل ل لعلمين" in doc.lines[0].alternatives


def test_detector_finds_the_lines_and_the_line_engine_reads_them(use_engines):
    boxes = [(10, 10, 400, 40), (10, 60, 400, 90), (10, 110, 40, 140)]  # the last one, too short for handwriting
    seen = []

    def lines(found):
        seen.extend(found)
        return [RecognitionCandidate(f"שורה {i}", 0.2 if i == 1 else 0.9, "kraken") for i in range(len(found))]

    def detected(image):
        return [tb("", box, 0.0) for box in boxes]

    page = make_engine("tesseract", scripts=frozenset({"hebrew"}))
    rapid = make_engine("rapidocr", scripts=frozenset({"latin"}), detect_lines=detected)
    kraken = make_engine(
        "kraken",
        level="line",
        handwriting=True,
        scripts=frozenset({"hebrew"}),
        print_languages=frozenset({"he"}),
        read_lines=lines,
    )
    use_engines(page, rapid, kraken)
    doc = OCR(languages=["he"]).read(inked(500, 300))
    # also the short one and what it barely reads: nothing else would read them. Doubtful on average, it was also
    # read upside down, and stays as it was: that way it does not read better
    assert sorted(seen[:3]) == sorted(boxes) and len(doc.lines) == 3 and doc.pages[0].rotation == 0
    assert doc.engines == ["kraken"] and page.calls == [] and rapid.calls == []  # no extra passes
    assert not any(line.handwritten for line in doc.lines)


def test_detector_failure_goes_to_the_page_engines(use_engines):
    def broken(image):
        raise RuntimeError("no detector")

    use_engines(
        make_engine("rapidocr", scripts=frozenset({"latin"}), detect_lines=broken),
        make_engine(
            "tesseract",
            reader(("שלום עולם", (10, 10, 400, 40), 0.95, "tesseract"), width=500),
            scripts=frozenset({"hebrew"}),
        ),
        make_engine(
            "kraken",
            level="line",
            handwriting=True,
            scripts=frozenset({"hebrew"}),
            print_languages=frozenset({"he"}),
            read_lines=lambda boxes: [None] * len(boxes),
        ),
    )
    assert OCR(languages=["he"], strategy="accurate").read(white(500, 300)).text == "שלום עולם"


def test_page_engines_read_if_the_line_engine_cannot_after_the_detector(use_engines):
    def no_model(boxes):
        raise RuntimeError("no network")

    use_engines(
        make_engine(
            "rapidocr", scripts=frozenset({"latin"}), detect_lines=lambda image: [tb("", (10, 10, 400, 40), 0.0)]
        ),
        make_engine(
            "tesseract",
            reader(("שלום עולם", (10, 10, 400, 40), 0.95, "tesseract"), width=500),
            scripts=frozenset({"hebrew"}),
        ),
        make_engine(
            "kraken",
            level="line",
            handwriting=True,
            scripts=frozenset({"hebrew"}),
            print_languages=frozenset({"he"}),
            read_lines=no_model,
        ),
    )
    assert OCR(languages=["he"]).read(inked(500, 300)).text == "שלום עולם"


def test_weak_engine_reads_after_all_if_the_others_barely_trust_their_reading(use_engines):
    def engines(confidence):
        tesseract = make_engine(
            "tesseract",
            reader(
                *[(f"tesseract {i}", (10, 10 + 30 * i, 280, 35 + 30 * i), confidence, "tesseract") for i in range(4)]
            ),
            cost=2,
        )
        rapid = make_engine(
            "rapidocr",
            reader(*[(f"rapidocr {i}", (10, 10 + 30 * i, 280, 35 + 30 * i), 0.9, "rapidocr") for i in range(4)]),
            weak_languages={"vi": 0.8},
        )
        use_engines(rapid, tesseract)
        return rapid, tesseract

    rapid, _ = engines(0.95)  # modern text: the weak one is not even loaded
    assert OCR(languages=["vi"], handwritten=False).read(white(300, 200)).engines == ["tesseract"] and rapid.calls == []
    engines(0.6)  # an old print
    assert OCR(languages=["vi"], handwritten=False).read(white(300, 200)).engines == ["rapidocr"]
    # the last one breaks: better the doubtful reading than nothing
    use_engines(
        make_engine("rapidocr", weak_languages={"vi": 0.8}, read=lambda image: 1 / 0),
        make_engine("tesseract", reader(("con dudas", (10, 10, 280, 35), 0.5, "tesseract")), cost=2),
    )
    assert OCR(languages=["vi"], handwritten=False, strategy="accurate").read(white(300, 200)).text == "con dudas"


def test_a_line_read_whole_absorbs_the_pieces_that_repeat_it(use_engines):
    # nobody reads the whole line (empty zone) and the pieces are doubtful: kraken reads all of them
    readings = {
        (0, 0, 1000, 100): "dit dia rebere de Pere Mas corredor de coll",
        (10, 10, 340, 90): "dit dia rebere de",
        (320, 10, 700, 90): "Pere Mas corredor",
        (680, 10, 990, 90): "corredor de coll",
    }
    page_lines = [("", (0, 0, 1000, 100), 0.0, "rapidocr")] + [
        (t[:8], b, 0.6, "rapidocr") for b, t in readings.items() if b[0] or b[2] != 1000
    ]

    def lines(boxes):
        return [RecognitionCandidate(readings[tuple(b)], 0.95, "kraken") for b in boxes]

    use_engines(
        make_engine("rapidocr", reader(*page_lines, width=1100)),
        make_engine("kraken", level="line", handwriting=True, read_lines=lines),
    )
    doc = OCR(languages=["ca"], strategy="accurate").read(white(1100, 200))
    assert doc.text == "dit dia rebere de Pere Mas corredor de coll"


def test_tesseract_uses_the_data_of_the_language_under_another_name_or_says_it_is_missing(monkeypatch, caplog):
    from types import SimpleNamespace

    from omnilector.engines import tesseract

    installed = ["eng", "fil", "spa"]
    fake = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd=""), get_languages=lambda config: installed)
    monkeypatch.setattr(tesseract, "require", lambda *a, **k: fake)
    monkeypatch.setattr(tesseract, "find_tesseract", lambda: "tesseract")

    def languages(script, *codes):
        return tesseract.TesseractEngine(OCRConfig(languages=list(codes)), script).load()[1]

    assert languages("latin", "tl") == "fil+eng"  # Debian calls Tagalog fil
    assert languages("latin", "es") == "spa+eng"
    assert languages("latin", "de") == "eng"  # its Latin in English, saying so
    assert "tesseract-ocr-deu" in caplog.text
    engine = tesseract.TesseractEngine(OCRConfig(languages=["he"]), "hebrew")  # another script: it would be garbage
    with pytest.raises(ModelNotAvailableError, match="tesseract-ocr-heb"):
        engine.load()
    assert engine.prepare() is None  # preparing a machine: it is said and the rest gets ready
    installed.append("chi_tra")
    assert languages("chinese", "zh-hant") == "chi_tra+eng"


def test_blank_page_does_not_escalate(use_engines):
    page_engine, other = make_engine("rapidocr"), make_engine("easyocr")
    use_engines(page_engine, other)
    doc = OCR(handwritten=False).read(white())
    assert doc.text == "" and len(page_engine.calls) == 1 and other.calls == []  # no variants or consensus


def test_pipeline_line_engine_only_on_doubtful_lines(use_engines):
    seen = []

    def lines(boxes):
        seen.append(list(boxes))
        return [RecognitionCandidate("scribble", 0.77, "trocr") for _ in boxes]

    page_reader = reader(
        ("Clear print", (10, 10, 200, 30), 0.98, "rapidocr"), ("scribb1e", (10, 50, 200, 70), 0.41, "rapidocr")
    )
    use_engines(
        make_engine("rapidocr", page_reader),
        make_engine("trocr", level="line", handwriting=True, scripts=frozenset({"latin"}), read_lines=lines),
    )
    doc = OCR(strategy="accurate").read(white(300, 100))
    assert seen == [[(10, 50, 200, 70)]]  # only the doubtful line, once (after fusing the passes)
    handwritten = next(line for line in doc.lines if line.text == "scribble")
    assert handwritten.handwritten and handwritten.engines == ["trocr"] and handwritten.alternatives == ["scribb1e"]
    assert doc.engines == ["rapidocr", "trocr"] and doc.text == "Clear print\nscribble"


def test_pipeline_upscales_only_when_it_is_worth_it(use_engines):
    # 900 px and the engine only processes 1000: enlarging x1.11 would repeat pass 1
    engine = make_engine("rapidocr", reader(("a", (0, 0, 10, 10), 0.3, "rapidocr"), width=900), max_side=1000)
    use_engines(engine)
    OCR(handwritten=False).read(white(900, 500))
    assert {img.shape[1] for img in engine.calls} == {900}
    # with 1400 px, x1.5 fits (1350) and x2.0 would stay at x1.56: almost the same, read once
    engine = make_engine("rapidocr", reader(("a", (0, 0, 10, 10), 0.3, "rapidocr"), width=900), max_side=1400)
    use_engines(engine)
    OCR(handwritten=False).read(white(900, 500))
    assert {img.shape[1] for img in engine.calls} == {900, 1350}
    # without a limit the strategy's scales are read (x1.5 and x2.0 differ enough)
    engine = make_engine("rapidocr", reader(("a", (0, 0, 10, 10), 0.3, "rapidocr"), width=900))
    use_engines(engine)
    OCR(handwritten=False).read(white(900, 500))
    assert {img.shape[1] for img in engine.calls} == {900, 1350, 1800}
