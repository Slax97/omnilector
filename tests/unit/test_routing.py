# SPDX-License-Identifier: AGPL-3.0-or-later
"""Languages, situation analyzer and router (with fake engines)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from omnilector.config import OCRConfig
from omnilector.core.strategies import get_strategy
from omnilector.exceptions import EngineNotAvailableError
from omnilector.routing import (
    PageProfile,
    Router,
    analyze,
    main_script,
    normalize_languages,
)
from omnilector.routing.languages import (
    EASYOCR_SCRIPTS,
    TESSERACT_SCRIPTS,
    detect_start,
    easyocr_codes,
    legacy_encodings,
    mixed_scripts,
    script_share,
    scripts_to_detect,
    tesseract_codes,
)

from .conftest import make_engine, white

SCAN, PHOTO = frozenset({"scan"}), frozenset({"photo"})


def text_image(noise: float = 0.0, gradient: bool = False, w: int = 800, h: int = 600) -> np.ndarray:
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=28)
    for i in range(h // 55):
        draw.text((40, 30 + 55 * i), f"Line of text number {i} total 45,67", fill="black", font=font)
    arr = np.array(img).astype(np.float32)
    if gradient:  # side shadow, like a phone photo
        arr *= np.linspace(0.35, 1.0, w)[None, :, None]
    if noise:  # sensor noise
        arr += np.random.default_rng(0).normal(0, noise, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def router(strategy: str = "maximum_recall", **cfg) -> Router:
    return Router(OCRConfig(strategy=strategy, **cfg), get_strategy(strategy))


def names(plan) -> list[str]:
    return [str(e) for e in plan.engines]


def test_languages():
    assert normalize_languages("es, EN,spa") == ("es", "en")
    assert normalize_languages(["chi_sim", "jpn"]) == ("zh", "ja")
    with pytest.raises(ValueError):
        normalize_languages(["klingon"])
    with pytest.raises(ValueError):
        OCRConfig(languages=["xx"])
    assert main_script(("es", "en")) == "latin"
    assert main_script(("es", "ru")) == "east_slavic"
    assert main_script(("ar",)) == "arabic"
    # detect_languages: starts with Latin if any language is Latin; otherwise, with the first one's script
    assert detect_start(()) == "latin" and detect_start(("ru", "es")) == "latin"
    assert detect_start(("ru", "ar")) == "east_slavic" and detect_start(("zh", "ko")) == "chinese"
    assert scripts_to_detect(("es", "ar")) == {"latin", "arabic"}
    assert scripts_to_detect(("uk",)) == scripts_to_detect(("bg",)) == {"east_slavic", "cyrillic"}  # same letters
    assert tesseract_codes(("es",), "latin") == ["spa", "eng"]
    assert tesseract_codes((), "latin") == ["spa", "eng"]  # without languages, Spanish first (PRIORITY)
    assert tesseract_codes((), "arabic") == ["ara", "eng"]
    assert easyocr_codes((), "latin") == ["es", "en", "fr", "de"]  # without languages: its Latin model, with accents
    assert easyocr_codes(("en",), "latin") == ["en"]
    assert easyocr_codes(("es", "ca"), "latin") == ["es", "fr", "en"]  # Catalan: close languages
    assert easyocr_codes(("zh",), "chinese") == ["ch_sim", "en"]
    assert easyocr_codes(("es",), "east_slavic") == ["ru", "en"]  # models are not mixed
    assert legacy_encodings(("pl", "ru"))[:3] == ["cp1250", "cp1251", "cp1252"]
    assert legacy_encodings(())[0] == "cp1252"


def test_languages_without_a_rapidocr_model():
    assert normalize_languages("he,ben,mya,kat,hye,khm,lao,amh,pan,guj,mal") == (
        "he",
        "bn",
        "my",
        "ka",
        "hy",
        "km",
        "lo",
        "am",
        "pa",
        "gu",
        "ml",
    )
    assert main_script(("he", "en")) == "hebrew" and main_script(("bn",)) == "bengali"
    assert tesseract_codes(("he",), "hebrew") == ["heb", "eng"]
    assert tesseract_codes((), "ethiopic") == ["amh", "eng"]
    assert easyocr_codes(("bn",), "bengali") == ["bn", "en"]
    assert script_share("שלום", ["hebrew"]) == 1.0 and script_share("გამარჯობა", ["georgian"]) == 1.0
    assert mixed_scripts("שלוםabc") and not mixed_scripts("שלום abc")
    assert legacy_encodings(("he",))[0] == "cp1255"
    # the models that do not exist or do not load are not offered
    assert {"bengali"} <= EASYOCR_SCRIPTS and "tamil" not in EASYOCR_SCRIPTS and "hebrew" not in EASYOCR_SCRIPTS
    assert "hebrew" in TESSERACT_SCRIPTS


def test_traditional_chinese_is_its_own_language():
    assert normalize_languages(["zh-Hant", "chi_tra", "ZH_TW", "zh-hk"]) == ("zh-hant",)
    assert normalize_languages(["chi_sim", "zh-CN", "zh-Hans"]) == ("zh",)
    assert main_script(("zh-hant",)) == "chinese"
    assert tesseract_codes(("zh-hant",), "chinese") == ["chi_tra", "eng"]
    assert easyocr_codes(("zh-hant",), "chinese") == ["ch_tra", "en"]
    assert easyocr_codes(("zh", "zh-hant"), "chinese") == ["ch_sim", "en"]  # two models that cannot go together
    assert legacy_encodings(("zh-hant",))[0] == "big5"


def test_analyzer_kinds_and_ink():
    assert analyze(text_image()).kind == "scan"  # clean render (screenshot, rasterized PDF)
    assert analyze(text_image(noise=4)).kind == "scan"  # paper with sensor noise
    assert analyze(text_image(noise=4, w=2480, h=3508)).kind == "scan"  # A4 at 300 DPI
    assert analyze(text_image(noise=6, gradient=True)).kind == "photo"  # shadow
    assert analyze(text_image(), hint="photo").kind == "photo"  # the loader's hint (EXIF) wins
    profile = analyze(text_image(), languages=("ru",))
    assert profile.script == "east_slavic" and profile.ink > 0.001
    assert analyze(white(800, 600)).ink == 0.0  # blank page


def test_router_prefers_engines_declared_for_the_kind(use_engines):
    use_engines(
        make_engine("rapidocr"),
        make_engine("tesseract", kinds=SCAN, cost=2),
        make_engine("easyocr", kinds=PHOTO, cost=3),
    )
    assert names(router().plan(PageProfile(kind="photo"))) == ["rapidocr", "easyocr", "tesseract"]
    assert names(router().plan(PageProfile(kind="scan"))) == ["rapidocr", "tesseract", "easyocr"]
    # an engine of your own that declares "scan" and is cheap competes without touching the router
    use_engines(make_engine("rapidocr", cost=2), make_engine("mio", kinds=SCAN, cost=1))
    assert names(router().plan(PageProfile(kind="scan"))) == ["mio", "rapidocr"]


def test_router_skips_unavailable_and_ensemble_by_strategy(use_engines):
    use_engines(
        make_engine("rapidocr"),
        make_engine("tesseract", reason="binary missing"),
        make_engine("easyocr", kinds=PHOTO, cost=3),
    )
    plan = router().plan(PageProfile(kind="scan"))
    assert names(plan) == ["rapidocr", "easyocr"]  # tesseract not installed: not even tried
    assert plan.ensemble is True
    assert router("balanced").plan(PageProfile(kind="scan")).ensemble is False


def test_router_filters_by_script(use_engines):
    use_engines(
        make_engine("rapidocr", scripts=frozenset({"latin", "arabic"})),
        make_engine("easyocr", scripts=frozenset({"latin"})),
    )
    assert names(router().plan(PageProfile(script="arabic"))) == ["rapidocr:arabic"]
    use_engines(make_engine("easyocr", scripts=frozenset({"latin"})))
    with pytest.raises(EngineNotAvailableError, match="arabic"):
        router().plan(PageProfile(script="arabic"))


def test_router_forced_engine(use_engines):
    use_engines(make_engine("rapidocr"), make_engine("tesseract", reason="pytesseract missing"))
    assert names(router(engine="rapidocr").plan(PageProfile(kind="photo"))) == ["rapidocr"]
    with pytest.raises(EngineNotAvailableError, match="pytesseract"):
        router(engine="tesseract").plan(PageProfile())
    with pytest.raises(EngineNotAvailableError, match="Unknown engine"):
        router(engine="nope").plan(PageProfile())
    # forcing an engine without a model for the script would only give letters of another one
    use_engines(make_engine("rapidocr", scripts=frozenset({"latin"})))
    with pytest.raises(EngineNotAvailableError, match="does not read the script 'hebrew'"):
        router(engine="rapidocr").plan(PageProfile(script="hebrew"))


def test_router_without_an_engine_says_what_to_install(use_engines):
    use_engines(
        make_engine("rapidocr", scripts=frozenset({"latin"})),
        make_engine("tesseract", scripts=frozenset({"latin", "hebrew"}), reason="binary missing", extra="tesseract"),
    )
    with pytest.raises(EngineNotAvailableError, match=r"'hebrew'.*omnilector\[tesseract\]"):
        router().plan(PageProfile(script="hebrew"))


def test_router_weak_engine_goes_behind_the_others_in_that_language(use_engines):
    use_engines(
        make_engine("rapidocr", weak_languages={"kn": 0.0}),
        make_engine("tesseract", kinds=SCAN, cost=2),
        make_engine("easyocr", kinds=PHOTO, cost=3),
    )
    scan, photo = PageProfile(kind="scan", script="kannada"), PageProfile(kind="photo", script="kannada")
    assert names(router(languages=["kn"]).plan(scan)) == ["tesseract:kannada", "easyocr:kannada", "rapidocr:kannada"]
    assert names(router(languages=["kn"]).plan(photo)) == ["easyocr:kannada", "tesseract:kannada", "rapidocr:kannada"]
    assert names(router(detect_languages=["kn"]).plan(scan))[0] == "tesseract:kannada"
    # another language of the same script, or without knowing the language: as always
    use_engines(
        make_engine("rapidocr", weak_languages={"vi": 0.0}),
        make_engine("tesseract", kinds=SCAN, cost=2),
    )
    assert names(router(languages=["es"]).plan(PageProfile()))[0] == "rapidocr"
    assert names(router().plan(PageProfile()))[0] == "rapidocr"
    assert names(router(languages=["vi"]).plan(PageProfile()))[0] == "tesseract"
    assert names(router(languages=["vi"], engine="rapidocr").plan(PageProfile())) == ["rapidocr"]  # forced
    # with nobody else, it stays
    use_engines(make_engine("rapidocr", weak_languages={"vi": 0.0}))
    assert names(router(languages=["vi"]).plan(PageProfile())) == ["rapidocr"]


def test_router_says_when_the_weak_engine_reads_after_all(use_engines):
    use_engines(make_engine("rapidocr", weak_languages={"vi": 0.8, "kn": 0.0}), make_engine("tesseract", cost=2))
    assert router(languages=["vi"]).plan(PageProfile()).min_confidence == 0.8
    assert router(languages=["kn"]).plan(PageProfile(script="kannada")).min_confidence == 0.0
    assert router(languages=["es"]).plan(PageProfile()).min_confidence == 0.0
    assert router("fast", languages=["vi"]).plan(PageProfile()).min_confidence == 0.0  # one single pass
    # with nobody in front of it, there is nobody to give way
    use_engines(make_engine("rapidocr", weak_languages={"vi": 0.8}))
    assert router(languages=["vi"]).plan(PageProfile()).min_confidence == 0.0


def test_router_engine_weak_only_at_times_stays_if_the_others_are_not_meant_for_that_page(use_engines):
    use_engines(
        make_engine("rapidocr", weak_languages={"vi": 0.8, "kn": 0.0}),
        make_engine("tesseract", kinds=SCAN, cost=2),
    )
    photo, scan = PageProfile(kind="photo"), PageProfile(kind="scan")
    plan = router(languages=["vi"]).plan(photo)  # Tesseract is not for photos
    assert names(plan) == ["rapidocr", "tesseract"] and plan.min_confidence == 0.0
    plan = router(languages=["vi"]).plan(scan)
    assert names(plan) == ["tesseract", "rapidocr"] and plan.min_confidence == 0.8
    # always weak: anyone reads better, even on a page that is not for them
    assert names(router(languages=["kn"]).plan(PageProfile(kind="photo", script="kannada"))) == [
        "tesseract:kannada",
        "rapidocr:kannada",
    ]
    # with one meant for photos, it goes last as always, and it does not join the consensus: its sure readings without
    # diacritics won the fusion
    use_engines(
        make_engine("rapidocr", weak_languages={"vi": 0.8}),
        make_engine("tesseract", kinds=SCAN, cost=2),
        make_engine("easyocr", kinds=PHOTO, cost=3),
    )
    plan = router(languages=["vi"]).plan(photo)
    assert names(plan) == ["easyocr", "tesseract", "rapidocr"] and plan.weak == {"rapidocr"} and plan.ensemble
    assert any("no consensus with rapidocr: weak in vi" in r for r in plan.reasons)
    assert router(languages=["es"]).plan(photo).weak == frozenset()


def test_router_line_engine_weak_in_the_language_does_not_read_its_doubtful_lines(use_engines):
    kraken = make_engine("kraken", level="line", handwriting=True, weak_languages={"vi": 0.0})
    use_engines(make_engine("rapidocr"), kraken)
    assert router(languages=["vi"]).plan(PageProfile()).line_engine is None  # it loses the diacritics
    assert router(languages=["es"]).plan(PageProfile()).line_engine.name == "kraken"


def test_router_weak_engine_stays_if_the_other_does_not_fit_in_memory(use_engines):
    use_engines(
        make_engine("rapidocr", weak_languages={"vi": 0.0}, memory_mb=150, mb_per_megapixel=150),
        make_engine("easyocr", kinds=PHOTO, cost=3, memory_mb=350, runtime="torch", mb_per_megapixel=770),
    )
    profile = PageProfile(kind="photo")
    assert names(router(languages=["vi"]).plan(profile))[0] == "easyocr"
    assert names(router(languages=["vi"], memory_mb=1024).plan(profile))[0] == "rapidocr"


def test_router_line_engine_that_reads_print_in_a_language(use_engines):
    kraken = make_engine(
        "kraken",
        level="line",
        handwriting=True,
        cost=2,
        scripts=frozenset({"arabic", "hebrew"}),
        print_languages=frozenset({"ar", "he"}),
    )
    rapid = make_engine(
        "rapidocr", scripts=frozenset({"arabic", "latin"}), weak_languages={"ur": 0.0}, detect_lines=list
    )
    use_engines(rapid, make_engine("tesseract", kinds=SCAN, cost=2, scripts=frozenset({"arabic", "hebrew"})), kraken)
    # RapidOCR has a model and a good detector: it goes on reading first, and kraken, every line
    plan = router(languages=["ar"]).plan(PageProfile(script="arabic"))
    assert plan.reads_print and plan.detector is None and names(plan)[0] == "rapidocr:arabic"
    # without a model in RapidOCR: its detector finds the lines, as Tesseract's lost many
    plan = router(languages=["he"]).plan(PageProfile(script="hebrew"))
    assert plan.reads_print and plan.detector.name == "rapidocr" and names(plan) == ["tesseract:hebrew"]
    # without handwriting it is still print, and balanced also reads it so
    for options in ({"handwritten": False}, {"strategy": "balanced"}):
        plan = router(**{"languages": ["he"], **options}).plan(PageProfile(script="hebrew"))
        assert plan.reads_print and plan.detector.name == "rapidocr" and plan.line_engine.name == "kraken"
    # a single quick pass: Tesseract alone; with Tesseract forced, it reads the page and kraken only what is doubtful
    plan = router(languages=["he"], strategy="fast").plan(PageProfile(script="hebrew"))
    assert not plan.reads_print and plan.detector is None and plan.line_engine is None
    assert names(plan) == ["tesseract:hebrew"]
    plan = router(languages=["he"], engine="tesseract").plan(PageProfile(script="hebrew"))
    assert not plan.reads_print and plan.detector is None and plan.line_engine.name == "kraken"
    # another language in that script: only the doubtful lines, and without handwriting, none
    plan = router(languages=["fa"]).plan(PageProfile(script="arabic"))
    assert not plan.reads_print and plan.line_engine.name == "kraken"
    assert router(languages=["fa"], handwritten=False).plan(PageProfile(script="arabic")).line_engine is None
    # without a page engine for the script, the detector and kraken are enough
    use_engines(rapid, kraken)
    plan = router(languages=["he"]).plan(PageProfile(script="hebrew"))
    assert plan.detector.name == "rapidocr" and plan.engines == []


def test_router_skips_gpu_only_engines_on_cpu(use_engines):
    use_engines(make_engine("vlm", gpu="required"), make_engine("rapidocr"))
    assert names(router().plan(PageProfile())) == ["rapidocr"]


def test_router_line_engine(use_engines):
    use_engines(
        make_engine("rapidocr"), make_engine("trocr", level="line", handwriting=True, scripts=frozenset({"latin"}))
    )
    assert router().plan(PageProfile(script="latin")).line_engine.name == "trocr"
    assert router().plan(PageProfile(script="east_slavic")).line_engine is None  # TrOCR only reads Latin
    assert router(handwritten=False).plan(PageProfile()).line_engine is None
    assert router("balanced").plan(PageProfile()).line_engine is None


def test_router_variants_by_kind(use_engines):
    use_engines(make_engine("rapidocr"))
    assert router().plan(PageProfile(kind="photo")).variants == ["original", "shadow", "clahe", "sharpen"]
    assert router().plan(PageProfile(kind="scan")).variants == ["original", "clahe", "sharpen"]
    assert router("fast").plan(PageProfile(kind="photo")).variants == []
