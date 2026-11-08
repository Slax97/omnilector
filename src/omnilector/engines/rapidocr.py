# SPDX-License-Identifier: AGPL-3.0-or-later
"""RapidOCR (PP-OCR in ONNX): default page engine. CPU, Apache-2.0, no extras.

One recognition model per script (those not installed are downloaded when used):
- PP-OCRv6 multi: Latin (es, en, fr, de, pt, ca...), Chinese and Japanese. Installed with the package.
- PP-OCRv5: Arabic, Cyrillic, East Slavic, Devanagari, Greek, Korean, Thai, Tamil, Telugu.
- PP-OCRv4: Kannada.

Decoding each line keeps the spaces the model sees (_SpacedDecoder), and on a Latin page the lines where
the multi model makes up ideograms are read again with the Latin-only recognizer.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..exceptions import InferenceError, ModelNotAvailableError
from ..result import RecognitionCandidate, TextBlock
from ..routing.languages import SCRIPT_OF, script_share, unspaced
from ..utils.deps import missing_modules, require
from ..utils.resources import MB_PER_MEGAPIXEL, PAGE_ENGINE_MB
from .base import Box, EngineInfo, OCREngine, crop, onnx_cuda, polygon_block

# script -> (lang_type, ocr_version) of the recognition model. None = PP-OCRv6 multi.
_REC_MODELS: dict[str, tuple[str, str] | None] = {
    "latin": None,
    "chinese": None,
    "japanese": None,
    "arabic": ("arabic", "PP-OCRv5"),
    "cyrillic": ("cyrillic", "PP-OCRv5"),
    "east_slavic": ("eslav", "PP-OCRv5"),
    "devanagari": ("devanagari", "PP-OCRv5"),
    "greek": ("el", "PP-OCRv5"),
    "korean": ("korean", "PP-OCRv5"),
    "thai": ("th", "PP-OCRv5"),
    "tamil": ("ta", "PP-OCRv5"),
    "telugu": ("te", "PP-OCRv5"),
    "kannada": ("ka", "PP-OCRv4"),
}
# model_size -> PP-OCRv6 ModelType (tiny / small / medium). "large" does not exist in RapidOCR 3.x.
_V6_SIZES = {"tiny": "TINY", "small": "SMALL", "large": "MEDIUM"}
REC_ONLY_MB = 50  # only the recognizer of a script (to try it on a few lines)
# Space between two letters even when the CTC blank wins: if its probability is above SPACE_PROB and the
# letters are at least SPACE_GAP times farther apart than the line's letter pitch, measured with at least
# SPACE_PAIRS letter pairs (in a short word, "TIPO", the pitch cannot be measured: it came out "TI PO")
SPACE_PROB = 0.02
SPACE_GAP = 1.3
SPACE_PAIRS = 4
# Latin-only recognizer, for the lines where the multi model makes up ideograms on a Latin page: those with
# confidence < IDEOGRAM_DOUBT, if the lines with ideograms are no more than IDEOGRAM_SHARE of the page
_LATIN_ONLY = ("latin", "PP-OCRv5")
IDEOGRAM_DOUBT = 0.7
IDEOGRAM_SHARE = 0.2


class _SpacedDecoder:
    """RapidOCR CTC decoding that keeps the spaces. The recognizer sees the space between two words but,
    when in doubt, the CTC blank wins and they come out glued: on tickets with monospaced type,
    "41001Sevilla", "CIF:B11223344" or "0.2 87.00" -> "0.287.00". Between two letters farther apart than
    the rest of their line, the space the model saw is put back, except in Chinese, Japanese and Thai, which
    do not separate words. Everything else, as in RapidOCR. On 10 Spanish tickets the words found went
    from 92.2 % to 95.6 %, and on 50 real forms (XFUND) from 91.8 % to 93.3 %, in the same time."""

    def __init__(self, decoder: object) -> None:
        self.decoder = decoder
        self.character: list[str] = decoder.character  # type: ignore[attr-defined]
        self.spaces = [i for i, c in enumerate(self.character) if c == " "]

    def __getattr__(self, name: str) -> object:
        if name == "decoder":  # without a decoder (a half-made copy): no infinite recursion
            raise AttributeError(name)
        return getattr(self.decoder, name)

    def __call__(self, preds: np.ndarray, return_word_box: bool = False, **kwargs: object) -> tuple[list, list]:
        if return_word_box or not self.spaces:
            return self.decoder(preds, return_word_box, **kwargs)  # type: ignore[operator]
        best, prob = preds.argmax(axis=2), preds.max(axis=2)
        return [self._line(preds[i], best[i], prob[i]) for i in range(len(best))], []

    def _line(self, preds: np.ndarray, best: np.ndarray, prob: np.ndarray) -> tuple[str, float]:
        keep = np.ones(len(best), dtype=bool)
        keep[1:] = best[1:] != best[:-1]
        keep &= best != 0  # the CTC blank
        cols = np.flatnonzero(keep)
        if not len(cols):
            return "", 0.0
        chars = [self.character[k] for k in best[cols]]
        confidence = float(np.mean(np.round(prob[cols].astype(np.float64), 5)).round(5))
        pairs = list(zip(cols, cols[1:], chars, chars[1:]))
        steps = [b - a for a, b, x, y in pairs if x != " " and y != " "]
        if len(steps) < SPACE_PAIRS:
            return "".join(chars), confidence
        gap = max(2.0, SPACE_GAP * float(np.median(steps)))
        out = [chars[0]]
        for a, b, x, y in pairs:
            if (
                x != " "
                and y != " "
                and b - a >= gap
                and not unspaced(x + y)
                and preds[a + 1 : b, self.spaces].max() > SPACE_PROB
            ):
                out.append(" ")
            out.append(y)
        return "".join(out), confidence


def _with_spaces(recognizer: object) -> object:
    """The RapidOCR recognizer with the decoding that keeps spaces (if it is the usual CTC one)."""
    decoder = getattr(recognizer, "postprocess_op", None)
    if isinstance(getattr(decoder, "character", None), list) and not isinstance(decoder, _SpacedDecoder):
        recognizer.postprocess_op = _SpacedDecoder(decoder)  # type: ignore[attr-defined]
    return recognizer


def _part(engine: object, name: str) -> object:
    """RapidOCR detector ("det"), classifier ("cls") or recognizer ("rec"). Since 3.10 each one is loaded
    the first time it is used (before, all of them when the engine was created)."""
    loader = getattr(engine, f"_load_{name}_model", None)
    return loader() if loader is not None else getattr(engine, f"text_{name}")


def _seq(value: object) -> list:
    return [] if value is None else list(value)  # type: ignore[call-overload]


def _warm_up_image() -> np.ndarray:
    """Image with text (BGR). RapidOCR >= 3.10 loads each model on first use and, without text to
    detect, never gets to load the recognizer."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 120), "white")
    ImageDraw.Draw(img).text((20, 30), "OCR 2026", fill="black", font=ImageFont.load_default(size=48))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class RapidOCREngine(OCREngine):
    info = EngineInfo(
        name="rapidocr",
        scripts=frozenset(_REC_MODELS),
        cost=1,
        memory_mb=PAGE_ENGINE_MB,
        mb_per_megapixel=MB_PER_MEGAPIXEL,
        # On a clean text, with confidence above 0.9: Kannada 29 % of the characters wrong (Tesseract and EasyOCR,
        # under 1 %), Urdu in Nastaliq 94 %, Vietnamese 9 % (its models do not have all its letters; the others, 0 %).
        # In Vietnamese that loss is small: on an old print Tesseract (confidence 0.62) gets 50 % wrong and RapidOCR 27 %,
        # while on modern text its confidence is 0.93 or more
        weak_languages={"kn": 0.0, "ur": 0.0, "vi": 0.8},
        detector=True,
        description="Printed text: scans, screenshots and photos. One model per script.",
    )

    def __init__(self, config, script: str = "latin") -> None:
        super().__init__(config, script)
        # RapidOCR reduces images larger than this internally: the pipeline does not enlarge beyond it
        self.max_side = config.max_image_side
        self._rec: object = None  # only the recognizer, as long as the whole engine is not needed
        self._latin: object = None  # Latin-only recognizer (False if it could not be loaded)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return missing_modules("rapidocr", "onnxruntime")

    def build_params(self, rec_model: tuple[str, str] | None = None) -> dict:
        """RapidOCR 3.x parameters (dotted keys; model types as Enum). `rec_model` (lang_type,
        ocr_version): a recognizer other than the one of the engine's script."""
        from rapidocr import LangRec, ModelType, OCRVersion

        cfg = self.config
        size = ModelType[_V6_SIZES[cfg.model_size]]
        params: dict = {
            # By default RapidOCR drops text with confidence < 0.5: nothing is dropped here
            "Global.text_score": 0.0,
            "Global.max_side_len": self.max_side,
            "Global.log_level": "debug" if (cfg.log_level or "").upper() == "DEBUG" else "error",
            "Det.box_thresh": cfg.det_box_thresh,
            "Det.model_type": size,
            "Rec.model_type": size,
            "EngineConfig.onnxruntime.use_cuda": onnx_cuda(cfg),
            # The per-line orientation classifier does not rotate anything on its own: it saw upright lines as upside
            # down (all Devanagari ones, and a clean line of a letter) and the whole line came out as garbage. What it
            # says is checked with the recognizer (_flip_verified)
            "Global.use_cls": False,
        }
        rec = rec_model or _REC_MODELS.get(self.script)
        if rec is not None:
            lang, version = rec
            params["Rec.lang_type"] = LangRec(lang)
            params["Rec.ocr_version"] = OCRVersion(version)
            params["Rec.model_type"] = ModelType.MOBILE
        if cfg.threads:  # several workers: each one with its own threads, without stepping on each other
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = cfg.threads
            params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
        if cfg.model_cache_dir:
            params["Global.model_root_dir"] = cfg.model_cache_dir
        return params

    def _build(self) -> object:
        engine = require("rapidocr").RapidOCR(params=self.build_params())
        engine(_warm_up_image())  # detection, orientation and recognition loaded (and downloaded)
        _with_spaces(_part(engine, "rec"))
        self._rec = None  # the whole engine already brings its recognizer
        return engine

    def resident_mb(self) -> int:
        latin = REC_ONLY_MB if self._latin else 0
        if self.loaded:
            return self.info.memory_mb + latin
        return (REC_ONLY_MB if self._rec is not None else 0) + latin

    def prepare(self) -> object:
        """The engine and, for Latin, also the Latin-only recognizer: ready without network."""
        engine = self.load()
        if self._ideograms_unexpected():
            self._latin_recognizer()
        return engine

    def _build_recognizer(self, rec_model: tuple[str, str] | None = None) -> object:
        """Only the recognizer of this script (or another one, `rec_model`), with the same configuration as the
        whole engine."""
        from rapidocr import RapidOCR
        from rapidocr.ch_ppocr_rec import TextRecognizer
        from rapidocr.utils.log import logger

        cfg = RapidOCR._load_config(None, None, self.build_params(rec_model))  # type: ignore[arg-type]
        logger.setLevel(cfg.Global.log_level.upper())
        cfg.Rec.engine_cfg = cfg.EngineConfig[cfg.Rec.engine_type.value]
        cfg.Rec.font_path = cfg.Global.font_path
        cfg.Rec.model_root_dir = cfg.Global.model_root_dir
        return _with_spaces(TextRecognizer(cfg.Rec))

    def _latin_recognizer(self) -> object | None:
        """The Latin-only recognizer (PP-OCRv5, 8 MB), loaded the first time it is needed. Without it (no
        network or cache), the lines stay as the multi model read them and it is not retried on every page."""
        with self._lock:
            if self._latin is None:
                try:
                    self._latin = self._build_recognizer(_LATIN_ONLY)
                except Exception:  # noqa: BLE001 - no network, no cache...: the multi model's reading is fine
                    self._latin = False
            return self._latin or None

    def _recognizer(self) -> object:
        """The recognizer. To try this script on a few lines it is enough: no detector or classifier, which the
        page already has (a third of the memory and loading time). If the page is read again with this
        script, the whole engine is loaded."""
        if self._model is None:
            with self._lock:
                if self._rec is None and self._load_error is None:
                    try:
                        self._rec = self._build_recognizer()
                    except (ImportError, AttributeError, TypeError, KeyError):
                        pass  # another RapidOCR version: the whole engine
                    except Exception as e:  # noqa: BLE001 - no network, no cache...: like load(), without retrying
                        self._load_error = ModelNotAvailableError(
                            f"rapidocr: could not load the model (offline? cache?): {e}"
                        )
                rec = self._rec
            if rec is not None:
                return rec
        return _part(self.load(), "rec")

    @classmethod
    def recognizer_for(cls, script: str) -> object:
        """Recognition model of a script: Latin, Chinese and Japanese share PP-OCRv6 multi."""
        return _REC_MODELS.get(script, script)

    def _detect_and_read(self, engine: object, bgr: np.ndarray) -> tuple[list, list, list]:
        """Boxes, texts and confidences of everything detected, including what the recognizer leaves empty:
        RapidOCR filters it internally, and a line in another script disappeared without a trace."""
        try:
            from rapidocr.utils.process_img import map_boxes_to_original

            h, w = bgr.shape[:2]
            img, record = engine.preprocess_img(bgr)  # type: ignore[attr-defined]
            det, _, rec, crops = engine.run_ocr_steps(img, record)  # type: ignore[attr-defined]
        except (AttributeError, ImportError, TypeError):  # another RapidOCR version: the public call
            out = engine(bgr)  # type: ignore[operator]
            return (
                _seq(getattr(out, "boxes", None)),
                _seq(getattr(out, "txts", None)),
                _seq(getattr(out, "scores", None)),
            )
        if det.boxes is None or rec.txts is None:
            return [], [], []
        txts, scores = _seq(rec.txts), _seq(rec.scores)
        self._flip_verified(engine, _seq(crops), txts, scores)
        if self._ideograms_unexpected():
            self._latin_verified(_seq(crops), txts, scores)
        return _seq(map_boxes_to_original(det.boxes, record, h, w)), txts, scores

    def _ideograms_unexpected(self) -> bool:
        """Whether this engine reads Latin pages with the multi model and no configured language is Chinese
        or Japanese: then a stray ideogram on the page is suspicious."""
        languages = (*self.config.languages, *self.config.detect_languages)
        return (
            self.script == "latin"
            and _REC_MODELS.get(self.script) is None
            and not any(SCRIPT_OF.get(lang) in ("chinese", "japanese") for lang in languages)
        )

    def _latin_verified(self, crops: list, txts: list, scores: list) -> None:
        """Lines with ideograms on a Latin page: where the multi model recognizes nothing (the asterisks of a
        card number, a logo) it makes up low-confidence ideograms ("**** 4821" -> "培塔培林 4821"). If they are
        few lines of the page, they are read again with the Latin-only recognizer and the most confident reading
        stays; a page with many lines in Chinese or Japanese (a sign) stays as it is."""
        if not crops or len(crops) != len(txts):
            return
        ideograms = [i for i, t in enumerate(txts) if script_share(str(t), ("japanese",)) > 0]
        doubtful = [i for i in ideograms if float(scores[i] or 0.0) < IDEOGRAM_DOUBT]
        if not doubtful or len(ideograms) > IDEOGRAM_SHARE * len(txts):
            return
        recognizer = self._latin_recognizer()
        if recognizer is None:
            return
        try:
            from rapidocr.ch_ppocr_rec import TextRecInput

            out = recognizer(TextRecInput(img=[crops[i] for i in doubtful]))  # type: ignore[operator]
        except Exception:  # noqa: BLE001 - without it, the lines stay as the multi model read them
            return
        for i, text, score in zip(doubtful, _seq(out.txts), _seq(out.scores)):
            if str(text).strip() and float(score or 0.0) >= float(scores[i] or 0.0):
                txts[i], scores[i] = text, score

    @staticmethod
    def _flip_verified(engine: object, crops: list, txts: list, scores: list) -> None:
        """Upside-down lines (a rotated photo, an inverted stamp): the lines the classifier sees rotated are
        also read rotated and the most confident reading stays. That way its mistakes (upright lines it saw
        inverted) do not cost the line, and a sideways or upside-down page is read the right way up."""
        if not crops or len(crops) != len(txts):
            return
        try:
            from rapidocr.ch_ppocr_rec import TextRecInput

            votes = getattr(_part(engine, "cls")(list(crops)), "cls_res", None) or []  # type: ignore[operator]
            turned = [i for i, (label, _) in enumerate(votes) if str(label) == "180"]
            if not turned:
                return
            out = _part(engine, "rec")(TextRecInput(img=[cv2.rotate(crops[i], cv2.ROTATE_180) for i in turned]))  # type: ignore[operator]
        except (AttributeError, ImportError, TypeError, ValueError):  # without a classifier: as it is
            return
        for i, text, score in zip(turned, _seq(out.txts), _seq(out.scores)):
            if str(text).strip() and float(score or 0.0) > float(scores[i] or 0.0):
                txts[i], scores[i] = text, score

    def read(self, image: np.ndarray) -> list[TextBlock]:
        engine = self.load()
        try:
            # RapidOCR treats ndarrays as BGR (OpenCV convention); omnilector works in RGB
            boxes, txts, scores = self._detect_and_read(engine, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        except Exception as e:
            raise InferenceError(f"RapidOCR failed: {e}") from e
        scores = scores or [0.0] * len(txts)
        return [
            polygon_block(bx, str(t).strip(), float(s or 0.0) if str(t).strip() else 0.0, self.info.name)
            for t, s, bx in zip(txts, scores, boxes)
            if bx is not None
        ]

    def detect(self, image: np.ndarray) -> list[TextBlock]:
        """Only the detector: it finds lines in any script, also in those without a recognition model."""
        engine = self.load()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            from rapidocr.main import RapidOCRError
            from rapidocr.utils.process_img import map_boxes_to_original

            h, w = bgr.shape[:2]
            img, record = engine.preprocess_img(bgr)  # type: ignore[attr-defined]
            try:
                _, det = engine.detect_and_crop(img, record)  # type: ignore[attr-defined]
            except RapidOCRError:  # it says so when it finds no text
                return []
            boxes = _seq(map_boxes_to_original(det.boxes, record, h, w))
        except (AttributeError, ImportError, TypeError):  # another RapidOCR version: the public call
            boxes = _seq(getattr(engine(bgr), "boxes", None))  # type: ignore[operator]
        except Exception as e:
            raise InferenceError(f"RapidOCR (detection) failed: {e}") from e
        return [polygon_block(box, "", 0.0, self.info.name) for box in boxes if box is not None]

    def read_lines(self, image: np.ndarray, boxes: list[Box]) -> list[RecognitionCandidate | None]:
        """Only the recognizer on line crops (to try another script on a few lines)."""
        crops = [
            (i, c)
            for i, box in enumerate(boxes)
            if (c := crop(image, box, pad=2)) is not None and min(c.shape[:2]) >= 4
        ]
        out: list[RecognitionCandidate | None] = [None] * len(boxes)
        if not crops:
            return out
        recognizer = self._recognizer()
        try:
            from rapidocr.ch_ppocr_rec import TextRecInput

            rec = recognizer(TextRecInput(img=[cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for _, c in crops]))  # type: ignore[operator]
        except Exception as e:
            raise InferenceError(f"RapidOCR (recognition) failed: {e}") from e
        for (i, _), text, score in zip(crops, _seq(rec.txts), _seq(rec.scores)):
            if str(text).strip():
                out[i] = RecognitionCandidate(str(text).strip(), float(score or 0.0), self.info.name)
        return out

    def upside_down(self, image: np.ndarray, boxes: list[Box]) -> bool | None:
        """RapidOCR orientation classifier (0 / 180 degrees) on a few lines, at once; each one votes with
        its confidence. The classifier is called directly: engine(..., use_det=False) would change the
        engine's configuration for the following reads."""
        engine = self.load()
        lines = [c for box in boxes if (c := crop(image, box, pad=2)) is not None]
        if not lines:
            return None
        try:
            out = _part(engine, "cls")([cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for c in lines])  # type: ignore[operator]
        except Exception as e:
            raise InferenceError(f"RapidOCR (orientation) failed: {e}") from e
        votes = [(str(label), float(score)) for label, score in getattr(out, "cls_res", None) or []]
        if not votes:
            return None
        return sum(score if label == "180" else -score for label, score in votes) > 0
