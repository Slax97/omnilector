# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adaptive pipeline per image: the Router picks the engines and the strategy how much effort to spend.

PASS1 best engine for the situation (if it fails, the next one) -> upright page (skewed, sideways or
upside down: rotated with the lines already read) -> PASS2 variants/scales if there are signs of missing
text -> consensus with other engines (maximum_recall) -> fusion -> PASS3 line engine (handwriting) on
what is doubtful -> rows in reading order.

In the languages where a line engine beats the page engines on print too, it reads every line in PASS3. If no page
engine can find the lines well, a detector finds them in PASS1 and there are no extra passes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from statistics import mean

import cv2
import numpy as np

from ..config import OCRConfig
from ..engines import get_engine_class
from ..exceptions import InferenceError, OmnilectorError
from ..preprocessing.pipeline import iter_variants
from ..result import RecognitionCandidate, TextBlock
from ..routing import PageProfile, Plan, Router, analyze
from ..routing.analyzer import BLANK_INK
from ..routing.languages import (
    SCRIPT_OF,
    detect_start,
    distinctive,
    guess_language,
    mixed_scripts,
    script_share,
)
from ..routing.router import EngineSpec
from ..utils.geometry import box_iou, coverage, reading_rows, scale_box
from ..utils.logging import get_logger
from ..utils.resources import input_side
from ..utils.text import right_to_left, without_marks
from .fusion import drop_repeats, fuse_blocks, join_pieces
from .layout import LayoutAnalyzer, TableGrid, Zone
from .model_manager import ModelManager
from .orientation import (
    FLIP_SAMPLE,
    VERTICAL_SCRIPTS,
    background,
    flip_sample,
    normalize,
    page_angle,
    rotate,
    straighten,
    transform_block,
)
from .strategies import get_strategy

SMALL_TEXT_PX = 15  # median line height below which the image is read again enlarged (x2)
MIN_SCALE_STEP = 1.3  # two scales closer than this give almost the same reading: only one is read
LINE_SHAPE = 1.5  # minimum width / height to re-read with the line engine (not icons or single letters)
SPECIALIST_MIN = 0.5  # the engine specialized in the language keeps the line if it reads it at least with this
# Automatic script (without languages): empty or doubtful lines -> the other models are tried
SCRIPT_DOUBT = 0.8  # confidence below which a line is doubtful
SCRIPT_SAMPLE = 12  # doubtful lines (the widest) that each candidate model reads
SCRIPT_MIN = 0.45  # a line "is" of another script if its model reads it with confidence x its own letters >= this
SCRIPT_GAIN = 0.2  # and beats the current reading of the same line by this much
SCRIPT_WINS = 0.25  # fraction of the tried doubtful lines that a script has to win
# ...and of all the lines of the page, to read it again whole with its model: in a Spanish form with
# dozens of lines, one of dots read as Kannada re-read the page with the Kannada model (dates came out
# with its zero, "೦9-08-2020"). Single lines of another script are adopted one by one
SCRIPT_PAGE = 0.25
SCRIPT_LETTERS = (
    2  # a reading in another script only wins its line with at least this many of its own letters ("ಪ....." does not)
)
FEW_CHARS = 0.3  # a reading with fewer characters than this x (width / height of its box) is doubtful
# the same for digit-only readings, which fill their box more: in 391 real lines, true numbers gave
# 1.3 or more; below 1 there were only leftovers of lines in another script ("24" of a Devanagari one)
DIGITS_FILL = 1.1
FLIP_MARGIN = 0.1  # rotate 180 degrees only if upside down reads at least this much better
# Below this mean confidence, a reading that cannot say how the page is turned (an engine without the corners of its
# lines, Tesseract; or a line engine on lines only detected, kraken) is also read upside down. On the pages of the
# comparison between engines, upside down or sideways Tesseract read garbage with 0.06 to 0.73 (upright, 0.34 to 0.96:
# the doubtful ones are read twice), and kraken read upside-down lines with 0.58 to 0.78 (upright, 0.92 to 1)
ORIENT_CHECK = 0.85
# Scripts with letters of the same shape (A, Α, А): on a page they are not mixed line by line
LOOKALIKE = frozenset({"latin", "cyrillic", "east_slavic", "greek"})
# What only the extra passes see, in a zone pass 1 did not, has to prove it is text. In the 231 cases of the comparison
# between engines: with 2 or more characters, below 0.7 all of it was noise ("TE" on a road sign), from 0.7 all of it
# was real. A single character is the usual noise (a "2", a "©"), so it needs more
NEW_ZONE_CONF = 0.7
NEW_ZONE_CONF_CHAR = 0.9


def median_height(blocks: list[TextBlock]) -> int:
    heights = sorted(b.box[3] - b.box[1] for b in blocks if b.box[3] > b.box[1])
    return heights[len(heights) // 2] if heights else 0


def _latin(text: str) -> bool:
    """Whether the letters of the text are mostly Latin (or it has no letters)."""
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if c <= "ɏ") * 2 >= len(letters)


def _too_short(text: str, box: tuple[int, int, int, int]) -> bool:
    """Very few characters for the space the line takes (a "." in a wide box: the model did not
    understand the script and only read a sign)."""
    w, h = box[2] - box[0], max(1, box[3] - box[1])
    return len(text.strip()) < FEW_CHARS * w / h


def _fit(text: str, conf: float, box: tuple[int, int, int, int], scripts: list[str]) -> float:
    """How well a line is read in those scripts: confidence x fraction of its letters in them;
    0 if the reading is garbage (mixes scripts in one word or falls short for its box)."""
    if not text.strip() or mixed_scripts(text) or _too_short(text, box):
        return 0.0
    return conf * script_share(text, scripts)


def _own_letters(text: str, script: str) -> int:
    """How many letters of the reading are in that script."""
    letters = sum(1 for c in text if c.isalpha())
    return round(script_share(text, [script]) * letters)


def _doubtful(block: TextBlock, covered: list[str]) -> bool:
    """A line that may be in another script: empty, unsure or with letters that do not fit (garbage).
    Digit-only lines (amounts, dates, codes) are not, if they fill their box: they are the same in any
    script; but "7 11 2026" in the box of a Russian sentence is what is left after losing its letters."""
    if not block.text.strip():
        return True
    if not any(
        c.isalpha() for c in block.text
    ):  # digits only: doubtful if it does not fill its box (letters were lost)
        w, h = block.box[2] - block.box[0], max(1, block.box[3] - block.box[1])
        return len(block.text.replace(" ", "")) < DIGITS_FILL * w / h
    return block.confidence < SCRIPT_DOUBT or _fit(block.text, 1.0, block.box, covered) == 0.0


def _adopt(fresh: list[TextBlock], old: list[TextBlock], script: str, covered: list[str]) -> list[TextBlock]:
    """Merges the re-reading with the model of the detected script (`fresh`) and the previous reading:
    in each zone, the one that best fits its script (the Latin lines of a bilingual sign keep the Latin
    model, unless the re-reading has letters that only exist in its script: "МЕНЮ" is not replaced by
    "MEHIC"); the other one goes to candidates. Nothing is lost."""
    out = list(fresh)
    for b in old:
        match = max(out, key=lambda f: box_iou(f.box, b.box), default=None)
        if match is None or box_iou(match.box, b.box) < 0.5:
            out.append(b)
            continue
        if not b.text:
            continue
        # the previous reading only stays if it is Latin text (the Latin lines of a bilingual sign) and fits
        # better: the Chinese "garbage" of the multi model on a Korean page does not
        keep_old = not match.text or (
            script_share(b.text, ["latin"]) >= 0.5
            and not distinctive(match.text, script)
            and _fit(b.text, b.confidence, b.box, covered)
            > _fit(match.text, match.confidence, match.box, [script]) + 0.05
        )
        if keep_old:
            match.text, match.confidence, match.engine = b.text, b.confidence, b.engine
        match.candidates.append(RecognitionCandidate(b.text, b.confidence, b.engine))
    return out


def _covered(cls: type, script: str) -> list[str]:
    """Scripts that the model of `script` reads well: those sharing its model, plus Latin (the models of
    each script also read the Latin of a bilingual sign)."""
    model = cls.recognizer_for(script)
    return list(dict.fromkeys([*(s for s in cls.info.scripts if cls.recognizer_for(s) == model), "latin"]))


def _then(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """2x3 matrix that applies `first` and then `second`."""
    return (np.vstack([second, (0, 0, 1)]) @ np.vstack([first, (0, 0, 1)]))[:2]


def _sureness(blocks: list[TextBlock]) -> tuple[float, int]:
    """Mean confidence of a reading, by characters, and how many characters it has."""
    read = [(b.confidence, len(b.text.strip())) for b in blocks if b.text.strip()]
    chars = sum(n for _, n in read)
    return (sum(c * n for c, n in read) / chars if chars else 0.0), chars


def _sure_chars(blocks: list[TextBlock]) -> float:
    """Characters of a reading weighted by their confidence: with the page skewed 8 degrees, Tesseract read nothing
    or a garbage shorter than the straightened page, even with a confidence of 0.86 (Georgian, 79 against 235)."""
    sure, chars = _sureness(blocks)
    return sure * chars


def _reads_better(other: list[TextBlock], reading: list[TextBlock]) -> bool:
    """Whether `other` reads clearly better than `reading`: surer by FLIP_MARGIN, without losing half of the text."""
    (sure, chars), (base, base_chars) = _sureness(other), _sureness(reading)
    return sure > base + FLIP_MARGIN and 2 * chars >= base_chars


def _letters(text: str) -> list[str]:
    return [c for c in text.lower() if c.isalnum()]


def _loses_marks(new: str, old: str) -> bool:
    """Whether `new` is `old` with fewer marks (accents, tones), the same letters otherwise: a line engine without the
    letters of the language read "quyên và tu do dó" where EasyOCR had read "quyền và tự do đó" (kraken in Vietnamese:
    on two photos, from 0.6 % and 0.5 % of the characters wrong to 4 % and 11 %). Only the same letters: in a line
    from the 16th century, "que pueda dis" is a better reading of "que epúeda dap", not one with fewer marks."""
    (a, marks_new), (b, marks_old) = without_marks(new), without_marks(old)
    return marks_new < marks_old and _letters(a) == _letters(b)


def _unread(block: TextBlock) -> TextBlock:
    """The line without its reading, to read it again."""
    return replace(block, text="", confidence=0.0, is_handwritten=False, candidates=[])


def _cut_glyphs(blocks: list[TextBlock]) -> None:
    """Two overlapping pieces of the same row read the common strip twice. If the left one ends with a
    single character that falls inside the box of the right one, it is the edge of the right one's first
    letter ("Firmado: I" + "Luis"): it is removed (the whole reading stays in candidates). A real single
    character ("y", "a") is separated by a space and does not fall inside the next box. In right-to-left
    scripts the last character is at the other edge: nothing is touched."""
    for a in blocks:
        head, space, last = a.text.rstrip().rpartition(" ")
        if not space or len(last) != 1 or not head.strip() or right_to_left(a.text):
            continue
        start = a.box[2] - (a.box[2] - a.box[0]) / len(a.text)  # where that last character starts
        for b in blocks:
            height = min(a.box[3] - a.box[1], b.box[3] - b.box[1])
            same_row = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1]) >= 0.5 * height
            if b is not a and same_row and b.text[:1].isalnum() and b.box[0] <= start and a.box[2] <= b.box[2]:
                a.candidates.append(RecognitionCandidate(a.text, a.confidence, a.engine))
                a.text = head.rstrip()
                break


def _adds_text(block: TextBlock, pass1: list[TextBlock]) -> bool:
    """Whether a reading of the extra passes is worth keeping: it reads a zone of pass 1 again (fusion decides) or
    it is text in a zone nobody had seen. Variants and scales also "find" stains and drawings, each one with a
    different letter: on signs they added more wrong characters than the sign had. A zone pass 1 found but could not
    read does not count: the stains are there. If pass 1 read nothing, whatever there is goes."""
    boxes = [p.box for p in pass1 if p.text.strip()]
    if not boxes or any(box_iou(block.box, box) >= 0.5 for box in boxes) or coverage(block.box, boxes) >= 0.5:
        return True
    length = len(block.text.replace(" ", ""))
    return length > 0 and block.confidence >= (NEW_ZONE_CONF if length >= 2 else NEW_ZONE_CONF_CHAR)


def _scaled(block: TextBlock, scale: float) -> TextBlock:
    """Block read on the image enlarged `scale` times -> coordinates of the original image."""
    quad = tuple((round(x / scale), round(y / scale)) for x, y in block.quad) if block.quad else None
    return replace(block, box=scale_box(block.box, scale), quad=quad)


@dataclass
class PageRead:
    """What was read in an image: rows in reading order with boxes of the already upright image."""

    rows: list[list[TextBlock]]
    size: tuple[int, int]  # (w, h) of the upright image: the coordinate system of the boxes
    rotation: float = 0.0  # degrees (counterclockwise) the received image was rotated to read it
    matrix: np.ndarray | None = None  # received image -> upright image; None if it was not rotated
    script: str = "latin"  # script it was read with
    zones: list[Zone] | None = None  # layout: zones in reading order (the rows come from them)
    tables: list[TableGrid] | None = None
    hurried: bool = False  # some extra pass skipped because time was running out (time_limit)

    def original_blocks(self) -> list[TextBlock]:
        """The blocks with their boxes in the image as it arrived (to combine them with the native text of
        the same page, which is not rotated)."""
        blocks = [b for row in self.rows for b in row]
        if self.matrix is None:
            return blocks
        inverse = cv2.invertAffineTransform(self.matrix)
        return [transform_block(b, inverse) for b in blocks]


class OCRPipeline:
    def __init__(self, config: OCRConfig, manager: ModelManager) -> None:
        self.config = config
        self.manager = manager
        self.log = get_logger("omnilector", config.log_level)
        self.strategy = get_strategy(config.strategy)
        self.router = Router(config, self.strategy)
        self.layout = LayoutAnalyzer(config)

    def run(
        self,
        image: np.ndarray,
        hint: str | None = None,
        escalate: bool = True,
        script: str | None = None,
        deadline: float | None = None,
    ) -> PageRead:
        """RGB image -> rows of blocks in reading order.

        `hint` comes from the loader (photo from camera EXIF). escalate=False (images inside a page with
        native text): no variants or consensus, only pass 1 and doubtful lines. `script`: the script detected
        on the previous page of the document (without languages), to start with it; without it, the one from
        detect_start. `deadline` (time.monotonic(), from time_limit): once passed, the rest is done without
        extra passes.
        """
        t0 = time.time()
        hurried = False

        def late() -> bool:
            """Whether there is no time left for the pass about to run (and then it is skipped)."""
            nonlocal hurried
            if deadline is not None and time.monotonic() >= deadline:
                hurried = True
            return hurried

        image, fit = self._fit(image)
        profile = analyze(image, hint=hint, languages=self.config.languages)
        if not self.config.languages:
            profile = replace(profile, script=script or detect_start(self.config.detect_languages))
        plan = self.router.plan(profile)
        self.log.debug("plan: %s", "; ".join(plan.reasons))
        blocks, spec = self._first_pass(image, plan)
        detected = spec == plan.detector  # only the lines, still without text: the line engine reads them
        if not any(b.text for b in blocks) and profile.ink < BLANK_INK:
            return PageRead([], (image.shape[1], image.shape[0]), script=profile.script)  # blank
        image, blocks, rotation, matrix = self._straighten(image, blocks, profile)
        if fit is not None:  # received image -> reduced -> upright
            matrix = fit if matrix is None else _then(fit, matrix)
        if not detected:
            if not late() and not any(b.quad for b in blocks if b.text.strip()):  # it does not say how the lines go
                image, blocks, rotation, matrix = self._orient_by_reading(
                    image, blocks, spec, profile, rotation, matrix
                )
            blocks, spec, plan, profile = self._auto_script(image, blocks, spec, plan, profile, late)
            if self._upside_down(image, blocks, spec):  # now with the model of its script, which does tell
                image, blocks, rotation, matrix = self._turn(image, blocks, rotation, matrix)
                blocks, spec, plan, profile = self._auto_script(image, blocks, spec, plan, profile, late)
            if (
                escalate
                and self.strategy.allow_second_pass()
                and self.strategy.needs_second_pass(blocks, image)
                and not late()
            ):
                extra = self._second_pass(image, plan, spec, blocks)
                if plan.ensemble and not late():
                    extra += self._ensemble(image, plan, spec)
                blocks = blocks + [b for b in extra if _adds_text(b, blocks)]
        blocks = drop_repeats(fuse_blocks(blocks))
        _cut_glyphs(blocks)
        if plan.line_engine is not None and (detected or not late()):
            blocks = join_pieces(blocks, self.config.handwriting_conf_threshold)
            self._handwriting(image, blocks, plan, profile, detected)
            if detected and not late() and self._reads_better_turned(image, blocks, plan.line_engine):
                image, blocks, rotation, matrix = self._turn(image, [_unread(b) for b in blocks], rotation, matrix)
                self._handwriting(image, blocks, plan, profile, detected)
            blocks = drop_repeats(blocks)  # a zone nobody had read can now read what its pieces repeat
        if detected and blocks and plan.engines and not any(b.text.strip() for b in blocks):
            self.log.warning("the line engine read nothing (model not available?): reading with the page engines")
            blocks, _ = self._first_pass(image, replace(plan, detector=None))
        zones, tables = self._layout(image, blocks, late) if escalate else (None, None)
        self.log.debug("image %sx%s read in %.2fs", image.shape[1], image.shape[0], time.time() - t0)
        rows = [r for z in zones for r in z.rows] if zones else reading_rows(blocks)
        return PageRead(
            rows, (image.shape[1], image.shape[0]), rotation, matrix, profile.script, zones, tables, hurried
        )

    def _fit(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """With limited memory, a huge image (48 Mpx photo, scanned drawing) is reduced on entry to
        resources.INPUT_SIDE times the side the OCR sees: its copies (rotations, variants) no longer reach
        the limit. Also returns the matrix of the reduction."""
        side = input_side(self.config.memory_mb, self.config.max_image_side)
        h, w = image.shape[:2]
        if side is None or max(h, w) <= side:
            return image, None
        scale = side / max(h, w)
        small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        self.log.debug(
            "image of %sx%s reduced to %sx%s (memory_mb=%s)",
            w,
            h,
            small.shape[1],
            small.shape[0],
            self.config.memory_mb,
        )
        return small, np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0]])

    def _layout(
        self, image: np.ndarray, blocks: list[TextBlock], late: Callable[[], bool] = lambda: False
    ) -> tuple[list[Zone] | None, list[TableGrid] | None]:
        """Zones in reading order and tables (if the strategy asks for it, the [layout] extra is there and there is time)."""
        if not self.config.layout or not any(b.text for b in blocks) or LayoutAnalyzer.unavailable_reason() or late():
            return None, None
        try:
            found = self.layout.analyze(image, blocks)
        except Exception as e:  # noqa: BLE001 - without layout analysis the usual order stays
            self.log.warning("layout analysis not available (%s): reading order by rows", e)
            return None, None
        return found if found else (None, None)

    def _first_pass(self, image: np.ndarray, plan: Plan) -> tuple[list[TextBlock], EngineSpec]:
        """Best engine of the plan (or only the lines, for the line engine); if it fails (no model, no network...),
        or barely trusts its reading in a language where the next ones are the last resort, the next one."""
        failures: list[Exception] = []
        if plan.detector is not None:
            try:
                return self.manager.get(plan.detector).detect(image), plan.detector
            except Exception as e:  # noqa: BLE001 - the page engines, if there are any, also find lines
                self.log.warning("%s failed (%s); trying the next engine", plan.detector, e)
                failures.append(e)
        doubtful: tuple[list[TextBlock], EngineSpec] | None = None
        for spec in plan.engines:
            try:
                blocks = self.manager.get(spec).read(image)
            except Exception as e:  # noqa: BLE001 - a broken engine does not bring the page down if there is another one
                self.log.warning("%s failed (%s); trying the next engine", spec, e)
                failures.append(e)
                continue
            read = [b.confidence for b in blocks if b.text.strip()]
            if plan.min_confidence and (mean(read) if read else 0.0) < plan.min_confidence:
                self.log.debug("%s barely trusts its reading: trying the next engine", spec)
                doubtful = blocks, spec
                continue
            return blocks, spec
        if doubtful is not None:
            return doubtful
        if len(failures) == 1 and isinstance(failures[0], OmnilectorError):
            raise failures[0]
        raise InferenceError("No engine could read the page: " + "; ".join(map(str, failures)))

    def _straighten(
        self, image: np.ndarray, blocks: list[TextBlock], profile: PageProfile
    ) -> tuple[np.ndarray, list[TextBlock], float, np.ndarray | None]:
        """Upright page (skewed or sideways) from the angle of its lines. What was read in pass 1 is still
        valid: only its boxes are moved. Upside down is checked later, with the model of its script."""
        angle = page_angle(blocks)
        if angle is None:
            return image, blocks, 0.0, None
        if abs(angle) > 45 and profile.script in VERTICAL_SCRIPTS:
            angle = 0.0  # Chinese and Japanese are written vertically: that is not a sideways page
        upright, moved, matrix = straighten(image, blocks, angle)
        if matrix is None:
            return image, blocks, 0.0, None
        self.log.debug("page rotated %.1f degrees to read it upright", normalize(angle))
        return upright, moved, normalize(angle), matrix

    def _turn(
        self, image: np.ndarray, blocks: list[TextBlock], rotation: float, matrix: np.ndarray | None
    ) -> tuple[np.ndarray, list[TextBlock], float, np.ndarray | None]:
        """The page turned 180 degrees, with its blocks moved."""
        turned, half = rotate(image, 180)
        rotation = normalize(rotation + 180)
        self.log.debug("page upside down: rotated %.1f degrees", rotation)
        return (
            turned,
            [transform_block(b, half) for b in blocks],
            rotation,
            half if matrix is None else _then(matrix, half),
        )

    def _orient_by_reading(
        self,
        image: np.ndarray,
        blocks: list[TextBlock],
        spec: EngineSpec,
        profile: PageProfile,
        rotation: float,
        matrix: np.ndarray | None,
    ) -> tuple[np.ndarray, list[TextBlock], float, np.ndarray | None]:
        """Upright page for an engine that does not give the corners of its lines (Tesseract): the lines a detector
        finds say whether it is skewed or sideways, and then it is read again straightened. Always: its confidence
        does not tell it (skewed 8 degrees, Tesseract read nothing on most pages, and Georgian, 88 % wrong with 0.90).
        The straightened reading stays if it is sideways or reads more sure characters.
        If the reading is doubtful, the page is also read upside down and the surer reading stays: neither RapidOCR's
        classifier nor Tesseract's OSD tell it in the scripts read by Tesseract (they saw upright pages as upside
        down). What was read the wrong way round is not kept: it is not text."""
        finder = self.router.detector([spec.name])
        if finder is None:
            return image, blocks, rotation, matrix
        try:
            lines = self.manager.get(finder).detect(image)
        except Exception as e:  # noqa: BLE001 - without its lines the page is read as it is
            self.log.debug("%s could not find the lines: %s", finder, e)
            return image, blocks, rotation, matrix
        if not lines:
            return image, blocks, rotation, matrix  # no text to turn (a photo without it)
        engine = self.manager.get(spec)
        angle = page_angle(lines)
        if angle is not None and not (abs(angle) > 45 and profile.script in VERTICAL_SCRIPTS):
            upright, _, m = straighten(image, [], angle, background(image))  # it is read again: no white corners
            straight = engine.read(upright) if m is not None else []
            # sideways, what was read is garbage; only a little skewed (a photo), Tesseract sometimes read worse
            # straightened (Burmese, 3 % -> 15 %): the reading with more sure characters stays
            if m is not None and (abs(angle) > 45 or _sure_chars(straight) > _sure_chars(blocks)):
                image, blocks, rotation = upright, straight, normalize(rotation + angle)
                matrix = m if matrix is None else _then(matrix, m)
                self.log.debug("page rotated %.1f degrees to read it upright (lines found by %s)", rotation, finder)
        if _sureness(blocks)[0] < ORIENT_CHECK:
            turned_image, half = rotate(image, 180)
            turned = engine.read(turned_image)
            if _reads_better(turned, blocks):
                rotation = normalize(rotation + 180)
                self.log.debug("page upside down: rotated %.1f degrees", rotation)
                image, blocks, matrix = turned_image, turned, half if matrix is None else _then(matrix, half)
        return image, blocks, rotation, matrix

    def _reads_better_turned(self, image: np.ndarray, blocks: list[TextBlock], spec: EngineSpec) -> bool:
        """Whether lines only detected, already read by the line engine, read clearly better upside down: if its
        reading is doubtful, a few of the longest lines are read turned. kraken read upside-down lines with a mean
        confidence of 0.58 to 0.78, and upright ones with 0.92 to 1."""
        if _sureness(blocks)[0] >= ORIENT_CHECK:
            return False
        read = [b for b in blocks if b.text.strip()]
        sample = sorted(read, key=lambda b: b.box[2] - b.box[0], reverse=True)[:FLIP_SAMPLE]
        if not sample:
            return False
        h, w = image.shape[:2]
        boxes = [(w - x1, h - y1, w - x0, h - y0) for x0, y0, x1, y1 in (b.box for b in sample)]
        try:
            found = self.manager.get(spec).read_lines(cv2.rotate(image, cv2.ROTATE_180), boxes)
        except Exception as e:  # noqa: BLE001 - without it, the page stays as it is
            self.log.debug("%s could not read the page upside down: %s", spec, e)
            return False
        turned = [TextBlock(c.text, box, c.confidence, c.engine) for c, box in zip(found, boxes) if c is not None]
        return _reads_better(turned, sample)

    def _upside_down(self, image: np.ndarray, blocks: list[TextBlock], spec: EngineSpec) -> bool:
        """Upside down if the engine's orientation classifier says so and the recognizer confirms it: the
        classifier does not know every script (it saw upright photos of Thai and Korean as upside down). If
        the recognizer reads nothing either way, the classifier decides."""
        boxes = flip_sample(blocks)
        engine = self.manager.get(spec)
        try:
            if not boxes or not engine.upside_down(image, boxes):
                return False
        except Exception as e:  # noqa: BLE001 - without knowing it, the page is read as it is
            self.log.debug("%s could not tell whether the page is upside down: %s", spec, e)
            return False
        h, w = image.shape[:2]
        turned_boxes = [(w - x1, h - y1, w - x0, h - y0) for x0, y0, x1, y1 in boxes]
        cls = get_engine_class(spec.name)
        covered = _covered(cls, spec.script)
        try:
            upright = engine.read_lines(image, boxes)
            turned = engine.read_lines(cv2.rotate(image, cv2.ROTATE_180), turned_boxes)
        except Exception:  # noqa: BLE001 - without a line recognizer: whatever the classifier says
            return True
        before, after = (
            mean(_fit(c.text, c.confidence, box, covered) if c else 0.0 for c, box in zip(found, boxes))
            if found
            else 0.0
            for found in (upright, turned)
        )
        return (before == after == 0.0 and not any(upright) and not any(turned)) or after > before + FLIP_MARGIN

    def _auto_script(
        self,
        image: np.ndarray,
        blocks: list[TextBlock],
        spec: EngineSpec,
        plan: Plan,
        profile: PageProfile,
        late: Callable[[], bool] = lambda: False,
    ) -> tuple[list[TextBlock], EngineSpec, Plan, PageProfile]:
        """Empty or doubtful lines are also read by the models of the other candidate scripts: those of
        languages (with ["ru", "en", "ar"], the Arabic lines of a page read with the Russian model), those of
        detect_languages, or all. Each one keeps the model that clearly understands it (with letters of its
        script); if a model wins enough of them, the whole page is read again with it (a document in Russian)
        and the other won lines keep their reading (a collage or a sign with several scripts)."""
        unchanged = blocks, spec, plan, profile
        if not self.config.auto_script:
            return unchanged
        cls = get_engine_class(spec.name)
        models = self._script_models(cls, profile.script)
        if not models:
            return unchanged  # no other candidate script (languages=["es"], detect_languages=["es", "en"])
        covered = _covered(cls, profile.script)
        lines = [b for b in blocks if b.box[2] - b.box[0] >= LINE_SHAPE * (b.box[3] - b.box[1])]
        doubtful = [b for b in lines if _doubtful(b, covered)]
        if not doubtful or late():
            return unchanged
        sample = sorted(doubtful, key=lambda b: b.box[2] - b.box[0], reverse=True)[:SCRIPT_SAMPLE]
        best = [(_fit(b.text, b.confidence, b.box, covered), None, None) for b in sample]  # (fit, script, reading)
        for scripts in models:
            try:
                found = self.manager.get(EngineSpec(spec.name, scripts[0])).read_lines(image, [b.box for b in sample])
            except Exception as e:  # noqa: BLE001 - without that model (no network...) the others are tried
                self.log.debug("script %s not available: %s", scripts[0], e)
                continue
            for i, (cand, b) in enumerate(zip(found, sample)):
                if cand is None:
                    continue
                # the script of the reading, among those of that model: the multi model reads Latin, Chinese and
                # Japanese, and a Latin line measured with Chinese letters never won
                script = max(scripts, key=lambda s: _own_letters(cand.text, s))
                fit = _fit(cand.text, cand.confidence, b.box, [script])
                if (
                    fit >= SCRIPT_MIN
                    and fit >= best[i][0] + SCRIPT_GAIN
                    and _own_letters(cand.text, script) >= SCRIPT_LETTERS
                ):
                    best[i] = (fit, script, cand)
        group = {s: scripts[0] for scripts in models for s in scripts}
        wins: dict[str, list[str]] = {}  # model (its first script) -> scripts of the lines it won
        for _, script, _ in best:
            if script:
                wins.setdefault(group[script], []).append(script)
        if not wins:
            return unchanged
        won = wins[max(wins, key=lambda m: (len(wins[m]), m))]
        if len(won) >= SCRIPT_WINS * len(sample) and len(won) >= SCRIPT_PAGE * len(lines):
            main = max(won, key=won.count)  # the script of most of its lines
            profile = replace(profile, script=main)
            plan = self.router.plan(profile)
            spec = next((s for s in plan.engines if s.name == spec.name), plan.engines[0])
            self.log.debug(
                "script %s (%d of %d doubtful lines): reading the page again with %s", main, len(won), len(sample), spec
            )
            blocks = _adopt(self.manager.get(spec).read(image), blocks, main, covered)
        for b, (fit, script, cand) in zip(sample, best):  # each won line, in the script that won it
            # between Latin, Cyrillic and Greek (same letters) only if its reading has letters of its own
            if not script or (
                script != profile.script
                and {script, profile.script} <= LOOKALIKE
                and not distinctive(cand.text, script)
            ):
                continue
            target = max(blocks, key=lambda x: box_iou(x.box, b.box))
            if box_iou(target.box, b.box) < 0.5 or _fit(target.text, target.confidence, target.box, [script]) >= fit:
                continue  # the re-read page already has it in its script, equal or better
            target.candidates.append(RecognitionCandidate(target.text, target.confidence, target.engine))
            target.text, target.confidence, target.engine = cand.text, cand.confidence, cand.engine
        return blocks, spec, plan, profile

    def _script_models(self, cls: type, script: str) -> list[list[str]]:
        """Models tried on the doubtful lines: those of the candidate scripts (detect_languages or all of the
        engine's) except the one of `script`, each with the scripts it reads. The Latin one goes first: the
        multi model is loaded with it, as when reading a page, and not twice."""
        allowed = cls.info.scripts & (self.config.detect_scripts or cls.info.scripts)
        models: dict[object, list[str]] = {}
        for s in sorted(allowed, key=lambda s: (s != "latin", s)):
            models.setdefault(cls.recognizer_for(s), []).append(s)
        models.pop(cls.recognizer_for(script), None)
        return list(models.values())

    def _scales(self, image: np.ndarray, plan: Plan, max_side: int | None, pass1: list[TextBlock]) -> list[float]:
        scales = list(plan.scales)
        small = median_height(pass1)
        if self.config.multi_scale and 0 < small < SMALL_TEXT_PX and 2.0 not in scales:
            scales.append(2.0)  # small text: read again enlarged
        out: list[float] = []
        side = max(image.shape[:2])
        for sc in sorted(scales):
            if sc > 1.0 and max_side:
                sc = min(sc, max_side / side)  # the engine reduces whatever goes over max_side
            if all(max(sc, o) / min(sc, o) >= MIN_SCALE_STEP for o in out):
                out.append(round(sc, 3))
        return out

    def _second_pass(self, image: np.ndarray, plan: Plan, spec: EngineSpec, pass1: list[TextBlock]) -> list[TextBlock]:
        """Preprocessing variants x scales with the same engine as pass 1."""
        engine = self.manager.get(spec)
        scales = self._scales(image, plan, engine.max_side, pass1)
        extra: list[TextBlock] = []
        for vname, vimg in iter_variants(image, plan.variants or ["original"]):
            for sc in scales:
                if vname == "original" and sc == 1.0:
                    continue  # that is pass 1
                simg = vimg if sc == 1.0 else cv2.resize(vimg, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                try:
                    found = engine.read(simg)
                except Exception as e:  # noqa: BLE001 - a failed variant does not cancel the others
                    self.log.debug("PASS2 %s %s@%s failed: %s", spec, vname, sc, e)
                    continue
                extra.extend(found if sc == 1.0 else (_scaled(b, sc) for b in found))
        return extra

    def _ensemble(self, image: np.ndarray, plan: Plan, spec: EngineSpec) -> list[TextBlock]:
        """The next engines of the plan read too; the fusion decides by consensus. Not those weak in the page's
        language: on a Vietnamese photo, RapidOCR's sure readings without diacritics replaced EasyOCR's (0 % -> 14 %)."""
        out: list[TextBlock] = []
        for other in plan.engines[plan.engines.index(spec) + 1 :]:
            if not self.router.fits(other.name, spec.name) or other.name in plan.weak:
                continue  # does not fit in memory next to the page engine, or would win with wrong readings
            try:
                out.extend(self.manager.get(other).read(image))
            except Exception as e:  # noqa: BLE001 - the consensus is optional
                self.log.warning("%s failed in the consensus: %s", other, e)
        return out

    def _language(self, blocks: list[TextBlock], script: str) -> str | None:
        """Language of the page: the one given in languages for its script; without languages, the one its words
        say, among those of detect_languages with that script (if there is only one, that one) or among all."""
        cfg = self.config
        if cfg.languages:
            return next((lang for lang in cfg.languages if SCRIPT_OF.get(lang) == script), None)
        if not cfg.detect_languages:
            return guess_language(b.text for b in blocks)
        among = [lang for lang in cfg.detect_languages if SCRIPT_OF.get(lang) == script]
        return among[0] if len(among) == 1 else guess_language((b.text for b in blocks), among)

    def _line_engine(self, blocks: list[TextBlock], plan: Plan, profile: PageProfile) -> EngineSpec:
        """Line engine for the language of the page (TrOCR for English; kraken for the rest or if it is not
        known)."""
        language = self._language(blocks, profile.script)
        name = self.router.line_engine(profile, language)
        if name is None or name == plan.line_engine.name:
            return plan.line_engine
        self.log.debug("language %s: doubtful lines with %s", language, name)
        return EngineSpec(name, profile.script)

    def _handwriting(
        self, image: np.ndarray, blocks: list[TextBlock], plan: Plan, profile: PageProfile, detected: bool = False
    ) -> None:
        """Doubtful lines -> line engine of the page's language. If the language is unknown, the general one
        (kraken) reads and, if its reading shows the page is in a language with a specialist (English: TrOCR,
        4.9 % against 10.4 % on IAM), those lines are read again with it. In the languages where it also beats
        the page engines on print, it reads every line, and if a detector found them, whatever their shape."""
        if plan.reads_print and plan.line_engine is not None:
            shape = 0.0 if detected else LINE_SHAPE
            self._line_pass(image, blocks, plan.line_engine, specialist=True, printed=True, shape=shape)
            return
        spec = self._line_engine(blocks, plan, profile)
        reread = self._line_pass(image, blocks, spec)
        if self.config.languages or not reread:
            return
        language = self._language(blocks, profile.script)
        name = self.router.line_engine(profile, language)
        if name and name != spec.name and language in get_engine_class(name).info.languages:
            self.log.debug("language %s after re-reading the handwriting: those lines also with %s", language, name)
            self._line_pass(image, reread, EngineSpec(name, profile.script), specialist=True)

    def _line_pass(
        self,
        image: np.ndarray,
        blocks: list[TextBlock],
        spec: EngineSpec,
        specialist: bool = False,
        printed: bool = False,
        shape: float = LINE_SHAPE,
    ) -> list[TextBlock]:
        """Low-confidence (or empty) lines -> line engine; the most confident reading wins, unless it only drops
        marks of the one there was (with `specialist`, all the given lines, and the specialist wins if its reading
        makes sense; with `printed`, always). Only those shaped like a line
        (an icon or a single letter is not handwriting: TrOCR made up words) and, if the engine only reads
        Latin, in Latin script (TrOCR does not read Chinese). With `printed`, reading a line again does not make
        it handwriting: only if it was doubtful and handwriting is looked for. Returns the lines read again."""
        threshold = self.config.handwriting_conf_threshold
        looks_for_hand = self.strategy.use_handwriting(self.config.handwritten)
        latin_only = get_engine_class(spec.name).info.scripts == {"latin"}
        targets = [
            b
            for b in blocks
            if (specialist or b.confidence < threshold or not b.text.strip())
            and b.box[2] - b.box[0] >= shape * (b.box[3] - b.box[1])
            and (not latin_only or _latin(b.text))
        ]
        if not targets:
            return []
        try:
            cands = self.manager.get(spec).read_lines(image, [b.box for b in targets])
        except Exception as e:  # noqa: BLE001 - without a line engine the printed reading stays
            self.log.warning("%s not available (%s); going on without it", spec, e)
            return []
        reread = []
        for b, cand in zip(targets, cands):
            if cand is None:
                continue
            b.candidates.append(cand)
            # on the print of its languages it is the best engine even when it doubts: on two old Persian books,
            # from 74 % and 79 % of the characters wrong to 51 % and 52 %
            if (
                (printed or cand.confidence >= SPECIALIST_MIN or not b.text.strip())
                if specialist
                else cand.confidence > b.confidence and not _loses_marks(cand.text, b.text)
            ):
                handwritten = not printed or (looks_for_hand and bool(b.text.strip()) and b.confidence < threshold)
                b.text, b.confidence, b.engine, b.is_handwritten = cand.text, cand.confidence, cand.engine, handwritten
                reread.append(b)
        return reread
