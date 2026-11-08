# SPDX-License-Identifier: AGPL-3.0-or-later
"""Router: page situation (PageProfile) + config -> Plan (which engines, which preprocessing)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..engines import get_engine_class, registered_engines
from ..exceptions import EngineNotAvailableError
from ..utils.deps import missing_modules
from ..utils.resources import LAYOUT_MB, RUNTIME_MB, USEFUL_SIDE, engine_side
from .analyzer import PageProfile
from .languages import SCRIPT_OF

if TYPE_CHECKING:
    from ..config import OCRConfig
    from ..core.strategies import OCRStrategy
    from ..engines import OCREngine


@dataclass(frozen=True)
class EngineSpec:
    """A concrete engine: name + script (each script can be a different model)."""

    name: str
    script: str = "latin"

    def __str__(self) -> str:
        return self.name if self.script == "latin" else f"{self.name}:{self.script}"


@dataclass
class Plan:
    engines: list[EngineSpec]  # best to worst: the 1st does pass 1; if it fails, the next one
    ensemble: bool = False  # maximum_recall: if text is missing, the other engines read too (consensus)
    line_engine: EngineSpec | None = None  # doubtful lines (handwriting)
    reads_print: bool = False  # the line engine reads every line: on the print of this language, it is the best
    detector: EngineSpec | None = None  # finds the lines for the line engine, if the page engines cannot do it well
    min_confidence: float = 0.0  # a page engine below this mean confidence gives way to the next one
    weak: frozenset[str] = frozenset()  # page engines weak in the page's language: not in the consensus
    variants: list[str] = field(default_factory=list)  # preprocessing of pass 2
    scales: tuple[float, ...] = (1.0,)
    reasons: list[str] = field(default_factory=list)  # why (log at DEBUG level)


class Router:
    def __init__(self, config: OCRConfig, strategy: OCRStrategy) -> None:
        self.config = config
        self.strategy = strategy

    def _usable(self, cls: type[OCREngine], script: str) -> bool:
        info = cls.info
        if script not in info.scripts or cls.unavailable_reason() is not None:
            return False
        return not (info.gpu == "required" and self.config.device == "cpu")

    @staticmethod
    def memory(name: str) -> int:
        """Memory of a loaded engine, with its library (MB)."""
        info = get_engine_class(name).info
        return info.memory_mb + RUNTIME_MB.get(info.runtime, 0)

    def room(self, page: str | None) -> int | None:
        """Memory for one more engine next to the page engine and the layout model; None without a limit."""
        budget = self.config.models_budget_mb
        if budget is None:
            return None
        layout = LAYOUT_MB if self.config.layout and missing_modules("rapid_layout", "rapid_table") is None else 0
        return budget - layout - (self.memory(page) if page else 0)

    def side(self, name: str) -> int | None:
        """Side at which that engine can read an image with the memory there is; None without a limit."""
        return engine_side(self.config.memory_mb, get_engine_class(name).info.mb_per_megapixel)

    def fits(self, name: str, page: str | None) -> bool:
        """Whether the engine fits in memory next to the page engine and can read at a useful size (EasyOCR needs
        5 times more memory per megapixel than RapidOCR). The page engine always does: without it there is no OCR."""
        room = self.room(page)
        if room is None or name == page:
            return True
        side = self.side(name)
        return self.memory(name) <= room and (side is None or side >= min(USEFUL_SIDE, self.config.max_image_side or 0))

    def page_engines(self, profile: PageProfile, language: str | None = None) -> list[str]:
        """Installed page engines that read that script: first those meant for that kind of page
        (EngineInfo.kinds) and, among them, the cheapest. An engine weak in the page's language goes last, if
        another one can take its place (fits in memory on its own). If it is only weak at times (it has a confidence
        below which it reads worse), that one has to be meant for this kind of page: Tesseract in a Vietnamese photo
        left out half the text (43 % character error) where RapidOCR, losing some accents, got 9 %."""
        usable = [c for c in registered_engines() if c.info.level == "page" and self._usable(c, profile.script)]
        weak = [c.info.weak_languages[language] for c in usable if language in c.info.weak_languages]
        others = [c for c in usable if language not in c.info.weak_languages and self.fits(c.info.name, None)]
        goes_last = bool(weak) and any(not max(weak) or profile.kind in c.info.kinds for c in others)

        def rank(c: type[OCREngine]) -> tuple[bool, bool, int]:
            behind = goes_last and (language in c.info.weak_languages or not self.fits(c.info.name, None))
            return behind, profile.kind not in c.info.kinds, c.info.cost

        return [c.info.name for c in sorted(usable, key=rank)]

    def min_confidence(self, names: list[str], language: str | None) -> float:
        """Mean confidence below which the engines in front of a weak one give way to it. Not in one single pass."""
        if not self.strategy.allow_second_pass():
            return 0.0
        return max((get_engine_class(n).info.weak_languages.get(language, 0.0) for n in names[1:]), default=0.0)

    @staticmethod
    def detector(names: list[str]) -> EngineSpec | None:
        """Engine that finds the lines for the line engine; none if the first page engine already does it well."""
        if names and get_engine_class(names[0]).info.detector:
            return None
        found = next((c for c in registered_engines() if c.info.detector and c.unavailable_reason() is None), None)
        return EngineSpec(found.info.name) if found else None

    @staticmethod
    def _no_engine(script: str) -> EngineNotAvailableError:
        extras = sorted({c.info.extra for c in registered_engines() if script in c.info.scripts and c.info.extra})
        hint = f': pip install "omnilector[{",".join(extras)}]"' if extras else ""
        return EngineNotAvailableError(f"No installed engine reads the script '{script}'{hint}")

    def line_engine(self, profile: PageProfile, language: str | None = None) -> str | None:
        """Line engine for handwriting: the one specialized in the page's language (TrOCR for English) and, if
        there is none or the language is unknown, the general one (kraken: 44 languages); the cheapest."""
        usable = [
            c
            for c in registered_engines()
            if c.info.level == "line"
            and c.info.handwriting
            and self._usable(c, profile.script)
            and language not in c.info.weak_languages  # its sure readings would replace better ones
        ]
        page = self.config.engine or next(iter(self.page_engines(profile)), None)
        usable = [c for c in usable if self.fits(c.info.name, page)]  # TrOCR (1.7 GB) does not fit in 2 GB
        if not usable:
            return None
        usable.sort(key=lambda c: (language not in c.info.languages, bool(c.info.languages), c.info.cost))
        return usable[0].info.name

    def print_engine(self, profile: PageProfile, language: str | None) -> str | None:
        """Line engine that reads the print of the page's language better than the page engines
        (EngineInfo.print_languages), so it reads every line: not with a forced engine or if the strategy does not
        let it (fast). Also without handwriting: it is print. The cheapest."""
        if language is None or self.config.engine or not self.strategy.use_line_engine_on_print():
            return None
        usable = [
            c
            for c in registered_engines()
            if c.info.level == "line" and language in c.info.print_languages and self._usable(c, profile.script)
        ]
        page = next(iter(self.page_engines(profile)), None)
        usable = [c for c in usable if self.fits(c.info.name, page)]
        return min(usable, key=lambda c: c.info.cost).info.name if usable else None

    def variants(self, profile: PageProfile) -> list[str]:
        base = list(self.strategy.variant_names(self.config.preprocessing))
        if profile.kind == "photo" and base and "shadow" not in base:
            return [base[0], "shadow", *base[1:]]  # shadows / uneven lighting
        return base

    def plan(self, profile: PageProfile) -> Plan:
        cfg = self.config
        reasons = [f"page {profile.kind}", f"script {profile.script}"]
        language = next((lang for lang in cfg.languages if SCRIPT_OF.get(lang) == profile.script), None)
        page_language = language or next(
            (lang for lang in cfg.detect_languages if SCRIPT_OF.get(lang) == profile.script), None
        )
        line = self.line_engine(profile, language) if self.strategy.use_handwriting(cfg.handwritten) else None
        printer = self.print_engine(profile, page_language)
        reads_print = printer is not None and line in (None, printer)
        if reads_print:
            line = printer
        detector = None
        if cfg.engine:
            cls = get_engine_class(cfg.engine)
            reason = cls.unavailable_reason()
            if reason:
                raise EngineNotAvailableError(f"{cfg.engine}: {reason}")
            if profile.script not in cls.info.scripts:
                raise EngineNotAvailableError(f"{cfg.engine} does not read the script '{profile.script}'")
            names = [cfg.engine]
            reasons.append(f"forced engine: {cfg.engine}")
        else:
            # with a line engine that reads everything, the page engine only has to find the lines
            names = self.page_engines(profile, None if reads_print else page_language)
            detector = self.detector(names) if reads_print else None
            if not names and detector is None:
                raise self._no_engine(profile.script)
            reasons.append(f"engines: {' > '.join(names) or 'none'}")
        if reads_print:
            reasons.append(f"every line: {line}" + (f", found by {detector.name}" if detector else ""))
        page = detector.name if detector else names[0]
        room = self.room(page)
        if room is not None:
            reasons.append(f"memory for more models: {room} MB")
            if room < 0:
                reasons.append(
                    f"{page} does not fit in memory_mb={cfg.memory_mb}: lower max_image_side or raise the memory"
                )
        # an engine weak in the language: its sure readings won the fusion (Vietnamese without diacritics)
        weak = frozenset(n for n in names if page_language in get_engine_class(n).info.weak_languages)
        consensus = [n for n in names[1:] if self.fits(n, page) and n not in weak]
        if self.strategy.use_ensemble():
            big = [n for n in names[1:] if not self.fits(n, page)]
            if big:
                reasons.append("no consensus with " + ", ".join(big) + ": they do not fit")
            if weak & set(names[1:]):
                reasons.append(f"no consensus with {', '.join(sorted(weak & set(names[1:])))}: weak in {page_language}")
        if line:
            reasons.append(f"doubtful lines: {line}")
        return Plan(
            engines=[EngineSpec(n, profile.script) for n in names],
            ensemble=self.strategy.use_ensemble() and bool(consensus),
            line_engine=EngineSpec(line, profile.script) if line else None,
            reads_print=reads_print,
            detector=detector,
            min_confidence=self.min_confidence(names, None if reads_print else page_language),
            weak=weak,
            variants=self.variants(profile),
            scales=tuple(self.strategy.scales(cfg.multi_scale, cfg.scales)),
            reasons=reasons,
        )
