# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public configuration of omnilector."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .core.strategies import get_strategy
from .routing.languages import normalize_languages, scripts_to_detect
from .utils import resources

T = TypeVar("T")

DEFAULT_STRATEGY = "maximum_recall"
_BOOLS = {
    "1": True,
    "true": True,
    "yes": True,
    "on": True,
    "0": False,
    "false": False,
    "no": False,
    "off": False,
}


def _env(name: str, parse: Callable[[str], T]) -> T | None:
    """Value of an environment variable; None if it is not set or empty. A value that is not valid raises an
    error naming the variable (whoever set it knows where to look)."""
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return parse(value)
    except ValueError as e:
        raise ValueError(f"{name}={value!r}: {e}") from None


def _strategy(value: str) -> str:
    return get_strategy(value).name  # ValueError if it does not exist


def _bool(value: str) -> bool:
    try:
        return _BOOLS[value.lower()]
    except KeyError:
        raise ValueError("use 1/0, true/false, yes/no or on/off") from None


@dataclass
class OCRConfig:
    """OCR configuration.

    Strategies: fast | balanced | accurate | maximum_recall.
    maximum_recall is the main strategy for evaluation.
    Everything works on CPU: device="cuda" without an available GPU warns and goes on with the CPU.
    Fields left as None take the value of the given environment variable or, without it, the usual one (or
    the strategy's): what is passed explicitly wins. Without threads or memory_mb, it adapts to what the
    process has (container, service or machine).
    """

    # None = OMNILECTOR_STRATEGY variable, or maximum_recall
    strategy: str | None = None
    device: str = "cpu"  # cpu | cuda | gpu | auto (GPU if there is one, otherwise CPU, without warning)
    # Re-read doubtful lines with a handwriting model; None = OMNILECTOR_HANDWRITTEN variable, or yes
    handwritten: bool | None = None
    preprocessing: bool = True
    multi_scale: bool = True
    model_cache_dir: str | None = None  # None = OMNILECTOR_MODEL_DIR variable, or each engine's cache
    # None = the application configures logging; "DEBUG" shows the situation and plan of each page
    log_level: str | None = None
    # Thresholds aimed at recall (low on purpose)
    det_box_thresh: float = 0.3
    handwriting_conf_threshold: float = 0.85
    # Scales of pass 2; None = the strategy's (balanced 1-1.5, accurate/maximum_recall 1-2)
    scales: tuple[float, ...] | None = None
    # RapidOCR model (PP-OCRv6): tiny | small | large (more recall, slower)
    model_size: str = "small"
    # Expected languages (ISO 639-1: "es", "en", "ru"...): they pick the model/engine per script. None =
    # OMNILECTOR_LANGUAGES variable ("es,en") or none; () = none even if the variable is set
    languages: tuple[str, ...] | None = None
    # Without languages: if there are lines the Latin/Chinese/Japanese model cannot read, the models of the
    # other scripts are tried (Cyrillic, Arabic, Korean, Greek...) and the page is read with the one that does
    auto_script: bool = True
    # Languages to detect among, without languages ("es", "ru", "ar"): only the models of their scripts are
    # tried (faster, and a Spanish form can no longer come out as Greek or Kannada; a Cyrillic language brings
    # both Cyrillic models), and the handwriting language and the encoding of legacy texts are chosen among
    # them. None = OMNILECTOR_DETECT_LANGUAGES variable ("es,ru,ar") or all; () = all even if the variable is
    # set
    detect_languages: tuple[str, ...] | None = None
    # Force a page engine ("rapidocr", "tesseract", "easyocr"); None = the router chooses
    engine: str | None = None
    # CPU threads per engine (servers with several workers); None = OMNILECTOR_THREADS variable or the
    # cores the process can use (its CPU quota and the cores assigned to it); without a limit, each engine uses
    # all of them
    threads: int | None = None
    # Memory for the OCR, in MB, per process; None = OMNILECTOR_MEMORY_MB variable, the memory limit of the
    # process's cgroup or the machine's RAM. It decides the maximum image side,
    # whether there is layout analysis and which engines fit (those that do not are not used unless forced)
    memory_mb: int | None = None
    # Maximum side the OCR sees; None = the strategy's (2000 fast/balanced, 4000 accurate/
    # maximum_recall) or less if it does not fit in memory_mb (2 GB: 2500)
    max_image_side: int | None = None
    # Pages that go through the OCR (scans, images, photos inside a PDF...), for safety: they are the
    # expensive ones. Pages with native text do not count. None = no limit
    max_ocr_pages: int | None = 50
    # Pages of any kind (PDF, TIFF, sheets, slides); None = all
    max_pages: int | None = None
    pdf_dpi: int = 300
    pdf_min_chars: int = 50  # fewer useful characters on a PDF page -> it is treated as scanned
    ocr_embedded_images: bool | None = None  # OCR of images inside PDF/docx/pptx; None = strategy
    # Layout: zones in reading order and tables with cells (extra [layout]); None = strategy (all but
    # fast) if it fits in memory_mb. In PDFs with text, tables come from the PDF itself
    layout: bool | None = None
    # Passwords for protected PDFs and encrypted files in a ZIP: tried in order
    passwords: tuple[str, ...] = ()
    # Seconds per document. After that time, the remaining pages do not go through the OCR and the one being
    # read is finished without extra passes (variants, consensus, handwriting, layout); a warning says so.
    # It is checked between steps: a single long read is not cut. None = OMNILECTOR_TIME_LIMIT variable or
    # no limit
    time_limit: float | None = None

    def __post_init__(self) -> None:
        if self.strategy is None:
            self.strategy = _env("OMNILECTOR_STRATEGY", _strategy) or DEFAULT_STRATEGY
        strategy = get_strategy(self.strategy)  # ValueError if it does not exist
        if self.handwritten is None:
            env = _env("OMNILECTOR_HANDWRITTEN", _bool)
            self.handwritten = True if env is None else env
        if self.languages is None:
            self.languages = _env("OMNILECTOR_LANGUAGES", normalize_languages) or ()
        if self.detect_languages is None:
            self.detect_languages = _env("OMNILECTOR_DETECT_LANGUAGES", normalize_languages) or ()
        if self.device not in {"cpu", "cuda", "gpu", "auto"}:
            raise ValueError("device must be cpu|cuda|gpu|auto")
        if self.model_size not in {"tiny", "small", "large"}:
            raise ValueError("model_size must be tiny|small|large")
        if self.threads is None:
            env = _env("OMNILECTOR_THREADS", int)
            self.threads = resources.cpu_limit() if env is None else env
        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be >= 1")
        if self.memory_mb is None:
            env = _env("OMNILECTOR_MEMORY_MB", resources.parse_memory)
            self.memory_mb = resources.available_memory_mb() if env is None else env
        if self.memory_mb is not None and self.memory_mb < 256:
            raise ValueError("memory_mb must be >= 256 (or None: whatever the process has)")
        if self.time_limit is None:
            self.time_limit = _env("OMNILECTOR_TIME_LIMIT", float)
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("time_limit must be > 0 seconds (or None, no limit)")
        if any(v is not None and v < 0 for v in (self.max_pages, self.max_ocr_pages)):
            raise ValueError("max_pages and max_ocr_pages must be >= 0 (or None, no limit)")
        if self.scales is None:
            self.scales = strategy.default_scales
        if self.max_image_side is None:
            self.max_image_side = strategy.max_image_side
            if self.memory_mb is not None:
                self.max_image_side = min(self.max_image_side, resources.image_side_for(self.memory_mb))
        if self.ocr_embedded_images is None:
            self.ocr_embedded_images = strategy.embedded_images
        if self.layout is None:
            budget = self.models_budget_mb
            self.layout = strategy.layout and (
                budget is None or budget >= resources.LAYOUT_MB + resources.PAGE_ENGINE_MB
            )
        self.languages = normalize_languages(self.languages)
        self.detect_languages = normalize_languages(self.detect_languages)
        self.passwords = (self.passwords,) if isinstance(self.passwords, str) else tuple(self.passwords)
        self.model_cache_dir = self.model_cache_dir or os.environ.get("OMNILECTOR_MODEL_DIR") or None
        if self.model_cache_dir:
            Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def models_budget_mb(self) -> int | None:
        """Memory for the loaded models (memory_mb minus the process and the image); None without a limit."""
        return resources.models_budget_mb(self.memory_mb, self.max_image_side)

    @property
    def detect_scripts(self) -> frozenset[str] | None:
        """Scripts to detect among: those of languages or, without them, those of detect_languages (and those
        sharing their letters); None = all."""
        return scripts_to_detect(self.languages or self.detect_languages or ()) or None

    @property
    def language_hints(self) -> tuple[str, ...]:
        """Expected languages: those of languages or, without them, those of detect_languages (their encodings
        are preferred when decoding a legacy text)."""
        return self.languages or self.detect_languages or ()
