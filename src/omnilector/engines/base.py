# SPDX-License-Identifier: AGPL-3.0-or-later
"""Common interface of the OCR engines and the capabilities each one declares (the router uses them)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..exceptions import DependencyMissingError, ModelNotAvailableError, OmnilectorError
from ..result import RecognitionCandidate, TextBlock
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..config import OCRConfig

Box = tuple[int, int, int, int]
PAGE_KINDS = frozenset({"scan", "photo"})


@dataclass(frozen=True)
class EngineInfo:
    """What an engine can do. The router decides with this, never with ifs on names."""

    name: str
    level: str = "page"  # page: detects + recognizes | line: only recognizes line crops
    scripts: frozenset[str] = frozenset({"latin"})
    kinds: frozenset[str] = PAGE_KINDS  # kinds of page where it is a good choice (scan, photo)
    handwriting: bool = False
    gpu: str = "optional"  # no | optional | required ("required" ones are not used with device="cpu")
    cost: int = 1  # 1 cheap ... 3 expensive (on CPU): breaks ties between equally suitable engines
    languages: frozenset[str] = frozenset()  # languages it is the best at (empty: good for any)
    # Line engines: languages whose printed text it reads better than the page engines, so it reads every line
    print_languages: frozenset[str] = frozenset()
    # Languages it gets wrong without doubting, so nothing would send those lines to another engine. A page engine
    # goes after the others and does not join their consensus (its sure readings would win the fusion); the value, the
    # mean confidence below which the others read worse than it does, and it reads after all (0: never). A line
    # engine does not read the doubtful lines of a page in that language
    weak_languages: dict[str, float] = field(default_factory=dict, hash=False)
    detector: bool = False  # page engines: detect() finds the lines of any script, even without a model to read them
    # Memory it takes when loaded, in MB (measured on Linux, CPU), without its library's (`runtime`): with
    # memory_mb, those that do not fit are not used unless forced, and loaded ones are released if needed
    memory_mb: int = 100
    runtime: str = ""  # library that stays loaded even if the model is released ("torch")
    # Memory to read an image, per megapixel (page engines): RapidOCR 150, EasyOCR 770
    mb_per_megapixel: int = 0
    extra: str = ""  # pip install "omnilector[extra]"
    description: str = ""


class OCREngine:
    """Adapter of one OCR. The model is loaded once, on the first read or with load().

    New engine = subclass with `info`, `_build()` (loads the model) and read() (level=page) or
    read_lines() (level=line) + engines.register_engine(). The pipeline and the router stay untouched.
    """

    info: ClassVar[EngineInfo]
    max_side: int | None = None  # max side it processes without reducing the image (None = no limit)

    def __init__(self, config: OCRConfig, script: str = "latin") -> None:
        self.config = config
        self.script = script
        self._lock = threading.Lock()
        self._model: object = None
        self._load_error: OmnilectorError | None = None

    @classmethod
    def unavailable_reason(cls) -> str | None:
        """None if it can be used; otherwise the reason (without importing anything or loading models)."""
        return None

    @classmethod
    def recognizer_for(cls, script: str) -> object:
        """Which model reads that script: two scripts with the same value share a model (so the same one is
        not tried twice when looking for the script of a page)."""
        return script

    def _build(self) -> object:
        """Loads the model (downloading it the first time). By default there is nothing to load."""
        return True

    @property
    def loaded(self) -> bool:
        """Whether the model is loaded (takes memory)."""
        return self._model is not None

    def resident_mb(self) -> int:
        """Memory its model takes now (0 if not loaded)."""
        return self.info.memory_mb if self.loaded else 0

    def load(self) -> object:
        """The model, loaded only once. If loading fails (no network, no cache...) it is not retried on every
        page: the same error is raised again."""
        with self._lock:
            if self._load_error is not None:
                raise self._load_error
            if self._model is None:
                try:
                    self._model = self._build()
                except DependencyMissingError:
                    raise
                except OmnilectorError as e:
                    self._load_error = e
                    raise
                except Exception as e:
                    self._load_error = ModelNotAvailableError(
                        f"{self.info.name}: could not load the model (offline? cache?): {e}"
                    )
                    raise self._load_error from e
            return self._model

    def prepare(self) -> object:
        """Everything the engine may load, already loaded (and downloaded): for load_models() and
        `omnilector download`. By default, its model."""
        return self.load()

    def read(self, image: np.ndarray) -> list[TextBlock]:
        """Whole RGB page -> blocks with coordinates in that image (level=page engines)."""
        raise NotImplementedError(f"{self.info.name} does not read whole pages")

    def detect(self, image: np.ndarray) -> list[TextBlock]:
        """Whole RGB page -> its lines without text, for a line engine to read them (info.detector)."""
        raise NotImplementedError(f"{self.info.name} does not only detect lines")

    def read_lines(self, image: np.ndarray, boxes: list[Box]) -> list[RecognitionCandidate | None]:
        """Line crops -> one candidate per box, None if it reads nothing (level=line engines)."""
        raise NotImplementedError(f"{self.info.name} does not recognize single lines")

    def upside_down(self, image: np.ndarray, boxes: list[Box]) -> bool | None:
        """Whether the lines in those (horizontal) boxes are upside down; None if the engine cannot tell.
        With it, the page is read the right way up and in its order."""
        return None


_warned_no_gpu = False


def _gpu(config: OCRConfig, probe: Callable[[], bool]) -> bool:
    """GPU if requested and `probe` sees it; otherwise CPU (server without a graphics card: warning, not error)."""
    global _warned_no_gpu
    if config.device == "cpu":
        return False
    try:
        if probe():
            return True
    except ImportError:
        pass
    if config.device != "auto" and not _warned_no_gpu:
        _warned_no_gpu = True
        get_logger("omnilector", config.log_level).warning(
            "device=%s but there is no GPU available: going on with the CPU", config.device
        )
    return False


def _torch_cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def _onnx_cuda() -> bool:
    import onnxruntime

    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


def torch_device(config: OCRConfig) -> str:
    return "cuda" if _gpu(config, _torch_cuda) else "cpu"


def onnx_cuda(config: OCRConfig) -> bool:
    return _gpu(config, _onnx_cuda)


def crop(image: np.ndarray, box: Box, pad: int = 0) -> np.ndarray | None:
    """Crop clipped to the image borders; None if nothing is left."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def polygon_block(points: object, text: str, confidence: float, engine: str) -> TextBlock:
    """4-point polygon (RapidOCR / EasyOCR format) -> TextBlock with its box and its corners (with them a
    skewed page is straightened)."""
    corners = [(float(p[0]), float(p[1])) for p in points]  # type: ignore[attr-defined]
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    quad = tuple((round(x), round(y)) for x, y in corners) if len(corners) == 4 else None
    return TextBlock(text=text, box=box, confidence=confidence, engine=engine, quad=quad)
