# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pluggable OCR engines. New engine = subclass of OCREngine + register_engine()."""

from __future__ import annotations

from dataclasses import asdict

from ..exceptions import EngineNotAvailableError
from ..result import RecognitionCandidate, TextBlock
from ..utils.resources import RUNTIME_MB
from .base import EngineInfo, OCREngine
from .easyocr import EasyOCREngine
from .kraken import KrakenEngine
from .rapidocr import RapidOCREngine
from .tesseract import TesseractEngine
from .trocr import TrOCREngine

_ENGINES: dict[str, type[OCREngine]] = {}


def register_engine(cls: type[OCREngine]) -> type[OCREngine]:
    """Registers (or replaces) an engine by its info.name. It can be used as a decorator."""
    _ENGINES[cls.info.name] = cls
    return cls


def get_engine_class(name: str) -> type[OCREngine]:
    try:
        return _ENGINES[name]
    except KeyError:
        raise EngineNotAvailableError(f"Unknown engine '{name}'. Registered: {', '.join(sorted(_ENGINES))}") from None


def registered_engines() -> list[type[OCREngine]]:
    return list(_ENGINES.values())


def list_engines() -> list[dict]:
    """Registered engines: whether they are available on this machine and how to install the missing ones."""
    out = []
    for cls in _ENGINES.values():
        reason = cls.unavailable_reason()
        info = asdict(cls.info) | {"scripts": sorted(cls.info.scripts), "kinds": sorted(cls.info.kinds)}
        info["memory_total_mb"] = cls.info.memory_mb + RUNTIME_MB.get(cls.info.runtime, 0)  # with its library
        out.append(info | {"available": reason is None, "reason": reason})
    return out


for _cls in (RapidOCREngine, TrOCREngine, KrakenEngine, TesseractEngine, EasyOCREngine):
    register_engine(_cls)

__all__ = [
    "EasyOCREngine",
    "EngineInfo",
    "KrakenEngine",
    "OCREngine",
    "RapidOCREngine",
    "RecognitionCandidate",
    "TesseractEngine",
    "TextBlock",
    "TrOCREngine",
    "get_engine_class",
    "list_engines",
    "register_engine",
    "registered_engines",
]
