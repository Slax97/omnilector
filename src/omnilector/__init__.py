# SPDX-License-Identifier: AGPL-3.0-or-later
"""omnilector: extracts the text of any file, picking the right OCR for each situation.

Common output for every engine (and their combinations): Document -> Page -> Line.
"""

from .config import OCRConfig
from .engines import list_engines, register_engine
from .exceptions import (
    DependencyMissingError,
    DocumentLoadError,
    EngineNotAvailableError,
    InferenceError,
    ModelNotAvailableError,
    OmnilectorError,
    UnsupportedFormatError,
)
from .loaders import register_loader
from .ocr import OCR, read, read_many
from .result import Document, Line, Page

__all__ = [
    "OCR",
    "DependencyMissingError",
    "Document",
    "DocumentLoadError",
    "EngineNotAvailableError",
    "InferenceError",
    "Line",
    "ModelNotAvailableError",
    "OCRConfig",
    "OmnilectorError",
    "Page",
    "UnsupportedFormatError",
    "list_engines",
    "read",
    "read_many",
    "register_engine",
    "register_loader",
]
__version__ = "1.0.0"
