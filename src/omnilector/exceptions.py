# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public exceptions of omnilector."""

from __future__ import annotations


class OmnilectorError(Exception):
    """Base of every error raised by the library."""


class DocumentLoadError(OmnilectorError):
    """Missing, corrupt, empty or unreadable input (image, PDF, docx...)."""


class UnsupportedFormatError(DocumentLoadError):
    """Unknown format, or no loader for it."""


class ModelNotAvailableError(OmnilectorError):
    """Model not downloaded / no network / no cache."""


class DependencyMissingError(OmnilectorError):
    """An optional dependency is missing (e.g. transformers for handwriting)."""

    def __init__(self, package: str, extra: str = "") -> None:
        hint = f' Install it with: pip install "omnilector[{extra}]"' if extra else ""
        super().__init__(f"Missing optional dependency '{package}'.{hint}")
        self.package = package
        self.extra = extra


class EngineNotAvailableError(OmnilectorError):
    """Unknown OCR engine, or not installed on this machine."""


class InferenceError(OmnilectorError):
    """A model failed during inference."""
