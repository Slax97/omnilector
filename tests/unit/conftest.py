# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fixtures of the unit tests. The helpers (fake engines, test files) live in tests/helpers.py and
are re-exported here for the tests' `from .conftest import ...`."""

from __future__ import annotations

import pytest

from omnilector import engines as engines_mod
from omnilector import ocr as ocr_mod
from omnilector.core.layout import LayoutAnalyzer
from omnilector.engines.base import OCREngine
from omnilector.utils import resources

from ..helpers import (
    make_engine,
    make_hybrid_pdf,
    make_mixed_pdf,
    png_bytes,
    reader,
    tb,
    tilted,
    white,
)

__all__ = ["make_engine", "make_hybrid_pdf", "make_mixed_pdf", "png_bytes", "reader", "tb", "tilted", "white"]


@pytest.fixture
def use_engines(monkeypatch):
    """Replaces the engine registry with the given fake ones (restored at the end)."""

    def _use(*classes: type[OCREngine]) -> None:
        monkeypatch.setattr(engines_mod, "_ENGINES", {c.info.name: c for c in classes})

    return _use


@pytest.fixture(autouse=True)
def _no_layout_models(monkeypatch):
    """Unit tests do not load the layout model (the layout tests use a fake one)."""
    monkeypatch.setattr(LayoutAnalyzer, "unavailable_reason", staticmethod(lambda: "no models in unit tests"))


@pytest.fixture(autouse=True)
def _no_server_limits(monkeypatch):
    """Tests do not depend on the machine: no limit of cores or memory unless the test sets one (memory_mb,
    threads), and without the environment configuration of whoever runs them."""
    monkeypatch.setattr(resources, "cpu_limit", lambda: None)
    monkeypatch.setattr(resources, "available_memory_mb", lambda: None)
    for name in ("THREADS", "MEMORY_MB", "STRATEGY", "LANGUAGES", "DETECT_LANGUAGES", "HANDWRITTEN", "TIME_LIMIT"):
        monkeypatch.delenv(f"OMNILECTOR_{name}", raising=False)


@pytest.fixture(autouse=True)
def _fresh_shared_ocr(monkeypatch):
    """omnilector.read() reuses instances: each test starts without any."""
    monkeypatch.setattr(ocr_mod, "_SHARED", {})
