# SPDX-License-Identifier: AGPL-3.0-or-later
"""Routing: what situation each page is in and which OCR it gets."""

from .analyzer import PageProfile, analyze
from .languages import main_script, normalize_languages
from .router import EngineSpec, Plan, Router

__all__ = ["EngineSpec", "PageProfile", "Plan", "Router", "analyze", "main_script", "normalize_languages"]
