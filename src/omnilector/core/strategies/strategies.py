# SPDX-License-Identifier: AGPL-3.0-or-later
"""Concrete strategies: fast / balanced / accurate / maximum_recall."""

from __future__ import annotations

import numpy as np

from ...result import TextBlock
from .base import OCRStrategy


class FastStrategy(OCRStrategy):
    name = "fast"
    max_image_side = 2000
    embedded_images = False  # PDF with text + images: only the native text
    default_scales = (1.0,)
    layout = False  # one pass and nothing else

    def variant_names(self, preprocessing: bool) -> list[str]:
        return []

    def scales(self, multi_scale: bool, scales: tuple[float, ...]) -> tuple[float, ...]:
        return (1.0,)  # a single pass

    def allow_second_pass(self) -> bool:
        return False

    def use_handwriting(self, handwritten: bool) -> bool:
        return False

    def use_line_engine_on_print(self) -> bool:
        return False  # one quick pass: on a book page kraken took 10 to 30 times longer than RapidOCR or Tesseract

    def needs_second_pass(self, blocks: list[TextBlock], image: np.ndarray) -> bool:
        return False


class BalancedStrategy(OCRStrategy):
    name = "balanced"
    max_image_side = 2000
    default_scales = (1.0, 1.5)

    def variant_names(self, preprocessing: bool) -> list[str]:
        return ["original", "clahe"] if preprocessing else ["original"]

    def use_handwriting(self, handwritten: bool) -> bool:
        return False  # balanced favours speed; handwriting only in accurate/maximum_recall


class AccurateStrategy(OCRStrategy):
    name = "accurate"

    def variant_names(self, preprocessing: bool) -> list[str]:
        return ["original", "clahe", "sharpen"] if preprocessing else ["original"]


class MaximumRecallStrategy(AccurateStrategy):
    name = "maximum_recall"

    def use_ensemble(self) -> bool:
        return True  # several engines + consensus (only when there are signs of missing text)


def get_strategy(name: str) -> OCRStrategy:
    """Factory: name -> instance. New strategy = new class + 1 line here."""
    mapping = {
        "fast": FastStrategy,
        "balanced": BalancedStrategy,
        "accurate": AccurateStrategy,
        "maximum_recall": MaximumRecallStrategy,
    }
    try:
        return mapping[name]()
    except KeyError:
        raise ValueError(f"strategy must be one of {set(mapping)}") from None
