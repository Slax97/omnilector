# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strategy + Template Method: each strategy decides the passes, with no ifs in the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...result import TextBlock


class OCRStrategy(ABC):
    """Template Method: the pipeline always calls the same hooks."""

    name: str = "base"
    max_image_side: int = 4000  # max side the OCR sees (larger images are reduced)
    embedded_images: bool = True  # OCR of the images inside PDF/docx/pptx
    default_scales: tuple[float, ...] = (1.0, 1.5, 2.0)  # scales of pass 2 (config.scales wins)
    layout: bool = True  # layout (zones, reading order, tables) on pages with OCR

    @abstractmethod
    def variant_names(self, preprocessing: bool) -> list[str]:
        """Which variants to try in PASS2. Always includes 'original' if there is a PASS2."""
        ...

    def scales(self, multi_scale: bool, scales: tuple[float, ...]) -> tuple[float, ...]:
        return scales if multi_scale else (1.0,)

    def allow_second_pass(self) -> bool:
        return True

    def use_handwriting(self, handwritten: bool) -> bool:
        return handwritten

    def use_line_engine_on_print(self) -> bool:
        """In the languages where a line engine reads print better than the page engines
        (EngineInfo.print_languages), it reads every line, with or without handwriting."""
        return True

    def use_ensemble(self) -> bool:
        """If text is missing after pass 1, also read with the other engines and fuse by consensus."""
        return False

    def needs_second_pass(self, blocks: list[TextBlock], image: np.ndarray) -> bool:
        """Signs of missing text: few blocks, low confidence, low coverage."""
        if not blocks:
            return True
        if len(blocks) < 3:
            return True
        mean_conf = sum(b.confidence for b in blocks) / len(blocks)
        if mean_conf < 0.5:
            return True
        h, w = image.shape[:2]
        area = sum((b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) for b in blocks) / (w * h)
        return area < 0.02
