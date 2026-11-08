# SPDX-License-Identifier: AGPL-3.0-or-later
"""EasyOCR: photos and scene text, more than 80 languages. Extra [easyocr]. Apache-2.0 license.

Works on CPU (slower than RapidOCR); uses the GPU only if device asks for it and torch sees it.
"""

from __future__ import annotations

import os

import numpy as np

from ..exceptions import InferenceError
from ..result import TextBlock
from ..routing.languages import EASYOCR_SCRIPTS, easyocr_codes
from ..utils.deps import missing_modules, require
from ..utils.resources import engine_side
from .base import EngineInfo, OCREngine, polygon_block, torch_device


class EasyOCREngine(OCREngine):
    info = EngineInfo(
        name="easyocr",
        scripts=EASYOCR_SCRIPTS,
        kinds=frozenset({"photo"}),
        cost=3,
        memory_mb=350,
        runtime="torch",
        mb_per_megapixel=770,  # its detector (CRAFT) at almost full resolution: an A4 page at 2560 px, +3.5 GB
        extra="easyocr",
        description="Photos and scene text, more than 80 languages",
    )

    canvas_size = 2560  # side to which EasyOCR reduces images (its own default)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return missing_modules("easyocr", "torch", extra=cls.info.extra)

    def _build(self) -> object:
        easyocr = require("easyocr", extra=self.info.extra)
        if self.config.threads:
            require("torch", extra=self.info.extra).set_num_threads(self.config.threads)
        cache = self.config.model_cache_dir
        return easyocr.Reader(
            easyocr_codes(self.config.language_hints, self.script),
            gpu=torch_device(self.config) == "cuda",
            model_storage_directory=os.path.join(cache, "easyocr") if cache else None,
            verbose=False,
        )

    def read(self, image: np.ndarray) -> list[TextBlock]:
        reader = self.load()
        # with memory_mb, the side at which its detector fits (at most its own, 2560 px)
        side = engine_side(self.config.memory_mb, self.info.mb_per_megapixel)
        canvas = min(side, self.canvas_size) if side else None
        try:
            results = reader.readtext(  # type: ignore[attr-defined]
                image, detail=1, paragraph=False, **({"canvas_size": canvas} if canvas else {})
            )
        except Exception as e:
            raise InferenceError(f"EasyOCR failed: {e}") from e
        return [
            polygon_block(pts, str(text), float(conf), self.info.name)
            for pts, text, conf in results
            if str(text).strip()
        ]
