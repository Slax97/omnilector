# SPDX-License-Identifier: AGPL-3.0-or-later
"""kraken with the PP-OCRv6 medium model: handwritten and printed lines in 44 languages and 10 scripts
(Latin, Cyrillic, Greek, Arabic...). Extra [handwritten]. Apache-2.0 (kraken and the model).

Measured on real handwriting: in Spanish (notarial pages) 40 % character error against 78 % for TrOCR,
and it keeps accents and ñ; in English (IAM) TrOCR is still better (4.9 % against 10.4 %). That is why
the router leaves TrOCR for English and this model for the rest. ~5 times faster.

On print it beats the page engines in some languages (print_languages), line by line on the boxes of RapidOCR:
pages of Arabic books 5.5 % character error against 49 % for RapidOCR, Hebrew ones 5-16 % against 22-75 % for
Tesseract, clean Urdu in Nastaliq 6 % against 94 %. The model card also declares Ethiopic and Malayalam, but there
Tesseract is clearly better, so they are not offered.
"""

from __future__ import annotations

import os
import threading
from statistics import mean

import numpy as np

from ..exceptions import InferenceError
from ..result import RecognitionCandidate
from ..utils.deps import missing_modules, require
from ..utils.text import nfc, right_to_left
from .base import Box, EngineInfo, OCREngine, torch_device

RTL_SCRIPTS = frozenset({"arabic", "hebrew"})


class KrakenEngine(OCREngine):
    info = EngineInfo(
        name="kraken",
        level="line",
        scripts=frozenset({"latin", "cyrillic", "east_slavic", "greek", "arabic", "hebrew", "georgian", "armenian"}),
        handwriting=True,
        print_languages=frozenset({"ar", "fa", "ur", "he", "ka", "hy"}),
        cost=2,
        memory_mb=380,
        runtime="torch",
        extra="handwritten",
        description="Handwritten and printed text line by line, 44 languages (kraken PP-OCRv6)",
    )
    repo, filename = "small-models-for-glam/kraken-ppocrv6-medium", "medium.safetensors"
    # Lines per batch: each batch is padded up to its longest line. With 2, on a deed of 58 lines, the same
    # error (25.9 %) as with 8, 420 MB less while reading and faster (2 cores: 44 s against 76 s)
    batch_size = 2

    def __init__(self, config, script: str = "latin") -> None:
        super().__init__(config, script)
        self._predict_lock = threading.Lock()

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return missing_modules("kraken", "huggingface_hub", extra=cls.info.extra)

    def _build(self) -> object:
        tasks = require("kraken.tasks", "kraken", self.info.extra)
        configs = require("kraken.configs", "kraken", self.info.extra)
        hub = require("huggingface_hub", extra=self.info.extra)
        cache = self.config.model_cache_dir
        path = hub.hf_hub_download(
            self.repo, self.filename, cache_dir=os.path.join(cache, "huggingface") if cache else None
        )
        net = tasks.RecognitionTaskModel.load_model(path)
        gpu = torch_device(self.config) == "cuda"
        # without helper processes: on Windows they hung; the threads, those of config.threads
        config = configs.RecognitionInferenceConfig(
            num_line_workers=0,
            num_threads=self.config.threads or os.cpu_count() or 1,
            batch_size=self.batch_size,
            accelerator="gpu" if gpu else "cpu",
            device=1,
            bidi_reordering=self.script not in RTL_SCRIPTS,  # there read_lines decides the direction of each line
        )
        return net, config

    def read_lines(self, image: np.ndarray, boxes: list[Box]) -> list[RecognitionCandidate | None]:
        """All the lines at once; the confidence, the mean of each character's."""
        net, config = self.load()  # type: ignore[misc]
        containers = require("kraken.containers", "kraken", self.info.extra)
        from PIL import Image

        out: list[RecognitionCandidate | None] = [None] * len(boxes)
        height, width = image.shape[:2]
        # clipped to the image: EasyOCR gives boxes that go over its edge, and kraken does not read those
        clipped = [
            (max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1))) for x0, y0, x1, y1 in boxes
        ]
        lines = [
            containers.BBoxLine(id=str(i), bbox=box)
            for i, box in enumerate(clipped)
            if box[2] - box[0] >= 4 and box[3] - box[1] >= 4
        ]
        if not lines:
            return out
        segmentation = containers.Segmentation(
            type="bbox", imagename="page", text_direction="horizontal-lr", script_detection=False, lines=lines
        )
        try:
            with self._predict_lock:
                records = list(net.predict(Image.fromarray(image), segmentation, config))
        except Exception as e:
            raise InferenceError(f"kraken failed: {e}") from e
        for record in records:
            if self.script in RTL_SCRIPTS:
                # kraken goes by the first letter: a misread Latin one at the left end of a Persian line swapped its words
                record = record.logical_order(base_dir="R" if right_to_left(str(record.prediction)) else "L")
            text = nfc(str(record.prediction).strip())  # kraken writes accents separately (NFD)
            if text:
                conf = float(mean(record.confidences)) if record.confidences else 0.0
                out[int(record.id)] = RecognitionCandidate(text=text, confidence=conf, engine=self.info.name)
        return out
