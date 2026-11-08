# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tesseract (pytesseract + system binary): clean scans, more than 100 languages. Extra [tesseract].

CPU only. Binary and languages: `apt install tesseract-ocr tesseract-ocr-spa` (Windows: UB Mannheim
installer). Apache-2.0 license. If it is not installed, the router simply does not use it.
"""

from __future__ import annotations

import os
import shutil

import numpy as np

from ..exceptions import InferenceError, ModelNotAvailableError
from ..result import TextBlock
from ..routing.languages import TESSERACT_RENAMED, TESSERACT_SCRIPTS, tesseract_codes
from ..utils.deps import missing_modules, require
from ..utils.logging import get_logger
from .base import EngineInfo, OCREngine

_WINDOWS_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def find_tesseract() -> str | None:
    """Path of the tesseract binary (PATH or the default install location on Windows)."""
    found = shutil.which("tesseract")
    if found:
        return found
    return next((p for p in _WINDOWS_PATHS if os.path.exists(p)), None)


def lines_from_data(data: dict) -> list[TextBlock]:
    """Words from pytesseract.image_to_data -> one TextBlock per line (block, paragraph, line)."""
    lines: dict[tuple, list[int]] = {}
    for i, word in enumerate(data.get("text", [])):
        if float(data["conf"][i]) < 0 or not str(word).strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)
    blocks: list[TextBlock] = []
    for idx in lines.values():
        box = (
            min(int(data["left"][i]) for i in idx),
            min(int(data["top"][i]) for i in idx),
            max(int(data["left"][i]) + int(data["width"][i]) for i in idx),
            max(int(data["top"][i]) + int(data["height"][i]) for i in idx),
        )
        blocks.append(
            TextBlock(
                text=" ".join(str(data["text"][i]).strip() for i in idx),
                box=box,
                confidence=round(sum(float(data["conf"][i]) for i in idx) / len(idx) / 100.0, 4),
                engine="tesseract",
            )
        )
    return blocks


class TesseractEngine(OCREngine):
    info = EngineInfo(
        name="tesseract",
        scripts=TESSERACT_SCRIPTS,
        kinds=frozenset({"scan"}),
        gpu="no",
        cost=2,
        memory_mb=20,
        mb_per_megapixel=15,  # the tesseract program, separate but on the same machine: A4 at 300 dpi, 124 MB
        extra="tesseract",
        description="Clean document scans, more than 100 languages",
    )

    @classmethod
    def unavailable_reason(cls) -> str | None:
        reason = missing_modules("pytesseract", extra=cls.info.extra)
        if reason:
            return reason
        if find_tesseract() is None:
            return "the tesseract binary is missing (apt install tesseract-ocr / Windows installer)"
        return None

    def _build(self) -> object:
        """(pytesseract, languages): the requested ones that are installed in Tesseract, also under another name
        (tgl, fil). Without the languages of the page, its Latin is read in English, with a warning; another
        script does not even load: in English it would only give garbage."""
        pt = require("pytesseract", extra=self.info.extra)
        if self.config.threads:  # Tesseract spreads the work with OpenMP; the program inherits the environment
            os.environ.setdefault("OMP_THREAD_LIMIT", str(self.config.threads))
        cmd = find_tesseract()
        if cmd:
            pt.pytesseract.tesseract_cmd = cmd
        try:
            installed = set(pt.get_languages(config=""))
        except Exception:  # noqa: BLE001 - without a list of languages: what was requested is tried
            installed = set()
        wanted = tesseract_codes(self.config.language_hints, self.script)
        if not installed:
            return pt, "+".join(wanted)
        wanted = [c if c in installed else TESSERACT_RENAMED.get(c, c) for c in wanted]
        found, missing = [c for c in wanted if c in installed], [c for c in wanted if c not in installed]
        if missing:
            packages = ", ".join(f"tesseract-ocr-{c.replace('_', '-')}" for c in missing)
            if self.script != "latin" and not set(found) - {"eng"}:
                raise ModelNotAvailableError(f"tesseract has no data for the script {self.script}: install {packages}")
            if self.config.language_hints:  # without languages, the Spanish of the default is not something asked for
                get_logger("omnilector", self.config.log_level).warning(
                    "tesseract has no data for %s: reading without it (install %s)", ", ".join(missing), packages
                )
        return pt, "+".join(found or ["eng"])

    def prepare(self) -> object | None:
        """Tesseract downloads nothing: if the data of a script is missing, it is said and the rest gets ready."""
        try:
            return self.load()
        except ModelNotAvailableError as e:
            get_logger("omnilector", self.config.log_level).warning("%s", e)
            return None

    def read(self, image: np.ndarray) -> list[TextBlock]:
        from PIL import Image

        pt, lang = self.load()  # type: ignore[misc]
        pil = Image.fromarray(image)
        pil.format = "PPM"  # pytesseract passes the image through a temporary file: PPM ~40x faster than PNG
        try:
            data = pt.image_to_data(pil, lang=lang, config="--oem 1 --psm 3", output_type=pt.Output.DICT)
        except Exception as e:
            raise InferenceError(f"Tesseract failed: {e}") from e
        return lines_from_data(data)
