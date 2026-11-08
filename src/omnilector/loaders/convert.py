# SPDX-License-Identifier: AGPL-3.0-or-later
"""Legacy Office (.doc, .xls, .ppt) and RTF: converted with headless LibreOffice to docx, xlsx or pptx and
read with the Office loader (tables by cell, embedded images with OCR...).

LibreOffice is optional, like the Tesseract binary: if it is not installed, the error says how to get
it. It is an external program (MPL-2.0): it is not linked with the library.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from ..exceptions import DocumentLoadError, UnsupportedFormatError
from .base import RawDocument, Source

if TYPE_CHECKING:
    from ..config import OCRConfig

TARGETS = {"doc": "docx", "rtf": "docx", "xls": "xlsx", "ppt": "pptx"}
TIMEOUT = 180  # seconds per document
_INSTALLED = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
)
_LOCK = threading.Lock()  # one conversion at a time per process: LibreOffice does not share its profile
_PROFILE = Path(tempfile.gettempdir()) / f"omnilector_libreoffice_{os.getpid()}"  # one per process (workers)


def find_soffice() -> str | None:
    """Path of LibreOffice (PATH or the default install location on Windows and macOS)."""
    found = shutil.which("soffice") or shutil.which("libreoffice")
    return found or next((p for p in _INSTALLED if os.path.exists(p)), None)


@atexit.register
def _remove_profile() -> None:
    shutil.rmtree(_PROFILE, ignore_errors=True)


def convert(data: bytes, kind: str, name: str) -> bytes:
    """Bytes of a .doc/.xls/.ppt/.rtf -> bytes of the equivalent docx/xlsx/pptx."""
    target = TARGETS[kind]
    soffice = find_soffice()
    if soffice is None:
        raise UnsupportedFormatError(
            f"{name}: reading .{kind} needs LibreOffice (apt install libreoffice-writer libreoffice-calc "
            f"libreoffice-impress, or libreoffice.org), or save it as {target} or PDF"
        )
    with _LOCK, tempfile.TemporaryDirectory(prefix="omnilector_") as tmp:
        source = Path(tmp) / f"documento.{kind}"
        source.write_bytes(data)
        cmd = [
            soffice,
            "--headless",
            "--norestore",
            "--nolockcheck",
            f"-env:UserInstallation={_PROFILE.as_uri()}",
            "--convert-to",
            target,
            "--outdir",
            tmp,
            str(source),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=TIMEOUT, check=False)
        except subprocess.TimeoutExpired as e:
            raise DocumentLoadError(f"{name}: LibreOffice did not finish converting it in {TIMEOUT} s") from e
        converted = source.with_suffix(f".{target}")
        if not converted.exists():
            raise DocumentLoadError(f"{name}: LibreOffice could not convert it (damaged or password protected?)")
        return converted.read_bytes()


def load(src: Source, config: OCRConfig, kind: str) -> RawDocument:
    from . import load as load_any  # deferred: loaders/__init__ imports this module

    data = convert(src.read_bytes(), kind, src.name)
    target = TARGETS[kind]
    inner = Source(name=f"{Path(src.name).stem}.{target}", data=data, depth=src.depth, budget=src.budget)
    raw = load_any(inner, config)
    raw.kind, raw.source = kind, src.name
    raw.metadata["converted"] = f".{kind} -> {target} with LibreOffice"
    return raw
