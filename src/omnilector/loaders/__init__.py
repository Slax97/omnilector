# SPDX-License-Identifier: AGPL-3.0-or-later
"""Loaders: any input -> pages with native text and/or an image for OCR.

New format = function (Source, OCRConfig) -> RawDocument + register_loader().
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from functools import partial
from typing import TYPE_CHECKING

from ..exceptions import UnsupportedFormatError
from . import archive, convert, email, image, msg, odf, office, pdf, text
from .base import (
    ImageRegion,
    OCRBudget,
    RawDocument,
    RawPage,
    Source,
    grid_rows,
    source_name,
    to_source,
)
from .detect import detect_kind

if TYPE_CHECKING:
    from ..config import OCRConfig

Loader = Callable[[Source, "OCRConfig"], RawDocument]

_LOADERS: dict[str, Loader] = {
    "image": image.load,
    "pdf": pdf.load,
    "epub": pdf.load_epub,
    "xps": pdf.load_xps,
    "ebook": pdf.load_ebook,
    "docx": office.load_docx,
    "xlsx": office.load_xlsx,
    "pptx": office.load_pptx,
    "odt": odf.load_odt,
    "ods": odf.load_ods,
    "odp": odf.load_odp,
    "html": text.load_html,
    "text": text.load_text,
    "email": email.load,
    "msg": msg.load,
    "zip": archive.load,
    "tar": partial(archive.load, kind="tar"),  # .tar, .tar.gz/.tgz, .tar.bz2, .tar.xz
    "compressed": partial(archive.load, kind="compressed"),  # single-file .gz, .bz2, .xz
    **{kind: partial(convert.load, kind=kind) for kind in convert.TARGETS},  # doc, xls, ppt, rtf
}
_EXTENSIONS: dict[str, str] = {}  # extension -> kind of the loaders registered by the user


def register_loader(kind: str, loader: Loader, extensions: tuple[str, ...] = ()) -> None:
    """Adds (or replaces) the loader of a kind. `extensions` (".rtf", ...) associate it with files."""
    _LOADERS[kind] = loader
    for ext in extensions:
        _EXTENSIONS[ext.lower() if ext.startswith(".") else f".{ext.lower()}"] = kind


def kind_of(src: Source) -> str:
    """The kind of the input, counting the loaders registered by the user."""
    return detect_kind(src, _EXTENSIONS)


def load(value: object, config: OCRConfig) -> RawDocument:
    """str | Path | bytes | stream | PIL | ndarray -> RawDocument (lazy pages)."""
    src = to_source(value)
    kind = kind_of(src)
    loader = _LOADERS.get(kind)
    if loader is None:
        raise UnsupportedFormatError(f"No loader for '{kind}': {src.name}")
    if src.budget is None:
        deadline = time.monotonic() + config.time_limit if config.time_limit else None
        src.budget = OCRBudget(config.max_ocr_pages, deadline=deadline)
    raw = loader(src, config)
    raw.source = raw.source or src.name
    raw.budget = src.budget
    return raw


def archive_members(src: Source, config: OCRConfig) -> tuple[list[str], Iterator[tuple[Source, str | None]]] | None:
    """If the input is a ZIP or a TAR, its files as separate inputs (see archive.members); otherwise None
    (a single-file .gz is that file)."""
    kind = kind_of(src)
    return archive.members(src, kind, config.passwords) if kind in ("zip", "tar") else None


__all__ = [
    "ImageRegion",
    "OCRBudget",
    "RawDocument",
    "RawPage",
    "Source",
    "archive_members",
    "detect_kind",
    "grid_rows",
    "kind_of",
    "load",
    "register_loader",
    "source_name",
    "to_source",
]
