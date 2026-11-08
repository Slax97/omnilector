# SPDX-License-Identifier: AGPL-3.0-or-later
"""Outlook .msg (OLE container) with olefile (BSD). Extra [msg].

Same as an .eml: headers + body (text or, if there is none, HTML) and each attachment with its loader,
also the body images (pasted screenshots) and attached emails (.msg inside the .msg). Each piece of
data is a __substg1.0_PPPPTTTT stream (PPPP = MAPI property, TTTT = type).
"""

from __future__ import annotations

import codecs
import contextlib
import datetime as dt
import struct
from collections.abc import Iterator
from typing import TYPE_CHECKING

from ..exceptions import DocumentLoadError
from ..utils.deps import require
from .base import MAX_DEPTH, RawDocument, RawPage, Source, grid_rows, nested_pages
from .text import decode_text, html_cells, text_cells

if TYPE_CHECKING:
    from ..config import OCRConfig

_HEADERS = (("0C1A", "From"), ("0E04", "To"), ("0E03", "CC"), ("date", "Date"), ("0037", "Subject"))
_KEYS = {"0C1A": "from", "0E04": "to", "0E03": "cc", "date": "date", "0037": "subject"}
_SUBMIT_TIME, _DELIVERY_TIME, _CODEPAGE, _INTERNET_CPID = 0x00390040, 0x0E060040, 0x3FFD0003, 0x3FDE0003


class _Message:
    """A message inside the OLE file: the main one or an attached one (storage)."""

    def __init__(self, ole, base: list[str], top: bool) -> None:
        self.ole, self.base = ole, base
        self.props = self._properties(32 if top else 24)
        cp = self.props.get(_CODEPAGE) or self.props.get(_INTERNET_CPID)
        self.codepage = "cp1252"
        if cp:
            with contextlib.suppress(LookupError):  # unknown encoding: cp1252 stays
                self.codepage = codecs.lookup(f"cp{cp & 0xFFFF}").name

    def _stream(self, *names: str) -> bytes | None:
        path = [*self.base, *names]
        return self.ole.openstream(path).read() if self.ole.exists("/".join(path)) else None

    def _properties(self, header: int, *where: str) -> dict[int, int]:
        """Fixed-size properties (dates, numbers) of the __properties_version1.0 stream."""
        data = self._stream(*where, "__properties_version1.0") or b""
        out: dict[int, int] = {}
        for offset in range(header, len(data) - 15, 16):
            tag, _, value = struct.unpack_from("<IIQ", data, offset)
            out[tag] = value
        return out

    def string(self, prop: str, *where: str) -> str:
        unicode = self._stream(*where, f"__substg1.0_{prop}001F")
        if unicode is not None:
            return unicode.decode("utf-16-le", "replace").rstrip("\x00")
        ansi = self._stream(*where, f"__substg1.0_{prop}001E")
        return ansi.decode(self.codepage, "replace").rstrip("\x00") if ansi is not None else ""

    def date(self) -> str:
        filetime = self.props.get(_SUBMIT_TIME) or self.props.get(_DELIVERY_TIME)
        if not filetime:
            return ""
        moment = dt.datetime(1601, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(microseconds=filetime // 10)
        return moment.strftime("%Y-%m-%d %H:%M UTC")

    def body_cells(self, languages: tuple[str, ...]) -> list[list[str]]:
        plain = self.string("1000")
        if plain.strip():
            return text_cells(plain)
        html = self._stream("__substg1.0_10130102") or self.string("1013").encode("utf-8")
        return html_cells(decode_text(html, languages)[0])[0] if html else []

    def header_cells(self) -> tuple[list[list[str]], dict]:
        values = {key: (self.date() if key == "date" else self.string(key)) for key, _ in _HEADERS}
        sender = self.string("5D01") or self.string("0C1F")
        if sender and sender not in values["0C1A"]:
            values["0C1A"] = f"{values['0C1A']} <{sender}>".strip()
        cells = [[f"{label}: {values[key]}"] for key, label in _HEADERS if values[key]]
        return cells, {_KEYS[key]: values[key] for key, _ in _HEADERS if values[key]}

    def attachments(self) -> list[tuple[str, str, bool, list[str] | None]]:
        """(storage, name, body image, storage of the email if it is an attached email)."""
        depth = len(self.base)
        storages = sorted(
            {
                entry[depth]
                for entry in self.ole.listdir(streams=True, storages=True)
                if entry[:depth] == self.base and len(entry) > depth and entry[depth].startswith("__attach_version1.0_")
            }
        )
        out = []
        for storage in storages:
            name = (
                self.string("3707", storage)
                or self.string("3704", storage)
                or self.string("3001", storage)
                or "attachment"
            )
            embedded = [*self.base, storage, "__substg1.0_3701000D"]
            if self.ole.exists("/".join(embedded)):
                out.append((storage, name, False, embedded))
            else:
                out.append((storage, name, bool(self.string("3712", storage)), None))
        return out

    def data(self, storage: str) -> bytes | None:
        return self._stream(storage, "__substg1.0_37010102")


def _pages(
    message: _Message,
    parent: Source,
    config: OCRConfig,
    depth: int,
    origin: str | None,
    warnings: list[str],
    counter: list[int],
) -> Iterator[RawPage]:
    cells, _ = message.header_cells()
    rows, size = grid_rows([*cells, [""], *message.body_cells(config.language_hints)])
    counter[0] += 1
    yield RawPage(counter[0], rows=rows, size=size, origin=origin)
    for storage, name, inline, embedded in message.attachments():
        prefix = "body image " if inline else "attachment "
        where = f"{origin}, " if origin else ""
        if embedded is not None:
            if depth + 1 >= MAX_DEPTH:
                warnings.append(f"{where}{prefix}{name}: nested too deep, skipped")
                continue
            yield from _pages(
                _Message(message.ole, embedded, top=False),
                parent,
                config,
                depth + 1,
                f"{where}{prefix}{name}",
                warnings,
                counter,
            )
            continue
        data = message.data(storage)
        if not data:
            continue
        for page in nested_pages(parent, name, data, config, where + prefix, warnings):
            counter[0] += 1
            page.number = counter[0]
            yield page


def load(src: Source, config: OCRConfig) -> RawDocument:
    olefile = require("olefile", extra="msg")
    try:
        ole = olefile.OleFileIO(src.stream())
    except OSError as e:
        raise DocumentLoadError(f".msg corrupt or unreadable: {src.name}: {e}") from e
    message = _Message(ole, [], top=True)
    _, metadata = message.header_cells()
    metadata["attachments"] = [name for _, name, inline, _ in message.attachments() if not inline]
    warnings: list[str] = []
    parent = Source(name=src.name, depth=src.depth, budget=src.budget)  # without the bytes of the email
    pages = _pages(message, parent, config, src.depth, None, warnings, [0])
    return RawDocument("msg", pages, metadata=metadata, warnings=warnings, on_close=ole.close)
