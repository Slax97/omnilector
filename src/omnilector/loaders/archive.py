# SPDX-License-Identifier: AGPL-3.0-or-later
"""ZIP, TAR (.tar, .tar.gz/.tgz, .tar.bz2, .tar.xz) and single compressed files (.gz, .bz2, .xz): each
file inside is read with its own loader, like the attachments of an email.

Two ways: a single document with the pages of all of them (load) or each file as a separate input
(members, for OCR.read_many with split_archives). With limits against malicious archives (zip bombs):
number of files, total uncompressed size and, in ZIPs, compression ratio; those over the limit are
skipped, with their reason.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import tarfile
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ..exceptions import DocumentLoadError
from .base import RawDocument, Source, nested_pages

if TYPE_CHECKING:
    from ..config import OCRConfig

MAX_ENTRIES = 1000  # files read from a ZIP or a TAR
MAX_TOTAL_BYTES = 512 * 1024 * 1024  # uncompressed in total
MAX_RATIO = 200  # uncompressed / compressed: more than this, in a large file, is almost surely a zip bomb
BOMB_MIN_BYTES = 16 * 1024 * 1024  # below this, even if it compresses a lot (a blank image), it is read
_SUFFIXES = (".gz", ".gzip", ".bz2", ".xz")

# (name, content, None) of each file inside, or (name, None, reason) if it is skipped
Files = Iterator[tuple[str, bytes | None, str | None]]


def compression(head: bytes) -> Callable | None:
    """What opens a compressed file (gzip, bzip2, xz), from its first bytes; None if it is not compressed."""
    if head.startswith(b"\x1f\x8b"):
        return gzip.open
    if head.startswith(b"BZh") and head[4:10] == b"1AY&SY":
        return bz2.open
    if head.startswith(b"\xfd7zXZ\x00"):
        return lzma.open
    return None


def _hidden(parts: tuple[str, ...]) -> bool:
    """macOS metadata and hidden files."""
    return not parts or parts[0] == "__MACOSX" or parts[-1].startswith(".")


# --------------------------------------------------------------------------- ZIP


def _open_zip(src: Source) -> tuple[zipfile.ZipFile, list[zipfile.ZipInfo], list[str]]:
    """The open ZIP, the files to read (no folders or hidden files, up to MAX_ENTRIES) and its warnings."""
    try:
        archive = zipfile.ZipFile(src.stream())
    except (zipfile.BadZipFile, ValueError, OSError) as e:
        raise DocumentLoadError(f"Corrupt ZIP: {src.name}: {e}") from e
    entries = [
        info for info in archive.infolist() if not info.is_dir() and not _hidden(PurePosixPath(info.filename).parts)
    ]
    warnings: list[str] = []
    if len(entries) > MAX_ENTRIES:
        warnings.append(f"ZIP with {len(entries)} files: only the first {MAX_ENTRIES} are read")
        entries = entries[:MAX_ENTRIES]
    return archive, entries, warnings


def _decrypt(archive: zipfile.ZipFile, info: zipfile.ZipInfo, passwords: tuple[str, ...]) -> bytes | None:
    """An encrypted file of the ZIP with the first password that works; None if none does."""
    for password in passwords:
        try:
            return archive.read(info, pwd=password.encode("utf-8"))
        except (
            RuntimeError,
            zipfile.BadZipFile,
        ):  # wrong (1 in 256 passes the first check and fails on the CRC)
            continue
    return None


def _zip_files(archive: zipfile.ZipFile, entries: list[zipfile.ZipInfo], passwords: tuple[str, ...]) -> Files:
    """Each file, decompressed when requested; encrypted ones without a working password, those too large
    (zip bomb) and those that cannot be decompressed are skipped."""
    total = 0
    for info in entries:
        if info.flag_bits & 0x1:
            try:
                data = _decrypt(archive, info, passwords)
            except Exception as e:  # noqa: BLE001 - AES encryption (zipfile does not read it) or broken data
                yield info.filename, None, f"encrypted, could not be read ({e}), skipped"
                continue
            if data is None:
                yield (
                    info.filename,
                    None,
                    "encrypted, skipped" + ("" if passwords else " (pass the password in passwords)"),
                )
            else:
                yield info.filename, data, None
            continue
        bomb = info.file_size > BOMB_MIN_BYTES and info.file_size > MAX_RATIO * max(info.compress_size, 1)
        if bomb or total + info.file_size > MAX_TOTAL_BYTES:
            yield info.filename, None, "too large when decompressed, skipped"
            continue
        total += info.file_size
        try:
            data = archive.read(info)
        except Exception as e:  # noqa: BLE001 - CRC, zlib, truncated, odd method, out of memory...: the rest is read
            yield info.filename, None, f"could not be decompressed ({e}), skipped"
            continue
        yield info.filename, data, None


# --------------------------------------------------------------------------- TAR


def _open_tar(src: Source) -> tarfile.TarFile:
    """The TAR (compressed or not) in stream mode: read in one pass, without decompressing it twice."""
    stream = src.stream()
    if isinstance(stream, str):
        return tarfile.open(name=stream, mode="r|*")
    return tarfile.open(fileobj=stream, mode="r|*")


def is_tar(src: Source) -> bool:
    """Whether it is a TAR, also a compressed one (.tar.gz, .tgz, .tar.bz2, .tar.xz): its first header is valid."""
    try:
        with _open_tar(src) as tar:
            tar.next()
    except Exception:  # noqa: BLE001 - not a TAR (a single-file .gz) or broken
        return False
    return True


def _tar_files(tar: tarfile.TarFile, warnings: list[str], names: list[str]) -> Files:
    """Each file of the TAR, in order (no folders, links or hidden files). Reading stops when the size limit
    is passed or broken data is found: in a compressed stream, skipping what follows would cost
    decompressing all of it."""
    total = 0
    try:
        for info in tar:
            if not info.isfile() or _hidden(PurePosixPath(info.name).parts):
                continue
            if len(names) == MAX_ENTRIES:
                warnings.append(f"TAR with more than {MAX_ENTRIES} files: only the first {MAX_ENTRIES} are read")
                return
            names.append(info.name)
            if total + info.size > MAX_TOTAL_BYTES:
                yield info.name, None, "too large when decompressed, skipped (and the rest of the TAR)"
                return
            total += info.size
            try:
                data = tar.extractfile(info).read()  # type: ignore[union-attr]
            except Exception as e:  # noqa: BLE001 - broken data: what was read so far is fine
                yield info.name, None, f"could not be decompressed ({e}), skipped (and the rest of the TAR)"
                return
            yield info.name, data, None
    except Exception as e:  # noqa: BLE001 - a broken header halfway: what was read so far is fine
        warnings.append(f"broken TAR ({e}): the rest is not read")


# --------------------------------------------------------------------------- single compressed file


def _inner_name(name: str) -> str:
    """invoice.pdf.gz -> invoice.pdf"""
    base = Path(name).name
    return next((base[: -len(s)] for s in _SUFFIXES if base.lower().endswith(s) and len(base) > len(s)), "decompressed")


def _decompress(src: Source) -> bytes:
    """A single-file .gz, .bz2 or .xz, decompressed up to MAX_TOTAL_BYTES (a zip bomb does not go beyond that)."""
    opener = compression(src.head(16))
    if opener is None:
        raise DocumentLoadError(f"Not a compressed file: {src.name}")
    try:
        with opener(src.stream()) as f:
            data = f.read(MAX_TOTAL_BYTES + 1)
    except Exception as e:  # truncated, CRC, broken data
        raise DocumentLoadError(f"Broken compressed file: {src.name}: {e}") from e
    if len(data) > MAX_TOTAL_BYTES:
        raise DocumentLoadError(f"Too large when decompressed (more than {MAX_TOTAL_BYTES // 2**20} MB): {src.name}")
    return data


# --------------------------------------------------------------------------- common


def _contents(src: Source, kind: str, passwords: tuple[str, ...]) -> tuple[list[str], list[str], Files, Callable]:
    """(warnings, names of the files inside, their contents, how to close it). The warnings and names of a
    TAR fill up while it is read."""
    if kind == "zip":
        archive, entries, warnings = _open_zip(src)
        return warnings, [info.filename for info in entries], _zip_files(archive, entries, passwords), archive.close
    if kind == "tar":
        try:
            tar = _open_tar(src)
        except Exception as e:  # truncated, broken compression
            raise DocumentLoadError(f"Corrupt TAR: {src.name}: {e}") from e
        warnings: list[str] = []
        names: list[str] = []
        return warnings, names, _tar_files(tar, warnings, names), tar.close
    name = _inner_name(src.name)
    return [], [name], iter([(name, _decompress(src), None)]), lambda: None


def load(src: Source, config: OCRConfig, kind: str = "zip") -> RawDocument:
    """One document with the pages of every file inside, each one with its provenance."""
    warnings, names, files, close = _contents(src, kind, config.passwords)

    def pages():
        number = 0
        for name, data, problem in files:
            if data is None:
                warnings.append(f"{name}: {problem}")
                continue
            for page in nested_pages(src, name, data, config, "file ", warnings):
                number += 1
                page.number = number
                yield page

    return RawDocument(kind, pages(), metadata={"files": names}, warnings=warnings, on_close=close)


def members(
    src: Source, kind: str = "zip", passwords: tuple[str, ...] = ()
) -> tuple[list[str], Iterator[tuple[Source, str | None]]]:
    """Each file of the ZIP or TAR as a separate input ("batch.zip/photos/ticket.jpg"), as if it had been
    passed separately: with its own OCR limit (max_ocr_pages). Returns the warnings of the archive (those
    of a TAR are added while it is read) and, for each file inside, its input and None, or the input without
    content and the reason it is skipped. DocumentLoadError if it is broken."""
    warnings, _, files, close = _contents(src, kind, passwords)

    def each() -> Iterator[tuple[Source, str | None]]:
        try:
            for name, data, problem in files:
                yield Source(name=f"{src.name}/{name}", data=data, depth=src.depth + 1), problem
        finally:
            close()

    return warnings, each()
