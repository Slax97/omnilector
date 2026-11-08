# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real kind of the input from its first bytes (the extension only breaks ties)."""

from __future__ import annotations

import re
import zipfile

from ..exceptions import DocumentLoadError, UnsupportedFormatError
from .archive import compression, is_tar
from .base import Source

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif", ".avif", ".jp2"}
HTML_EXTS = {".html", ".htm", ".xhtml"}
EMAIL_EXTS = {".eml"}
HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}
_AVIF_BRANDS = {b"avif", b"avis"}
_ZIP_KINDS = {"word/document.xml": "docx", "xl/workbook.xml": "xlsx", "ppt/presentation.xml": "pptx"}
# mimetype of the OpenDocument and EPUB ZIPs (templates, .ott..., are read the same way)
_ZIP_MIMETYPES = {
    "application/epub+zip": "epub",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
}
# Legacy Office and Outlook (OLE container): the kind comes from their internal streams or, without
# olefile, from the extension
_OLE_STREAMS = (
    ("__properties_version1.0", "msg"),
    ("WordDocument", "doc"),
    ("Workbook", "xls"),
    ("Book", "xls"),
    ("PowerPoint Document", "ppt"),
)
_OLE_EXTS = {".msg": "msg", ".doc": "doc", ".dot": "doc", ".xls": "xls", ".xlt": "xls", ".ppt": "ppt", ".pps": "ppt"}
_EMAIL_HEADER = re.compile(
    rb"^(from|to|subject|date|received|return-path|message-id|mime-version|delivered-to|cc|reply-to)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


def is_heif(head: bytes) -> bool:
    """HEIC/HEIF (iPhone photos): PIL only opens them with pillow-heif."""
    return head[4:8] == b"ftyp" and head[8:12] in HEIF_BRANDS


def is_image(head: bytes) -> bool:
    return (
        head.startswith(
            (
                b"\x89PNG\r\n\x1a\n",
                b"\xff\xd8\xff",
                b"GIF87a",
                b"GIF89a",
                b"II*\x00",
                b"MM\x00*",
                b"\x00\x00\x00\x0cjP  \r\n\x87\n",  # JPEG 2000
            )
        )
        or (head.startswith(b"BM") and head[6:10] == b"\x00\x00\x00\x00")
        or (head.startswith(b"RIFF") and head[8:12] == b"WEBP")
        or is_heif(head)
        or (head[4:8] == b"ftyp" and head[8:12] in _AVIF_BRANDS)
    )


def _zip_kind(src: Source) -> str:
    try:
        with zipfile.ZipFile(src.stream()) as z:
            names = set(z.namelist())
            for marker, kind in _ZIP_KINDS.items():
                if marker in names:
                    return kind
            mimetype = z.read("mimetype").decode("ascii", "ignore").strip() if "mimetype" in names else ""
    except (zipfile.BadZipFile, ValueError, OSError) as e:  # truncated: zipfile sometimes raises ValueError
        raise DocumentLoadError(f"Corrupt ZIP/Office file: {src.name}") from e
    if any(name.lower().endswith(".fdseq") for name in names):
        return "xps"
    for prefix, kind in _ZIP_MIMETYPES.items():
        if mimetype.startswith(prefix):
            return kind
    if mimetype:
        raise UnsupportedFormatError(f"Format '{mimetype}' not supported yet: {src.name}")
    return "zip"  # any compressed file: each file inside with its loader


def _ole_kind(src: Source) -> str:
    """OLE container: .msg (Outlook), .doc, .xls or .ppt."""
    try:
        import olefile
    except ImportError:
        kind = _OLE_EXTS.get(src.suffix)
        if kind is None:
            raise UnsupportedFormatError(
                f"OLE file (legacy Office or Outlook) without a known extension: {src.name}. "
                'With pip install "omnilector[msg]" it is recognized by its content'
            ) from None
        return kind
    try:
        with olefile.OleFileIO(src.stream()) as ole:
            streams = {"/".join(entry) for entry in ole.listdir(streams=True, storages=True)}
    except OSError as e:
        raise DocumentLoadError(f"Corrupt OLE file: {src.name}: {e}") from e
    for marker, kind in _OLE_STREAMS:
        if marker in streams:
            return kind
    raise UnsupportedFormatError(f"Unrecognized OLE file (neither Outlook nor Office): {src.name}")


def _looks_html(head: bytes) -> bool:
    start = head[:2048].lstrip().lower()
    return start.startswith((b"<!doctype html", b"<html")) or (b"<html" in start and b"<body" in head.lower())


def _looks_email(head: bytes) -> bool:
    first = head.lstrip()[:4096]
    return bool(re.match(rb"[\w-]+\s*:", first)) and len(_EMAIL_HEADER.findall(first)) >= 2


def _looks_text(head: bytes) -> bool:
    if head.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):
        return True
    if b"\x00" in head:
        return False
    text = head.decode("utf-8", errors="replace")
    bad = sum(1 for ch in text if ch == "�" or (ord(ch) < 32 and ch not in "\t\n\r\f"))
    if bad <= len(text) * 0.05:
        return True
    latin = head.decode("cp1252", errors="replace")  # legacy Windows texts
    return sum(1 for ch in latin if ord(ch) < 32 and ch not in "\t\n\r\f") <= len(latin) * 0.02


def detect_kind(src: Source, extensions: dict[str, str] | None = None) -> str:
    """image | pdf | docx | xlsx | pptx | odt | ods | odp | epub | xps | ebook | zip | tar | compressed |
    msg | doc | xls | ppt | rtf | html | email | text, or whatever `extensions` says (user loaders)."""
    if src.obj is not None:
        return "image"
    if extensions and src.suffix in extensions:
        return extensions[src.suffix]
    head = src.head()
    if not head:
        raise DocumentLoadError(f"Empty input: {src.name}")
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if is_image(head):
        return "image"
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):  # the second one, an empty ZIP
        return _zip_kind(src)
    if head[257:262] == b"ustar":
        return "tar"
    if compression(head) is not None:  # .tar.gz, .tgz... or a single compressed file (.pdf.gz)
        return "tar" if is_tar(src) else "compressed"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return _ole_kind(src)
    if head.startswith(b"{\\rtf"):
        return "rtf"
    if head[60:68] == b"BOOKMOBI":
        return "ebook"  # MOBI / AZW
    if b"<FictionBook" in head[:2048]:
        return "ebook"  # FB2
    if src.suffix in EMAIL_EXTS or (src.suffix not in HTML_EXTS and _looks_email(head)):
        return "email"
    if src.suffix in HTML_EXTS or _looks_html(head):
        return "html"
    if src.suffix in IMAGE_EXTS:
        return "image"  # odd header: let PIL try and give a clear error if it cannot
    if _looks_text(head):
        return "text"
    raise UnsupportedFormatError(f"Unrecognized format: {src.name}")
