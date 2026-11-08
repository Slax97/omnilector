# SPDX-License-Identifier: AGPL-3.0-or-later
"""New formats: EPUB/FB2 (PyMuPDF), ZIP, OpenDocument, Outlook .msg and legacy Office/RTF."""

from __future__ import annotations

import io
import zipfile

import pytest

from omnilector.config import OCRConfig
from omnilector.exceptions import DocumentLoadError, UnsupportedFormatError
from omnilector.loaders import Source, convert, detect_kind, load

from ..helpers import msg_file, ole_file, png_bytes


def pages(source, **cfg):
    with load(source, OCRConfig(**cfg)) as raw:
        return raw, list(raw.pages)


def texts(page) -> list[str]:
    return [" ".join(b.text for b in row) for row in page.rows]


def zipped(files: dict[str, str | bytes], first: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if first:  # mimetype goes first and uncompressed (EPUB, OpenDocument)
            z.writestr(first, files.pop(first), compress_type=zipfile.ZIP_STORED)
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


# --------------------------------------------------------------------------- ebooks


def test_epub_and_fb2_are_read_with_pymupdf():
    epub = zipped(
        {
            "mimetype": "application/epub+zip",
            "META-INF/container.xml": '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="c.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
            "c.opf": '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">x</dc:identifier><dc:title>T</dc:title>'
            '<dc:language>es</dc:language></metadata><manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
            '</manifest><spine><itemref idref="c1"/></spine></package>',
            "c1.xhtml": '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Chapter one</h1>'
            "<p>Text of the ebook.</p></body></html>",
        },
        first="mimetype",
    )
    raw, ps = pages(epub)
    assert raw.kind == "epub" and texts(ps[0])[:2] == ["Chapter one", "Text of the ebook."]
    fb2 = (
        b'<?xml version="1.0" encoding="utf-8"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
        b"<body><section><p>First paragraph of the book.</p><p>Second paragraph.</p></section></body></FictionBook>"
    )
    raw, ps = pages(fb2)
    assert raw.kind == "ebook" and "First paragraph of the book." in texts(ps[0])
    xps = zipped({"FixedDocumentSequence.fdseq": "<FixedDocumentSequence/>", "Documents/1/FixedDocument.fdoc": "<x/>"})
    assert detect_kind(Source(name="<bytes>", data=xps)) == "xps"


# --------------------------------------------------------------------------- zip


def test_zip_reads_every_file_with_its_loader():
    inner = zipped({"inside.txt": "Text of the inner zip"})
    data = zipped(
        {
            "invoice.txt": "Total: 45,67 EUR",
            "photos/ticket.png": png_bytes(320, 120, "TICKET"),
            "nested.zip": inner,
            "__MACOSX/._factura.txt": "basura de macOS",
            ".DS_Store": "hidden",
            "folder/": "",
        }
    )
    raw, ps = pages(data)
    assert raw.kind == "zip" and raw.metadata["files"] == ["invoice.txt", "photos/ticket.png", "nested.zip"]
    assert [(p.number, p.origin) for p in ps] == [
        (1, "file invoice.txt"),
        (2, "file photos/ticket.png"),
        (3, "file nested.zip, file inside.txt"),
    ]
    assert texts(ps[0]) == ["Total: 45,67 EUR"] and ps[1].image is not None


def test_zip_bomb_and_unreadable_entries_are_skipped(monkeypatch):
    from omnilector.loaders import archive

    monkeypatch.setattr(archive, "BOMB_MIN_BYTES", 1000)
    data = zipped({"bomb.txt": "0" * 500_000, "good.txt": "Hello"})
    raw, ps = pages(data)
    assert [p.origin for p in ps] == ["file good.txt"]
    assert any("bomb.txt" in w and "large" in w for w in raw.warnings)


# --------------------------------------------------------------------------- tar, gz, bz2, xz


def tarred(files: dict[str, bytes], mode: str = "w:gz") -> bytes:
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode=mode) as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_tar_in_every_compression_and_loose_compressed_files():
    import bz2
    import gzip
    import lzma

    files = {"invoice.txt": b"Total: 45,67 EUR", "photos/ticket.png": png_bytes(320, 120, "TICKET"), ".hidden": b"no"}
    for mode in ("w", "w:gz", "w:bz2", "w:xz"):
        raw, ps = pages(tarred(files, mode))
        assert raw.kind == "tar" and raw.metadata["files"] == ["invoice.txt", "photos/ticket.png"], mode
        assert [p.origin for p in ps] == ["file invoice.txt", "file photos/ticket.png"] and texts(ps[0]) == [
            "Total: 45,67 EUR"
        ]
    for compress, suffix in ((gzip.compress, ".gz"), (bz2.compress, ".bz2"), (lzma.compress, ".xz")):
        src = Source(name=f"note.txt{suffix}", data=compress(b"Hello from a compressed file"))
        raw, ps = pages(src)
        assert raw.kind == "compressed" and [(p.origin, texts(p)) for p in ps] == [
            ("file note.txt", ["Hello from a compressed file"])
        ]


def test_compression_bombs_and_broken_tars(monkeypatch):
    import gzip

    from omnilector.loaders import archive

    monkeypatch.setattr(archive, "MAX_TOTAL_BYTES", 1000)
    with pytest.raises(DocumentLoadError, match="large"):  # a .gz that decompresses endlessly: it is cut at the limit
        pages(gzip.compress(b"0" * 100_000))
    raw, ps = pages(tarred({"a.txt": b"uno", "big.txt": b"0" * 5000, "b.txt": b"dos"}))
    assert [p.origin for p in ps] == ["file a.txt"]  # after one that is too large, reading stops
    assert raw.warnings == ["big.txt: too large when decompressed, skipped (and the rest of the TAR)"]
    raw, ps = pages(tarred({"a.txt": b"uno" * 100}, "w")[:600])  # truncated halfway through its first file
    assert ps == [] and raw.warnings[0].startswith("a.txt: could not be decompressed")


ODF_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
    'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
    'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
    'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
    'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0"'
)


def odf(kind: str, body: str, styles: str = "", extra: dict | None = None) -> bytes:
    files = {
        "mimetype": f"application/vnd.oasis.opendocument.{kind}",
        "content.xml": f'<?xml version="1.0"?><office:document-content {ODF_NS}><office:body>{body}</office:body></office:document-content>',
        "styles.xml": f'<?xml version="1.0"?><office:document-styles {ODF_NS}>{styles}</office:document-styles>',
        "meta.xml": f'<?xml version="1.0"?><office:document-meta {ODF_NS}><office:meta><dc:title>Test</dc:title></office:meta></office:document-meta>',
        **(extra or {}),
    }
    return zipped(files, first="mimetype")


def test_odt():
    body = (
        "<office:text>"
        "<text:h>Grant application</text:h>"
        "<text:p>Name:<text:s/><text:span>Luis</text:span><text:s text:c='2'/>Perez<text:note><text:note-citation>1</text:note-citation>"
        "<text:note-body><text:p>Footnote of the name</text:p></text:note-body></text:note></text:p>"
        "<text:tracked-changes><text:changed-region><text:deletion><text:p>Deleted text</text:p></text:deletion></text:changed-region></text:tracked-changes>"
        "<text:list><text:list-item><text:p>Primer punto</text:p></text:list-item></text:list>"
        "<text:p><draw:frame><draw:text-box><text:p>Text box</text:p></draw:text-box></draw:frame>Paragraph text</text:p>"
        "<table:table><table:table-row><table:table-cell><text:p>Total</text:p></table:table-cell>"
        "<table:table-cell><text:p>45,67 EUR</text:p></table:table-cell></table:table-row></table:table>"
        "</office:text>"
    )
    styles = (
        "<office:master-styles><style:master-page><style:header><text:p>City Council letterhead</text:p></style:header>"
        "<style:footer><text:p>Page footer</text:p></style:footer></style:master-page></office:master-styles>"
    )
    raw, (page,) = pages(odf("text", body, styles))
    assert raw.kind == "odt" and raw.metadata["title"] == "Test"
    assert texts(page) == [
        "City Council letterhead",
        "Grant application",
        "Name: Luis  Perez",
        "Footnote of the name",
        "Primer punto",
        "Paragraph text",
        "Text box",
        "Total 45,67 EUR",
        "Page footer",
    ]  # without the deleted text


def test_ods_sheets_and_repeated_cells():
    body = (
        "<office:spreadsheet>"
        "<table:table table:name='Sales'><table:table-row><table:table-cell><text:p>Product</text:p></table:table-cell>"
        "<table:table-cell table:number-columns-repeated='2'/><table:table-cell><text:p>Amount</text:p></table:table-cell>"
        "<table:table-cell table:number-columns-repeated='16000'/></table:table-row>"
        "<table:table-row table:number-rows-repeated='1048000'><table:table-cell table:number-columns-repeated='1024'/></table:table-row>"
        "</table:table>"
        "<table:table table:name='Notes'><table:table-row><table:table-cell><text:p>Just a note</text:p></table:table-cell></table:table-row></table:table>"
        "</office:spreadsheet>"
    )
    raw, ps = pages(odf("spreadsheet", body))
    assert raw.kind == "ods" and raw.metadata["sheets"] == ["Sales", "Notes"]
    assert [p.origin for p in ps] == ["sheet Sales", "sheet Notes"]
    assert [b.text for b in ps[0].blocks] == ["Product", "Amount"] and ps[0].blocks[1].box[0] > ps[0].blocks[0].box[0]


def test_odp_slides_and_notes():
    body = (
        "<office:presentation><draw:page draw:name='p1'>"
        "<draw:frame svg:x='1cm' svg:y='1cm' svg:width='20cm' svg:height='2cm'><draw:text-box><text:p>Resultados 2026</text:p></draw:text-box></draw:frame>"
        "<draw:frame svg:x='1cm' svg:y='5cm' svg:width='10cm' svg:height='2cm'><draw:text-box><text:p>Sales +12%</text:p></draw:text-box></draw:frame>"
        "<presentation:notes><draw:frame><draw:text-box><text:p>Speaker note</text:p></draw:text-box></draw:frame></presentation:notes>"
        "</draw:page></office:presentation>"
    )
    styles = "<office:automatic-styles><style:page-layout><style:page-layout-properties fo:page-width='28cm' fo:page-height='15.75cm'/></style:page-layout></office:automatic-styles>"
    raw, ps = pages(odf("presentation", body, styles))
    assert raw.kind == "odp" and [p.origin for p in ps] == ["slide 1", "slide 1, notes"]
    assert texts(ps[0]) == ["Resultados 2026", "Sales +12%"] and texts(ps[1]) == ["Speaker note"]
    assert ps[0].size == (1058, 595) and ps[0].blocks[1].box[:2] == (38, 189)  # cm -> px at 96 dpi


def test_odf_images_and_damaged_files():
    body = "<office:text><text:p>With photo<draw:frame><draw:image xlink:href='Pictures/ticket.png'/></draw:frame></text:p></office:text>"
    _, ps = pages(odf("text", body, extra={"Pictures/ticket.png": png_bytes(320, 120, "TICKET")}))
    assert len(ps) == 2 and ps[1].image is not None and ps[1].origin == "image ticket.png"
    with pytest.raises(DocumentLoadError):  # broken (or encrypted) content.xml
        pages(odf("text", "<office:text>"))
    with pytest.raises(DocumentLoadError):  # truncated ZIP
        pages(odf("text", "<office:text/>")[:300])


# --------------------------------------------------------------------------- Outlook and legacy Office


def test_outlook_msg():
    pytest.importorskip("olefile")
    data = msg_file(
        "Invoice 7",
        "Please find the invoice attached.\nRegards",
        [
            ("note.txt", b"Total: 45,67 EUR", None),
            ("image001.png", png_bytes(320, 120, "CAPTURA"), "image001@x"),
            ("icono.png", png_bytes(16, 16, ""), "icono@x"),
        ],
        forwarded={"subject": "Pedido anterior", "body": "Text of the forwarded email"},
    )
    raw, ps = pages(data)
    assert (
        raw.kind == "msg" and raw.metadata["subject"] == "Invoice 7" and raw.metadata["date"] == "2026-09-11 10:44 UTC"
    )
    assert raw.metadata["attachments"] == ["note.txt", "forwarded.msg"]
    assert texts(ps[0]) == [
        "From: Example Shop <shop@example.com>",
        "To: Customer",
        "Date: 2026-09-11 10:44 UTC",
        "Subject: Invoice 7",
        "",
        "Please find the invoice attached.",
        "Regards",
    ]
    assert [(p.number, p.origin, p.image is not None) for p in ps[1:]] == [
        (2, "attachment note.txt", False),
        (3, "body image image001.png", True),  # pasted screenshot; the 16 px icon is not
        (4, "attachment forwarded.msg", False),
    ]
    assert texts(ps[3])[-1] == "Text of the forwarded email"


def test_old_office_is_recognized_by_its_streams():
    pytest.importorskip("olefile")
    cases = {"WordDocument": "doc", "Workbook": "xls", "PowerPoint Document": "ppt"}
    for stream, kind in cases.items():
        data = ole_file({stream: b"contenido", "\x05SummaryInformation": b"x"})
        assert detect_kind(Source(name="no_extension", data=data)) == kind
    with pytest.raises(DocumentLoadError):  # OLE header but broken content
        detect_kind(Source(name="broken.doc", data=b"\xd0\xcf\x11\xe0" + b"\x00" * 100))


def test_old_office_needs_libreoffice(monkeypatch):
    pytest.importorskip("olefile")
    monkeypatch.setattr(convert, "find_soffice", lambda: None)
    with pytest.raises(UnsupportedFormatError, match="LibreOffice"):
        pages(ole_file({"WordDocument": b"contenido"}))
    with pytest.raises(UnsupportedFormatError, match="LibreOffice"):
        pages(b"{\\rtf1\\ansi Hello}")
