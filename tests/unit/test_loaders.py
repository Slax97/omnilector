# SPDX-License-Identifier: AGPL-3.0-or-later
"""Loaders: detection by bytes, pages per format and coordinates. No OCR."""

from __future__ import annotations

import datetime as dt
import io
from email.message import EmailMessage

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw

from omnilector import loaders as loaders_mod
from omnilector.config import OCRConfig
from omnilector.exceptions import DocumentLoadError, UnsupportedFormatError
from omnilector.loaders import Source, detect_kind, load, register_loader
from omnilector.loaders.base import single_page
from omnilector.loaders.pdf import useful_chars
from omnilector.result import TextBlock
from omnilector.utils.image import load_image

from ..helpers import zipcrypto
from .conftest import make_hybrid_pdf, make_mixed_pdf, png_bytes


def pages(source, **cfg):
    with load(source, OCRConfig(**cfg)) as raw:
        return raw, list(raw.pages)


def _empty_zip() -> bytes:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("notes.txt", "hello")
    return buf.getvalue()


def test_detect_kind_by_content(tmp_path):
    cases = {
        "a.png": (png_bytes(), "image"),
        "no_extension": (png_bytes(), "image"),  # no extension: the content decides
        "doc.txt": (b"hello world\nline 2", "text"),
        "page.html": (b"<!DOCTYPE html><html><body><p>hello</p></body></html>", "html"),
        "email.eml": (b"From: a@b.es\nTo: c@d.es\nSubject: hello\n\nbody", "email"),
    }
    for name, (data, kind) in cases.items():
        path = tmp_path / name
        path.write_bytes(data)
        assert detect_kind(Source(name=str(path), path=path)) == kind, name
    assert detect_kind(Source(name="<bytes>", data=b"%PDF-1.7\n%...")) == "pdf"
    assert detect_kind(Source(name="<bytes>", data=b"From: x@y.z\nSubject: s\nDate: hoy\n\nhello")) == "email"
    assert detect_kind(Source(name="<bytes>", data=b"{\\rtf1\\ansi Hello}")) == "rtf"
    assert detect_kind(Source(name="<bytes>", data=b"PK\x03\x04" + _empty_zip()[4:])) == "zip"
    with pytest.raises(UnsupportedFormatError):
        detect_kind(Source(name="x.bin", data=bytes(range(256)) * 4))
    with pytest.raises(DocumentLoadError):
        detect_kind(Source(name="empty.txt", data=b""))


def test_missing_or_invalid_input(tmp_path):
    with pytest.raises(DocumentLoadError):
        load(tmp_path / "nope.pdf", OCRConfig())
    with pytest.raises(DocumentLoadError):
        load(12345, OCRConfig())


def test_image_multipage_tiff(tmp_path):
    frames = [Image.new("RGB", (100, 50), c) for c in ("white", "gray")]
    path = tmp_path / "multi.tif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    raw, ps = pages(path)
    assert raw.kind == "image" and raw.metadata["frames"] == 2
    assert [p.number for p in ps] == [1, 2]
    assert all(p.image is not None and p.image.shape == (50, 100, 3) for p in ps)


def test_image_camera_exif_is_photo_hint(tmp_path):
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "Apple", "iPhone 15"
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (64, 64), "white").save(path, exif=exif.tobytes())
    raw, ps = pages(path)
    assert raw.metadata["camera"] == "Apple iPhone 15" and ps[0].hint == "photo"


def test_image_in_memory():
    raw, ps = pages(Image.new("RGB", (40, 20), "white"))
    assert raw.kind == "image" and ps[0].size == (40, 20)


def test_huge_jpeg_is_decoded_already_reduced():
    buf = io.BytesIO()
    Image.new("RGB", (4000, 3000), "white").save(buf, "JPEG")
    # with 256 MB the OCR sees up to 1000 px and the image comes in up to 2000: libjpeg decodes it at half size
    _, (page,) = pages(buf.getvalue(), memory_mb=256)
    assert page.size == (2000, 1500) and page.image.shape == (1500, 2000, 3)
    _, (page,) = pages(buf.getvalue())  # without a memory limit, whole
    assert page.size == (4000, 3000)


def test_transparent_and_unusual_images():
    # black signature on a transparent background: convert("RGB") used to leave everything black
    signature = Image.new("RGBA", (200, 60), (0, 0, 0, 0))
    ImageDraw.Draw(signature).rectangle((20, 20, 180, 40), fill=(0, 0, 0, 255))
    arr, _ = load_image(signature)
    assert arr[5, 5].tolist() == [255, 255, 255] and arr[30, 100].tolist() == [0, 0, 0]
    # white text on transparent: black background so it shows
    logo = Image.new("RGBA", (200, 60), (255, 255, 255, 0))
    ImageDraw.Draw(logo).rectangle((20, 20, 180, 40), fill=(255, 255, 255, 255))
    arr, _ = load_image(logo)
    assert arr[5, 5].tolist() == [0, 0, 0] and arr[30, 100].tolist() == [255, 255, 255]
    # arrays from other libraries: float 0-1, 16 bits, RGBA
    gray = np.full((10, 10), 0.5)
    assert load_image(gray)[0][0, 0].tolist() == [128, 128, 128]
    assert load_image(np.full((10, 10, 3), 65535, np.uint16))[0].min() == 255
    rgba = np.zeros((10, 10, 4), np.uint8)
    assert load_image(rgba)[0].min() == 255  # transparent: white, not black


def test_pdf_decides_per_page(tmp_path):
    path = tmp_path / "mixed.pdf"
    make_mixed_pdf(path)
    raw, (native, scanned) = pages(path)
    assert raw.kind == "pdf" and raw.metadata["pages"] == 2
    # page with text: no OCR, real boxes (they used to be (0,0,0,0))
    assert native.image is None and len(native.blocks) == 6
    w, h = native.size
    assert all(0 < b.box[0] < b.box[2] <= w and 0 < b.box[1] < b.box[3] <= h for b in native.blocks)
    assert native.blocks[0].engine == "native" and native.blocks[0].confidence == 1.0
    # scanned page: image for OCR (it used to be lost if another page had text)
    assert scanned.image is not None and scanned.blocks == [] and scanned.hint is None  # the analyzer decides
    assert scanned.image.shape[:2] == (scanned.size[1], scanned.size[0])


def test_pdf_hybrid_page(tmp_path):
    path = tmp_path / "hybrid.pdf"
    make_hybrid_pdf(path)
    _, ps = pages(path)
    (region,) = ps[0].regions  # the OCR will only read the crop of the image, not the whole page
    assert ps[0].image is None and len(ps[0].blocks) == 5
    h, w = region.image.shape[:2]
    x, y = region.origin
    assert (x, y) == (208, 833) and (w, h) == (2084, 2084)  # Rect(50, 200, 550, 700) at 300 DPI
    assert region.image.min() < 255  # the embedded image is there
    _, ps_fast = pages(path, strategy="fast")  # fast: only the native text
    assert ps_fast[0].regions == []


def stamp_pdf(path, pages_with_logo: int = 1, text_over_image: bool = False) -> None:
    """Pages of native text with a small stamp (~5 % of the page) at the bottom right."""
    stamp = Image.new("RGB", (700, 300), "white")
    ImageDraw.Draw(stamp).rectangle((20, 100, 680, 200), outline="black", width=6)
    data = io.BytesIO()
    stamp.save(data, "PNG")
    doc = pymupdf.open()
    for _ in range(pages_with_logo):
        page = doc.new_page()
        for i in range(10):
            page.insert_text((72, 80 + 16 * i), f"Native text of the letter, line {i}", fontsize=11)
        page.insert_image(pymupdf.Rect(300, 600, 550, 707), stream=data.getvalue())
        if text_over_image:
            page.insert_text((320, 640), "Signed: the secretary", fontsize=11)
    doc.save(path)
    doc.close()


def test_pdf_small_images_are_read_but_icons_are_not(tmp_path):
    path = tmp_path / "stamp.pdf"
    stamp_pdf(path, pages_with_logo=2)
    _, ps = pages(path)
    assert [len(p.regions) for p in ps] == [1, 1]  # before, below 25 % of the page, nothing
    assert ps[0].regions[0].key == ps[1].regions[0].key is not None  # same logo: read once
    # the native text on top of the image is covered (and the image no longer counts as repeated)
    stamp_pdf(path, text_over_image=True)
    _, (page,) = pages(path)
    (region,) = page.regions
    x, y = region.origin
    signed = next(b for b in page.blocks if b.text.startswith("Signed"))
    x0, y0, x1, y1 = signed.box
    assert region.image[y0 - y : y1 - y, x0 - x : x1 - x].min() == 255 and region.key is None
    # a 10 x 10 pt icon is not worth OCR
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(10):
        page.insert_text((72, 80 + 16 * i), f"Native text of the letter, line {i}", fontsize=11)
    page.insert_image(pymupdf.Rect(50, 50, 60, 60), stream=png_bytes(40, 40))
    doc.save(tmp_path / "icono.pdf")
    _, (page,) = pages(tmp_path / "icono.pdf")
    assert page.regions == []


def test_pdf_native_columns_and_rotated_margin(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    for col, x in (("IZQ", 50), ("DER", 320)):
        for i in range(6):
            page.insert_text((x, 100 + 14 * i), f"{col} line {i} with filler text", fontsize=9)
    page.insert_text((30, 700), "CSV: ABC123 verifiable at the e-office", fontsize=8, rotate=90)
    path = tmp_path / "columns.pdf"
    doc.save(path)
    _, (page,) = pages(path)
    texts = [" ".join(b.text for b in row) for row in page.rows]
    expected = [f"{c} line {i} with filler text" for c in ("IZQ", "DER") for i in range(6)]
    assert texts == [*expected, "CSV: ABC123 verifiable at the e-office"]  # before: LEFT and RIGHT mixed up


def test_ocr_page_limit_only_counts_ocr_pages(tmp_path):
    doc = pymupdf.open()
    for i in range(4):
        page = doc.new_page()
        if i % 2:
            page.insert_image(pymupdf.Rect(50, 50, 550, 300), stream=png_bytes(800, 300, "SCANNED"))
        else:
            for j in range(5):
                page.insert_text((72, 100 + 20 * j), f"Page {i} with enough native text, line {j}", fontsize=11)
    path = tmp_path / "mixed.pdf"
    doc.save(path)
    raw, ps = pages(path, max_ocr_pages=1)
    assert [p.image is not None for p in ps] == [False, True, False, False]  # the 2nd one scanned, without OCR
    assert all(p.blocks for p in ps[::2]) and raw.budget.skipped == 1  # native pages do not spend it
    raw, ps = pages(path, max_pages=2)
    assert len(ps) == 2 and "max_pages" in raw.warnings[0]


def test_pdf_rotated_page_boxes(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    for i in range(4):
        page.insert_text((20, 40 + 20 * i), "Rotated text with enough letters", fontsize=10)
    page.set_rotation(90)
    path = tmp_path / "rotado.pdf"
    doc.save(path)
    _, ps = pages(path)
    w, h = ps[0].size
    assert w < h  # 300x200 rotated 90 -> 200x300
    assert all(b.box[2] <= w + 1 and b.box[3] <= h + 1 for b in ps[0].blocks)


def test_pdf_garbage_text_counts_as_no_text():
    assert useful_chars([TextBlock(text="�" * 40 + "ab", box=(0, 0, 1, 1))]) == 0
    assert useful_chars([TextBlock(text="Hello world", box=(0, 0, 1, 1))]) == 10


def test_pdf_form_attachments_and_password(tmp_path):
    path = tmp_path / "formulario.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((100, 60), "Application form for school canteen grants", fontsize=11)
    page.insert_text((100, 100), "I accept the conditions", fontsize=11)
    page.insert_text((100, 130), "Quiero publicidad", fontsize=11)
    for name, y, value in (("acepto", 88, True), ("publi", 118, False)):
        box = pymupdf.Widget()
        box.field_type, box.field_name, box.field_value = pymupdf.PDF_WIDGET_TYPE_CHECKBOX, name, value
        box.rect = pymupdf.Rect(80, y, 94, y + 14)
        page.add_widget(box)
    page.insert_text((80, 172), "4", fontname="zadb", fontsize=11)  # the mark of an already flattened form
    page.insert_text((100, 172), "Data reviewed", fontsize=11)
    doc.embfile_add("invoice.xml", b"<Invoice><Total>45.67</Total></Invoice>", filename="invoice.xml")
    page.add_file_annot(pymupdf.Point(300, 300), b"Note attached to the form", "note.txt")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="1234", owner_pw="dueno")
    doc.close()
    with pytest.raises(DocumentLoadError, match="password"):
        pages(path)
    raw, ps = pages(path, passwords=("mala", "1234"))  # the first one that works
    form = [" ".join(b.text for b in row) for row in ps[0].rows]
    assert form[1:] == ["☑ I accept the conditions", "☐ Quiero publicidad", "✔ Data reviewed"]  # before, a stray "3"
    assert [(p.number, p.origin, p.blocks[0].text) for p in ps[1:]] == [
        (2, "attachment invoice.xml", "<Invoice><Total>45.67</Total></Invoice>"),
        (3, "attachment note.txt", "Note attached to the form"),
    ]
    assert raw.metadata["attachments"] == ["invoice.xml", "note.txt"]


def test_zip_entry_with_password():
    data = zipcrypto("secret.txt", b"Total: 45,67 EUR", "1234")
    raw, ps = pages(data)
    assert ps == [] and raw.warnings == ["secret.txt: encrypted, skipped (pass the password in passwords)"]
    raw, ps = pages(data, passwords=("mala", "1234"))
    assert [(p.origin, p.blocks[0].text) for p in ps] == [("file secret.txt", "Total: 45,67 EUR")]


def test_docx(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Invoice 7", level=1)
    d.add_paragraph("Customer: Luis")
    table = d.add_table(rows=2, cols=2)
    for (r, c), text in {(0, 0): "Total", (0, 1): "45,67 EUR", (1, 0): "IVA", (1, 1): "21%"}.items():
        table.cell(r, c).text = text
    d.sections[0].footer.paragraphs[0].text = "Page footer"
    d.add_picture(io.BytesIO(png_bytes(200, 80)))
    path = tmp_path / "f.docx"
    d.save(path)
    raw, ps = pages(path)
    texts = [b.text for b in ps[0].blocks]
    assert raw.kind == "docx" and texts[:2] == ["Invoice 7", "Customer: Luis"] and texts[-1] == "Page footer"
    total, value = (next(b for b in ps[0].blocks if b.text == t) for t in ("Total", "45,67 EUR"))
    assert total.box[1] == value.box[1] and total.box[0] < value.box[0]  # same row, another column
    assert len(ps) == 2 and ps[1].image is not None and ps[1].origin.startswith("image ")
    _, no_images = pages(path, ocr_embedded_images=False)
    assert len(no_images) == 1


W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
MC_NS = 'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
VML_NS = 'xmlns:v="urn:schemas-microsoft-com:vml"'


def test_docx_text_that_python_docx_skips(tmp_path):
    docx = pytest.importorskip("docx")
    from docx.oxml import parse_xml

    d = docx.Document()
    body = d.element.body
    run = '<w:r><w:t xml:space="preserve">{}</w:t></w:r>'
    box = "<w:pict><v:shape><v:textbox><w:txbxContent><w:p>{}</w:p></w:txbxContent></v:textbox></v:shape></w:pict>"
    xml = [
        f"<w:sdt {W_NS}><w:sdtContent><w:p>{run.format('Block control')}</w:p></w:sdtContent></w:sdt>",
        f"<w:p {W_NS}>{run.format('Name: ')}<w:sdt><w:sdtContent>{run.format('Luis Perez')}</w:sdtContent></w:sdt></w:p>",
        (
            f'<w:p {W_NS}>{run.format("Amount ")}<w:ins w:id="1" w:author="a">{run.format("45,67")}</w:ins>'
            f'<w:del w:id="2" w:author="a"><w:r><w:delText>99,99</w:delText></w:r></w:del></w:p>'
        ),
        f'<w:p {W_NS}>{run.format("File no. ")}<w:fldSimple w:instr=" MERGEFIELD Exp ">{run.format("2026/0042")}</w:fldSimple></w:p>',
        # modern text box: the same text in mc:Choice and in its VML copy (mc:Fallback)
        (
            f'<w:p {W_NS} {MC_NS} {VML_NS}><w:r><mc:AlternateContent><mc:Choice Requires="wps">'
            f"{box.format(run.format('Text box'))}</mc:Choice><mc:Fallback>{box.format(run.format('Text box'))}"
            "</mc:Fallback></mc:AlternateContent></w:r></w:p>"
        ),
    ]
    for item in xml:
        body.insert(len(body) - 1, parse_xml(item))
    table = d.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Cell"
    table.cell(0, 1).add_table(1, 1).cell(0, 0).text = "Nested table"
    header = d.sections[0].header
    header.add_table(1, 2, docx.shared.Inches(6)).rows[0].cells[1].text = "Letterhead in a table"
    path = tmp_path / "formulario.docx"
    d.save(path)
    _, (page, *_) = pages(path)
    rows = [" ".join(b.text for b in row) for row in page.rows]
    assert rows[0] == "Letterhead in a table"
    for expected in (
        "Block control",
        "Name: Luis Perez",
        "Amount 45,67",
        "File no. 2026/0042",
        "Cell",
        "Nested table",
    ):
        assert expected in rows, expected
    assert rows.count("Text box") == 1 and not any(
        "99,99" in r for r in rows
    )  # without the VML copy or what was deleted


def test_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Product", "Amount"])
    ws.append(["Diesel", 45.5])
    ws.append(["Date", dt.datetime(2026, 9, 11)])  # noqa: DTZ001 - Excel does not store a time zone
    wb.create_sheet("Notes")["B2"] = "only in B2"
    path = tmp_path / "v.xlsx"
    wb.save(path)
    raw, ps = pages(path)
    assert raw.metadata["sheets"] == ["Sales", "Notes"]
    assert [p.origin for p in ps] == ["sheet Sales", "sheet Notes"]
    assert [b.text for b in ps[0].blocks] == ["Product", "Amount", "Diesel", "45.5", "Date", "2026-09-11"]
    assert ps[1].blocks[0].text == "only in B2" and ps[1].blocks[0].box[0] > 0  # column B


def test_pptx(tmp_path):
    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches

    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Resultados 2026"
    slide.shapes.add_textbox(Inches(1), Inches(3), Inches(4), Inches(1)).text_frame.text = "Sales +12%"
    slide.notes_slide.notes_text_frame.text = "Speaker note"
    slide.shapes.add_picture(io.BytesIO(png_bytes(300, 100)), Inches(5), Inches(4))
    path = tmp_path / "s.pptx"
    prs.save(path)
    _, ps = pages(path)
    assert [b.text for b in ps[0].blocks] == ["Resultados 2026", "Sales +12%"]  # reading order
    box = next(b.box for b in ps[0].blocks if b.text == "Sales +12%")
    assert box[:2] == (96, 288)  # real position: 1 and 3 inches at 96 px/inch
    assert ps[0].origin == "slide 1"
    assert ps[1].origin == "slide 1, notes" and [b.text for b in ps[1].blocks] == ["Speaker note"]
    assert ps[2].image is not None and ps[2].origin.startswith("slide 1, ")
    assert [p.number for p in ps] == [1, 2, 3]


def test_text_encodings_and_csv(tmp_path):
    latin = tmp_path / "latin1.txt"
    latin.write_bytes("Año 2026: el señor Muñoz pagó la cuota del camión en efectivo.".encode("cp1252"))
    _, ps = pages(latin)  # no hint: cp1252 wins the tie with cp1250 ("ñ", not "ń")
    assert ps[0].blocks[0].text == "Año 2026: el señor Muñoz pagó la cuota del camión en efectivo."
    polish = tmp_path / "pl.txt"
    polish.write_bytes("Zakład pracy: żółta ćma".encode("cp1250"))
    _, ps = pages(polish, languages=["pl"])  # with a language hint: its legacy encoding
    assert ps[0].blocks[0].text == "Zakład pracy: żółta ćma"
    _, ps = pages(polish, detect_languages=["ru", "pl"])  # the languages detection is limited to are a hint too
    assert ps[0].blocks[0].text == "Zakład pracy: żółta ćma"  # without a hint it came out "Zak³ad pracy: ¿ó³ta æma"
    russian = tmp_path / "ru.txt"
    russian.write_bytes("Счёт № 7: итого 45,67 рублей, оплачено наличными.".encode("cp1251"))
    _, ps = pages(russian, languages=["es", "ru"])  # several languages: the one that really fits wins
    assert ps[0].blocks[0].text == "Счёт № 7: итого 45,67 рублей, оплачено наличными."
    utf16 = tmp_path / "u16.txt"
    utf16.write_bytes("hello\nworld".encode("utf-16"))
    _, ps = pages(utf16)
    assert [b.text for b in ps[0].blocks] == ["hello", "world"]
    sheet = tmp_path / "data.csv"
    sheet.write_text("Product;Amount\nDiesel;45,50\n", encoding="utf-8")
    _, ps = pages(sheet)
    blocks = ps[0].blocks
    assert [b.text for b in blocks] == ["Product", "Amount", "Diesel", "45,50"]
    assert blocks[0].box[1] == blocks[1].box[1] and blocks[0].box[0] < blocks[1].box[0]


def test_html():
    html = (
        b"<html><head><title>Invoice</title><style>p{color:red}</style><script>var x=1</script></head>"
        b"<body><h1>Petrol station</h1><p>Total<br>paid</p><table><tr><th>Item</th><th>Amount</th></tr>"
        b"<tr><td>Diesel</td><td>45,67 EUR</td></tr></table></body></html>"
    )
    raw, ps = pages(io.BytesIO(html))
    assert raw.kind == "html" and raw.metadata["title"] == "Invoice"
    assert [b.text for b in ps[0].blocks] == [
        "Petrol station",
        "Total",
        "paid",
        "Item",
        "Amount",
        "Diesel",
        "45,67 EUR",
    ]


def test_email_reads_images_pasted_in_the_body():
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = "a@b.es", "c@d.es", "Incidencia"
    msg.set_content("Ver screenshot")
    msg.add_alternative('<p>Ver screenshot</p><img src="cid:screenshot"><img src="cid:icono">', subtype="html")
    html = msg.get_payload()[1]
    html.add_related(png_bytes(320, 120, "CAPTURA"), "image", "png", cid="<screenshot>")
    html.add_related(png_bytes(16, 16, ""), "image", "png", cid="<icono>")  # signature icon: no OCR
    msg.add_attachment(b"Total: 45,67 EUR", maintype="text", subtype="plain", filename="note.txt")
    raw, ps = pages(msg.as_bytes())
    assert [(p.origin, p.image is not None) for p in ps[1:]] == [
        ("body image screenshot", True),
        ("attachment note.txt", False),
    ]
    assert raw.metadata["attachments"] == ["note.txt"] and raw.metadata["inline_images"] == ["screenshot", "icono"]
    assert "Ver screenshot" in [b.text for b in ps[0].blocks]  # the HTML of the same body is not repeated


def test_email_with_attachments():
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = "shop@example.com", "customer@example.com", "Your invoice"
    msg.set_content("Hello,\nwe attach the invoice.")
    msg.add_attachment(b"Total: 45,67 EUR", maintype="text", subtype="plain", filename="notes.txt")
    msg.add_attachment(png_bytes(), maintype="image", subtype="png", filename="ticket.png")
    msg.add_attachment(
        b"\xd0\xcf\x11\xe0" + b"\x00" * 64, maintype="application", subtype="msword", filename="viejo.doc"
    )
    raw, ps = pages(msg.as_bytes())
    assert raw.kind == "email" and raw.metadata["subject"] == "Your invoice"
    body = [b.text for b in ps[0].blocks]
    assert "Subject: Your invoice" in body and "we attach the invoice." in body
    assert ps[1].origin == "attachment notes.txt" and ps[1].blocks[0].text == "Total: 45,67 EUR"
    assert ps[2].origin == "attachment ticket.png" and ps[2].image is not None
    assert [p.number for p in ps] == [1, 2, 3]
    assert any("viejo.doc" in w for w in raw.warnings)  # unsupported attachment: a warning, not an error


def test_register_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders_mod, "_LOADERS", dict(loaders_mod._LOADERS))
    monkeypatch.setattr(loaders_mod, "_EXTENSIONS", {})
    register_loader("rtf", lambda src, cfg: single_page("rtf", [["Test RTF"]]), (".rtf",))
    path = tmp_path / "a.rtf"
    path.write_bytes(b"{\\rtf1 hello}")
    raw, ps = pages(path)
    assert raw.kind == "rtf" and ps[0].blocks[0].text == "Test RTF"
