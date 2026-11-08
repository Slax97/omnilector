# SPDX-License-Identifier: AGPL-3.0-or-later
"""Legacy Office and RTF with real LibreOffice: generated from modern Office and read back."""

from __future__ import annotations

import io
import subprocess

import pytest

from omnilector import OCR
from omnilector.loaders.convert import find_soffice

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def soffice():
    found = find_soffice()
    if found is None:
        pytest.skip("no LibreOffice")
    return found


def legacy(soffice: str, tmp_path, name: str, data: bytes, target: str) -> tuple[str, bytes]:
    """Converts a docx/xlsx/pptx to the legacy format `target` (doc, xls, ppt, rtf) with LibreOffice: its name
    and its content. With the name, the extension gives the format; without it, a .doc, .xls or .ppt is only
    recognized with olefile (extra [msg]): the three of them, and .msg, are the same OLE container."""
    source = tmp_path / name
    source.write_bytes(data)
    profile = (tmp_path / "perfil").as_uri()
    cmd = [
        soffice,
        "--headless",
        f"-env:UserInstallation={profile}",
        "--convert-to",
        target,
        "--outdir",
        str(tmp_path),
        str(source),
    ]
    subprocess.run(cmd, capture_output=True, timeout=180, check=False)
    out = source.with_suffix(f".{target}")
    if not out.exists():
        pytest.skip(f"LibreOffice did not produce the .{target}")
    return out.name, out.read_bytes()


def test_doc_and_rtf(soffice, tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Grant application")
    table = d.add_table(rows=1, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Total", "45,67 EUR"
    buf = io.BytesIO()
    d.save(buf)
    for target in ("doc", "rtf"):
        doc = OCR(strategy="fast").read(legacy(soffice, tmp_path, "input.docx", buf.getvalue(), target))
        assert doc.kind == target and "LibreOffice" in doc.metadata["converted"]
        assert doc.text.startswith("Grant application") and "45,67 EUR" in doc.text
    # RTF keeps the table (to .doc, LibreOffice exports python-docx's table without widths as a paragraph)
    assert "Total 45,67 EUR" in doc.text.splitlines()


def test_xls_and_ppt(soffice, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    pptx = pytest.importorskip("pptx")
    wb = openpyxl.Workbook()
    wb.active.title = "Sales"
    wb.active.append(["Product", "Amount"])
    buf = io.BytesIO()
    wb.save(buf)
    doc = OCR(strategy="fast").read(legacy(soffice, tmp_path, "sheet.xlsx", buf.getvalue(), "xls"))
    assert doc.kind == "xls" and doc.pages[0].origin == "sheet Sales" and doc.text == "Product Amount"
    prs = pptx.Presentation()
    prs.slides.add_slide(prs.slide_layouts[5]).shapes.title.text = "Resultados 2026"
    buf = io.BytesIO()
    prs.save(buf)
    doc = OCR(strategy="fast").read(legacy(soffice, tmp_path, "diapos.pptx", buf.getvalue(), "ppt"))
    assert doc.kind == "ppt" and doc.text == "Resultados 2026"
