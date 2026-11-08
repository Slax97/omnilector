# SPDX-License-Identifier: AGPL-3.0-or-later
"""Several files at once: one result or one error per file, ZIP as a group or separately, JSON output,
command line with several files and configuration through environment variables."""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
import logging
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

import omnilector
from omnilector import OCR, Document, OCRConfig
from omnilector.exceptions import DocumentLoadError, InferenceError
from omnilector.loaders import Source, archive, kind_of
from omnilector.result import Block, Cell, Line, Page, Table

from .conftest import make_engine, png_bytes, reader

HELLO = reader(("Hello 123", (10, 30, 120, 60), 0.9, "rapidocr"), width=320)  # what it "reads" in any image


def zipped(files: dict[str, str | bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def ocr(use_engines):
    use_engines(make_engine("rapidocr", HELLO))
    return OCR(strategy="fast", handwritten=False)


# --------------------------------------------------------------------------- one result per file


def test_one_result_or_error_per_file(ocr, tmp_path):
    good = tmp_path / "note.txt"
    good.write_text("Total: 45,67 EUR", encoding="utf-8")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 10)
    unknown = tmp_path / "data.bin"
    unknown.write_bytes(bytes(range(256)) * 4)
    missing = tmp_path / "missing.pdf"
    docs = list(ocr.read_many([good, missing, broken, unknown, ("upload.png", png_bytes()), good]))
    assert [d.source for d in docs] == [str(good), str(missing), str(broken), str(unknown), "upload.png", str(good)]
    assert [type(d.error).__name__ if d.error else None for d in docs] == [
        None,
        "DocumentLoadError",
        "DocumentLoadError",
        "UnsupportedFormatError",
        None,
        None,
    ]
    assert docs[0].text == docs[5].text == "Total: 45,67 EUR" and docs[4].text == "Hello 123"
    assert [d.kind for d in docs[1:4]] == ["unknown", "image", "unknown"]  # the kind, when it is known
    assert all(d.pages == [] and d.text == "" for d in docs[1:4])
    (doc,) = omnilector.read_many([png_bytes()], strategy="fast", handwritten=False)  # the shortcut
    assert doc.text == "Hello 123" and doc.source == "<bytes>"


def test_results_come_as_each_file_is_read(use_engines):
    engine = make_engine("rapidocr", HELLO)
    use_engines(engine)
    docs = OCR(strategy="fast", handwritten=False).read_many([png_bytes(), png_bytes()])
    assert next(docs).text == "Hello 123" and len(engine.calls) == 1  # the second one has not been read yet
    assert len(list(docs)) == 1 and len(engine.calls) == 2
    with pytest.raises(TypeError):  # a single path is not a list of files (it would be read letter by letter)
        OCR(strategy="fast").read_many("invoice.pdf")


def test_a_file_whose_pages_all_fail_keeps_its_warnings(use_engines):
    def broken(image):
        raise InferenceError("it broke")

    use_engines(make_engine("rapidocr", broken))
    ocr = OCR(strategy="fast", handwritten=False)
    with pytest.raises(InferenceError):  # read() of one file still raises
        ocr.read(png_bytes())
    (doc,) = ocr.read_many([png_bytes()])
    assert isinstance(doc.error, InferenceError) and doc.kind == "image" and len(doc.pages) == 1
    assert doc.warnings == ["page 1: it broke"]
    assert doc.error.__traceback__ is None  # without its traceback: it does not keep the page's memory alive


def test_unexpected_errors_are_kept_and_logged(ocr, monkeypatch, caplog):
    from omnilector import loaders

    def boom(src, config):
        raise RuntimeError("a library failure")

    monkeypatch.setitem(loaders._LOADERS, "text", boom)
    with caplog.at_level(logging.ERROR, logger="omnilector"):
        docs = list(ocr.read_many([("a.txt", b"hello"), png_bytes()]))
    assert isinstance(docs[0].error, RuntimeError) and docs[0].kind == "text" and docs[1].text == "Hello 123"
    (record,) = caplog.records
    assert (
        record.getMessage() == "a.txt: unexpected error while reading it" and record.exc_info
    )  # with its traceback, to look into it


# --------------------------------------------------------------------------- ZIP: as a group or separately


def test_zip_as_one_document_or_each_file_apart(ocr, monkeypatch):
    monkeypatch.setattr(archive, "BOMB_MIN_BYTES", 1000)
    data = zipped(
        {
            "invoice.txt": "Total: 45,67 EUR",
            "photos/ticket.png": png_bytes(),
            "nested.zip": zipped({"inside.txt": "Text of the inner zip"}),
            "bomb.txt": "0" * 500_000,
            ".DS_Store": "hidden",
        }
    )
    (group,) = ocr.read_many([("batch.zip", data)])
    assert group.kind == "zip" and group.source == "batch.zip"
    assert [p.origin for p in group.pages] == [
        "file invoice.txt",
        "file photos/ticket.png",
        "file nested.zip, file inside.txt",
    ]
    assert group.warnings == ["bomb.txt: too large when decompressed, skipped"]
    apart = list(ocr.read_many([("batch.zip", data)], split_archives=True))
    assert [(d.source, d.kind, d.text) for d in apart] == [
        ("batch.zip/invoice.txt", "text", "Total: 45,67 EUR"),
        ("batch.zip/photos/ticket.png", "image", "Hello 123"),
        ("batch.zip/nested.zip/inside.txt", "text", "Text of the inner zip"),  # the inner one, separately too
        ("batch.zip/bomb.txt", "unknown", ""),
    ]
    assert (
        isinstance(apart[3].error, DocumentLoadError) and str(apart[3].error) == "too large when decompressed, skipped"
    )
    assert all([(p.number, p.origin) for p in d.pages] == [(1, None)] for d in apart[:3])  # each one, as if on its own


def test_ocr_limit_is_for_the_whole_zip_or_for_each_file(ocr):
    limited = OCR(strategy="fast", handwritten=False, max_ocr_pages=2)
    data = zipped({f"photo{i}.png": png_bytes() for i in range(3)})
    (group,) = limited.read_many([("batch.zip", data)])
    assert len(group.pages) == 2 and group.warnings == ["1 pages or images without OCR: limit of 2 (max_ocr_pages)"]
    apart = list(limited.read_many([("batch.zip", data)], split_archives=True))
    assert [d.text for d in apart] == ["Hello 123"] * 3 and not any(d.warnings for d in apart)


def test_split_zip_empty_truncated_or_broken(ocr, monkeypatch):
    for name, data in (("empty.zip", zipped({})), ("mac.zip", zipped({"__MACOSX/._a.txt": "x", ".DS_Store": "y"}))):
        (empty,) = ocr.read_many([(name, data)], split_archives=True)  # nothing inside: the ZIP itself, empty
        assert (empty.source, empty.kind, empty.pages, empty.error) == (name, "zip", [], None)
    monkeypatch.setattr(archive, "MAX_ENTRIES", 1)
    docs = list(ocr.read_many([("batch.zip", zipped({"a.txt": "one", "b.txt": "two"}))], split_archives=True))
    assert [d.source for d in docs] == ["batch.zip/a.txt"]
    assert docs[0].warnings == [
        "batch.zip: ZIP with 2 files: only the first 1 are read"
    ]  # the ZIP's warning, with its first file
    (broken,) = ocr.read_many([("broken.zip", b"PK\x03\x04" + b"\x00" * 64)], split_archives=True)
    assert isinstance(broken.error, DocumentLoadError) and broken.source == "broken.zip"


def test_split_tar_and_its_late_warnings(ocr, monkeypatch):
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in (("a.txt", b"one"), ("b.txt", b"two"), ("c.txt", b"three")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(archive, "MAX_ENTRIES", 2)
    docs = list(ocr.read_many([("batch.tar.gz", buf.getvalue())], split_archives=True))
    assert [(d.source, d.text) for d in docs] == [("batch.tar.gz/a.txt", "one"), ("batch.tar.gz/b.txt", "two")]
    # the TAR is read in one pass: the warning that there are more comes at the end, with the last one read
    assert docs[0].warnings == [] and docs[1].warnings == [
        "batch.tar.gz: TAR with more than 2 files: only the first 2 are read"
    ]


def test_a_damaged_file_inside_a_zip_is_skipped(ocr):
    data = bytearray(zipped({"broken.txt": "text that is going to be damaged " * 50, "good.txt": "Hello from the ZIP"}))
    info = zipfile.ZipFile(io.BytesIO(bytes(data))).getinfo("broken.txt")
    name_len, extra_len = struct.unpack("<HH", data[info.header_offset + 26 : info.header_offset + 30])
    start = info.header_offset + 30 + name_len + extra_len
    data[start : start + 4] = (
        b"\xff\xff\xff\xff"  # impossible deflate: zlib.error (it used to bring the whole ZIP down)
    )
    doc = ocr.read(bytes(data))
    assert doc.text == "Hello from the ZIP" and doc.warnings[0].startswith("broken.txt: could not be decompressed")
    apart = list(ocr.read_many([("batch.zip", bytes(data))], split_archives=True))
    assert [(d.source, d.error is None) for d in apart] == [
        ("batch.zip/broken.txt", False),
        ("batch.zip/good.txt", True),
    ]


def test_a_file_that_cannot_be_opened_is_a_load_error(tmp_path):
    with pytest.raises(DocumentLoadError, match="Could not read"):  # no permission, drive down... (a folder, here)
        kind_of(Source(name="no-permission.pdf", path=tmp_path))


# --------------------------------------------------------------------------- JSON


def test_json_has_every_field_and_reads_back():
    page = Page(
        number=1,
        lines=[Line("Привет, señor", (0.1, 0.1, 0.5, 0.2), 0.97, ["rapidocr"], ["Привет, сеnor"], row=0, block=0)],
        method="ocr",
        rotation=90.0,
        script="cyrillic",
        blocks=[Block("title", (0.1, 0.1, 0.5, 0.2), [0])],
        tables=[
            Table(
                (0.1, 0.5, 0.9, 0.7),
                np.int64(1),
                2,
                [
                    Cell(0, 0, "Total", (0.1, 0.5, 0.5, 0.7)),
                    Cell(0, 1, "45,67", (0.5, 0.5, 0.9, 0.7)),
                ],
            )
        ],
    )
    metadata = {
        "created": dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.timezone.utc),
        "files": ("a.txt", "b.png"),
    }  # a docx brings its date like this
    doc = Document(pages=[page], source="letter.jpg", kind="image", metadata=metadata, warnings=["a warning"])
    data = doc.to_dict()
    assert json.loads(doc.to_json()) == data and "Привет, señor" in doc.to_json()  # not escaped
    assert list(data) == ["source", "kind", "text", "error", "warnings", "metadata", "pages"]
    assert {f.name for f in dataclasses.fields(Document)} <= set(data)  # no field is left out
    assert data["text"] == "Привет, señor" and data["error"] is None
    assert data["metadata"] == {"created": "2026-10-31T12:00:00+00:00", "files": ["a.txt", "b.png"]}
    (p,) = data["pages"]
    assert list(p) == ["number", "method", "origin", "rotation", "script", "text", "lines", "blocks", "tables"]
    assert p["lines"] == [
        {
            "text": "Привет, señor",
            "box": [0.1, 0.1, 0.5, 0.2],
            "confidence": 0.97,
            "engines": ["rapidocr"],
            "alternatives": ["Привет, сеnor"],
            "handwritten": False,
            "row": 0,
            "block": 0,
        }
    ]
    assert p["blocks"] == [{"kind": "title", "box": [0.1, 0.1, 0.5, 0.2], "lines": [0]}]
    assert p["tables"][0]["rows"] == 1 and p["tables"][0]["cells"][1] == {
        "row": 0,
        "col": 1,
        "text": "45,67",
        "box": [0.5, 0.5, 0.9, 0.7],
        "rowspan": 1,
        "colspan": 1,
    }
    failed = Document(source="broken.png", error=DocumentLoadError("Corrupt image"))
    assert failed.to_dict()["error"] == {"type": "DocumentLoadError", "message": "Corrupt image"}
    assert json.loads(failed.to_json(indent=2))["pages"] == []


# --------------------------------------------------------------------------- environment variables


def test_settings_from_environment_and_explicit_wins(monkeypatch):
    monkeypatch.setenv("OMNILECTOR_STRATEGY", "balanced")
    monkeypatch.setenv("OMNILECTOR_LANGUAGES", "es, ru")
    monkeypatch.setenv("OMNILECTOR_HANDWRITTEN", "no")
    cfg = OCRConfig()
    assert (cfg.strategy, cfg.languages, cfg.handwritten) == ("balanced", ("es", "ru"), False)
    assert cfg.scales == (1.0, 1.5)  # the strategy from the variable decides its own settings
    explicit = OCRConfig(strategy="fast", languages=(), handwritten=True)  # what is passed wins; () = none
    assert (explicit.strategy, explicit.languages, explicit.handwritten) == ("fast", (), True)
    monkeypatch.setenv("OMNILECTOR_DETECT_LANGUAGES", "es, rus,ar")
    assert OCRConfig().detect_languages == ("es", "ru", "ar")
    # detection among the scripts of languages (here, the variable: es, ru) and, without them, those of detect_languages
    assert OCRConfig().detect_scripts == {"latin", "east_slavic", "cyrillic"}  # both Cyrillic models
    assert OCRConfig(languages=()).detect_scripts == {"latin", "east_slavic", "cyrillic", "arabic"}
    assert OCRConfig(languages=(), detect_languages=()).detect_scripts is None  # () = all, even if the variable is set
    assert OCR().config.strategy == "balanced" and OCR(strategy="accurate").config.strategy == "accurate"
    monkeypatch.setenv("OMNILECTOR_STRATEGY", "")  # empty: as if it were not set
    assert OCRConfig().strategy == "maximum_recall"
    for name, value in (
        ("STRATEGY", "fastest"),
        ("LANGUAGES", "klingon"),
        ("DETECT_LANGUAGES", "klingon"),
        ("HANDWRITTEN", "maybe"),
        ("THREADS", "two"),
    ):
        monkeypatch.setenv(f"OMNILECTOR_{name}", value)
        with pytest.raises(ValueError, match=f"OMNILECTOR_{name}="):  # the error says which variable to look at
            OCRConfig()
        monkeypatch.delenv(f"OMNILECTOR_{name}")


# --------------------------------------------------------------------------- command line


def test_cli_with_several_files_folders_and_json(ocr, tmp_path):
    from typer.testing import CliRunner

    from omnilector.cli import app

    folder = tmp_path / "batch"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_text("One", encoding="utf-8")
    (folder / "sub" / "b.txt").write_text("Two", encoding="utf-8")
    (folder / ".hidden.txt").write_text("no", encoding="utf-8")
    (folder / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\nbasura")
    single = tmp_path / "c.txt"
    single.write_text("Three", encoding="utf-8")
    runner = CliRunner()
    out = runner.invoke(app, ["read", str(single), str(folder), "--strategy", "fast"])
    assert out.exit_code == 1  # one could not be read; the others come out anyway
    assert out.stdout.splitlines() == [
        f"==> {single} <==",
        "Three",
        "",
        f"==> {folder / 'a.txt'} <==",
        "One",
        "",
        f"==> {folder / 'sub' / 'b.txt'} <==",
        "Two",
    ]
    assert out.stderr.startswith("error: Corrupt image") and "broken.png" in out.stderr
    out = runner.invoke(app, ["read", str(folder), "-f", "json", "--strategy", "fast"])
    found = [json.loads(line) for line in out.stdout.splitlines()]
    assert [(Path(d["source"]).name, d["text"], d["error"] and d["error"]["type"]) for d in found] == [
        ("a.txt", "One", None),
        ("broken.png", "", "DocumentLoadError"),
        ("b.txt", "Two", None),
    ]
    assert out.exit_code == 1 and out.stderr == ""  # in JSON, warnings and errors go inside
    batch = tmp_path / "batch.zip"
    batch.write_bytes(zipped({"x.txt": "Ex", "y.txt": "Why"}))
    out = runner.invoke(app, ["read", str(batch), "--split-archives", "--strategy", "fast"])
    assert out.exit_code == 0 and out.stdout.splitlines() == [
        f"==> {batch}/x.txt <==",
        "Ex",
        "",
        f"==> {batch}/y.txt <==",
        "Why",
    ]
    out = runner.invoke(app, ["read", str(tmp_path / "nothing.pdf")])
    assert out.exit_code == 1 and out.stderr.strip() == f"error: File not found: {tmp_path / 'nothing.pdf'}"
    (tmp_path / "empty").mkdir()
    out = runner.invoke(app, ["read", str(tmp_path / "empty")])
    assert out.exit_code == 1 and out.stderr.strip() == "error: no files to read"
    polish = tmp_path / "pl.txt"
    polish.write_bytes("Zakład pracy: żółta ćma".encode("cp1250"))
    out = runner.invoke(app, ["read", str(polish), "--detect", "ru,pl"])  # language hint for the encoding
    assert out.exit_code == 0 and out.stdout.strip() == "Zakład pracy: żółta ćma"
    for option in (["--detect", "klingon"], ["--lang", "klingon"], ["--strategy", "fastest"]):
        out = runner.invoke(
            app, ["read", str(polish), *option]
        )  # an option that is not valid: its message, no traceback
        assert out.exit_code == 2 and out.stderr.startswith("error: ") and "Traceback" not in out.output
    out = runner.invoke(app, ["download", "--lang", "klingon"])
    assert out.exit_code == 2 and out.stderr.startswith("error: Unsupported language 'klingon'")
