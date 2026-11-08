# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLI: omnilector read files or folders (pdf, jpg, docx...) · omnilector pdf (searchable PDF) ·
omnilector engines · omnilector download."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from omnilector import OCR, Document

app = typer.Typer(
    help="omnilector: extracts the text of any file (image, PDF, Office, HTML, email).",
    no_args_is_help=True,
)

# OCR options, the same in read and in pdf
Strategy = Annotated[
    str | None,
    typer.Option(help="fast | balanced | accurate | maximum_recall (default: OMNILECTOR_STRATEGY, or maximum_recall)"),
]
Lang = Annotated[
    str, typer.Option(help="Expected languages, e.g. es,en: they pick the model/engine (default: OMNILECTOR_LANGUAGES)")
]
Detect = Annotated[
    str,
    typer.Option(
        "--detect",
        help="Without --lang: languages to detect among, e.g. es,ru,ar (default: OMNILECTOR_DETECT_LANGUAGES, or all)",
    ),
]
Engine = Annotated[str | None, typer.Option(help="Force an engine: rapidocr | tesseract | easyocr")]
Handwritten = Annotated[
    bool | None,
    typer.Option(
        "--handwritten/--no-handwritten",
        help="Read handwriting (default: OMNILECTOR_HANDWRITTEN, or yes)",
        show_default=False,
    ),
]
Device = Annotated[str, typer.Option(help="cpu | cuda | auto")]
Threads = Annotated[int | None, typer.Option(help="CPU threads per engine (default: the cores of the process)")]
Memory = Annotated[
    str | None,
    typer.Option("--memory", "-m", help="Memory for the OCR: 2g, 1536m... (default: the container's or the machine's)"),
]
Passwords = Annotated[
    list[str] | None,
    typer.Option("--password", help="Password of protected PDFs and encrypted ZIPs (can be repeated; tried in order)"),
]
TimeLimit = Annotated[
    float | None,
    typer.Option(
        "--time-limit", help="Seconds per file; after them, the rest is read without OCR or without extra passes"
    ),
]
Verbose = Annotated[
    bool, typer.Option("--verbose", "-v", help="Show the situation seen on each page and the engine chosen")
]


def _ocr(
    strategy: str | None,
    lang: str,
    detect: str,
    engine: str | None,
    handwritten: bool | None,
    device: str,
    threads: int | None,
    memory: str | None,
    passwords: list[str] | None,
    time_limit: float | None,
    verbose: bool,
) -> OCR:
    return _new_ocr(
        strategy=strategy,
        languages=lang or None,
        detect_languages=detect or None,
        engine=engine,
        handwritten=handwritten,
        device=device,
        threads=threads,
        memory_mb=_memory(memory),
        log_level="DEBUG" if verbose else None,
        passwords=tuple(passwords or ()),
        time_limit=time_limit,
    )


def _new_ocr(**options: object) -> OCR:
    """The OCR with those options. One that is not valid (a language, a strategy, an environment variable)
    comes out as an error with its message and exit code 2, without the traceback."""
    from omnilector import OCR

    try:
        return OCR(**options)  # type: ignore[arg-type]
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from None


@app.command()
def read(
    paths: Annotated[list[str], typer.Argument(help="Files or folders (with their subfolders)", show_default=False)],
    strategy: Strategy = None,
    lang: Lang = "",
    detect: Detect = "",
    engine: Engine = None,
    handwritten: Handwritten = None,
    device: Device = "cpu",
    threads: Threads = None,
    memory: Memory = None,
    fmt: str = typer.Option(
        "text", "--format", "-f", help="text | md (with a header per page) | json (one line per file)"
    ),
    split_archives: bool = typer.Option(
        False, "--split-archives", help="Each file inside a ZIP or TAR, separately (otherwise it is one document)"
    ),
    passwords: Passwords = None,
    time_limit: TimeLimit = None,
    verbose: Verbose = False,
) -> None:
    """Extracts the text of one or more files (or of those in a folder). If one cannot be read, it goes on
    with the rest and ends with exit code 1."""
    if fmt not in ("text", "md", "json"):
        raise typer.BadParameter("use text, md or json", param_hint="--format")
    files = _files(paths)
    if not files:
        typer.echo("error: no files to read", err=True)
        raise typer.Exit(1)
    ocr = _ocr(strategy, lang, detect, engine, handwritten, device, threads, memory, passwords, time_limit, verbose)
    several = len(files) > 1 or split_archives
    shown = failed = 0
    for doc in ocr.read_many(files, split_archives=split_archives):
        if fmt == "json":  # warnings and error go inside
            typer.echo(doc.to_json())
        elif doc.error is None:
            if several and fmt == "text":
                typer.echo(("\n" if shown else "") + f"==> {doc.source} <==")
            typer.echo(doc.to_markdown() if fmt == "md" else doc.text)
            shown += 1
        failed += doc.error is not None
        if fmt != "json":
            _report(doc, several)
    if failed:
        raise typer.Exit(1)


@app.command()
def pdf(
    source: Annotated[str, typer.Argument(help="Scanned PDF or image (jpg, png, multi-page tiff...)")],
    output: Annotated[str, typer.Argument(help="Searchable PDF to write")],
    strategy: Strategy = None,
    lang: Lang = "",
    detect: Detect = "",
    engine: Engine = None,
    handwritten: Handwritten = None,
    device: Device = "cpu",
    threads: Threads = None,
    memory: Memory = None,
    passwords: Passwords = None,
    time_limit: TimeLimit = None,
    verbose: Verbose = False,
) -> None:
    """Searchable PDF: the scanned PDF (or the image) as it is, with the recognized text invisible on top, to
    search and copy it in any viewer."""
    from omnilector import OmnilectorError

    ocr = _ocr(strategy, lang, detect, engine, handwritten, device, threads, memory, passwords, time_limit, verbose)
    try:
        doc = ocr.searchable_pdf(source, output)
    except OmnilectorError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from None
    _report(doc, several=False)
    lines = sum(1 for p in doc.pages if p.method != "native" for line in p.lines if "native" not in line.engines)
    typer.echo(f"{output}: {len(doc.pages)} pages, {lines} lines read with OCR", err=True)


def _report(doc: Document, several: bool) -> None:
    """Warnings and error of the document on stderr; with several files, with the name of their file."""
    where = f"{doc.source}: " if several else ""
    for warning in doc.warnings:
        typer.echo(f"warning: {where}{warning}", err=True)
    if doc.error is not None:
        message = str(doc.error)
        named = doc.source and doc.source in message  # "File not found: a.pdf" already says it
        typer.echo(f"error: {message}" if named else f"error: {doc.source}: {message}", err=True)


def _files(paths: list[str]) -> list[str]:
    """The paths as they are and, in place of each folder, its files (with subfolders, without hidden files or
    macOS metadata, as in ZIPs), in order."""
    out: list[str] = []
    for value in paths:
        folder = Path(value)
        if not folder.is_dir():
            out.append(value)  # a file or something that does not exist: its error comes out in its result
            continue
        for path in sorted(folder.rglob("*")):
            parts = path.relative_to(folder).parts
            if path.is_file() and not any(p.startswith(".") or p == "__MACOSX" for p in parts):
                out.append(str(path))
    return out


@app.command()
def engines() -> None:
    """Registered OCR engines and whether they are available on this machine."""
    from omnilector import list_engines

    for e in list_engines():
        status = "yes" if e["available"] else "no"
        line = f"{status:3} {e['name']:10} {e['level']:5} ~{e['memory_total_mb']:>4} MB  {e['description']}"
        if not e["available"]:
            line += f"  [{e['reason']}]"
        typer.echo(line)


@app.command()
def download(
    lang: str = typer.Option("", help="Languages to prepare, e.g. es,ru (each script has its own model)"),
    engine: Annotated[
        list[str] | None, typer.Option(help="Only these engines (default: all the installed ones)")
    ] = None,
) -> None:
    """Downloads and loads the models of the installed engines, to use them later without network."""
    for spec in _new_ocr(languages=lang or None).load_models(engine or None):
        typer.echo(f"ready: {spec}")


def _memory(value: str | None) -> int | None:
    from omnilector.utils.resources import parse_memory

    try:
        return parse_memory(value) if value else None
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--memory") from e


def main() -> None:
    # The extracted text can be in any script: on Windows, redirected to a file or pipe, stdout would be
    # cp1252 and would fail with Cyrillic, Chinese... Output is always UTF-8.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    app()


if __name__ == "__main__":
    main()
