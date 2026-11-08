# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public API: omnilector.read(anything) -> Document (common output for every engine) and
omnilector.read_many(several) -> one Document per input."""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterable, Iterator
from dataclasses import astuple, replace

import cv2

from .config import OCRConfig
from .core.layout import LayoutAnalyzer
from .core.model_manager import ModelManager
from .core.pipeline import OCRPipeline
from .engines import OCREngine, registered_engines
from .exceptions import (
    DependencyMissingError,
    DocumentLoadError,
    EngineNotAvailableError,
    OmnilectorError,
    UnsupportedFormatError,
)
from .loaders import (
    ImageRegion,
    OCRBudget,
    RawPage,
    Source,
    archive_members,
    kind_of,
    load,
    source_name,
    to_source,
)
from .loaders.base import MAX_DEPTH
from .result import Document, Page, TextBlock, make_page
from .routing import EngineSpec
from .routing.languages import SCRIPT_OF, detect_start
from .utils.geometry import reading_rows
from .utils.logging import get_logger
from .utils.resources import PAGE_ENGINE_MB


class OCR:
    """Extracts text from any input, picking the right OCR for each situation.

    Reuses models between calls: create one instance per process (or worker) and reuse it.
    """

    def __init__(
        self,
        strategy: str | None = None,
        device: str = "cpu",
        handwritten: bool | None = None,
        preprocessing: bool = True,
        multi_scale: bool = True,
        model_cache_dir: str | None = None,
        log_level: str | None = None,
        model_size: str = "small",
        languages: str | list[str] | tuple[str, ...] | None = None,
        engine: str | None = None,
        threads: int | None = None,
        memory_mb: int | None = None,
        **options: object,
    ) -> None:
        """strategy, handwritten and languages left as None: those of the OMNILECTOR_STRATEGY,
        OMNILECTOR_HANDWRITTEN and OMNILECTOR_LANGUAGES variables or, without them, maximum_recall, yes and
        none (the rest of the options, in OCRConfig)."""
        self.config = OCRConfig(
            strategy=strategy,
            device=device,
            handwritten=handwritten,
            preprocessing=preprocessing,
            multi_scale=multi_scale,
            model_cache_dir=model_cache_dir,
            log_level=log_level,
            model_size=model_size,
            languages=languages,  # type: ignore[arg-type]
            engine=engine,
            threads=threads,
            memory_mb=memory_mb,
            **options,  # type: ignore[arg-type]
        )
        self.log = get_logger("omnilector", log_level)
        if threads or os.environ.get("OMNILECTOR_THREADS"):
            # set explicitly: OpenCV does not go over those cores either. It is a setting of the whole process,
            # which is why the container's cores do not touch it: OpenCV already respects them on its own
            cv2.setNumThreads(self.config.threads)
        self.manager = ModelManager(self.config)
        self.pipeline = OCRPipeline(self.config, self.manager)
        self._log_resources()

    def _log_resources(self) -> None:
        """How it adapted to the server (DEBUG) and a warning if not even the page engine fits."""
        cfg = self.config
        self.log.debug(
            "resources: threads per engine: %s, memory (MB): %s -> image up to %s px, models (MB): %s, layout: %s",
            cfg.threads or "all",
            cfg.memory_mb or "no limit",
            cfg.max_image_side,
            self.manager.budget if self.manager.budget is not None else "no limit",
            "yes" if cfg.layout else "no",
        )
        if self.manager.budget is not None and self.manager.budget < PAGE_ENGINE_MB:
            self.log.warning(
                "memory_mb=%s is not enough for max_image_side=%s: it may run out of memory (lower max_image_side or raise memory_mb)",
                cfg.memory_mb,
                cfg.max_image_side,
            )

    def read(self, source: object) -> Document:
        """Any input -> Document. Path, bytes, (name, bytes), stream, PIL or ndarray of: image (jpg, png, webp,
        multi-page tiff, heic...), PDF, docx, xlsx, pptx, html, txt/csv, eml with attachments.

        A page that fails stays empty with a warning in Document.warnings; if all of them fail, error.
        """
        doc = self._read(source)
        if doc.error is not None:
            raise doc.error
        return doc

    def read_many(self, sources: Iterable[object], split_archives: bool = False) -> Iterator[Document]:
        """Several inputs (of those read() accepts) -> one Document for each, in order and as they are read. One
        that cannot be read does not stop the rest: it comes out with doc.error and without pages.

        A ZIP is one document with the pages of all its files and one OCR limit (max_ocr_pages) for all
        of them; with split_archives, each file inside is a document ("batch.zip/photos/ticket.jpg") with
        its own limit, as if it had been passed in the list.
        """
        if isinstance(sources, (str, bytes, os.PathLike)):
            raise TypeError("read_many() reads several inputs (a list...); for a single one, read()")
        return (doc for source in sources for doc in self._documents(source, split_archives))

    def _documents(self, source: object, split: bool) -> Iterator[Document]:
        """The document of an input or, with split and if it is a ZIP, the one of each file inside."""
        try:
            src = to_source(source)
            inside = archive_members(src, self.config) if split and src.depth < MAX_DEPTH else None
        except Exception as e:  # noqa: BLE001 - a broken file does not stop the batch
            yield self._failed(source, e)
            return
        if inside is None:
            yield self._document(src)
            return
        notes, members = inside
        docs = (
            doc
            for member, problem in members
            for doc in (
                [self._failed(member, DocumentLoadError(problem))] if problem else self._documents(member, split)
            )
        )
        # the warnings of the ZIP or TAR go with the file being read when they were raised (earlier ones, with
        # the first); each document comes out when there is another one after it, or at the end, to get the last ones
        held: Document | None = None
        for doc in docs:
            if held is not None:
                yield held
            doc.warnings[:0] = [f"{src.name}: {note}" for note in notes]
            notes.clear()
            held = doc
        if held is None:  # nothing inside: the file itself, empty
            yield self._document(src)
            return
        held.warnings += [f"{src.name}: {note}" for note in notes]
        yield held

    def _document(self, src: Source) -> Document:
        """The one from read(), without raising: if it cannot be read, the error goes in doc.error."""
        try:
            doc = self._read(src)
        except Exception as e:  # noqa: BLE001 - a broken file does not stop the batch
            return self._failed(src, e)
        if doc.error is not None:
            _without_traceback(doc.error)
        return doc

    def _failed(self, source: object, error: Exception) -> Document:
        """Document of an input that could not be read: no pages, with the error and its kind if known."""
        name = source_name(source)
        if not isinstance(error, OmnilectorError):  # not a broken file nor a missing extra: a failure to look into
            self.log.error("%s: unexpected error while reading it", name, exc_info=error)
        kind = "unknown"
        if isinstance(source, Source) and (source.path is not None or source.data):
            with contextlib.suppress(Exception):
                kind = kind_of(source)
        return Document(source=name, kind=kind, error=_without_traceback(error))

    def searchable_pdf(self, source: object, output: str | os.PathLike) -> Document:
        """Searchable PDF in `output`: the scanned PDF (or the image, or the TIFF) as it is, with the
        recognized text invisible on top of each line, to search and copy it in any viewer. Returns the
        Document read. From a PDF, only what went through the OCR gets new text."""
        from .searchable import write

        src = to_source(source)
        kind = kind_of(src)
        if kind not in ("pdf", "image"):
            raise UnsupportedFormatError(f"Searchable PDF only from PDFs and images: {src.name} is {kind}")
        overlay: dict = {}
        doc = self._read(src, overlay)
        if doc.error is not None:
            raise doc.error
        write(src, kind, overlay, output, self.config)
        return doc

    def _read(self, source: object, overlay: dict | None = None) -> Document:
        """Like read(), but if every page fails it returns the document with doc.error (and its warnings).
        With `overlay`, it stores there what the OCR read on each page, in place (for searchable_pdf)."""
        pages: list[Page] = []
        failures: list[OmnilectorError] = []
        seen: dict[object, list[TextBlock]] = {}  # images repeated on several pages (letterhead)
        with load(source, self.config) as raw:
            for raw_page in raw.pages:
                page = self._page(raw_page, raw.warnings, failures, seen, self._script(pages), raw.budget, overlay)
                pages.append(page)
        budget = raw.budget
        if budget is not None and budget.skipped:
            raw.warnings.append(
                f"{budget.skipped} pages or images without OCR: limit of {self.config.max_ocr_pages} (max_ocr_pages)"
            )
        if budget is not None and (budget.late or budget.hurried):
            parts = [f"{budget.late} pages or images without OCR"] if budget.late else []
            parts += [f"{budget.hurried} read without the extra passes"] if budget.hurried else []
            raw.warnings.append(f"time limit of {self.config.time_limit:g} s (time_limit): " + " and ".join(parts))
        doc = Document(pages=pages, source=raw.source, kind=raw.kind, metadata=raw.metadata, warnings=raw.warnings)
        if failures and len(failures) == len(pages):
            doc.error = failures[0]
        return doc

    @staticmethod
    def _script(pages: list[Page]) -> str | None:
        """Script of the last page read with OCR: the next one starts with it."""
        return next((p.script for p in reversed(pages) if p.script), None)

    def _page(
        self,
        raw: RawPage,
        warnings: list[str],
        failures: list[OmnilectorError],
        seen: dict[object, list[TextBlock]],
        script: str | None = None,
        budget: OCRBudget | None = None,
        overlay: dict | None = None,
    ) -> Page:
        deadline = budget.deadline if budget is not None else None
        keep = overlay is not None and raw.origin is None  # a page of the file itself, not of an attachment
        if raw.image is not None:  # the whole page goes to the OCR
            try:
                read = self.pipeline.run(raw.image, hint=raw.hint, script=script, deadline=deadline)
            except (DependencyMissingError, EngineNotAvailableError):
                raise  # installation/configuration problem: not about this page
            except OmnilectorError as e:  # a broken page does not bring the document down
                warnings.append(f"page {raw.number}: {e}")
                failures.append(e)
                return make_page([], raw.size, raw.number, "ocr", raw.origin)
            if read.hurried and budget is not None:
                budget.hurried += 1
            if keep:
                overlay[raw.number] = (raw.size, read.original_blocks())  # type: ignore[index]
            return make_page(
                read.rows, read.size, raw.number, "ocr", raw.origin, read.rotation, read.script, read.zones, read.tables
            )
        found: list[TextBlock] = []
        for region in raw.regions:  # images inside a page with text: photos, stamps, signatures
            try:
                found += self._region(region, seen, deadline)
            except (DependencyMissingError, EngineNotAvailableError):
                raise
            except OmnilectorError as e:  # the native text of the page is still valid
                warnings.append(f"page {raw.number}: an image could not be read: {e}")
                if not raw.rows:
                    failures.append(e)
        if not found:
            return make_page(raw.rows, raw.size, raw.number, "native", raw.origin, tables=raw.tables)
        if keep:
            overlay[raw.number] = (raw.size, found)  # type: ignore[index]
        return make_page(
            reading_rows(raw.blocks + found), raw.size, raw.number, "hybrid", raw.origin, tables=raw.tables
        )

    def _region(
        self, region: ImageRegion, seen: dict[object, list[TextBlock]], deadline: float | None = None
    ) -> list[TextBlock]:
        """OCR of an image of the page, with the boxes on the page. The same image on another page (the
        letterhead logo) is read once."""
        if region.key is not None and region.key in seen:
            blocks = seen[region.key]
        else:
            blocks = self.pipeline.run(region.image, escalate=False, deadline=deadline).original_blocks()
            if region.key is not None:
                seen[region.key] = blocks
        ox, oy = region.origin
        return [
            replace(
                b,
                box=(b.box[0] + ox, b.box[1] + oy, b.box[2] + ox, b.box[3] + oy),
                quad=tuple((x + ox, y + oy) for x, y in b.quad) if b.quad else None,
                candidates=list(b.candidates),
            )
            for b in blocks
        ]

    def load_models(self, engines: Iterable[str] | None = None) -> list[str]:
        """Loads (and downloads the first time) the models of the installed engines for the configured
        languages (without languages, those of the scripts that can be detected: the ones of
        detect_languages or all) and the layout models: to start workers ready or to prepare a machine or
        image without network."""
        wanted = set(engines) if engines else None
        loaded: list[str] = []
        for cls in registered_engines():
            if (wanted and cls.info.name not in wanted) or cls.unavailable_reason() is not None:
                continue
            models: set[object] = set()
            scripts = self._scripts_for(cls)
            if cls.info.detector and not set(scripts) & cls.info.scripts:
                scripts = ["latin"]  # it reads none of them, but it finds their lines with the model it comes with
            for script in scripts:
                if script not in cls.info.scripts or cls.recognizer_for(script) in models:
                    continue
                models.add(cls.recognizer_for(script))
                spec = EngineSpec(cls.info.name, script)
                engine = self.manager.get(spec)
                engine.prepare()
                if engine.loaded:
                    loaded.append(str(spec))
        if self.config.layout and (not wanted or "layout" in wanted) and LayoutAnalyzer.unavailable_reason() is None:
            self.pipeline.layout.load()
            loaded.append("layout")
        return loaded

    def _scripts_for(self, cls: type[OCREngine]) -> list[str]:
        """Scripts whose models are prepared: those of the given languages; without languages, the script the
        pages start with. For the page engine that detects (auto_script), also the others it detects among
        (those of languages with the other Cyrillic model, those of detect_languages, or all). The first is
        the one that reads the pages: the model it shares with others (the multi one: Latin, Chinese and
        Japanese) is loaded with it."""
        cfg = self.config
        detects = cfg.auto_script and cls.info.level == "page" and cls.read_lines is not OCREngine.read_lines
        if cfg.languages:
            scripts = [SCRIPT_OF[lang] for lang in cfg.languages]
            return list(dict.fromkeys([*scripts, *sorted(cfg.detect_scripts or ())] if detects else scripts))
        start = detect_start(cfg.detect_languages)
        if not detects:
            return [start]
        return [start, *sorted((cfg.detect_scripts or cls.info.scripts) - {start})]


def _without_traceback(error: Exception) -> Exception:
    """The exception without its traceback or those of its causes: stored in the Document of a batch, it
    would keep alive what was in memory when it failed (pages, images) as long as the document exists."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current.__traceback__ = None
        current = current.__cause__ or current.__context__
    return error


_SHARED: dict[tuple, OCR] = {}
_SHARED_LOCK = threading.Lock()
_SHARED_MAX = 4


def _shared(options: dict[str, object]) -> OCR:
    """The OCR (and its models) the shortcuts reuse for that configuration ("es" and ["es"] are the same)."""
    key = astuple(OCRConfig(**options))  # type: ignore[arg-type]
    with _SHARED_LOCK:
        ocr = _SHARED.get(key)
        if ocr is None:
            if len(_SHARED) >= _SHARED_MAX:
                _SHARED.pop(next(iter(_SHARED)))
            ocr = _SHARED[key] = OCR(**options)  # type: ignore[arg-type]
    return ocr


def read(source: object, **options: object) -> Document:
    """Shortcut: omnilector.read("invoice.pdf", languages=["es"]). Reuses the OCR (and its models)
    between calls with the same configuration."""
    return _shared(options).read(source)


def read_many(sources: Iterable[object], split_archives: bool = False, **options: object) -> Iterator[Document]:
    """Shortcut: omnilector.read_many(["a.pdf", "b.jpg"], languages=["es"]) (see OCR.read_many)."""
    return _shared(options).read_many(sources, split_archives=split_archives)
