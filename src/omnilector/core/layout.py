# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layout of a page read with OCR: zones in reading order (titles, paragraphs, tables, figures, headers,
stamps...) with PP-DocLayout, and the structure of tables (rows, columns, merged cells) with
SLANet-plus. Extra [layout]: rapid-layout and rapid-table (Apache-2.0, ONNX on CPU).

The reading order comes from the model, which knows about columns, boxes and captions; the lines of
each zone go by rows. Without the extra, or if the model finds no zones, the order is the usual one
(utils.geometry.reading_rows).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..exceptions import ModelNotAvailableError, OmnilectorError
from ..result import TextBlock
from ..utils.deps import missing_modules, require
from ..utils.geometry import group_rows, intersection_area, reading_rows

if TYPE_CHECKING:
    from ..config import OCRConfig

# PP-DocLayout classes -> zone kinds of the common output
KINDS = {
    "doc_title": "title",
    "title": "title",
    "paragraph_title": "heading",
    "text": "text",
    "content": "text",
    "abstract": "text",
    "reference": "text",
    "reference_content": "text",
    "aside_text": "text",
    "algorithm": "text",
    "vertical_text": "text",
    "list": "list",
    "table": "table",
    "figure_title": "caption",
    "table_title": "caption",
    "chart_title": "caption",
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "header_image": "figure",
    "footer_image": "figure",
    "seal": "stamp",
    "header": "header",
    "footer": "footer",
    "number": "page_number",
    "footnote": "footnote",
    "vision_footnote": "footnote",
    "formula": "formula",
    "display_formula": "formula",
    "inline_formula": "formula",
    "formula_number": "formula",
}
MIN_SCORE = 0.45  # less confident zones do not count
IN_ZONE = 0.5  # fraction of a line inside a zone to belong to it
TABLE_MARGIN = 8  # px around a table when cropping it for SLANet


@dataclass
class Zone:
    """Zone of the page (pixels of the image read) with its lines by rows."""

    kind: str
    box: tuple[int, int, int, int]
    rows: list[list[TextBlock]] = field(default_factory=list)


@dataclass
class TableCell:
    row: int
    col: int
    rowspan: int
    colspan: int
    box: tuple[int, int, int, int]
    text: str = ""


@dataclass
class TableGrid:
    """Table in pixels of the image read; `zone` is its index in the list of zones."""

    zone: int
    box: tuple[int, int, int, int]
    rows: int
    cols: int
    cells: list[TableCell]


def _bbox(points: object) -> tuple[int, int, int, int]:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return int(arr[:, 0].min()), int(arr[:, 1].min()), int(np.ceil(arr[:, 0].max())), int(np.ceil(arr[:, 1].max()))


def _inside(block: TextBlock, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = block.box
    area = max(1, (x1 - x0) * (y1 - y0))
    return intersection_area(block.box, box) / area


@contextmanager
def _quiet() -> Iterator[None]:
    """Without the INFO messages rapid-layout and rapid-table print when loading their models (their loggers
    are created at that moment, with INFO level and their own handler). Only while loading, once."""
    previous = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(previous)


def _model_file(processor: object, model_type: object, cache_dir: str | None, folder: str) -> str | None:
    """Path of the model in model_cache_dir, downloaded (with its SHA256) the first time; None = the folder
    of the package itself (it may not be writable: that is what model_cache_dir is for)."""
    if not cache_dir:
        return None
    info = processor.model_map[model_type.value]  # type: ignore[attr-defined]
    path = Path(cache_dir) / folder / Path(info["model_dir_or_path"]).name
    if not path.exists():
        download = require(f"{processor.__module__.split('.')[0]}.utils.download_file", extra="layout")
        path.parent.mkdir(parents=True, exist_ok=True)
        params = download.DownloadFileInput(
            file_url=info["model_dir_or_path"], sha256=info["SHA256"], save_path=path, logger=processor.logger
        )  # type: ignore[attr-defined]
        download.DownloadFile.run(params)
    return str(path)


class LayoutAnalyzer:
    """Zones and tables of a page. Loads the models once (on the first page or with load()); if they
    cannot be loaded, it is not retried on every page."""

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._models: tuple[object, object] | None = None
        self._error: OmnilectorError | None = None

    @staticmethod
    def unavailable_reason() -> str | None:
        return missing_modules("rapid_layout", "rapid_table", extra="layout")

    def load(self) -> tuple[object, object]:
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._models is None:
                try:
                    self._models = self._build()
                except OmnilectorError as e:
                    self._error = e
                    raise
                except PermissionError as e:  # rapid-table creates a folder inside its package when imported
                    self._error = ModelNotAvailableError(
                        f"layout: no permission for {e.filename}: import rapid_table once as the user that installed it"
                    )
                    raise self._error from e
                except Exception as e:
                    self._error = ModelNotAvailableError(f"layout: could not load the model (offline? cache?): {e}")
                    raise self._error from e
            return self._models

    def _build(self) -> tuple[object, object]:
        layout_mod = require("rapid_layout", extra="layout")
        table_mod = require("rapid_table", extra="layout")
        from rapid_layout.model_handler.utils import ModelProcessor as LayoutModels
        from rapid_table.model_processor.main import ModelProcessor as TableModels

        with _quiet():
            return self._create(layout_mod, table_mod, LayoutModels, TableModels)

    def _create(self, layout_mod, table_mod, LayoutModels, TableModels) -> tuple[object, object]:
        cache = self.config.model_cache_dir
        threads = self.config.threads
        engine_cfg = {"intra_op_num_threads": threads, "inter_op_num_threads": 1} if threads else {}
        layout_type, table_type = layout_mod.ModelType.PP_DOC_LAYOUTV3, table_mod.ModelType.SLANETPLUS
        layout = layout_mod.RapidLayout(
            layout_mod.RapidLayoutInput(
                model_type=layout_type,
                conf_thresh=MIN_SCORE,
                engine_cfg=dict(engine_cfg),
                model_dir_or_path=_model_file(LayoutModels, layout_type, cache, "rapid_layout"),
            )
        )
        table = table_mod.RapidTable(
            table_mod.RapidTableInput(
                model_type=table_type,
                use_ocr=False,
                engine_cfg=dict(engine_cfg),
                model_dir_or_path=_model_file(TableModels, table_type, cache, "rapid_table"),
            )
        )
        return layout, table

    def zones(self, image: np.ndarray) -> list[Zone]:
        """Zones of the page in reading order (the one the model gives)."""
        layout, _ = self.load()
        with self._lock:
            out = layout(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # type: ignore[operator]
        names = list(out.class_names) if out.class_names is not None else []
        boxes = [] if out.boxes is None else list(out.boxes)
        return [Zone(KINDS.get(str(n), "text"), _bbox(b)) for n, b in zip(names, boxes)]

    def table(self, image: np.ndarray, zone: Zone, index: int) -> TableGrid | None:
        """Rows, columns and cells of the table of a zone; the text of each cell, from its lines."""
        _, model = self.load()
        x0, y0, x1, y1 = zone.box
        h, w = image.shape[:2]
        x0, y0, x1, y1 = (
            max(0, x0 - TABLE_MARGIN),
            max(0, y0 - TABLE_MARGIN),
            min(w, x1 + TABLE_MARGIN),
            min(h, y1 + TABLE_MARGIN),
        )
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        crop = cv2.cvtColor(np.ascontiguousarray(image[y0:y1, x0:x1]), cv2.COLOR_RGB2BGR)
        with self._lock:
            try:  # the RapidTable steps without its progress bar (it printed it on every table)
                structures, cell_boxes = model.table_structure([crop])  # type: ignore[attr-defined]
                logic_points = model.table_matcher.decode_logic_points(structures)  # type: ignore[attr-defined]
            except AttributeError:  # another rapid-table version: the public call
                out = model(crop)  # type: ignore[operator]
                cell_boxes, logic_points = out.cell_bboxes, out.logic_points
        if not len(cell_boxes) or not len(cell_boxes[0]):
            return None
        cells = []
        for points, logic in zip(cell_boxes[0], logic_points[0]):
            r0, r1, c0, c1 = (int(v) for v in logic)
            bx0, by0, bx1, by1 = _bbox(points)
            cells.append(TableCell(r0, c0, r1 - r0 + 1, c1 - c0 + 1, (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)))
        lines = [b for row in zone.rows for b in row]
        for line in lines:  # each line, to the cell where most of it falls (or to the nearest one)
            best = max(cells, key=lambda c: _inside(line, c.box))
            if _inside(line, best.box) == 0:
                cx, cy = (line.box[0] + line.box[2]) / 2, (line.box[1] + line.box[3]) / 2
                best = min(
                    cells, key=lambda c: abs((c.box[0] + c.box[2]) / 2 - cx) + abs((c.box[1] + c.box[3]) / 2 - cy)
                )
            best.text = f"{best.text} {line.text}".strip()
        rows = max(c.row + c.rowspan for c in cells)
        cols = max(c.col + c.colspan for c in cells)
        return TableGrid(index, zone.box, rows, cols, cells)

    def analyze(self, image: np.ndarray, blocks: list[TextBlock]) -> tuple[list[Zone], list[TableGrid]] | None:
        """Zones with their lines in reading order, and the tables; None if the model sees no zones (and the
        usual order is used). Lines that fall in no zone are placed by height."""
        zones = self.zones(image)
        if not zones:
            return None
        members: list[list[TextBlock]] = [[] for _ in zones]
        loose: list[TextBlock] = []
        for b in (b for b in blocks if b.text.strip()):
            scores = [_inside(b, z.box) for z in zones]
            best = int(np.argmax(scores))
            (members[best] if scores[best] >= IN_ZONE else loose).append(b)
        for zone, found in zip(zones, members):
            zone.rows = group_rows(found)
        for row in reading_rows(loose):  # lines outside any zone: before the first zone that starts below them
            top = min(b.box[1] for b in row)
            at = next((i for i, z in enumerate(zones) if z.box[1] > top), len(zones))
            zones.insert(
                at,
                Zone(
                    "text",
                    (min(b.box[0] for b in row), top, max(b.box[2] for b in row), max(b.box[3] for b in row)),
                    [row],
                ),
            )
        zones = [z for z in zones if z.rows or z.kind in ("table", "figure", "stamp")]
        tables = [t for i, z in enumerate(zones) if z.kind == "table" and (t := self.table(image, z, i)) is not None]
        return zones, tables
