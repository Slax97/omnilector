# SPDX-License-Identifier: AGPL-3.0-or-later
"""Box geometry (x0, y0, x1, y1): areas, IoU, angles, rows, reading order, rescaling."""

from __future__ import annotations

import math
from itertools import pairwise
from statistics import median

from .text import right_to_left

Box = tuple[float, float, float, float]

ELONGATED = 2.0  # long side / short side above which the direction of a line is known
SLANTED_DEG = 20  # text more than this off the horizontal (vertical margin, skewed stamp): its own row
VERTICAL_BOX = 2.0  # box without corners taller than this times its width, with several letters: vertical text
COLUMN_GAP = 1.0  # minimum gap between two text columns, in line heights
COLUMN_CLEARANCE = 0.25  # what no line may invade on each side of the cut between columns
COLUMN_ROWS = 3  # rows with text on both sides of the gap to accept the columns
PROSE_CHARS = 20  # columns of running text; amounts, codes or short cells stay in rows


def box_area(b: Box) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection_area(a: Box, b: Box) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def box_iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    if not inter:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union else 0.0


def coverage(box: Box, others: list[Box]) -> float:
    """Fraction of `box` covered by `others` (sum of intersections, at most 1; 0 if it has no area)."""
    area = box_area(box)
    if not area:
        return 0.0
    return min(1.0, sum(intersection_area(box, o) for o in others) / area)


def scale_box(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    """Converts coordinates of image*scale to the original: x/scale."""
    x0, y0, x1, y1 = box
    return (int(x0 / scale), int(y0 / scale), int(x1 / scale), int(y1 / scale))


def line_angle(quad: tuple[tuple[float, float], ...]) -> tuple[float, float] | None:
    """(angle in degrees between -90 and 90, length) of the long side of a 4-corner line; None if it is not
    clearly elongated (with one or two letters the direction of the text is unknown)."""
    (x0, y0), (x1, y1), (x2, y2) = quad[0], quad[1], quad[2]
    first, second = (x1 - x0, y1 - y0), (x2 - x1, y2 - y1)
    a, b = math.hypot(*first), math.hypot(*second)
    (dx, dy), long, short = (first, a, b) if a >= b else (second, b, a)
    if long == 0 or long < ELONGATED * short:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return angle, long


def slanted(block) -> bool:
    """Text that is not horizontal: the vertical CSV of the margin, a watermark, a skewed stamp. Its box
    takes the height of many lines, so it cannot share a row with them."""
    if block.quad is not None:
        found = line_angle(block.quad)
        return found is not None and abs(found[0]) > SLANTED_DEG
    w, h = block.box[2] - block.box[0], block.box[3] - block.box[1]
    return h > VERTICAL_BOX * w and len(block.text.strip()) >= 3


def same_row(a, b) -> bool:
    """Same visual row: they overlap vertically by at least half the height of the smaller one. It does not
    depend on the scale or the font size (a large amount next to its small label)."""
    overlap = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
    lower = min(a.box[3] - a.box[1], b.box[3] - b.box[1])
    return lower > 0 and overlap >= 0.5 * lower


def _text(blocks: list) -> str:
    return " ".join(b.text for b in blocks)


def in_row_order(row: list) -> list:
    """The pieces of a row in reading order: left to right or, if the row is in a right-to-left script
    (Arabic, Persian, Urdu...), right to left."""
    if right_to_left(_text(row)):
        return sorted(row, key=lambda b: -b.box[2])
    return sorted(row, key=lambda b: b.box[0])


def group_rows(blocks: list) -> list[list]:
    """Rows top to bottom and, within each row, in its reading direction (in_row_order)."""
    rows: list[list] = []
    for b in sorted(blocks, key=lambda b: (b.box[1], b.box[0])):
        row = next((r for r in rows if same_row(b, r[0])), None)
        if row is None:
            rows.append([b])
        else:
            row.append(b)
    rows.sort(key=lambda r: min(x.box[1] for x in r))
    return [in_row_order(r) for r in rows]


def reading_rows(blocks: list) -> list[list]:
    """Rows in reading order, the same for the OCR and for native text: top to bottom and left to right
    ("TOTAL" and "48,22" in the same row; in Arabic, right to left), but column by column when the page
    has columns of running text. Rotated text goes at the end, each line in its own row. Zones nobody could
    read (no text) do not count: a large one pulled into its row everything it crosses."""
    blocks = [b for b in blocks if b.text.strip()]
    flat = [b for b in blocks if not slanted(b)]
    rotated = sorted((b for b in blocks if slanted(b)), key=lambda b: (b.box[0], b.box[1]))
    heights = [b.box[3] - b.box[1] for b in flat if b.box[3] > b.box[1]]
    rows = _columns(group_rows(flat), median(heights)) if heights else group_rows(flat)
    return rows + [[b] for b in rotated]


def reading_order(blocks: list) -> list:
    """Blocks in reading order (the rows of reading_rows, one after another)."""
    return [b for row in reading_rows(blocks) for b in row]


def _crosses(row: list, cut: float, clearance: float) -> bool:
    return any(b.box[0] < cut + clearance and b.box[2] > cut - clearance for b in row)


def _column_run(rows: list[list], cut: float, at: int, height: float) -> tuple[int, int, int] | None:
    """Consecutive rows around `at` that do not cross the cut -> (start, end, rows with text on both
    sides) if they form two columns of running text; None otherwise."""
    clearance = COLUMN_CLEARANCE * height

    def joins(
        row: list,
    ) -> bool:  # the stretch goes on: it does not cross the cut and has running text (a table below does not)
        return not _crosses(row, cut, clearance) and any(len(b.text.strip()) >= PROSE_CHARS // 2 for b in row)

    start = at
    while start > 0 and joins(rows[start - 1]):
        start -= 1
    end = at + 1
    while end < len(rows) and joins(rows[end]):
        end += 1
    left = [len(b.text.strip()) for r in rows[start:end] for b in r if b.box[2] <= cut]
    right = [len(b.text.strip()) for r in rows[start:end] for b in r if b.box[0] >= cut]
    both = sum(1 for r in rows[start:end] if any(b.box[2] <= cut for b in r) and any(b.box[0] >= cut for b in r))
    if both < COLUMN_ROWS or median(left) < PROSE_CHARS or median(right) < PROSE_CHARS:
        return None
    return start, end, both


def _columns(rows: list[list], height: float) -> list[list]:
    """Reorders column by column the stretches of the page with columns of running text. A gap in a row
    between two long lines proposes a cut; it holds if it stays in the surrounding rows and there is running
    text on both sides (tables of amounts and tickets do not have it). In Arabic, the right column goes
    first."""
    best: tuple[int, int, int, float] | None = None
    for i, row in enumerate(rows):
        for a, b in pairwise(sorted(row, key=lambda x: x.box[0])):
            if b.box[0] - a.box[2] < COLUMN_GAP * height:
                continue
            if min(len(a.text.strip()), len(b.text.strip())) < PROSE_CHARS:
                continue
            cut = (a.box[2] + b.box[0]) / 2
            run = _column_run(rows, cut, i, height)
            if run is not None and (best is None or run[2] > best[2]):
                best = (*run, cut)
    if best is None:
        return rows
    start, end, _, cut = best
    band = [b for r in rows[start:end] for b in r]
    left = [b for b in band if b.box[2] <= cut]
    right = [b for b in band if b.box[2] > cut]
    first, second = (right, left) if right_to_left(_text(band)) else (left, right)
    return (
        _columns(rows[:start], height)
        + _columns(group_rows(first), height)
        + _columns(group_rows(second), height)
        + _columns(rows[end:], height)
    )
