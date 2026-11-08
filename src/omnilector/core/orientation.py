# SPDX-License-Identifier: AGPL-3.0-or-later
"""Page orientation: skewed photos, tilted scans, sideways or upside down.

The angle comes from the lines of pass 1 (page engines give their 4 corners), so straightening costs no
extra read: the image is rotated and the boxes already read are moved. With the page upright, fusion,
rows, reading order and line crops (handwriting) work as on a straight scan. Skewed boxes overlap
between neighbouring lines: at 8 degrees they merged rows and at 15 lines were lost.
"""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np

from ..result import TextBlock
from ..utils.geometry import line_angle

MIN_ANGLE = 0.5  # below this the page is taken as straight: it is not rotated and the boxes do not change
PEAK_DEG = 5  # lines closer than this to the dominant angle count to measure it precisely
DOMINANT = 0.5  # fraction of the text (by length) that has to go in that direction
FLIP_SAMPLE = 7  # lines looked at to know whether the page is upside down
VERTICAL_SCRIPTS = frozenset({"chinese", "japanese"})  # written vertically: not a rotated page


def _distance(a: float, b: float) -> float:
    """Difference between two line directions (180 degrees = the same direction)."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def page_angle(blocks: list[TextBlock]) -> float | None:
    """Direction of the page's text in degrees (between -90 and 90; 0 = horizontal), measured with the corners
    of its lines. None if the engine does not give them or no direction dominates (loose text in every
    direction, as in a photo of a street)."""
    lines = [found for b in blocks if b.quad and (found := line_angle(b.quad))]
    if not lines:
        return None
    total = sum(length for _, length in lines)
    peak = max((a for a, _ in lines), key=lambda c: sum(n for a, n in lines if _distance(a, c) <= PEAK_DEG))
    near = [(a, n) for a, n in lines if _distance(a, peak) <= PEAK_DEG]
    if sum(n for _, n in near) < DOMINANT * total:
        return None
    # mean of directions (with doubled angles: 89 and -89 degrees are almost the same direction)
    sin = sum(n * math.sin(math.radians(2 * a)) for a, n in near)
    cos = sum(n * math.cos(math.radians(2 * a)) for a, n in near)
    return math.degrees(math.atan2(sin, cos)) / 2


def normalize(angle: float) -> float:
    """Equivalent angle between -180 (excluded) and 180."""
    angle = math.fmod(angle, 360)
    if angle <= -180:
        angle += 360
    elif angle > 180:
        angle -= 360
    return angle


def rotation_matrix(size: tuple[int, int], angle: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Matrix (2x3) that rotates an image of `size` (w, h) by `angle` degrees counterclockwise without
    cropping anything, and the size of the rotated image. Exact for 90-degree rotations."""
    w, h = size
    quarter = angle / 90
    if abs(quarter - round(quarter)) < 1e-9:
        turns = round(quarter) % 4
        matrices = {
            0: ([[1, 0, 0], [0, 1, 0]], (w, h)),
            1: ([[0, 1, 0], [-1, 0, w]], (h, w)),
            2: ([[-1, 0, w], [0, -1, h]], (w, h)),
            3: ([[0, -1, h], [1, 0, 0]], (h, w)),
        }
        m, new = matrices[turns]
        return np.array(m, dtype=np.float64), new
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = math.ceil(h * sin + w * cos), math.ceil(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return m, (nw, nh)


def background(image: np.ndarray) -> tuple[int, ...]:
    """Color of the edges of the image (median), to fill the corners a rotation uncovers before reading the page
    again: white corners on a darker photo made Tesseract lose whole lines (Gurmukhi, from 6 % of the characters wrong
    to 80 %). Not where RapidOCR has already read it: on two signs it read worse afterwards (Athens, 27 % -> 36 %)."""
    edges = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    return tuple(int(v) for v in np.median(edges.reshape(len(edges), -1), axis=0))


def rotate(image: np.ndarray, angle: float, fill: tuple[int, ...] = (255, 255, 255)) -> tuple[np.ndarray, np.ndarray]:
    """Image rotated by `angle` degrees (counterclockwise), whole and on a white background (or `fill`) + matrix
    original -> rotated."""
    h, w = image.shape[:2]
    m, (nw, nh) = rotation_matrix((w, h), angle)
    turns = {
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        -180: cv2.ROTATE_180,
        -90: cv2.ROTATE_90_CLOCKWISE,
    }
    exact = turns.get(round(angle)) if abs(angle - round(angle)) < 1e-9 else None
    if exact is not None:
        return cv2.rotate(image, exact), m
    rotated = cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=fill)
    return rotated, m


def transform_block(block: TextBlock, m: np.ndarray) -> TextBlock:
    """Block with its corners (or those of its box) carried by the matrix `m`; the new box covers them. A
    skewed line, once straightened, gets a tight box again."""
    x0, y0, x1, y1 = block.box
    points = np.array(block.quad or ((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=np.float64)
    moved = points @ m[:, :2].T + m[:, 2]
    xs, ys = moved[:, 0], moved[:, 1]
    box = (math.floor(xs.min()), math.floor(ys.min()), math.ceil(xs.max()), math.ceil(ys.max()))
    quad = tuple((round(float(x)), round(float(y))) for x, y in moved)
    return replace(block, box=box, quad=quad)


def straighten(
    image: np.ndarray, blocks: list[TextBlock], angle: float, fill: tuple[int, ...] = (255, 255, 255)
) -> tuple[np.ndarray, list[TextBlock], np.ndarray | None]:
    """(rotated image on `fill`, moved blocks, matrix) to read the page upright; unchanged (and matrix None) if the
    rotation is negligible."""
    angle = normalize(angle)
    if abs(angle) < MIN_ANGLE:
        return image, blocks, None
    rotated, m = rotate(image, angle, fill)
    return rotated, [transform_block(b, m) for b in blocks], m


def flip_sample(blocks: list[TextBlock]) -> list[tuple[int, int, int, int]]:
    """Boxes of the longest, clearly horizontal lines: with them the engine says whether the page is upside
    down."""
    lines = [b for b in blocks if len(b.text.strip()) >= 3 and b.box[2] - b.box[0] >= 2 * (b.box[3] - b.box[1]) > 0]
    lines.sort(key=lambda b: b.box[2] - b.box[0], reverse=True)
    return [b.box for b in lines[:FLIP_SAMPLE]]
