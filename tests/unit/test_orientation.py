# SPDX-License-Identifier: AGPL-3.0-or-later
"""Skewed, sideways or upside-down pages: straightened with the lines of pass 1 (fake engines)."""

from __future__ import annotations

import numpy as np
import pytest

from omnilector import OCR
from omnilector.core.orientation import background, page_angle, rotate, transform_block
from omnilector.core.pipeline import PageRead
from omnilector.result import RecognitionCandidate

from .conftest import make_engine, tb, tilted, white

TICKET = [
    "LUIS PETROL STATION SL CIF B12345678",
    "12 Liberty Avenue, Murcia",
    "Date: 11/09/2026 Time: 18:42",
    "Diesel A 32,50 L x 1,405",
    "Total amount: 45,67 EUR",
    "VAT 21% included: 7,93 EUR",
    "Plate: 2124 CZT",
    "Thank you for your visit",
]


def ticket(angle: float, size: tuple[int, int] = (1000, 800)) -> list:
    """The lines of the ticket (600 x 30 px, every 45 px) in a photo rotated `angle` degrees."""
    center = (size[0] / 2, size[1] / 2)
    return [tilted(text, 200, 200 + 45 * i, 600, 30, angle, center) for i, text in enumerate(TICKET)]


def test_page_angle_from_line_corners():
    assert page_angle(ticket(15)) == pytest.approx(-15, abs=0.5)  # text going up to the right
    assert page_angle(ticket(-4)) == pytest.approx(4, abs=0.5)
    assert abs(page_angle(ticket(90))) == pytest.approx(90, abs=0.5)  # sideways: 90 and -90 are the same
    assert page_angle([tb("no corners", (0, 0, 100, 20), 0.9)]) is None  # engine without quadrilaterals
    scattered = [tilted("loose text", 100, 100 * i, 300, 30, a, (500, 500)) for i, a in enumerate((0, 35, 70, -40))]
    assert page_angle(scattered) is None  # no direction dominates (text of a street photo)


def test_rotate_keeps_the_whole_image_and_moves_boxes_with_it():
    image = white(400, 200)
    image[20:40, 300:380] = 0  # a black "line" at the top right
    for angle in (90, 180, -90, 30):
        rotated, m = rotate(image, angle)
        x0, y0, x1, y1 = transform_block(tb("x", (300, 20, 380, 40), 1.0), m).box
        inside = rotated[max(0, y0) : y1, max(0, x0) : x1]
        assert inside.min() == 0 and (rotated < 128).sum() <= (inside < 128).sum() + 40  # the ink falls in the box
    assert rotate(image, 90)[0].shape == (400, 200, 3)  # sideways: height and width swap


@pytest.mark.parametrize("angle", [8, 15, -20])
def test_tilted_photo_keeps_every_line_in_its_own_row(use_engines, angle):
    use_engines(make_engine("rapidocr", lambda image: ticket(angle)))
    page = OCR(strategy="fast", handwritten=False).read(white(1000, 800)).pages[0]
    assert page.text.splitlines() == TICKET  # before: rows merged at 8 degrees and lines lost at 15
    assert page.rotation == pytest.approx(-angle, abs=0.5)
    assert all(line.box[3] - line.box[1] < 0.06 for line in page.lines)  # tight boxes, not the diagonal


def photo(angle: float) -> np.ndarray:
    """Blank photo with a black mark above the ticket (rotated with it): with it the fake engine knows
    whether the page is upside down, like RapidOCR's orientation classifier."""
    image = white(1000, 800)
    x0, y0, x1, y1 = tilted("marca", 200, 140, 40, 40, angle, (500, 400)).box
    image[y0:y1, x0:x1] = 0
    return image


def mark_below_text(image: np.ndarray, boxes: list) -> bool:
    ys, _ = np.nonzero(image[:, :, 0] < 128)
    return bool(ys.mean() > min(b[1] for b in boxes))


@pytest.mark.parametrize("angle", [90, 180, -90, 170])
def test_sideways_and_upside_down_pages_are_read_in_order(use_engines, angle):
    use_engines(make_engine("rapidocr", lambda image: ticket(angle), flipped=mark_below_text))
    page = OCR(strategy="fast", handwritten=False).read(photo(angle)).pages[0]
    assert page.text.splitlines() == TICKET  # before: sideways, everything in one line; upside down, reversed
    assert page.rotation == pytest.approx(-angle if angle != 180 else 180, abs=0.5)


def test_straight_page_is_left_alone(use_engines):
    engine = make_engine("rapidocr", lambda image: ticket(0.2))
    use_engines(engine)
    doc = OCR(strategy="fast", handwritten=False).read(white(1000, 800))
    assert doc.pages[0].rotation == 0.0 and doc.text.splitlines() == TICKET
    assert doc.lines[0].box == pytest.approx((0.2, 0.25, 0.8, 0.29), abs=0.01)  # boxes of the received image


def test_boxes_go_back_to_the_image_as_received():
    image = white(1000, 800)
    rotated, m = rotate(image, -15)
    straight = [transform_block(b, m) for b in ticket(15)]
    back = PageRead([straight], (rotated.shape[1], rotated.shape[0]), -15, m).original_blocks()
    for before, after in zip(ticket(15), back):
        assert np.allclose(before.box, after.box, atol=3)


def test_rapidocr_flips_a_line_only_if_it_reads_better_upside_down():
    from types import SimpleNamespace

    from omnilector.engines.rapidocr import RapidOCREngine

    pytest.importorskip("rapidocr")
    crops = [np.full((40, 200, 3), v, np.uint8) for v in (0, 1, 2)]
    # the classifier sees all three as inverted; rotated, only the second reads better (an upside-down photo);
    # the first is the upright line it saw inverted (Devanagari, the letter on Linux)
    rotated = {0: ("weiver rof", 0.4), 1: ("Total 45,67", 0.97), 2: ("", 0.0)}
    engine = SimpleNamespace(
        _load_cls_model=lambda: lambda imgs: SimpleNamespace(cls_res=[("180", 0.99)] * len(imgs)),
        _load_rec_model=lambda: (
            lambda inp: SimpleNamespace(
                txts=[rotated[int(c[0, 0, 0])][0] for c in inp.img],
                scores=[rotated[int(c[0, 0, 0])][1] for c in inp.img],
            )
        ),
    )
    txts, scores = ["appeal for review", "L9'St lPtoT", "Signed"], [0.99, 0.5, 0.9]
    RapidOCREngine._flip_verified(engine, crops, txts, scores)
    assert txts == ["appeal for review", "Total 45,67", "Signed"] and scores[1] == 0.97


def test_rotation_fills_the_corners_with_the_color_of_the_page_if_asked():
    dark = np.full((200, 400, 3), (120, 110, 100), np.uint8)
    assert tuple(rotate(dark, 30)[0][0, 0]) == (255, 255, 255)  # what RapidOCR has already read: as always
    # to read it again with Tesseract: white corners on a photo made it lose lines
    assert tuple(rotate(dark, 30, background(dark))[0][0, 0]) == (120, 110, 100)


def mark_at_the_bottom(image: np.ndarray) -> bool:
    ys, _ = np.nonzero(image[:, :, 0] < 128)
    return bool(ys.mean() > image.shape[0] / 2)


def without_corners(upright: float, turned: float, calls: list) -> list:
    """Engines for a script only a page engine without line corners reads (Tesseract), and a detector that finds
    the lines of the ticket straight: the reading has `upright` confidence with the mark of photo() at the top of the
    page and `turned` with it at the bottom."""

    def page(image):
        calls.append(image.shape[:2])
        conf = turned if mark_at_the_bottom(image) else upright
        return [tb(text, (200, 200 + 45 * i, 800, 230 + 45 * i), conf, "tesseract") for i, text in enumerate(TICKET)]

    return [
        make_engine("tesseract", page, scripts=frozenset({"gurmukhi"})),
        make_engine("rapidocr", scripts=frozenset({"latin"}), detect_lines=lambda image: ticket(0)),
    ]


def test_engine_without_corners_reads_the_page_again_straightened(use_engines):
    def page(image):  # skewed, Tesseract read nothing
        skewed = image.shape[:2] == (800, 1000)
        return (
            []
            if skewed
            else [tb(t, (200, 200 + 45 * i, 800, 230 + 45 * i), 0.95, "tesseract") for i, t in enumerate(TICKET)]
        )

    tesseract = make_engine("tesseract", page, scripts=frozenset({"gurmukhi"}))
    use_engines(tesseract, make_engine("rapidocr", scripts=frozenset({"latin"}), detect_lines=lambda image: ticket(8)))
    read = OCR(languages=["pa"], strategy="fast").read(photo(8)).pages[0]
    assert read.text.splitlines() == TICKET and read.rotation == pytest.approx(-8, abs=0.5)
    assert len(tesseract.calls) == 2  # the skewed page and the straightened one


def test_engine_without_corners_keeps_the_surer_reading_if_the_page_is_barely_skewed(use_engines):
    def page(image):  # straightened, it reads worse (Tesseract with a photo in Burmese)
        sure = 0.9 if image.shape[:2] == (800, 1000) else 0.6
        return [tb(t, (200, 200 + 45 * i, 800, 230 + 45 * i), sure, "tesseract") for i, t in enumerate(TICKET)]

    tesseract = make_engine("tesseract", page, scripts=frozenset({"gurmukhi"}))
    use_engines(tesseract, make_engine("rapidocr", scripts=frozenset({"latin"}), detect_lines=lambda image: ticket(2)))
    read = OCR(languages=["pa"], strategy="fast").read(photo(2)).pages[0]
    assert read.rotation == 0.0 and len(tesseract.calls) == 2  # it was also read straightened


@pytest.mark.parametrize(
    ("upright", "turned", "rotation", "reads"),
    [
        (0.3, 0.95, 180, 2),  # upside down: read turned, it is surer
        (0.6, 0.3, 0, 2),  # an old print, doubtful: turned it is worse, and stays as it is
        (0.95, 0.3, 0, 1),  # a sure reading is not read turned
    ],
)
def test_engine_without_corners_reads_the_page_turned_if_its_reading_is_doubtful(
    use_engines, upright, turned, rotation, reads
):
    calls: list = []
    use_engines(*without_corners(upright, turned, calls))
    page = OCR(languages=["pa"], strategy="fast").read(photo(0)).pages[0]
    assert page.text.splitlines() == TICKET and page.rotation == rotation and len(calls) == reads


def test_lines_only_detected_are_read_upside_down_if_that_is_surer(use_engines):
    def lines(boxes):  # as kraken: upright (the lines low on the page) 0.99; upside down, 0.7
        seen.extend(boxes)
        return [RecognitionCandidate(f"שורה {i}", 0.99 if b[1] > 400 else 0.7, "kraken") for i, b in enumerate(boxes)]

    def detected(top):
        return lambda image: [tb("", (100, top + 50 * i, 900, top + 40 + 50 * i), 0.0) for i in range(4)]

    kraken = make_engine(
        "kraken",
        level="line",
        handwriting=True,
        scripts=frozenset({"hebrew"}),
        print_languages=frozenset({"he"}),
        read_lines=lines,
    )
    for top, rotation, reads in ((100, 180, 4 + 4 + 4), (500, 0, 4)):
        seen: list = []
        use_engines(make_engine("rapidocr", scripts=frozenset({"latin"}), detect_lines=detected(top)), kraken)
        page = OCR(languages=["he"]).read(photo(0)).pages[0]
        assert page.rotation == rotation and all(line.confidence == 0.99 for line in page.lines)
        assert len(seen) == reads  # upside down: every line, the sample turned and every line again
