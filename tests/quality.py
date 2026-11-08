# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measurable quality: synthetic cases with their correct text and error measures (CER, WER, lines).

The cases are generated on the fly, always the same (no files in git): skewed, sideways or upside-down
tickets, a thermal printer ticket, an A4 letter from a scanner, noise, small type, shadow, a transparent
signature, columns, a PDF with a stamp, a docx form and an email with a screenshot. Used by
benchmark.py and tests/integration/test_quality.py. A TrueType font with accents is needed (Arial,
DejaVu); the thermal ticket uses a monospaced one (Lucida Console, DejaVu Sans Mono) if available. The
documents are in Spanish on purpose: accents and ñ are part of what is measured.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = (
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)

TICKET = [
    "GASOLINERA LUIS SL",
    "CIF B12345678",
    "Avda. de la Libertad 12, Murcia",
    "Fecha: 11/09/2026 Hora: 18:42",
    "Gasóleo A 32,50 L x 1,405",
    "Importe total: 45,67 EUR",
    "IVA 21% incluido: 7,93 EUR",
    "Matrícula: 2124 CZT",
    "Gracias por su visita",
]
# Supermarket ticket from a thermal printer (42 columns): the columns are aligned with spaces
THERMAL_ROWS = [
    "     SUPERMERCADOS VALLE VERDE, S.A.",
    "       Avda. de la Constitución, 45",
    "              41001 Sevilla",
    "              NIF A41234567",
    "14/10/2026 19:05                   OP: 117",
    "FACTURA SIMPLIFICADA: 2317-014-552093",
    "------------------------------------------",
    " 1 LECHE SEMIDESNATADA 6X1L          5,34",
    " 1 PAN DE MOLDE INTEGRAL             1,45",
    " 2 YOGUR NATURAL PACK 4      1,20    2,40",
    " 1 PLÁTANO DE CANARIAS",
    "      1,215 kg      2,49 €/kg        3,03",
    " 1 JAMÓN COCIDO EXTRA LONCHAS        2,10",
    "------------------------------------------",
    "TOTAL (€)                           14,32",
    "TARJETA BANCARIA                    14,32",
    "          GRACIAS POR SU COMPRA",
]
THERMAL = [" ".join(row.split()) for row in THERMAL_ROWS if row.strip("-")]
MONO_FONTS = (
    "C:/Windows/Fonts/lucon.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
)
LETTER = [
    "El solicitante declara que los datos aportados en esta solicitud son ciertos",
    "y se compromete a comunicar cualquier cambio que se produzca en los mismos.",
    "La Administración podrá comprobar la veracidad de la información facilitada",
    "y requerir la documentación que considere necesaria para su tramitación.",
    "Contra esta resolución, que pone fin a la vía administrativa, podrá interponerse",
    "recurso potestativo de reposición en el plazo de un mes desde su notificación.",
    "En Murcia, a 11 de septiembre de 2026. El jefe de servicio, Luis Pérez Muñoz.",
]
LEFT = [
    "La primera columna empieza aquí y sigue",
    "con un texto corrido que se lee de arriba",
    "abajo antes de pasar a la otra columna,",
    "como en un boletín o en un artículo.",
]
RUSSIAN = [
    "Счёт № 7 от 11 сентября 2026 года",
    "Покупатель: Иван Петрович Смирнов",
    "Итого к оплате: 45,67 рублей",
]
GREEK = ["ΟΔΟΣ ΦΥΛΑΣΙΩΝ", "Αθήνα 2026"]
TITLE = "Boletín informativo de la comarca"
TABLE = [["Concepto", "Unidades", "Importe"], ["Gasóleo A", "32", "45,67"], ["Lavado", "1", "8,00"]]
RIGHT = [
    "La segunda columna continúa el texto",
    "y solo se lee cuando ya ha terminado",
    "la primera, aunque sus líneas queden",
    "a la misma altura que las de la otra.",
]


def font_path() -> str | None:
    return next((p for p in FONTS if Path(p).exists()), None)


def widest(font: ImageFont.FreeTypeFont, lines: list[str]) -> int:
    """Width of the longest line in that font: DejaVu (Linux) is wider than Arial (Windows) and, with fixed
    positions, the columns overlapped and the lines went out of the image."""
    return max(int(font.getlength(line)) + 1 for line in lines)


def render(lines: list[str], size: int = 40, width: int = 1000) -> Image.Image:
    """Lines in a real TrueType font (PIL's default one does not draw accents or ñ)."""
    path = font_path()
    if path is None:
        raise RuntimeError("no TrueType font with accents (Arial or DejaVu)")
    font = ImageFont.truetype(path, size)
    width = max(width, 80 + widest(font, lines))
    img = Image.new("RGB", (width, 80 + len(lines) * (size + 30)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * (size + 30)), line, fill="black", font=font)
    return img


def _rotated(lines: list[str], angle: float, **kw: int) -> np.ndarray:
    return np.array(render(lines, **kw).rotate(angle, expand=True, fillcolor="white", resample=Image.BICUBIC))


def _noisy() -> np.ndarray:
    img = render(TICKET).filter(ImageFilter.GaussianBlur(1.2))
    arr = np.asarray(img).astype(np.float32) + np.random.default_rng(7).normal(0, 18, (img.height, img.width, 3))
    return np.clip(arr, 0, 255).astype(np.uint8)


def _shadow() -> np.ndarray:
    arr = np.asarray(render(TICKET)).astype(np.float32)
    arr *= np.linspace(0.35, 1.0, arr.shape[1])[None, :, None]
    return arr.astype(np.uint8)


def _small() -> np.ndarray:
    img = render(TICKET, size=40, width=1000)
    return np.array(img.resize((img.width * 3 // 10, img.height * 3 // 10), Image.LANCZOS))


def _jpeg(arr: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=quality)
    return np.array(Image.open(buf).convert("RGB"))


def _degraded(img: Image.Image, blur: float, noise: float, quality: int, seed: int = 7) -> np.ndarray:
    img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img).astype(np.float32) + np.random.default_rng(seed).normal(0, noise, (img.height, img.width, 3))
    return _jpeg(np.clip(arr, 0, 255).astype(np.uint8), quality)


def _tiny() -> np.ndarray:
    img = render(TICKET, size=40, width=1000)
    return np.array(img.resize((img.width // 5, img.height // 5), Image.LANCZOS))


def _low_contrast() -> np.ndarray:
    arr = np.asarray(render(TICKET)).astype(np.float32)
    arr = 150 + arr * (55 / 255)  # gray text (150) on light gray (205)
    return _degraded(Image.fromarray(arr.astype(np.uint8)), 0.6, 8, 60)


def _phone_photo() -> np.ndarray:
    """Phone photo of a ticket: skewed, with shadow, noise and JPEG."""
    img = render(TICKET).rotate(6, expand=True, fillcolor=(235, 230, 220), resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) * np.linspace(0.45, 1.0, img.width)[None, :, None]
    return _degraded(Image.fromarray(arr.astype(np.uint8)), 1.0, 14, 45)


def _thermal() -> np.ndarray:
    """80 mm thermal printer ticket (576 dots at 203 dpi) scanned at 300 dpi and saved as JPEG, as scanners
    and phones leave it: monospaced type without smoothing (the dots of the thermal head). With that type
    and the compression the recognizer lost the spaces between words ("NIFA41234567",
    "1 LECHESEMIDESNATADA 6X1L 5,34"): of 15 lines, 3 came out glued."""
    path = next((p for p in MONO_FONTS if Path(p).exists()), None) or font_path()
    size = 20
    while widest(font := ImageFont.truetype(path, size), THERMAL_ROWS) > 560:
        size -= 1  # 42 columns across the paper width with any font
    strip = Image.new("L", (576, 120 + 30 * len(THERMAL_ROWS)), 255)
    draw = ImageDraw.Draw(strip)
    draw.fontmode = "1"
    for i, row in enumerate(THERMAL_ROWS):
        draw.text((8, 60 + 30 * i), row, fill=0, font=font)
    scan = strip.resize((strip.width * 300 // 203, strip.height * 300 // 203), Image.BILINEAR)
    page = Image.new("L", (scan.width + 120, scan.height + 120), 255)
    page.paste(scan, (60, 60))
    return _jpeg(np.array(page.convert("RGB")), 85)


def _letter_a4() -> np.ndarray:
    size = 40
    while widest(font := ImageFont.truetype(font_path(), size), LETTER) > 2480 - 2 * 160:
        size -= 2  # so it fits between the margins with any font
    page = Image.new("RGB", (2480, 3508), "white")
    draw = ImageDraw.Draw(page)
    for i, line in enumerate(LETTER):
        draw.text((160, 300 + 70 * i), line, fill="black", font=font)
    return np.array(page.rotate(2, fillcolor="white", resample=Image.BICUBIC))


def _signature() -> bytes:
    font = ImageFont.truetype(font_path(), 44)
    img = Image.new("RGBA", (max(760, 40 + widest(font, ["Firmado: Luis Pérez Muñoz"])), 120), (0, 0, 0, 0))
    # black ink on transparent: with a plain convert("RGB") the background came out black too
    ImageDraw.Draw(img).text((20, 30), "Firmado: Luis Pérez Muñoz", fill=(0, 0, 0, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _columns_image() -> np.ndarray:
    font = ImageFont.truetype(font_path(), 30)
    right_x = max(800, 40 + widest(font, LEFT) + 190)  # the same gap between columns with any font
    img = Image.new("RGB", (max(1500, right_x + widest(font, RIGHT) + 40), 420), "white")
    draw = ImageDraw.Draw(img)
    for i, (left, right) in enumerate(zip(LEFT, RIGHT)):
        draw.text((40, 60 + 70 * i), left, fill="black", font=font)
        draw.text((right_x, 60 + 70 * i), right, fill="black", font=font)
    return np.array(img)


def _layout_page() -> np.ndarray:
    """Scanned page: title, two text columns and a table with lines."""
    font = ImageFont.truetype(font_path(), 30)
    big = ImageFont.truetype(font_path(), 46)
    right_x = max(900, 100 + widest(font, LEFT) + 230)
    x0, y0, cw, rh = 100, 600, max(450, widest(font, [t for row in TABLE for t in row]) + 40), 70
    img = Image.new("RGB", (max(1700, right_x + widest(font, RIGHT) + 100, x0 + 3 * cw + 100), 1400), "white")
    draw = ImageDraw.Draw(img)
    draw.text((100, 60), TITLE, fill="black", font=big)
    for i, (left, right) in enumerate(zip(LEFT, RIGHT)):
        draw.text((100, 200 + 50 * i), left, fill="black", font=font)
        draw.text((right_x, 200 + 50 * i), right, fill="black", font=font)
    for r, row in enumerate(TABLE):
        for c, text in enumerate(row):
            box = (x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            draw.rectangle(box, outline="black", width=3)
            draw.text((box[0] + 20, box[1] + 18), text, fill="black", font=font)
    return np.array(img)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _stamp_pdf() -> bytes:
    import pymupdf

    stamp = Image.new("RGB", (700, 240), "white")
    draw = ImageDraw.Draw(stamp)
    draw.rectangle((10, 10, 690, 230), outline=(30, 30, 150), width=8)
    draw.text((40, 80), "REGISTRO 2026/0042", fill=(30, 30, 150), font=ImageFont.truetype(font_path(), 60))
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(LETTER[:4]):
        page.insert_text((60, 100 + 18 * i), line, fontsize=10)
    page.insert_image(pymupdf.Rect(320, 600, 520, 669), stream=_png(stamp))
    return doc.tobytes()


def _columns_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    for x, lines in ((60, LEFT), (320, RIGHT)):
        for i, line in enumerate(lines):
            page.insert_text((x, 120 + 16 * i), line, fontsize=10)
    page.insert_text((30, 760), "CSV: ABC123 verificable en la sede", fontsize=8, rotate=90)
    return doc.tobytes()


def _form_docx() -> bytes:
    import docx
    from docx.oxml import parse_xml

    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    d = docx.Document()
    d.add_paragraph("Solicitud de ayuda")
    body = d.element.body
    for xml in (
        f'<w:p {ns}><w:r><w:t xml:space="preserve">Nombre: </w:t></w:r><w:sdt><w:sdtContent><w:r><w:t>Luis Pérez</w:t></w:r></w:sdtContent></w:sdt></w:p>',
        f'<w:p {ns}><w:r><w:t xml:space="preserve">Importe: </w:t></w:r><w:ins w:id="1" w:author="a"><w:r><w:t>45,67 EUR</w:t></w:r></w:ins></w:p>',
    ):
        body.insert(len(body) - 1, parse_xml(xml))
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _email() -> bytes:
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = "a@b.es", "c@d.es", "Incident"
    msg.set_content("Screenshot attached")
    msg.add_alternative('<p>Screenshot attached</p><img src="cid:screenshot">', subtype="html")
    msg.get_payload()[1].add_related(_png(render(["Error 404 on the website"])), "image", "png", cid="<screenshot>")
    return msg.as_bytes()


@dataclass
class Case:
    """An input and the text that should come out (one line per row, in reading order)."""

    name: str
    make: Callable[[], object]
    truth: list[str]
    max_cer: float = 0.05  # acceptable character error in tests/integration/test_quality.py
    tags: tuple[str, ...] = field(default_factory=tuple)
    requires: tuple[str, ...] = ()  # optional modules needed to generate it


CASES = [
    Case("ticket", lambda: np.array(render(TICKET)), TICKET),
    Case("ticket_8_degrees", lambda: _rotated(TICKET, 8), TICKET),
    Case("ticket_15_degrees", lambda: _rotated(TICKET, 15), TICKET),
    Case("ticket_sideways", lambda: _rotated(TICKET, 90), TICKET),
    Case("ticket_upside_down", lambda: _rotated(TICKET, 180), TICKET),
    Case("ticket_noise", _noisy, TICKET, max_cer=0.1),
    Case("ticket_shadow", _shadow, TICKET, max_cer=0.1),
    Case("ticket_small", _small, TICKET, max_cer=0.1),
    Case("ticket_tiny", _tiny, TICKET, max_cer=0.4),
    Case("ticket_jpeg_noise", lambda: _degraded(render(TICKET), 1.6, 32, 20), TICKET, max_cer=0.2),
    Case("ticket_low_contrast", _low_contrast, TICKET, max_cer=0.1),
    Case("ticket_phone_photo", _phone_photo, TICKET, max_cer=0.15),
    Case("ticket_thermal", _thermal, THERMAL),
    Case("letter_a4_2_degrees", _letter_a4, LETTER),
    Case("transparent_signature", _signature, ["Firmado: Luis Pérez Muñoz"]),
    Case("two_columns", _columns_image, LEFT + RIGHT),
    Case("russian_no_language", lambda: np.array(render(RUSSIAN)), RUSSIAN, max_cer=0.08),  # script detected on its own
    Case("greek_no_language", lambda: np.array(render(GREEK, size=56)), GREEK, max_cer=0.1),
    Case(
        "layout_table",
        _layout_page,
        [TITLE, *LEFT, *RIGHT, *(" ".join(r) for r in TABLE)],
        max_cer=0.08,
        requires=("rapid_layout",),
    ),
    Case("pdf_stamp", _stamp_pdf, [*LETTER[:4], "REGISTRO 2026/0042"], tags=("pdf",)),
    Case("pdf_columns", _columns_pdf, LEFT + RIGHT + ["CSV: ABC123 verificable en la sede"], 0.0, ("pdf", "native")),
    Case(
        "docx_form",
        _form_docx,
        ["Solicitud de ayuda", "Nombre: Luis Pérez", "Importe: 45,67 EUR"],
        0.0,
        ("native",),
        ("docx",),
    ),
    Case(
        "email_screenshot",
        _email,
        ["From: a@b.es", "To: c@d.es", "Subject: Incident", "Screenshot attached", "Error 404 on the website"],
    ),
]


# --------------------------------------------------------------------------- measures


def normalize(text: str) -> list[str]:
    """Lines without extra spaces or empty lines."""
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def edit_distance(a: list | str, b: list | str) -> int:
    """Levenshtein (inserting, deleting or changing an element costs 1)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


@dataclass
class Score:
    cer: float  # misread characters / characters of the correct text (line order included)
    wer: float  # the same with words
    lines: float  # correct lines that come out whole as one row (without losing or merging lines)
    # words of the correct text that come out, regardless of order: what is found by searching in what was
    # read (in real documents the order of the correct text is approximate and the CER penalizes it)
    words: float = 0.0


def score(predicted: str, truth: list[str]) -> Score:
    pred, gold = normalize(predicted), normalize("\n".join(truth))
    pred_text, gold_text = "\n".join(pred), "\n".join(gold)
    cer = edit_distance(pred_text, gold_text) / max(1, len(gold_text))
    wer = edit_distance(pred_text.split(), gold_text.split()) / max(1, len(gold_text.split()))
    found = sum(1 for line in gold if line in pred)
    wanted, got = Counter(gold_text.split()), Counter(pred_text.split())
    words = sum(min(n, got[w]) for w, n in wanted.items()) / max(1, sum(wanted.values()))
    return Score(round(cer, 4), round(wer, 4), round(found / max(1, len(gold)), 4), round(words, 4))


def ground_truth(path: Path) -> list[str] | None:
    """Correct text of a real file: invoice.pdf.gt.txt or invoice.gt.txt next to it."""
    for candidate in (path.with_name(path.name + ".gt.txt"), path.with_suffix(".gt.txt")):
        if candidate.exists():
            return normalize(candidate.read_text(encoding="utf-8"))
    return None


__all__ = ["CASES", "Case", "Score", "edit_distance", "font_path", "ground_truth", "render", "score"]
