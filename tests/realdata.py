# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real images from the internet to check what cannot be fabricated: photos of signs in other scripts, a
scanned table and tickets. Only images with a license free for any use (CC0, CC BY, CC BY-SA,
Apache-2.0, MIT), each one with its license noted. They are not in git: downloaded the first time to
tests/data/net/ (ignored). Used by tests/integration/test_real_data.py.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

NET = Path(__file__).parent / "data" / "net"
AGENT = "omnilector-tests (https://github.com/Slax97/omnilector)"
_EASYOCR = "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples"
_PADDLE = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main"
_COMMONS = "https://thumb.wikimedia.org/wikipedia/commons/thumb"


@dataclass
class Sample:
    """A real image and what has to come out of it."""

    name: str
    url: str
    license: str
    expect: list[str] = field(default_factory=list)  # texts that have to appear in doc.text
    scripts: tuple[str, ...] = ()  # accepted detected script (without languages)
    table: tuple[int, int] | None = None  # (rows, columns) of the table that has to come out
    requires: tuple[str, ...] = ()  # optional modules needed


SAMPLES = [
    Sample(
        "moscow.jpg",
        f"{_COMMONS}/1/11/Commemorative_street_name_plaque_for_Ulitsa_Grina_Moscow.jpg/1280px-Commemorative_street_name_plaque_for_Ulitsa_Grina_Moscow.jpg",
        "CC BY 4.0, Wikimedia Commons",
        ["Названа в 1986 году", "Александре Грине"],
        ("cyrillic", "east_slavic"),
    ),
    Sample(
        "kyiv.jpg",
        f"{_COMMONS}/d/de/Kyiv_Zhylianska_street_sign_01.jpg/1280px-Kyiv_Zhylianska_street_sign_01.jpg",
        "CC0, Wikimedia Commons",
        ["вулиця Жилянська", "Zhylianska vulytsia"],
        ("cyrillic", "east_slavic"),
    ),
    Sample(
        "tarnovo.jpg",
        f"{_COMMONS}/3/38/Mednikarska_Street%2C_Veliko_Tarnovo_%28the_street_sign%29.jpg/1280px-Mednikarska_Street%2C_Veliko_Tarnovo_%28the_street_sign%29.jpg",
        "CC BY-SA 4.0, Wikimedia Commons",
        ["Медникарска", "Mednikarska Str."],
        ("cyrillic", "east_slavic"),
    ),
    Sample(
        "athens.jpg",
        f"{_COMMONS}/3/3e/Old_street_sign_for_Flysasion_Street_in_Athens%2C_Greece.jpg/1280px-Old_street_sign_for_Flysasion_Street_in_Athens%2C_Greece.jpg",
        "CC BY-SA 2.0, Wikimedia Commons",
        ["ΦΥΛΑΣΙΩΝ"],
        ("greek",),
    ),
    Sample("korean.png", f"{_EASYOCR}/korean.png", "Apache-2.0, EasyOCR", ["서울", "Seoul"], ("korean",)),
    Sample("thai.jpg", f"{_EASYOCR}/thai.jpg", "Apache-2.0, EasyOCR", ["เพชรบุรี"], ("thai",)),
    Sample(
        "arabic.jpg",
        f"{_PADDLE}/docs/version2.x/ppocr/blog/images/arabic_0.jpg",
        "Apache-2.0, PaddleOCR",
        ["حلويات", "Sweets"],
        ("arabic",),
    ),
    Sample(
        "devanagari.png", f"{_EASYOCR}/example3.png", "Apache-2.0, EasyOCR", ["मार्किट", "BASRURKAR"], ("devanagari",)
    ),
    Sample(
        "japanese.jpg", f"{_EASYOCR}/japanese.jpg", "Apache-2.0, EasyOCR", ["ポイ捨て禁止", "MINATO CITY"], ("latin",)
    ),
    # signs in Arabic, English and Russian at once (the plate says "REPUPLIC")
    Sample(
        "egypt_embassy.jpg",
        f"{_COMMONS}/6/6f/Embassy_of_Egypt_in_Moscow%2C_plate.jpg/1280px-Embassy_of_Egypt_in_Moscow%2C_plate.jpg",
        "CC BY-SA 3.0, Wikimedia Commons",
        ["سفارة جمهورية مصر العربية", "EMBASSY OF THE ARAB REPUPLIC OF EGYPT", "ЕГИПЕТ"],
        ("latin", "cyrillic", "east_slavic", "arabic"),
    ),
    Sample(
        "hurghada.jpg",
        f"{_COMMONS}/8/83/Hurgada_russian_pelmeni.jpg/1280px-Hurgada_russian_pelmeni.jpg",
        "CC BY-SA 3.0, Wikimedia Commons",
        ["ПЕЛЬМЕННАЯ", "РУССКИЕ ПЕЛЬМЕНИ", "Cocktail", "كوكتيل"],
        ("latin", "cyrillic", "east_slavic", "arabic"),
    ),
    Sample(
        "pubtabnet_table.png",
        f"{_PADDLE}/docs/datasets/images/table_PubTabNet_demo/PMC524509_007_00.png",
        "Apache-2.0, PaddleOCR (PubTabNet)",
        ["516.1"],
        table=(6, 8),
        requires=("rapid_layout",),
    ),
    # tickets in Spanish: a skewed photo from a Madrid cafe and a Mexican thermal ticket (phone scanner,
    # 4000x11000). With their type the recognizer lost spaces: "FECHA:22/01/2026"
    Sample(
        "madrid_cafe_ticket.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/6/61/-sablazosmadrid_%27Caf%C3%A9_%26_Te%27--_%22en_fin_de_semana_cobramos_el_desayuno_por_separado%22_-haztefan_pero_no_vayas_%286225790880%29.jpg",
        "CC BY-SA 2.0, Wikimedia Commons (#sablazosmadrid)",
        ["CIA DEL TROPICO DE CAFE Y TE S.L.", "2 BARRITA TOMATE 2,30 4,60", "GRACIAS POR SU VISITA"],
        ("latin",),
    ),
    Sample(
        "mexico_ticket.jpg",
        "https://raw.githubusercontent.com/PabloIAIN/instrumento-ocr-tickets-mx/main/imagenes/TICKET_003.jpg",
        "MIT, PabloIAIN/instrumento-ocr-tickets-mx",
        ["No. TICKET: 742627", "FECHA: 22/01/2026", "0.435 40.00 0.00 17.40", "Público en General"],
        ("latin",),
    ),
]


def fetch(sample: Sample) -> Path:
    """Local path of the sample, downloaded the first time."""
    path = NET / sample.name
    if not path.exists():
        NET.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(sample.url, headers={"User-Agent": AGENT})
        path.write_bytes(urllib.request.urlopen(request, timeout=60).read())
    return path
