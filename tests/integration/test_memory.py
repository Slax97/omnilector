# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OCR fits in the memory it is given: each case runs in a new process with memory_mb=2048 (the
default strategy, handwriting included) and its memory peak has to stay below it. Without the
adjustment, an A4 page at 4000 px already went over 1.7 GB and a handwritten deed over 2.1 GB."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

from .. import realdata

pytestmark = pytest.mark.integration

LIMIT_MB = 2048
ROOT = Path(__file__).parents[2]

# What runs in the new process: reads the case and prints its memory peak (MB)
CHILD = """
import json, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from omnilector import OCR

def page(w, h, lines):
    img = Image.new("RGB", (w, h), "white")
    draw, size = ImageDraw.Draw(img), h // 70
    font = ImageFont.load_default(size=size)
    for i in range(lines):
        draw.text((size * 2, size * (2 + 1.6 * i)), f"Line {i} of the test document, amount {i * 7}.50 EUR", fill="black", font=font)
    return img

case, path = sys.argv[1], sys.argv[2]
if case == "a4":
    source = np.array(page(2480, 3508, 40))  # A4 at 300 dpi
elif case == "photo":
    source = np.array(page(4000, 3000, 30))  # 12 Mpx photo
elif case == "pdf":
    pages = [page(2480, 3508, 40) for _ in range(4)]
    pages[0].save(path, save_all=True, append_images=pages[1:])  # 4-page scanned PDF
    source = path
else:
    source = path
doc = OCR(memory_mb=int(sys.argv[3])).read(source)
try:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)
except ImportError:  # Windows
    import psutil
    peak = psutil.Process().memory_info().peak_wset / 2**20
print(json.dumps({"peak_mb": round(peak), "chars": len(doc.text)}))
"""


@pytest.mark.parametrize("case", ["a4", "photo", "pdf", "sign"])
def test_fits_in_2gb(case, tmp_path):
    pytest.importorskip("rapidocr")
    if sys.platform == "win32" and importlib.util.find_spec("psutil") is None:
        pytest.skip("on Windows psutil is needed to measure the memory")
    path = tmp_path / "doc.pdf"
    if case == "sign":  # a real photo: a street sign in Russian, its script detected without languages
        sample = next(s for s in realdata.SAMPLES if s.name == "moscow.jpg")
        try:
            path = realdata.fetch(sample)
        except (urllib.error.URLError, OSError) as e:
            pytest.skip(f"no network or no sample: {e}")
    out = subprocess.run(
        [sys.executable, "-c", CHILD, case, str(path), str(LIMIT_MB)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=900,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["chars"] > 0
    assert result["peak_mb"] < LIMIT_MB, f"{case}: peak of {result['peak_mb']} MB with memory_mb={LIMIT_MB}"
