# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared text normalization."""

from __future__ import annotations

import re
import unicodedata

_SPACES = re.compile(r"\s+")


def nfc(text: str) -> str:
    """Composed Unicode form: "í" as one character and not "i" + accent (kraken writes NFD) so that
    searching for "Martínez" finds it."""
    return unicodedata.normalize("NFC", text)


def normalize_space(text: str) -> str:
    """Runs of spaces and line breaks -> one space, no spaces at the ends."""
    return _SPACES.sub(" ", text).strip()


def text_key(text: str) -> str:
    """Key to compare readings: ignores differences in spacing and case."""
    return nfc(normalize_space(text)).lower()


_STROKES = str.maketrans("đĐøØłŁħĦ", "dDoOlLhH")  # letters with a stroke: NFD does not split them


def without_marks(text: str) -> tuple[str, int]:
    """The text without its marks (accents, tones, cedillas, the stroke of đ) and how many it had."""
    letters = unicodedata.normalize("NFD", text)
    base = "".join(c for c in letters.translate(_STROKES) if not unicodedata.combining(c))
    marks = sum(1 for c in letters if unicodedata.combining(c)) + sum(
        1 for a, b in zip(letters, letters.translate(_STROKES)) if a != b
    )
    return base, marks


def right_to_left(text: str) -> bool:
    """Whether it is written right to left (Arabic, Persian, Urdu, Hebrew...): most of its letters with
    their own direction are (digits and punctuation do not count)."""
    strong = [d for d in map(unicodedata.bidirectional, text) if d in ("L", "R", "AL")]
    return sum(1 for d in strong if d != "L") * 2 > len(strong)
