# SPDX-License-Identifier: AGPL-3.0-or-later
"""Languages (ISO 639-1) -> script, which decides the model/engine, and each engine's own codes."""

from __future__ import annotations

from collections.abc import Iterable

_SCRIPT_LANGS = {
    "latin": "af az bs ca cs cy da de en es et eu fi fr ga gl hr hu id is it ku la lb lt lv mi ms mt "
    "nl no oc pl pt qu rm ro sk sl sq sv sw tl tr uz vi",
    "east_slavic": "ru uk be",
    "cyrillic": "bg sr mk mn kk ky tg tt ba cv",
    "arabic": "ar fa ur ug ps",
    "devanagari": "hi mr ne sa",
    "chinese": "zh zh-hant",
    "japanese": "ja",
    "korean": "ko",
    "greek": "el",
    "thai": "th",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    # RapidOCR has no model for these: they are read with the engines that do and are not detected automatically
    "bengali": "bn",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "malayalam": "ml",
    "burmese": "my",
    "khmer": "km",
    "lao": "lo",
    "ethiopic": "am",
    "hebrew": "he",
    "georgian": "ka",
    "armenian": "hy",
}
SCRIPT_OF: dict[str, str] = {lang: s for s, langs in _SCRIPT_LANGS.items() for lang in langs.split()}
SCRIPTS = frozenset(_SCRIPT_LANGS)

# Letters of each script (Unicode blocks): with them we know whether a model really read that script
_HAN = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))
_CYRILLIC = ((0x0400, 0x052F), (0x1C80, 0x1C8F), (0xA640, 0xA69F))
_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "latin": ((0x0041, 0x024F), (0x1E00, 0x1EFF)),
    "cyrillic": _CYRILLIC,
    "east_slavic": _CYRILLIC,
    "greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "chinese": _HAN,
    "japanese": ((0x3040, 0x30FF), (0x31F0, 0x31FF), *_HAN),
    "korean": ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)),
    "thai": ((0x0E00, 0x0E7F),),
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "bengali": ((0x0980, 0x09FF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
    "gujarati": ((0x0A80, 0x0AFF),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "burmese": ((0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)),
    "khmer": ((0x1780, 0x17FF), (0x19E0, 0x19FF)),
    "lao": ((0x0E80, 0x0EFF),),
    "ethiopic": ((0x1200, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)),
    "hebrew": ((0x0590, 0x05FF), (0xFB1D, 0xFB4F)),
    "georgian": ((0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x2D00, 0x2D2F)),
    "armenian": ((0x0530, 0x058F), (0xFB13, 0xFB17)),
}

# Codes of other systems (ISO 639-2 / Tesseract, RapidOCR, EasyOCR) -> ISO 639-1
_ALIASES = {
    "spa": "es",
    "eng": "en",
    "fra": "fr",
    "fre": "fr",
    "deu": "de",
    "ger": "de",
    "ita": "it",
    "por": "pt",
    "cat": "ca",
    "glg": "gl",
    "eus": "eu",
    "baq": "eu",
    "nld": "nl",
    "dut": "nl",
    "pol": "pl",
    "ron": "ro",
    "rum": "ro",
    "tur": "tr",
    "swe": "sv",
    "dan": "da",
    "nor": "no",
    "fin": "fi",
    "ces": "cs",
    "cze": "cs",
    "hun": "hu",
    "rus": "ru",
    "ukr": "uk",
    "bel": "be",
    "bul": "bg",
    "srp": "sr",
    "rs_cyrillic": "sr",
    "ara": "ar",
    "fas": "fa",
    "per": "fa",
    "urd": "ur",
    "hin": "hi",
    "mar": "mr",
    "nep": "ne",
    "zho": "zh",
    "chi": "zh",
    "chi_sim": "zh",
    "ch": "zh",
    "ch_sim": "zh",
    "zh_cn": "zh",
    "zh-cn": "zh",
    "zh_hans": "zh",
    "zh-hans": "zh",
    "chi_tra": "zh-hant",
    "ch_tra": "zh-hant",
    "zh_hant": "zh-hant",
    "zh_tw": "zh-hant",
    "zh-tw": "zh-hant",
    "zh_hk": "zh-hant",
    "zh-hk": "zh-hant",
    "jpn": "ja",
    "japan": "ja",
    "jp": "ja",
    "kor": "ko",
    "korean": "ko",
    "ell": "el",
    "gre": "el",
    "tha": "th",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "ben": "bn",
    "pan": "pa",
    "guj": "gu",
    "mal": "ml",
    "mya": "my",
    "bur": "my",
    "khm": "km",
    "lao": "lo",
    "amh": "am",
    "heb": "he",
    "iw": "he",
    "kat": "ka",
    "geo": "ka",
    "hye": "hy",
    "arm": "hy",
}

_TESSERACT = {
    "es": "spa",
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "ca": "cat",
    "gl": "glg",
    "eu": "eus",
    "nl": "nld",
    "pl": "pol",
    "ro": "ron",
    "tr": "tur",
    "sv": "swe",
    "da": "dan",
    "no": "nor",
    "fi": "fin",
    "cs": "ces",
    "hu": "hun",
    "sk": "slk",
    "sl": "slv",
    "hr": "hrv",
    "bs": "bos",
    "sq": "sqi",
    "lt": "lit",
    "lv": "lav",
    "et": "est",
    "is": "isl",
    "ga": "gle",
    "cy": "cym",
    "mt": "mlt",
    "la": "lat",
    "af": "afr",
    "id": "ind",
    "ms": "msa",
    "vi": "vie",
    "tl": "tgl",
    "sw": "swa",
    "az": "aze",
    "uz": "uzb",
    "ru": "rus",
    "uk": "ukr",
    "be": "bel",
    "bg": "bul",
    "sr": "srp",
    "mk": "mkd",
    "mn": "mon",
    "kk": "kaz",
    "ar": "ara",
    "fa": "fas",
    "ur": "urd",
    "hi": "hin",
    "mr": "mar",
    "ne": "nep",
    "sa": "san",
    "zh": "chi_sim",
    "zh-hant": "chi_tra",
    "ja": "jpn",
    "ko": "kor",
    "el": "ell",
    "th": "tha",
    "ta": "tam",
    "te": "tel",
    "kn": "kan",
    "bn": "ben",
    "pa": "pan",
    "gu": "guj",
    "ml": "mal",
    "my": "mya",
    "km": "khm",
    "lo": "lao",
    "am": "amh",
    "he": "heb",
    "ka": "kat",
    "hy": "hye",
}
# The same language under another name in some distributions: Debian packages Tagalog as fil
TESSERACT_RENAMED = {"tgl": "fil"}
_TESSERACT_DEFAULT = {  # no languages; for Latin, Spanish (PRIORITY), and always + English
    "latin": "spa",
    "east_slavic": "rus",
    "cyrillic": "bul",
    "arabic": "ara",
    "devanagari": "hin",
    "chinese": "chi_sim",
    "japanese": "jpn",
    "korean": "kor",
    "greek": "ell",
    "thai": "tha",
    "tamil": "tam",
    "telugu": "tel",
    "kannada": "kan",
    "bengali": "ben",
    "gurmukhi": "pan",
    "gujarati": "guj",
    "malayalam": "mal",
    "burmese": "mya",
    "khmer": "khm",
    "lao": "lao",
    "ethiopic": "amh",
    "hebrew": "heb",
    "georgian": "kat",
    "armenian": "hye",
}

# EasyOCR: languages of the same group share a model; those of "other" scripts only go with English.
_EASYOCR_GROUP = {"east_slavic": "cyrillic"}
_EASYOCR_LATIN = frozenset(
    {
        "af",
        "az",
        "bs",
        "cs",
        "cy",
        "da",
        "de",
        "en",
        "es",
        "et",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
    }
)
# Without Tamil, although EasyOCR lists it: in 1.7.2 its model has 143 characters and the code expects 127, and it
# does not load
_EASYOCR_CODES = {
    "cyrillic": {"ru": "ru", "uk": "uk", "be": "be", "bg": "bg", "mn": "mn", "sr": "rs_cyrillic"},
    "arabic": {"ar": "ar", "fa": "fa", "ug": "ug", "ur": "ur"},
    "devanagari": {"hi": "hi", "mr": "mr", "ne": "ne", "sa": "sa"},
    "chinese": {"zh": "ch_sim", "zh-hant": "ch_tra"},
    "japanese": {"ja": "ja"},
    "korean": {"ko": "ko"},
    "thai": {"th": "th"},
    "telugu": {"te": "te"},
    "kannada": {"kn": "kn"},
    "bengali": {"bn": "bn"},
}
_EASYOCR_DEFAULT = {"cyrillic": "ru", "arabic": "ar", "devanagari": "hi"}
# Latin languages EasyOCR does not ship: they share the Latin model, languages with similar characters are enough
_EASYOCR_NEAR = {"ca": ("es", "fr"), "gl": ("es", "pt"), "eu": ("es",), "fi": ("sv", "et")}

# Scripts each external engine reads, taken from its own language tables
TESSERACT_SCRIPTS = frozenset(_TESSERACT_DEFAULT)
EASYOCR_SCRIPTS = frozenset({"latin", *_EASYOCR_GROUP, *_EASYOCR_CODES})

# Usual legacy (pre-UTF-8) encoding of each script; Latin ones use cp1252 except these
_LEGACY_BY_SCRIPT = {
    "east_slavic": "cp1251",
    "cyrillic": "cp1251",
    "arabic": "cp1256",
    "greek": "cp1253",
    "thai": "cp874",
    "japanese": "cp932",
    "chinese": "gb18030",
    "korean": "cp949",
    "hebrew": "cp1255",
}
_LEGACY_BY_LANGUAGE = {"zh-hant": "big5"}  # within a script, a language with its own encoding
_LEGACY_LATIN = {
    **dict.fromkeys(("pl", "cs", "sk", "sl", "hr", "bs", "hu", "ro", "sq"), "cp1250"),
    **dict.fromkeys(("lt", "lv", "et"), "cp1257"),
    **dict.fromkeys(("tr", "az"), "cp1254"),
    "vi": "cp1258",
}
_LEGACY_PREVALENCE = (
    "cp1252",
    "cp1250",
    "cp1251",
    "cp1254",
    "cp1257",
    "cp1253",
    "cp1256",
    "cp1258",
    "cp874",
    "cp932",
    "gb18030",
    "cp949",
)


def normalize_languages(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """'es,en' | ['spa', 'EN'] -> ('es', 'en'). ValueError if the language is unknown."""
    if not value:
        return ()
    items = value.replace(";", ",").split(",") if isinstance(value, str) else list(value)
    out: list[str] = []
    for raw in items:
        code = str(raw).strip().lower()
        if not code:
            continue
        code = _ALIASES.get(code, code)
        if code not in SCRIPT_OF:
            raise ValueError(f"Unsupported language '{raw}'. Use ISO 639-1 codes: es, en, fr, de, ru, ar, zh, ja...")
        if code not in out:
            out.append(code)
    return tuple(out)


# Letter families to see whether a word mixes scripts ("中YAAΣION": garbage from a model that does
# not know the script); Chinese and Japanese share the ideograms
_FAMILIES = {
    "latin": "latin",
    "cyrillic": "cyrillic",
    "east_slavic": "cyrillic",
    "greek": "greek",
    "arabic": "arabic",
    "devanagari": "devanagari",
    "chinese": "cjk",
    "japanese": "cjk",
    "korean": "korean",
    "thai": "thai",
    "tamil": "tamil",
    "telugu": "telugu",
    "kannada": "kannada",
    "bengali": "bengali",
    "gurmukhi": "gurmukhi",
    "gujarati": "gujarati",
    "malayalam": "malayalam",
    "burmese": "burmese",
    "khmer": "khmer",
    "lao": "lao",
    "ethiopic": "ethiopic",
    "hebrew": "hebrew",
    "georgian": "georgian",
    "armenian": "armenian",
}


def _family(ch: str) -> str | None:
    code = ord(ch)
    return next((_FAMILIES[s] for s, ranges in _SCRIPT_RANGES.items() if any(a <= code <= b for a, b in ranges)), None)


def mixed_scripts(text: str) -> bool:
    """Whether some word has letters of two different scripts (does not happen in real text)."""
    for word in text.split():
        families = {_family(c) for c in word if c.isalpha()} - {None}
        if len(families) > 1:
            return True
    return False


# Letters that only exist in their script: none of the others look like them. Those of the same shape (А В Е К
# М Н О Р С Т Х, Α Β Ε Η Ι Κ Μ Ν Ο Ρ Τ Υ Χ, and lowercase а е о р с у х) also come out when that model
# reads a text in the other script; these do not. Cyrillic lowercase letters that look like a Latin one in
# italics or like another letter (в к м н т, п like n, г like r, и like u) do not count either
_DISTINCT = {
    "cyrillic": frozenset("БГДЁЖЗИЙЛПФЦЧШЩЪЫЬЭЮЯбдёжзйлфцчшщъыьэюяЄєЇїҐґЎўЉљЊњЂђЋћЏџ"),
    "greek": frozenset("ΓΔΘΛΞΠΣΦΨΩαβγδζηθλξπσςφψω"),
    "latin": frozenset("fjlqstvwzDFGJLNQRSUVWZ"),
}
_DISTINCT["east_slavic"] = _DISTINCT["cyrillic"]
DISTINCT_SHARE = 0.2  # share of the reading's letters that have to be unique to its script


def distinctive(text: str, script: str) -> bool:
    """Whether the reading has letters that only exist in that script (at least one in five): "МЕНЮ",
    thanks to its Ю, is Cyrillic and not "MEHIC"; "РЕСТОРАН", all letters that are also Latin, cannot be told."""
    letters = [c for c in text if c.isalpha()]
    own = sum(1 for c in letters if c in _DISTINCT.get(script, ()))
    return own > 0 and own >= DISTINCT_SHARE * len(letters)


# Scripts that do not separate words with spaces: Chinese, Japanese (with their full-width signs) and Thai
_UNSPACED = ((0x3000, 0x30FF), (0x31F0, 0x31FF), *_HAN, (0xFF00, 0xFFEF), (0x0E00, 0x0E7F))


def unspaced(text: str) -> bool:
    """Whether the text has some character of a script that does not separate words with spaces."""
    return any(a <= ord(c) <= b for c in text for a, b in _UNSPACED)


def script_share(text: str, scripts: Iterable[str]) -> float:
    """Fraction of the text's letters that belong to one of those scripts (0 if it has no letters)."""
    ranges = [r for s in scripts for r in _SCRIPT_RANGES.get(s, ())]
    letters = [ord(c) for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if any(a <= c <= b for a, b in ranges)) / len(letters)


# Very frequent words of each language: enough to know which language a page is in
_STOPWORDS = {
    "en": "the and of to in is that for it with as was on be by this are from at or an which have not you a i "
    "he she his her we they but all so if my your me him them our their there what when who would could will "
    "can do did has had been were one",
    "es": "de la que el en y los del se las por un para con no una su al lo como mas pero sus le ya este esta "
    "es son fue era muy yo tu ella hay porque cuando todo esto nos ni hasta donde",
    "fr": "le la les de des et est un une du en que qui dans pour pas sur au par avec ce il elle je vous nous "
    "mais ou sont aux cette",
    "de": "der die und in den von zu das mit sich des auf fur ist im dem nicht ein eine als auch es an ich sie "
    "wir aber oder sind war",
    "it": "di e il la che in per un una del non sono le si con da lo gli al della ma come anche io",
    "pt": "de a o que e do da em um para com nao uma os no se na por mais as dos mas foi ao ele ela",
    "ca": "de la el i que a en les per un una amb del no els es al com dels pero aquesta",
}
_STOPWORD_SETS = {lang: frozenset(words.split()) for lang, words in _STOPWORDS.items()}
# Languages in order of preference when none is given: decides between near ties when guessing the
# language of a page, and which languages Tesseract and EasyOCR read in Latin script
PRIORITY = ("es", "en", "fr", "de", "it", "pt", "ca")


def guess_language(texts: Iterable[str], among: Iterable[str] | None = None) -> str | None:
    """Language of a page's text from its most frequent words (en, es, fr, de, it, pt, ca; with `among`,
    only among those); None if there is little text. If another language nearly ties with the top one
    (it does not win by a margin), the first in PRIORITY: "de la casa y el perro" is Spanish, not Catalan."""
    import unicodedata

    sets = _STOPWORD_SETS if among is None else {lang: _STOPWORD_SETS[lang] for lang in among if lang in _STOPWORD_SETS}
    raw = unicodedata.normalize("NFKD", " ".join(texts).lower())
    words = "".join(c if c.isalpha() else " " for c in raw if not unicodedata.combining(c)).split()
    if len(words) < 4 or not sets:
        return None
    hits = {lang: sum(w in sw for w in words) for lang, sw in sets.items()}
    best = max(hits.values())
    if best < (2 if len(words) < 15 else 3):
        return None
    return next(lang for lang in PRIORITY if lang in hits and 1.5 * hits[lang] > best)


def main_script(languages: Iterable[str]) -> str:
    """Script that decides the model: the first non-Latin one if there is any (its models also read
    digits and basic Latin); otherwise, Latin."""
    for lang in languages:
        script = SCRIPT_OF.get(lang, "latin")
        if script != "latin":
            return script
    return "latin"


# Scripts with the same letters and a different model each: a language of one also brings the other to
# detection. On real signs in Russian and Ukrainian, the general Cyrillic model read clean lines that the
# East Slavic one mixed with Latin letters ("вYлиця", "PYCCKИE ПEЛЬMЕHИ")
_SAME_LETTERS = {"cyrillic": "east_slavic", "east_slavic": "cyrillic"}


def scripts_to_detect(languages: Iterable[str]) -> frozenset[str]:
    """Scripts to detect among with those languages (detect_languages): theirs and those sharing their
    letters."""
    scripts = {SCRIPT_OF[lang] for lang in languages}
    return frozenset(scripts | {_SAME_LETTERS[s] for s in scripts if s in _SAME_LETTERS})


def detect_start(languages: Iterable[str]) -> str:
    """Script detection starts with, among those languages (detect_languages): Latin if any of them is and,
    otherwise, the first one's. Detection only works well in that direction: what the Latin model does not
    understand comes out doubtful and is tried with the others; not the other way round, because the models
    of the other scripts also read basic Latin and a Spanish page read with them does not come out doubtful."""
    scripts = [SCRIPT_OF.get(lang, "latin") for lang in languages]
    return scripts[0] if scripts and "latin" not in scripts else "latin"


def tesseract_codes(languages: Iterable[str], script: str) -> list[str]:
    """Requested languages as Tesseract codes ('spa', 'eng'...); if there are none, the script's + English
    (for Latin, Spanish + English: as with languages=["es"])."""
    codes = [_TESSERACT[lang] for lang in languages if lang in _TESSERACT]
    if not codes:
        codes = [_TESSERACT_DEFAULT.get(script, "eng")]
    if "eng" not in codes:
        codes.append("eng")
    return codes


def easyocr_codes(languages: Iterable[str], script: str) -> list[str]:
    """Mutually compatible languages for an easyocr.Reader of that script (+ English). Without languages, for
    Latin, the first four of PRIORITY: the same Latin model with its letters (with only "en", its English
    model has no accents or ñ: on real Spanish forms, 80 % of the words against 90 %)."""
    group = _EASYOCR_GROUP.get(script, script)
    codes: list[str] = []
    for lang in languages:
        if group == "latin" and SCRIPT_OF.get(lang) == "latin":
            candidates = [lang] if lang in _EASYOCR_LATIN else list(_EASYOCR_NEAR.get(lang, ()))
        else:
            mapped = _EASYOCR_CODES.get(group, {}).get(lang)
            candidates = [mapped] if mapped else []
        codes += [c for c in candidates if c not in codes]
    if group == "chinese":
        codes = codes[:1]  # simplified and traditional are two models that cannot go together
    if not codes and group == "latin":
        codes = list(PRIORITY[:4])
    if not codes and group in _EASYOCR_DEFAULT:
        codes.append(_EASYOCR_DEFAULT[group])
    if not codes and group in _EASYOCR_CODES:
        codes.append(next(iter(_EASYOCR_CODES[group].values())))
    if "en" not in codes:
        codes.append("en")
    return codes


def legacy_encodings(languages: Iterable[str]) -> list[str]:
    """Legacy encodings to prefer, in order: those of the given languages and then the most common ones."""
    hinted = []
    for lang in languages:
        script = SCRIPT_OF.get(lang, "latin")
        if script == "latin":
            hinted.append(_LEGACY_LATIN.get(lang, "cp1252"))
        else:
            hinted.append(_LEGACY_BY_LANGUAGE.get(lang) or _LEGACY_BY_SCRIPT.get(script))
    return list(dict.fromkeys(e for e in (*hinted, *_LEGACY_PREVALENCE) if e))
