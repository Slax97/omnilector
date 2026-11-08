# omnilector

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
![Python 3.10 to 3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

Extracts the text of **any file** (photos, scans, PDF, Word, Excel, PowerPoint, OpenDocument, HTML, Outlook or .eml emails, EPUB, ZIP...) picking **the right OCR for each situation**, and always returns it in **the same format**, whichever engine is used or however several are combined. Local, no paid services, **runs on CPU**. Free software: GNU AGPL-3.0 license (see [Licenses](#licenses)).

## Installation

From GitHub, with the extras you need (Python 3.10 or later):

```bash
pip install "omnilector @ git+https://github.com/Slax97/omnilector.git"                  # base
pip install "omnilector[office,layout] @ git+https://github.com/Slax97/omnilector.git"   # with extras
omnilector download --lang es   # optional: download the models now (to work offline)
omnilector download             # without a language: those of every script detected automatically + layout
```

| Extra | What it adds |
|---|---|
| (base) | images, PDF, EPUB, OpenDocument, html, txt, eml, zip (RapidOCR, CPU) |
| `office` | docx, xlsx, pptx |
| `layout` | layout analysis: zones, reading order and tables in scans (~140 MB of models) |
| `msg` | Outlook .msg |
| `handwritten` | handwriting: kraken (64 MB) + TrOCR for English (~1.4 GB the first time) |
| `easyocr` | EasyOCR: photos, 80+ languages |
| `tesseract` | Tesseract (and the program: `apt install tesseract-ocr tesseract-ocr-spa`) |
| `heic` | iPhone photos |
| `all` | all of the above (no GPU) |
| `gpu` | onnxruntime-gpu, instead of onnxruntime |

With `handwritten` or `easyocr`, install the CPU build of torch in the same command (otherwise pip downloads the CUDA one, ~3 GB): `pip install "omnilector[handwritten] @ git+https://github.com/Slax97/omnilector.git" --extra-index-url https://download.pytorch.org/whl/cpu`.

Every dependency is allowed up to its next major version (PyMuPDF 2.0 changed table extraction without notice): the bound is raised once the tests pass with it.

Legacy Office and RTF (.doc, .xls, .ppt, .rtf) need LibreOffice installed (`apt install libreoffice-writer-nogui libreoffice-calc-nogui libreoffice-impress-nogui`, or the installer from libreoffice.org); if it is missing, the error says so.

To work on omnilector itself (environment, tests and style): [CONTRIBUTING.md](CONTRIBUTING.md).

## Usage

```python
import omnilector

doc = omnilector.read("invoice.pdf")  # or .jpg .heic .tiff .docx .odt .xlsx .pptx .html .eml .msg .epub .zip .doc .rtf...
print(doc.text)

ocr = omnilector.OCR(languages=["es"])  # reuses the models: create one per process/worker
doc = ocr.read("ticket.jpg")
for line in doc.lines:
    print(line.text, line.confidence, line.engines, line.alternatives, line.box)
for table in doc.tables:                 # tables with rows, columns and cells (scanned or from the file)
    print(table.grid(), table.to_markdown())
print(doc.to_markdown())                 # titles, paragraphs and tables: ready for an LLM
print(doc.to_json())                     # everything as JSON: pages, lines with box and confidence, tables, warnings

for doc in ocr.read_many(["a.pdf", "b.jpg", "batch.zip"]):  # several: one document (or one error) per file
    print(doc.source, doc.error or doc.text)

omnilector.read("photo.jpg", engine="easyocr")  # force an engine
omnilector.read("letter.pdf", detect_languages=["es", "ru"])  # the language is detected, but only among these
omnilector.list_engines()  # engines available on this machine and how to install the rest
```

## Common output

The same structure whether the text comes from RapidOCR, Tesseract, EasyOCR, kraken, TrOCR, from the file itself (PDF with text, Word, Excel...) or from several engines combined:

```
Document
  text          full text (pages separated by a blank line)
  pages         [Page]
  kind          image | pdf | docx | xlsx | pptx | odt | ods | odp | epub | xps | ebook | html | text | email | msg | zip | doc | xls | ppt | rtf
  source, metadata, warnings
  error         None, or why it could not be read (with read_many; read() raises the exception)
Page
  number, origin ("sheet Sales", "attachment invoice.pdf", "slide 2, notes", "body image image001.png"...)
  method        native (text of the file) | ocr | hybrid
  rotation      degrees the image was rotated to read it upright (skewed photo, sideways or upside-down scan); 0 = as is
  script        script the OCR read with (latin, cyrillic, arabic, korean...); None = text of the file
  text          one line per row
  lines         [Line] in reading order
  blocks        [Block] zones in reading order: title | heading | text | list | table | caption | figure |
                stamp | header | footer | page_number | footnote | formula (with layout analysis)
  tables        [Table] rows, cols, cells [Cell(row, col, text, box, rowspan, colspan)], grid(), to_markdown(), to_html()
Line
  text          "TOTAL 48,22 EUR"
  box           (0.05, 0.62, 0.41, 0.68)    box relative to the page, 0-1
  confidence    0.97                        0-1; 1.0 = native text of the file
  engines       ["rapidocr", "tesseract"]   engines that read this text ("native" if it comes from the file)
  alternatives  ["TOTAL 48.22 EUR"]         other different readings of the same zone
  handwritten   False
  row           3                           visual row: lines with the same row are at the same height
  block         2                           zone of Page.blocks it belongs to (None without layout analysis)
```

- If one engine reads "TOTAL 48,22 EUR" whole and another splits it into "TOTAL" + "48,22 EUR", `page.text` comes out the same: the pieces of a row are joined with a space (and share `row`).
- When engines are combined (`maximum_recall`), each line says how many agree (`engines`) and what the others read (`alternatives`): nothing is lost.
- Boxes are relative (0-1), so they do not depend on the resolution, the PDF's DPI or the engine. On a rotated page (`rotation` other than 0) they refer to the upright page, whole (not cropped). In formats without geometry (docx, xlsx, odt, html, txt, email) they are an approximate row/column grid; the rows are the file's own.
- Reading order: on pages with OCR and layout analysis (extra `[layout]`), the order of the zones the model gives (columns, boxes, captions) and by rows inside each one. Without layout analysis, and for native text: by rows, top to bottom; column by column when the page has columns of running text; tables and tickets stay in rows. Rotated text (the vertical CSV in the margin of a signed PDF, a watermark, a skewed stamp) goes at the end of the page, without merging with the rows. In right-to-left scripts (Arabic, Persian, Urdu, Hebrew), the pieces of a row and the columns go right to left.
- Tables: `Page.tables` has the structure wherever the table comes from: a scan or a photo (SLANet-plus), a PDF with text (the PDF's own lines), a docx, odt, pptx, html (only data tables: not the ones used to lay out emails and websites) or a spreadsheet (xlsx, ods: each sheet is a table). `doc.to_markdown()` puts them in place.
- `doc.to_json()` and `doc.to_dict()`: all of this as JSON (see [Several files](#several-files)).

## Several files

```python
ocr = omnilector.OCR(languages=["es"])
for doc in ocr.read_many(["invoice.pdf", "broken.jpg", ("upload.msg", data)]):
    if doc.error:                        # that file could not be read; the others go on
        print(doc.source, type(doc.error).__name__, doc.error)
    else:
        print(doc.source, doc.text)

ocr.read_many(["batch.zip"], split_archives=True)  # batch.zip/invoice.pdf, batch.zip/photos/ticket.jpg...
omnilector.read_many(paths, languages=["es"])     # shortcut, like omnilector.read
```

- One `Document` per input, in order and as they are read (it does not wait to have them all). An input that cannot be read (missing, corrupt, unsupported format, missing extra, or all its pages failed) comes out with `doc.error` (the exception) and no pages, and the rest goes on. An unexpected error (from omnilector or from a library it uses) too, and it also goes to the `omnilector` log with its traceback. `read()`, with a single input, still raises the exception.
- What arrives in memory, with its name: `(name, bytes)` or `(name, stream)`. The name is `doc.source`, and its extension helps recognize the format (without the `[msg]` extra, a .doc, .xls or .ppt is only recognized by it; it also breaks ties between .eml and .html).
- ZIP: by default, one document with the pages of all its files (`page.origin` says which file each one comes from) and one OCR limit (`max_ocr_pages`) for the whole ZIP. With `split_archives=True`, one document per file inside (`batch.zip/photos/ticket.jpg`; those of a ZIP inside the ZIP too), each one with its own limit, as if they had been passed in the list; the files the ZIP skips (encrypted, too large, broken) come out with their error. An email (.eml, .msg) is always one document, with its attachments inside.
- JSON: `doc.to_json()` (or `doc.to_dict()`): the fields of the [common output](#common-output), plus the text of the document and of each page, and the error as `{"type": "DocumentLoadError", "message": "..."}`. The text goes as it is, not escaped (accents, Cyrillic...), and dates in ISO format.

```
{"source": "invoice.pdf", "kind": "pdf", "text": "...", "error": null, "warnings": [], "metadata": {"pages": 1},
 "pages": [{"number": 1, "method": "ocr", "origin": null, "rotation": 0.0, "script": "latin", "text": "...",
   "lines": [{"text": "TOTAL 48,22 EUR", "box": [0.05, 0.62, 0.41, 0.68], "confidence": 0.97, "engines": ["rapidocr"],
              "alternatives": ["TOTAL 48.22 EUR"], "handwritten": false, "row": 3, "block": 2}],
   "blocks": [{"kind": "text", "box": [...], "lines": [0]}],
   "tables": [{"box": [...], "rows": 2, "cols": 2, "block": 4, "cells": [{"row": 0, "col": 0, "text": "Item", ...}]}]}]}
```

## Searchable PDF

```python
ocr.searchable_pdf("scan.pdf", "scan_searchable.pdf")   # returns the Document read
ocr.searchable_pdf("photo.jpg", "photo.pdf")            # or an image, or a multi-page TIFF
```

The same document, looking the same (no image is touched; a JPEG goes as it is, without compressing it again), with the recognized text invisible on top of each line: in any viewer it can be searched, selected and copied, as with ocrmypdf. From a PDF, only what went through the OCR gets new text (scanned pages and the images of hybrid pages; native text was already searchable). Each line goes in its place even if the page was skewed, sideways or had `/Rotate`. In any script: the text goes character by character, without shaping, so it is extracted exactly (shaping lost letters in Devanagari); PyMuPDF fonts (Helvetica, Droid Sans Fallback and Noto). Arabic is copied with the presentation forms of each letter; NFKC normalization, the one search engines apply, turns them back into the normal ones.

## Formats

| Input | How it is read |
|---|---|
| jpg, png, webp, bmp, gif, multi-page tiff, jpeg2000 | OCR. Transparent areas (signatures, logos) go on white |
| heic/heif (iPhone photos) | OCR, extra `[heic]` |
| PDF | per page: native text / OCR if it is scanned or its text is unreadable / the OCR reads only the crop of each image (photos, stamps, signatures, logos), with the native text covered; the same image on several pages (the letterhead), once. No OCR if nothing is left (a scanned PDF with a text layer). Form checkboxes come out as ☑ / ☐ (◉ / ○ for radio buttons) and the PDF's attachments (a PDF/A-3 invoice with its XML, a PDF portfolio, the paper clip of a comment) are each read with their loader. With a password, `passwords` |
| EPUB, XPS, MOBI, FB2 | like a PDF, with PyMuPDF |
| docx, xlsx, pptx | native text (tables by cell; one page per sheet, slide and notes) + embedded images as image pages, extra `[office]`. From docx, also content controls (forms), text boxes, inserted changes, fields, nested tables, headers, footers and notes |
| odt, ods, odp (LibreOffice) | same as Office, no dependencies |
| html, txt, md, csv/tsv, json... | native text, encoding detected |
| eml | headers + body + attachments + images pasted in the body, each one with its loader |
| msg (Outlook) | same as eml (attached emails too), extra `[msg]` |
| zip | each file inside with its loader, with limits against zip bombs: all in one document or, with `split_archives`, one per file. Encrypted ones (ZipCrypto), with `passwords` |
| tar, tar.gz/tgz, tar.bz2, tar.xz | like ZIP; read in a single pass (without decompressing it twice) |
| gz, bz2, xz | the file inside, with its loader (invoice.pdf.gz is a PDF), decompressed up to 512 MB |
| doc, xls, ppt, rtf | converted with LibreOffice (if installed) to docx/xlsx/pptx |

It accepts a path, bytes, a stream, a `PIL.Image` or a `numpy` array (also float 0-1, 16 bits or with alpha). The kind is detected from the first bytes, not from the extension.

## Which OCR is used in each situation

| Situation | How it is detected | Engine | If it is not installed |
|---|---|---|---|
| Digital text | the loader finds text | none: native extraction | - |
| Scanned PDF page or one with unreadable text | < 50 useful characters on that page | RapidOCR | Tesseract / EasyOCR |
| Scan, screenshot or rasterized PDF | light, even background (analyzer, ~10 ms) | RapidOCR | Tesseract |
| Phone photo | camera EXIF, uneven lighting or dark background | RapidOCR + a shadow-free variant | EasyOCR |
| Skewed photo, sideways or upside-down scan | angle of the lines from pass 1 + RapidOCR's 0/180 classifier (~15 ms) | the page is straightened without reading it again | read as it is |
| The same, with an engine that does not give the corners of its lines (Tesseract) or lines read by kraken | RapidOCR's detector finds the lines; if the reading is doubtful (mean confidence < 0.85), it is also read upside down: RapidOCR's classifier and Tesseract's OSD do not know those scripts | Tesseract reads the page again straightened, and upside down if that reads clearly surer; kraken, a few lines upside down and, if they read surer, the whole page | read as it is |
| Blank page | no ink | one pass and nothing else | - |
| Small text | line height < 15 px | re-read enlarged x2 | - |
| Handwriting | lines with confidence < 0.85 and shaped like a line; the pieces of a line that overlap go together, read whole | kraken line by line (44 languages in Latin, Cyrillic, Greek, Arabic, Hebrew, Georgian and Armenian) `[handwritten]`; if the page is in English (`languages=["en"]` or its words say so), TrOCR. A reading with the same letters that only drops the marks of the one there was ("quyên và tu do" for "quyền và tự do") does not replace it | without kraken, TrOCR (Latin only, not in Vietnamese); without either, the printed reading |
| Another script, language not given | lines the Latin/Chinese/Japanese model cannot read (empty, garbage, few letters): the models of the other scripts try them (with `detect_languages`, only those of its languages) | RapidOCR with the model of the script that reads them; on signs with several scripts, each line with its own | the Latin reading stays |
| Another script, language given | `languages=["ru"]`, `["ar"]`, `["ko"]`... | RapidOCR with the model of that script; the lines it does not understand, with those of the other languages of the list | EasyOCR / Tesseract |
| Arabic, Persian, Urdu, Hebrew, Georgian or Armenian, language given | `languages=["ar"]`, `["he"]`... (also with `handwritten=False`: it is print; not with `fast`, as kraken takes 10 to 30 times longer on a book page) | kraken reads every line, on the lines found by RapidOCR: it also beats the page engines on print | the usual page engines |
| Kannada, Vietnamese or Urdu, language given | RapidOCR gets them wrong without doubting (`EngineInfo.weak_languages`) | the other page engines go first: Tesseract on scans, EasyOCR on photos. In Vietnamese, RapidOCR keeps its place if none of them is meant for that kind of page (a photo without EasyOCR), and reads after all if they barely trust their reading (an old print). It does not join the consensus of `maximum_recall`: its sure readings without diacritics won the fusion. With kraken, Urdu stays with RapidOCR + kraken | RapidOCR |
| Bengali, Punjabi, Gujarati, Malayalam, Burmese, Khmer, Lao or Amharic, language given | RapidOCR has no model for them | Tesseract `[tesseract]` (Bengali on photos, EasyOCR `[easyocr]`) | an error says what to install |
| Layout (columns, boxes, tables) | PP-DocLayout on pages with OCR (~0.7 s); tables of the PDF itself with PyMuPDF | zones in reading order + SLANet-plus for tables `[layout]` | order by rows |
| `maximum_recall` with signs of missing text | few blocks, low confidence, low coverage | + the other engines, fusion by consensus. In a zone nobody had read, what they find only stays if it is text (2 or more characters with confidence >= 0.7, or >= 0.9): the variants also "find" stains and drawings | - |
| Made-up ideograms on a Latin page | lines with Chinese or Japanese characters and confidence < 0.7, if they are few on the page (the asterisks of a card number, a logo) and no Chinese or Japanese among the languages | read again with the Latin-only recognizer (PP-OCRv5, 8 MB) and the most confident reading stays | the multi model's |

In everything RapidOCR reads, the space between two words that the model sees but does not pick is put back: it happened with the monospaced type of tickets in JPEG ("41001Sevilla", "0.2 87.00" -> "0.287.00"). It is checked between two letters farther apart than the rest of their line, and not in Chinese, Japanese or Thai, which do not separate words.

The order of the engines comes from what each one declares (`EngineInfo.kinds`: scan/photo, `cost`, `weak_languages` and `print_languages`). With `log_level="DEBUG"` (or `omnilector read -v`) you can see, page by page, the detected situation, the rotation applied and why each engine was chosen.

## Languages

```python
omnilector.OCR(languages=["es"])                     # the language is known: read with its model
omnilector.OCR(detect_languages=["es", "ru", "ar"])  # each page can be in any of these
omnilector.OCR()                                     # in any: detected among all scripts
```

- `languages`: the document is in those languages. It is read with the model of their script (with several, the first non-Latin one: Russian in `["es", "ru"]`, which also reads basic Latin), and the lines that model does not understand are tried with those of the other languages of the list: on a sign with `["ru", "en", "ar"]`, the Arabic is no longer lost. For "each page in Spanish or in Russian", `detect_languages` is better, as it starts with Latin.
- `detect_languages`: each page can be in any of them. It starts with the Latin model if any language is Latin (otherwise, with the first one's), and the lines it does not understand are only tried with the models of the other scripts of the list. Faster (with `["es", "ru", "ar"]`, 3 models to try instead of 10), fewer downloads (`omnilector download --lang es,ru,ar` gets only those), and a Spanish form can no longer come out as Greek or Kannada. What is not in the list is not detected.
- Neither: detected among the 13 scripts RapidOCR has a model for. The other 11 (Bengali, Gurmukhi, Gujarati, Malayalam, Burmese, Khmer, Lao, Ethiopic, Hebrew, Georgian and Armenian) are not detected: their language has to be given, in `languages` or as the only script of `detect_languages`. Tesseract's script detection (OSD) took 8 of them for another script on every page tried (Georgian and Armenian for Latin, Arabic or Cyrillic, Lao for Thai) and 2 more on some, and trying every engine on the lines no model understands would cost seconds on every handwritten page.

Spanish goes first when no language is given, then English, French and German: EasyOCR reads Latin script in those four (with English only, its model had no accents or ñ: on real Spanish forms, from 80 % to 90 % of the words), Tesseract in Spanish and English, and when guessing the language of a page, if another one nearly ties, the first of that list wins.

A Cyrillic language brings both Cyrillic models, the general one and the East Slavic one, also with `languages`: on real signs in Russian and Ukrainian, the general one reads clean lines that the other mixes with Latin letters ("вYлиця"). Without `languages`, within a document each page starts with the script of the previous one. The `detect_languages` list also decides the handwriting language (if it is the only one of the list with that script, that one: with `["en", "ru"]` English goes straight to TrOCR, and without `en` TrOCR is not used), the encoding of legacy texts (txt, csv, emails that are not UTF-8) and the languages Tesseract and EasyOCR read with. If both are given, `languages` wins; `auto_script=False` turns detection off. Also configurable with `OMNILECTOR_DETECT_LANGUAGES` and `--detect`. ISO 639-1 codes (`es`, `ru`, `zh`...) or Tesseract's (`spa`, `rus`, `chi_sim`...). `zh` is simplified Chinese and `zh-hant` (or `zh-tw`, `chi_tra`), traditional, which decides the model of Tesseract and EasyOCR; RapidOCR reads both with the same one.

Scripts: Latin (es, en, fr, de, it, pt, ca, pl, tr, vi...), Cyrillic (bg, sr, mk, kk...), East Slavic (ru, uk, be), Arabic (ar, fa, ur), Devanagari (hi, mr, ne), Chinese, Japanese, Korean, Greek, Thai, Tamil, Telugu and Kannada. With other engines: Hebrew (he), Georgian (ka) and Armenian (hy), with `[handwritten]` or `[tesseract]`; Bengali (bn), with `[tesseract]` or `[easyocr]`; and Gurmukhi (pa), Gujarati (gu), Malayalam (ml), Burmese (my), Khmer (km), Lao (lo) and Ethiopic (am), with `[tesseract]` and its language data (`apt install tesseract-ocr-pan`...).

Tesseract without the data of the language: its Latin is read in English, with a warning; another script does not use it, since it would only give garbage.

## Using it as a library

- One `OCR` instance per process (or worker), reused: the models are loaded once. `ocr.load_models()` at startup, so the first request does not wait.
- Errors: all inherit from `omnilector.OmnilectorError` (`DependencyMissingError` says which extra to install). A page that fails does not bring the document down: it goes to `doc.warnings`. With `read_many`, a file that fails does not bring the batch down: it goes to its `doc.error`.
- Logs: logger `omnilector`, with no handlers of its own: your application's logging configuration decides.
- Types: the package ships `py.typed`; your editor and mypy see those of `Document`, `Page`, `Line`...
- License: AGPL-3.0. If you distribute your application or offer it as a network service, its code also has to be published under the AGPL-3.0 (see [Licenses](#licenses)).
- Memory and cores: it adapts on its own to what is free (see "Any server"). With `threads` set explicitly, OpenCV and torch also use those threads in your whole process (they are global settings of those libraries), and Tesseract gets `OMP_THREAD_LIMIT`.

## Environment variables

To configure omnilector from outside (a service, a container, a shell) without touching code:

| Variable | Option | Example | Without it |
|---|---|---|---|
| `OMNILECTOR_STRATEGY` | `strategy` | `balanced` | `maximum_recall` |
| `OMNILECTOR_LANGUAGES` | `languages` | `es,en` | none: the script is detected automatically |
| `OMNILECTOR_DETECT_LANGUAGES` | `detect_languages` | `es,ru,ar` | all: detected among every script |
| `OMNILECTOR_HANDWRITTEN` | `handwritten` | `1`/`0`, `true`/`false`, `yes`/`no`, `on`/`off` | yes |
| `OMNILECTOR_MEMORY_MB` | `memory_mb` | `2048`, `2g` | what is free (container or machine) |
| `OMNILECTOR_THREADS` | `threads` | `2` | the cores of the process |
| `OMNILECTOR_MODEL_DIR` | `model_cache_dir` | `/models` | each engine's cache |
| `OMNILECTOR_TIME_LIMIT` | `time_limit` | `120` (seconds per document) | no limit |

What is passed in code (`OCR(strategy="fast")`) or on the command line (`--strategy fast`) wins over the variable. An empty variable is as if it were not set, and an invalid value gives an error naming the variable.

## Any server: cores and memory

Without configuring anything, omnilector looks at what the process has and adapts:

| | Where it comes from | What it decides |
|---|---|---|
| Cores | the CPU quota of the process (its cgroup) and the cores assigned to it | threads of each engine: RapidOCR, layout, kraken, TrOCR, EasyOCR, Tesseract and OpenCV. Without a limit, each one uses its own |
| Memory | what is free when the OCR is created: the memory limit of the process (its cgroup) or the machine's RAM, minus what your application and the other processes already use | maximum image size, whether there is layout analysis, which engines are used and how many models stay loaded |

What is used depending on the memory (default strategy, `maximum_recall`):

| Memory | Image up to | Layout | Handwriting | Consensus |
|---|---|---|---|---|
| 1 GB | 1400 px | no | no | Tesseract |
| 1.5 GB | 1700 px | yes | no | Tesseract |
| 2 GB | 2000 px | yes | kraken | Tesseract |
| 4 GB | 3000 px | yes | kraken and TrOCR (English) | Tesseract |
| 8 GB | 4000 px | yes | kraken and TrOCR | Tesseract and EasyOCR (at 1900 px) |

Measured on Linux (CPU), what each part takes: reading an image with RapidOCR, ~150 MB per megapixel (a 2000 px page, 0.66 GB; a 4000 px one, 2.2 GB); EasyOCR, ~770 MB per megapixel (an A4 page at its own size, 2560 px, +3.5 GB); RapidOCR loaded, 150 MB; layout, 320 MB; kraken, 380 MB; TrOCR, 1.6 GB; EasyOCR, 350 MB; torch, 200 MB (once); Tesseract, ~15 MB per megapixel. With a limit, the loaded models do not go over what is left (before loading one, the least recently used is released) and huge images are reduced on entry to 2 times the side the OCR sees (with 2 GB, 4000 px); JPEGs, already while decoding (a 48 Mpx photo with `memory_mb=2048`: peak from 1.10 GB to 0.83 GB, measured on Windows). Checked with the process limited to 2 GB of memory, without swap, and 2 cores: 44 documents in a row in one process (A4 scans at 300 dpi, 12 and 48 Mpx photos, a 6-page scanned PDF, signs in 9 scripts, tables, layout, handwriting), with a peak of 1.9 GB counting the file cache and never reaching the limit; the program's memory, with RapidOCR, layout and kraken loaded, stays at ~1.2 GB. With 2 cores, a scanned page takes 5 to 10 s and a whole handwritten deed, 1-1.5 min.

Explicit settings always win:

```python
OCR(memory_mb=2048, threads=2)              # or OMNILECTOR_MEMORY_MB=2048 and OMNILECTOR_THREADS=2
OCR(memory_mb=2048, max_image_side=2500)    # more detail per page, at the cost of memory
OCR(memory_mb=2048, engine="easyocr")       # a forced engine is used even if it does not fit
```

```bash
omnilector read invoice.pdf --memory 2g --threads 2 -v   # -v: what it adapted to and which engines do not fit
omnilector engines                                       # how much each engine takes
```

- `memory_mb` is per process. If it is not set, each OCR counts what is already in use when it is created (your application, the workers that already started); if the workers create the OCR at the same time, split the memory (`OMNILECTOR_MEMORY_MB`) and the cores yourself.
- If what is set explicitly does not fit (for example `max_image_side=4000` with 2 GB), it warns and tries anyway.
- Linux with a memory limit: start the workers with `MALLOC_ARENA_MAX=2` in the environment. glibc keeps a memory arena for each thread that allocates and does not give it back, so with the threads of several engines the process holds more than it uses and can reach the limit; the 2 GB check above was made with it.

## Server without a GPU

Everything works on CPU and it is the default mode (`device="cpu"`). If `device="cuda"` is requested and there is no GPU, it warns and goes on with the CPU; `device="auto"` uses the GPU if there is one, without warning.

- torch for CPU (extras `handwritten` and `easyocr`), in the same installation as the extras so ~3 GB of CUDA are not downloaded (see [Installation](#installation)). Installing it before and separately is not enough: kraken asks for `torch<=2.14` and pip replaced it with the PyPI one, with CUDA.
- Linux without a desktop: the base uses OpenCV without a GUI (`opencv-python-headless`, no libGL). With `[layout]`, rapid-layout also installs `opencv-python`; if `libGL.so.1` is missing on import: `pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless`.
- One `OCR` instance per worker and `OCR(...).load_models()` when it starts, so the first request does not wait for models to load.
- Several workers on the same machine: split cores and memory (`threads=N` and `memory_mb=M`, or `OMNILECTOR_THREADS` and `OMNILECTOR_MEMORY_MB`). Each worker is a process and, without splitting, each one thinks it has the whole machine: they step on each other and perform worse than a single one.
- Offline: `omnilector download --lang es,...` once with network; `model_cache_dir` (or `OMNILECTOR_MODEL_DIR`) decides where they are stored. If a model cannot be loaded, that engine is not retried on every page.
- Safety limits: `max_ocr_pages=50` pages or images with OCR per document (attachments included; pages with native text do not count, all of them are read; a ZIP is one document and, with `split_archives`, so is each file inside) and `max_pages` (all of them, no limit by default). What is skipped goes to `doc.warnings`.
- Maximum time per document: `time_limit` in seconds (or `OMNILECTOR_TIME_LIMIT`, `--time-limit`). After it, the remaining pages do not go through the OCR (nor are they rasterized) and the one being read is finished without extra passes (variants, consensus, handwriting, layout, script detection); `doc.warnings` says so. It is checked between steps: a single long read is not cut.
- Passwords: `passwords=["1234", ...]` (or `--password`, which can be repeated) for protected PDFs and encrypted files in a ZIP; tried in order. WinZip AES encryption is not read by zipfile: that file is skipped with a warning.
- Streaming pages: a 50-page PDF does not take 50 images in memory.
- A page that fails stays empty with a warning in `doc.warnings`; if none can be read, the error is raised (with `read_many`, it goes in `doc.error`). If the OCR of the images of a page with text fails, its native text stays.
- Logs: the library adds no handlers. Without configuring anything, warnings go to stderr; with your application's logging configured, they come out once, in its format.
- Rough CPU performance: a 1200x700 ticket, ~0.3-0.8 s if the first pass is enough and ~3.5 s if it escalates to variants + EasyOCR; a blank A4 page, ~2 s; a scanned PDF with a text layer, ~0.1 s per page (no OCR); straightening a page does not read it again, except where Tesseract reads (it does not give the corners of its lines: RapidOCR's detector finds them, and a skewed page is read again). `strategy="balanced"` drops handwriting and consensus (in Arabic, Persian, Urdu, Hebrew, Georgian and Armenian kraken still reads the print: on a book page, 7 to 37 s instead of 1 to 3 s).

## Strategies

`fast`: 1 pass (in PDFs with images, only the native text; no layout analysis or tables). `balanced`: + variants/scale if text is missing, and layout analysis. `accurate`: + handwriting. `maximum_recall` (default): + consensus between engines. The values that depend on the strategy (maximum image side, OCR of embedded images, `scales`, `layout`) can be set explicitly: explicit settings win. The script is detected automatically in all of them: among the 13, among those of `detect_languages` or, with `languages`, among those of its languages (`auto_script=False` turns it off).

## Architecture

```
src/omnilector/
  ocr.py config.py result.py exceptions.py cli.py
  loaders/        what the file is -> pages with native text (by rows) and/or an image    detect, image, pdf, office, odf, text, email, msg, archive, convert
  routing/        what situation it is -> which engine                                    analyzer, router, languages
  engines/        the OCRs, all with the same interface                                   rapidocr, kraken, trocr, tesseract, easyocr
  core/           pipeline (passes + consensus), orientation, layout (zones and tables), fusion, model_manager, strategies
  preprocessing/  variants (clahe, sharpen, shadow...), one at a time and only those requested
  utils/          geometry (rows, columns, reading order), text, image, optional dependencies
```

`read(input)` -> loader -> per page: native (no OCR), image, or native with image crops -> analyzer -> router plan -> PASS1 with the best engine (if it fails, the next one) -> upright page (skew and 90-degree turns, with the lines already read) -> script (without `languages`: the lines the Latin model cannot read, with those of the other scripts) -> upside down (with the model of its script) -> PASS2 variants/scales if text is missing -> consensus with other engines (`maximum_recall`) -> fusion -> PASS3 handwriting on doubtful lines (kraken; if it turns out to be English, those lines also with TrOCR) -> layout (zones in reading order and tables) or rows in reading order -> common output (`result.make_page`).

Rules: text is never dropped because of low confidence (RapidOCR filtered < 0.5 by default: turned off); every reading stays in `alternatives`, also those contained in a larger one.

### Extending

```python
from omnilector import register_engine, register_loader
from omnilector.engines import EngineInfo, OCREngine, TextBlock

@register_engine
class MyOCR(OCREngine):
    info = EngineInfo(name="my_ocr", scripts=frozenset({"latin"}), kinds=frozenset({"scan"}), cost=2)

    def _build(self):  # loads the model once (called by load(), also by `omnilector download`)
        ...

    def read(self, image):  # RGB image -> list[TextBlock] (boxes in pixels; quad = 4 corners, if it has them)
        model = self.load()
        ...

register_loader("wpd", my_wpd_loader, extensions=(".wpd",))  # (Source, OCRConfig) -> RawDocument
```

A new engine only returns `TextBlock` (text, box in pixels, confidence and, if it has them, the 4 corners of the line: with them skewed pages are straightened; without them, a detector finds the lines and the page is read again straightened); the conversion to the common output is the same for all. The router places it according to its declared capabilities (scripts, `kinds`, page/line level, cost, GPU, languages it is good or bad at), without touching the pipeline. Engines with `gpu="required"` are not used with `device="cpu"`. If it can tell whether some lines are upside down, `upside_down(image, boxes)`.

## CLI

```bash
omnilector read invoice.pdf --lang es               # text
omnilector read invoice.pdf --lang es --format md   # text with a header per page
omnilector read a.pdf b.jpg invoices/               # several files and folders (with subfolders, without hidden ones)
omnilector read invoices/ -f json > batch.jsonl     # one JSON line per file, as each one finishes
omnilector read batch.zip --split-archives          # one result per file of the ZIP
omnilector read letters/ --detect es,ru,ar          # the language of each page, only among these
omnilector read photo.jpg -v                        # + situation, rotation and engines of each page
omnilector read protected.pdf --password 1234       # password-protected PDF (or encrypted ZIP)
omnilector read batch/ --time-limit 120             # at most 2 minutes per file
omnilector pdf scan.pdf scan_searchable.pdf         # searchable PDF: the scan with its invisible text on top
omnilector engines                                  # which engines there are and how to install the rest
omnilector download --lang es,ru                    # download models to work offline
```

With several files, in `text` each one goes after `==> name <==` and in `md` after its `# name`; warnings and errors go to stderr with the name of the file (in `json`, inside its line). They are read in order, with the models loaded once. If one cannot be read, it goes on with the rest and ends with exit code 1.

## Licenses

Copyright (C) 2026 Slax97. omnilector is free software: you can use, study, modify and distribute it, also for commercial purposes, under the terms of the **GNU Affero General Public License, version 3 or later** (`LICENSE`). It comes without any warranty.

- If you distribute omnilector, with or without changes, or a program that uses it, you have to give whoever receives it the complete source code under the same license.
- If you offer it to others as a network service (a website, an API), too: its users have the right to receive the source code of what runs on that server, with your changes.
- Using it for yourself, without distributing it or offering it to others, requires nothing.

### Third-party licenses

Everything omnilector uses is open source and compatible with the AGPL-3.0. The repository and the package do not include third-party code or models: pip installs the dependencies, with their license files, and the models are downloaded the first time they are needed.

- PyMuPDF (PDF, EPUB, XPS, MOBI and FB2) has the same license as omnilector, AGPL-3.0 (or Artifex's commercial license, https://artifex.com/licensing), and omnilector always needs it.
- The rest are permissive (MIT, BSD, Apache-2.0, MIT-CMU) or weak copyleft (MPL-2.0 for tqdm, LGPL-3.0 for python-bidi, LGPL for FFmpeg in OpenCV and for libheif in pillow-heif): all of them can be combined with an AGPL-3.0 program. Complying with the AGPL-3.0 and keeping each package's license notices covers them all.

| Dependency | License | What for |
|---|---|---|
| pymupdf | AGPL-3.0 or commercial (Artifex) | PDF, EPUB, XPS, MOBI, FB2 |
| rapidocr | Apache-2.0 | default OCR (PP-OCR in ONNX) |
| python-bidi | LGPL-3.0 | right-to-left order (Arabic, Persian, Urdu) in RapidOCR |
| onnxruntime | MIT | running the ONNX models |
| opencv-python-headless | Apache-2.0 (its wheels include FFmpeg, LGPL-2.1) | images |
| numpy | BSD-3-Clause | |
| pillow | MIT-CMU | images |
| typer | MIT | command line |
| charset-normalizer | MIT | text encodings |
| Dependencies of the above | omegaconf, shapely, networkx, psutil (BSD-3-Clause); pyclipper, pyyaml, colorlog, six, tabulate (MIT); requests (Apache-2.0); tqdm (MPL-2.0 and MIT) | |

| Extra | Packages and licenses |
|---|---|
| `office` | python-docx, openpyxl, python-pptx: MIT |
| `msg` | olefile: BSD-2-Clause |
| `layout` | rapid-layout, rapid-table: Apache-2.0 |
| `handwritten` | kraken, transformers, huggingface_hub, sentencepiece: Apache-2.0; torch, torchvision: BSD-3-Clause (torch includes Apache-2.0, MIT and BSL-1.0 parts); kraken's dependencies: lightning, htrmopo, pyarrow, regex (Apache-2.0), scikit-learn, scikit-image, scipy, coremltools, click, lxml (BSD), rich (MIT) |
| `easyocr` | easyocr: Apache-2.0 |
| `tesseract` | pytesseract: Apache-2.0 |
| `heic` | pillow-heif: BSD-3-Clause (includes libheif, LGPL-3.0) |
| `gpu` | onnxruntime-gpu: MIT |

| Model | License | Source |
|---|---|---|
| PP-OCRv6 / PP-OCRv5 / PP-OCRv4 (detection, orientation and recognition per script) | Apache-2.0 | PaddleOCR, converted to ONNX by RapidOCR |
| PP-DocLayoutV3 (layout) and SLANet-plus (tables) | Apache-2.0 | PaddleOCR / PaddleX, through rapid-layout and rapid-table |
| kraken-ppocrv6-medium (handwriting, 44 languages) | Apache-2.0 | https://huggingface.co/small-models-for-glam/kraken-ppocrv6-medium |
| microsoft/trocr-base-handwritten (English handwriting) | MIT | https://huggingface.co/microsoft/trocr-base-handwritten |
| EasyOCR: CRAFT detection and recognition | CRAFT: MIT (NAVER Clova); EasyOCR: Apache-2.0 | https://github.com/JaidedAI/EasyOCR |
| Tesseract (tessdata) | Apache-2.0 | https://github.com/tesseract-ocr/tessdata_best |

The YOLO / DocLayout-YOLO models (AGPL-3.0) that rapid-layout also offers are not used. Optional external programs, not linked: Tesseract (Apache-2.0) for the `tesseract` engine and LibreOffice (MPL-2.0) to convert .doc, .xls, .ppt and .rtf.

This is a summary to find your way, not legal advice: for a specific case, check the text of each license or ask a lawyer.

## Tests / benchmark

```bash
pytest -q -m "not integration"  # no models (fake engines)
pytest -q -m integration        # real models (downloaded the first time), quality and LibreOffice
pytest -q -m integration tests/integration/test_real_data.py  # real images from the internet (to tests/data/net)
python benchmark.py                                  # quality: CER/WER and whole lines per strategy
python benchmark.py --dir my_documents --lang es     # your files, with their correct text next to them
python benchmark.py --dir my_documents --limit 15 --strategy all   # the first 15, with every strategy
```

`benchmark.py` measures the character error rate (CER), the word error rate (WER) and how many lines come out whole in their row, with synthetic cases generated on the fly (`tests/quality.py`: skewed, sideways and upside-down tickets, a thermal printer ticket in JPEG, noise, JPEG, tiny type, low contrast, a phone photo, an A4 letter from a scanner, a transparent signature, columns, a PDF with a stamp, a docx form, an email with a screenshot) or with real files: `invoice.pdf` + `invoice.pdf.gt.txt` (one line per row, in reading order). `tests/integration/test_quality.py` fails if a case gets worse. The "words" column says how many words of the correct text come out, regardless of order: in real documents the order of the correct text is approximate and the CER penalizes it.

Measured locally with 15 real scanned forms in Spanish from [XFUND](https://github.com/doc-analysis/XFUND) (A4 at 300 dpi, without `languages`). Their license, CC BY-NC-SA 4.0, does not allow commercial use: they are not included nor downloaded with the project.

| Strategy | Time | Words | Whole lines |
|---|---|---|---|
| fast | 2.3 s per form | 92.4 % | 24 % |
| accurate | 4.3 s | 92.6 % | 35 % |
| maximum_recall | 4.0 s | 92.6 % | 35 % |

Times and whole lines with the `[layout]` extra. The words count with the spaces the model saw but did not pick put back (without that step, 90.6-90.7 %).

On these clean scans the extra passes do not find more words; layout analysis does leave more lines whole. What is missing is mostly loose punctuation (colons, filler dots) and fields filled in on a dotted line, which come out glued to it ("calle..rojo.12...").

What cannot be fabricated is checked with real images from the internet (`tests/realdata.py`: only links to images with a license free for any use, each one with its own; downloaded the first time and not stored in git): photos of signs in Cyrillic, Greek, Arabic, Korean, Thai, Devanagari and Japanese without giving the language and two in Arabic, English and Russian at once, a scanned table (PubTabNet) and two tickets (a cafe in Madrid and a shop in Mexico). `tests/integration/test_real_data.py` says which text has to come out of each one.

## Known limitations

- Handwriting: kraken reads Spanish much better than TrOCR (on 18th-century notarial deeds, 40 % character error per line against 78 %), but old handwriting is still hard: with `accurate` and without `languages`, ~28 % error on those deeds and ~10 % on modern English (IAM). kraken reads Latin, Cyrillic, Greek, Arabic, Hebrew, Georgian and Armenian; handwriting in other scripts keeps the printed reading.
- kraken pins versions of lightning, scipy (and with it numpy < 2.5) and scikit-learn: if they clash with those of your project, install `[handwritten]` in its own environment.
- Without the `[layout]` extra, columns are detected if they are running text (lines of about 20 characters or more); a two-column table of long sentences is also read column by column.
- The script is detected per line. Between scripts with similar letters (Latin, Cyrillic, Greek), a line takes the other script if its reading has letters that only exist in it ("МЕНЮ", thanks to its Ю, is not replaced by "MEHIO"); a word made only of letters shared by both ("РЕСТОРАН" / "PECTOPAH") keeps the page's script: without knowing the language it cannot be told. The other mixes (Latin with Arabic, Korean, Thai, Devanagari...), line by line. The first detection loads the models of every script (a few seconds and ~10 MB each; with `detect_languages`, only those of its scripts). In a document, a Spanish page that comes after a Russian one is read with the Cyrillic model if that one reads it without doubts (it also reads Latin): on 3 real forms, 93.0 % of the words against 93.7 % with the Latin one.
- Vertical Chinese and Japanese text is not rotated (those are columns, not a sideways page), but its column order goes left to right.
- Where Tesseract reads (Kannada and Vietnamese on scans, the scripts without a model in RapidOCR, or forced) or kraken reads lines only detected (Hebrew, Georgian, Armenian), whether the page is upside down is only checked if its reading is doubtful (mean confidence < 0.85): read upside down, Tesseract gave 0.06 to 0.73 and kraken 0.58 to 0.78. Finding the lines of a Tesseract page takes RapidOCR's detector ~0.2-0.4 s more, and a skewed or doubtful page is read again.
- The engine chosen for each language (`weak_languages`, `print_languages`) comes from a comparison with few samples per language (a clean text, a simulated photo and up to two book pages), checked afterwards with 40 new documents (another passage and typeface, another photo, pages of other books, handwritten letters): the rules held, and the two flaws they showed were fixed (Vietnamese photos, old Persian books). Still, 2 to 5 samples per language: enough to choose an engine, not to say how well it reads your documents.
- With EasyOCR reading first (photos in Kannada, Bengali and Vietnamese), an upside-down page is not detected: telling it would take another EasyOCR reading of every doubtful photo. The EXIF orientation of photos is applied when loading them.
- Outlook .msg with the body only in compressed RTF (rare): headers and attachments are read, not the body.
- MOBI and XPS are read with PyMuPDF; only EPUB and FB2 were tested.
- Tickets: RapidOCR's model does not have the "¡" ("¡Gracias por su visita!" comes out "Gracias por su visita!") and the asterisks of a card number ("**** **** **** 4821") sometimes come out halfway. On 10 tickets in Spanish (4 real and 6 synthetic), 95.6 % of the words.

## Troubleshooting

- `DependencyMissingError` -> install the extra the message names.
- `EngineNotAvailableError` -> `omnilector engines` shows what is missing.
- `.doc/.xls/.ppt/.rtf` -> `UnsupportedFormatError` with the instructions if LibreOffice is missing.
- No internet -> `omnilector download` with network once.
- Text in a legacy encoding decoded wrongly -> pass `languages` or `detect_languages` (e.g. `["pl"]` prefers cp1250; without a language, cp1252).
- AGPL PyMuPDF in a commercial product -> buy a license or do not process PDFs.
