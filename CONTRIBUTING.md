# Contributing

omnilector is free software (GNU AGPL-3.0 or later): contributions are published under that same license.

## Environment

```bash
git clone https://github.com/Slax97/omnilector.git
cd omnilector
python -m venv .venv
.venv/bin/pip install -e ".[dev]"     # on Windows: .venv\Scripts\pip install -e ".[dev]"
# with every engine (CPU build of torch):
.venv/bin/pip install -e ".[all,dev]" --extra-index-url https://download.pytorch.org/whl/cpu
```

## Tests

```bash
pytest -q -m "not integration"   # fast, no models: fake engines (tests/unit/conftest.py)
pytest -q -m integration         # real models, quality, LibreOffice and real images (downloaded the first time)
python benchmark.py              # quality per strategy (CER, WER, lines and words)
```

- A change in what the OCR reads is measured with `benchmark.py` before and after (the synthetic cases and,
  with `--dir`, real documents with their `.gt.txt`) and the numbers go in the commit message.
- No test files are committed: test data is generated in code (`tests/helpers.py`, `tests/quality.py`) or, if
  a real image is needed, linked in `tests/realdata.py` with its license, which has to allow any use (CC0,
  CC BY, CC BY-SA, Apache-2.0, MIT...).

## Style

```bash
ruff check .
ruff format .
```

- Python 3.10 or later, with type hints and docstrings. Comments, docstrings and messages in English.
- Every `.py` file starts with `# SPDX-License-Identifier: AGPL-3.0-or-later`.
- Logging with the `omnilector` logger, never `print` (the command line writes with `typer.echo`).
- Text is never dropped because of low confidence: the other readings go to `alternatives`. The only exception is what
  an extra pass (variants, scales, consensus) sees in a zone pass 1 did not read: it has to prove it is text.
- New engines with `register_engine` and new formats with `register_loader` (see "Extending" in the README):
  the router chooses by what each engine declares, with no `if` on engine or strategy names in the pipeline.
- Dependencies up to their next major version; optional ones, in an extra of `pyproject.toml`.
