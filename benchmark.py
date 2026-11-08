# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quality benchmark: CER/WER (misread characters/words) and whole lines, per strategy.

    python benchmark.py                                  # synthetic cases (tests/quality.py)
    python benchmark.py --strategy maximum_recall
    python benchmark.py --dir my_documents --lang es     # real files with their .gt.txt next to them
    python benchmark.py --dir forms --limit 15 --strategy all   # the first 15, every strategy
    python benchmark.py --dir forms --detect es,en       # detecting only among those languages
    python benchmark.py --save folder                    # save the synthetic cases to look at them

Correct text of a real file: invoice.pdf -> invoice.pdf.gt.txt (or invoice.gt.txt), one line per row in
reading order.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import typer

from tests.quality import CASES, Score, ground_truth, score

app = typer.Typer(add_completion=False)
STRATEGIES = ("fast", "balanced", "accurate", "maximum_recall")


def _inputs(directory: str | None, only: str | None, limit: int = 0) -> list[tuple[str, object, list[str]]]:
    if directory:
        files = sorted(p for p in Path(directory).rglob("*") if p.is_file() and not p.name.endswith(".gt.txt"))
        found = [(p.name, p, truth) for p in files if (truth := ground_truth(p)) is not None]
        found = found[:limit] if limit else found
        if not found:
            raise typer.BadParameter(f"no file in {directory} has its .gt.txt next to it", param_hint="--dir")
        return found
    available = [c for c in CASES if all(importlib.util.find_spec(m) for m in c.requires)]
    return [(c.name, c.make(), c.truth) for c in available if not only or only in c.name]


@app.command()
def main(
    strategy: str = typer.Option("fast,maximum_recall", help="Strategies separated by commas, or all"),
    directory: str | None = typer.Option(None, "--dir", help="Real files with their .gt.txt"),
    lang: str = typer.Option("", help="Languages, e.g. es,en"),
    detect: str = typer.Option("", help="Without --lang: languages to detect among, e.g. es,ru,ar"),
    only: str | None = typer.Option(None, help="Only the synthetic cases whose name contains this"),
    save: str | None = typer.Option(None, help="Save the synthetic cases to this folder and exit"),
    limit: int = typer.Option(0, help="With --dir: only the first N files (0 = all)"),
) -> None:
    from omnilector import OCR

    if save:
        Path(save).mkdir(parents=True, exist_ok=True)
        for name, data, truth in _inputs(None, only):
            if isinstance(data, np.ndarray):
                from PIL import Image

                Image.fromarray(data).save(Path(save) / f"{name}.png")
            else:
                (Path(save) / f"{name}.bin").write_bytes(data)  # type: ignore[arg-type]
            (Path(save) / f"{name}.gt.txt").write_text("\n".join(truth) + "\n", encoding="utf-8")
        typer.echo(f"casos guardados en {save}")
        return
    names = STRATEGIES if strategy == "all" else tuple(s.strip() for s in strategy.split(","))
    inputs = _inputs(directory, only, limit)
    typer.echo(f"{'case':24} {'strategy':15} {'time':>7} {'CER':>7} {'WER':>7} {'lines':>7} {'words':>8}  engines")
    for name in names:
        # one instance per strategy: models loaded once
        ocr = OCR(strategy=name, languages=lang or None, detect_languages=detect or None)
        scores: list[Score] = []
        total = 0.0
        for case, data, truth in inputs:
            t0 = time.time()
            try:
                doc = ocr.read(data)
            except Exception as e:  # noqa: BLE001 - a broken file does not stop the benchmark
                typer.echo(f"{case:24} {name:15} ERROR {e}")
                continue
            elapsed = time.time() - t0
            total += elapsed
            s = score(doc.text, truth)
            scores.append(s)
            engines = ",".join(doc.engines) or "-"
            typer.echo(
                f"{case[:24]:24} {name:15} {elapsed:6.2f}s {s.cer:7.2%} {s.wer:7.2%} {s.lines:7.0%} {s.words:8.1%}  {engines}"
            )
        if scores:
            n = len(scores)
            cer, wer, lines, words = (sum(getattr(s, k) for s in scores) / n for k in ("cer", "wer", "lines", "words"))
            typer.echo(f"{'MEDIA':24} {name:15} {total:6.1f}s {cer:7.2%} {wer:7.2%} {lines:7.0%} {words:8.1%}\n")


if __name__ == "__main__":
    app()
