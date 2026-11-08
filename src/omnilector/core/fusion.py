# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fusion of several readings (passes, variants, engines) of the same page.

- Same zone (IoU >= 0.5) read several times -> one block; the consensus decides the text (how many
  engines and readings agree) and then the confidence.
- Reading contained in a larger one (part of the line, a piece of a letter, the tight box of another
  engine) -> its readings go to the larger one.
- Nothing is ever lost: every reading stays in candidates (and comes out in Line.alternatives).
- Different zones are not touched: "123" and "456" close to each other stay separate.
- A stretch of a line already read by other readings of that line, which only partly overlap each other, is the
  same text: its readings go to them (drop_repeats).
- The doubtful pieces of a line that overlap go together to the line engine (join_pieces).
"""

from __future__ import annotations

from dataclasses import replace
from difflib import SequenceMatcher

from ..result import RecognitionCandidate, TextBlock
from ..utils.geometry import box_area, box_iou, coverage, in_row_order
from ..utils.text import text_key

REPEAT_WIDTH = 0.8  # share of a block's width the other readings of its line have to cover
REPEAT_TEXT = 0.6  # share of its characters they have to read too: another line that overlaps it would not match
# Share of the height of the shorter piece inside the taller one to join them. In the 265 cases of the comparison between
# engines: 0.82 or more on pieces of one line, 0.59 on two neighbouring slanted lines, read as one they are garbage
JOIN_INSIDE = 0.8


def _merge_candidates(blocks: list[TextBlock]) -> list[RecognitionCandidate]:
    """Candidates of every reading, without repeats (text, engine): the highest confidence stays."""
    best: dict[tuple[str, str], RecognitionCandidate] = {}
    for c in (c for b in blocks for c in b.candidates):
        key = (c.text, c.engine)
        if key not in best or c.confidence > best[key].confidence:
            best[key] = c
    return list(best.values())


def _resolve(cluster: list[TextBlock]) -> TextBlock:
    if len(cluster) == 1:
        return cluster[0]
    votes: dict[str, list[TextBlock]] = {}
    for b in cluster:
        votes.setdefault(text_key(b.text), []).append(b)
    if len(votes) > 1:
        votes.pop("", None)  # a zone some engine could not read: any reading with text wins

    def score(key: str) -> tuple[int, int, float]:
        members = votes[key]
        return (len({b.engine for b in members}), len(members), max(b.confidence for b in members))

    winner = max(votes[max(votes, key=score)], key=lambda b: b.confidence)
    return replace(winner, candidates=_merge_candidates(cluster))


def fuse_blocks(blocks: list[TextBlock], iou_thresh: float = 0.5, contain_thresh: float = 0.8) -> list[TextBlock]:
    clusters: list[list[TextBlock]] = []
    for b in sorted(blocks, key=lambda x: -x.confidence):
        cluster = next((c for c in clusters if box_iou(b.box, c[0].box) >= iou_thresh), None)
        if cluster is None:
            clusters.append([b])
        else:
            cluster.append(b)
    kept: list[TextBlock] = []
    for b in sorted((_resolve(c) for c in clusters), key=lambda x: box_area(x.box), reverse=True):
        # a zone without text (something detected that no engine read) does not take the lines inside it
        container = next(
            (k for k in kept if (k.text.strip() or not b.text.strip()) and coverage(b.box, [k.box]) >= contain_thresh),
            None,
        )
        if container is None:
            kept.append(b)
        else:
            container.candidates = _merge_candidates([container, b])
    return kept


def _same_line(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """They overlap vertically by at least half the height of the taller one: not a line inside a whole paragraph."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap >= 0.5 * max(a[3] - a[1], b[3] - b[1])


def _width_covered(block: TextBlock, others: list[TextBlock]) -> float:
    """Share of the block's width covered by the others, without counting what they overlap twice."""
    x0, x1 = block.box[0], block.box[2]
    covered, reached = 0, x0
    for start, end in sorted((max(o.box[0], x0), min(o.box[2], x1)) for o in others):
        covered += max(0, end - max(start, reached))
        reached = max(reached, end)
    return covered / (x1 - x0) if x1 > x0 else 1.0


def _matched_share(text: str, others: str) -> float:
    """Share of the characters of `text` that `others` also has, in the same order."""
    blocks = SequenceMatcher(None, text, others, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(text)


def _repeated_in(block: TextBlock, kept: list[TextBlock]) -> TextBlock | None:
    """The kept block that already reads this stretch of a line, if the others of the line read it all."""
    line = [k for k in kept if k.text.strip() and _same_line(block.box, k.box)]
    if not line or _width_covered(block, line) < REPEAT_WIDTH:
        return None
    text = text_key(block.text)
    if text and _matched_share(text, " ".join(text_key(k.text) for k in line)) < REPEAT_TEXT:
        return None
    return max(line, key=lambda k: min(block.box[2], k.box[2]) - max(block.box[0], k.box[0]))


def drop_repeats(blocks: list[TextBlock]) -> list[TextBlock]:
    """Without the stretches read twice: engines and variants cut the same line in different places, and each
    piece repeated part of the text of its neighbours ("fabrer guanter de" + "er guanter de Bara"). What goes
    away stays in the candidates of the line that reads it."""
    kept: list[TextBlock] = []
    for b in sorted(blocks, key=lambda x: box_area(x.box), reverse=True):
        target = _repeated_in(b, kept)
        if target is None:
            kept.append(b)
        else:
            target.candidates = _merge_candidates([target, b])
    return kept


def _span(blocks: list[TextBlock]) -> tuple[int, int, int, int]:
    boxes = [b.box for b in blocks]
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _pieces_of_a_line(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """On the same line and the shorter one almost whole inside the height of the other: two slanted lines one above
    the other also overlap, but only partly."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return _same_line(a, b) and overlap >= JOIN_INSIDE * min(a[3] - a[1], b[3] - b[1])


def join_pieces(blocks: list[TextBlock], threshold: float) -> list[TextBlock]:
    """The doubtful pieces of a line that overlap, as a single block, for the line engine to read the line whole:
    on a handwritten line cut in pieces by the detector and the variants, kraken got 79 % of the characters wrong and,
    read whole, 8 %. Each reading stays in the candidates; until it is read, the block has the text of its pieces and
    the mean confidence of their characters: a sure "kay" of 3 letters does not make a doubtful line of 60 look sure."""
    pieces = sorted((b for b in blocks if b.confidence < threshold or not b.text.strip()), key=lambda b: b.box[0])
    groups: list[list[TextBlock]] = []
    for b in pieces:
        group = next((g for g in groups if b.box[0] < _span(g)[2] and _pieces_of_a_line(b.box, _span(g))), None)
        if group is None:
            groups.append([b])
        else:
            group.append(b)
    out = [b for b in blocks if not (b.confidence < threshold or not b.text.strip())]
    for group in groups:
        if len(group) == 1:
            out.append(group[0])
            continue
        read = [b for b in in_row_order(group) if b.text.strip()]
        out.append(
            TextBlock(
                text=" ".join(b.text for b in read),
                box=_span(group),
                confidence=sum(b.confidence * len(b.text) for b in read) / sum(len(b.text) for b in read)
                if read
                else 0.0,
                engine=max(group, key=lambda b: b.confidence).engine,
                candidates=_merge_candidates(group),
            )
        )
    return out
