# SPDX-License-Identifier: AGPL-3.0-or-later
"""TrOCR (microsoft/trocr-base-handwritten): handwriting line by line. Extra [handwritten].

Trained on English handwriting (IAM): it is the best for English (4.9 % error on real IAM lines), but in
Spanish it loses accents and ñ; for the other languages the router chooses kraken.
Licenses: MIT checkpoint, transformers Apache-2.0, torch BSD. ~1.4GB, works on CPU.
"""

from __future__ import annotations

import math

import numpy as np

from ..exceptions import InferenceError
from ..result import RecognitionCandidate
from ..utils.deps import missing_modules, require
from .base import Box, EngineInfo, OCREngine, crop, torch_device


def sequence_confidence(logprobs: list[float]) -> float:
    """Confidence of a generated text: geometric mean of the probability of its tokens."""
    if not logprobs:
        return 0.0
    return float(math.exp(sum(logprobs) / len(logprobs)))


def token_logprobs(tokens: list[int], scores: list[float], eos_id: int | None) -> list[float]:
    """Log-probs of the tokens generated up to the end of sequence (what follows is padding)."""
    out: list[float] = []
    for tok, lp in zip(tokens, scores):
        if math.isfinite(lp):
            out.append(lp)
        if tok == eos_id:
            break
    return out


class TrOCREngine(OCREngine):
    info = EngineInfo(
        name="trocr",
        level="line",
        scripts=frozenset({"latin"}),
        handwriting=True,
        cost=3,
        memory_mb=1600,  # 1.3 GB of weights in float32
        runtime="torch",
        extra="handwritten",
        languages=frozenset({"en"}),
        # Without Vietnamese letters: 25 % to 84 % of the characters wrong on every page, worse than any page engine
        weak_languages={"vi": 0.0},
        description="English handwriting, line by line",
    )
    model_id = "microsoft/trocr-base-handwritten"
    batch_size = 8

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return missing_modules("transformers", "torch", extra=cls.info.extra)

    def _build(self) -> object:
        torch = require("torch", extra=self.info.extra)
        transformers = require("transformers", extra=self.info.extra)
        if self.config.threads:
            torch.set_num_threads(self.config.threads)
        cache = self.config.model_cache_dir
        processor = transformers.TrOCRProcessor.from_pretrained(self.model_id, cache_dir=cache)
        model = transformers.VisionEncoderDecoderModel.from_pretrained(self.model_id, cache_dir=cache)
        device = torch_device(self.config)
        return processor, model.to(device).eval(), device

    def read_lines(self, image: np.ndarray, boxes: list[Box]) -> list[RecognitionCandidate | None]:
        """Lines in batches; the confidence comes from the real probabilities of the generation."""
        processor, model, device = self.load()  # type: ignore[misc]
        import torch
        from PIL import Image

        results: list[RecognitionCandidate | None] = [None] * len(boxes)
        jobs = []
        for i, box in enumerate(boxes):
            c = crop(image, box, pad=2)
            if c is not None and min(c.shape[:2]) >= 4:
                jobs.append((i, c))
        eos_id = processor.tokenizer.eos_token_id
        for start in range(0, len(jobs), self.batch_size):
            chunk = jobs[start : start + self.batch_size]
            try:
                pil = [Image.fromarray(c).convert("RGB") for _, c in chunk]
                pv = processor(images=pil, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    out = model.generate(pv, max_new_tokens=64, output_scores=True, return_dict_in_generate=True)
                texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
                trans = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            except Exception as e:
                raise InferenceError(f"TrOCR failed: {e}") from e
            for k, (i, _) in enumerate(chunk):
                text = texts[k].strip()
                if not text:
                    continue
                generated = out.sequences[k, 1:].tolist()  # the first one is the decoder start token
                lps = token_logprobs(generated, trans[k].tolist(), eos_id)
                results[i] = RecognitionCandidate(text=text, confidence=sequence_confidence(lps), engine=self.info.name)
        return results
