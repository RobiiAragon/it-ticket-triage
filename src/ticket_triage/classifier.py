"""ONNX Runtime inference: tokenizer (pure Python) + exported model. No torch needed."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import providers
from .taxonomy import PRIORITY_NAMES
from .tokenizer import WordPieceTokenizer

VARIANTS = {"fp32": "triage.fp32.onnx", "int8": "triage.int8.onnx", "w8a16": "triage.w8a16.onnx"}


@dataclass
class Prediction:
    category: str
    category_confidence: float
    priority: str
    priority_name: str
    priority_confidence: float
    category_probs: dict[str, float]
    priority_probs: dict[str, float]
    latency_ms: float


class TicketClassifier:
    def __init__(self, models_dir: Path, variant: str = "fp32", provider: str = "cpu") -> None:
        meta = json.loads((models_dir / "model_meta.json").read_text(encoding="utf-8"))
        self.categories: list[str] = meta["categories"]
        self.priorities: list[str] = meta["priorities"]
        self.max_len: int = meta["max_len"]
        self.variant = variant
        self.path = models_dir / VARIANTS[variant]
        self.tokenizer = WordPieceTokenizer.from_file(models_dir / "vocab.txt")
        # Accelerator EPs (CoreML, QNN, NNAPI) need static shapes: batch 1, padded to max_len.
        self.static = provider not in ("cpu", "xnnpack", "cuda")
        dims = {"batch": 1, "seq": self.max_len} if self.static else None
        self.session = providers.session(str(self.path), provider, dims)

    def _encode(self, texts: list[str]) -> dict[str, np.ndarray]:
        enc = self.tokenizer.encode_batch(texts, self.max_len)
        if self.static:
            pad = self.max_len - enc["input_ids"].shape[1]
            enc = {k: np.pad(v, ((0, 0), (0, pad))) for k, v in enc.items()}
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def probabilities(self, texts: list[str], batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
        batch = 1 if self.static else batch
        cats, pris = [], []
        for i in range(0, len(texts), batch):
            c, p = self.session.run(None, self._encode(texts[i : i + batch]))
            cats.append(c)
            pris.append(p)
        return np.concatenate(cats), np.concatenate(pris)

    def predict(self, text: str) -> Prediction:
        t0 = time.perf_counter()
        c, p = self.probabilities([text])
        ms = (time.perf_counter() - t0) * 1000
        ci, pi = int(c[0].argmax()), int(p[0].argmax())
        return Prediction(
            self.categories[ci],
            float(c[0, ci]),
            self.priorities[pi],
            PRIORITY_NAMES[self.priorities[pi]],
            float(p[0, pi]),
            {k: round(float(v), 4) for k, v in zip(self.categories, c[0], strict=True)},
            {k: round(float(v), 4) for k, v in zip(self.priorities, p[0], strict=True)},
            ms,
        )
