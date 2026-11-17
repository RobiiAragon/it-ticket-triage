"""Accuracy, per-class precision/recall/F1, macro-F1 and confusion matrices (NumPy only)."""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        m[idx[t], idx[p]] += 1
    return m


def report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    m = confusion(y_true, y_pred, labels)
    tp = np.diag(m).astype(float)
    prec = np.divide(tp, m.sum(0), out=np.zeros_like(tp), where=m.sum(0) > 0)
    rec = np.divide(tp, m.sum(1), out=np.zeros_like(tp), where=m.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    present = m.sum(1) > 0  # macro average over classes that occur in y_true
    return {
        "accuracy": float(tp.sum() / m.sum()) if m.sum() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            c: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(m[i].sum()),
            }
            for i, c in enumerate(labels)
        },
        "labels": labels,
        "confusion": m.tolist(),
        "n": int(m.sum()),
    }


def within_one(y_true: list[str], y_pred: list[str], order: list[str]) -> float:
    """Share of priority predictions at most one level off (P2 for a P1 is 'near miss')."""
    pos = {p: i for i, p in enumerate(order)}
    return float(np.mean([abs(pos[t] - pos[p]) <= 1 for t, p in zip(y_true, y_pred, strict=True)]))
