"""Evaluate the ONNX variants (and the TF-IDF baseline) on the synthetic test split and the
external KameronB set; measure batch-1 latency per execution provider."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import metrics
from .classifier import TicketClassifier
from .data import Example
from .taxonomy import CATEGORY_LIST, PRIORITIES


def score(clf: TicketClassifier, test: list[Example], external: list[Example]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    c, p = clf.probabilities([e.text for e in test])
    out["batch_seconds"] = time.perf_counter() - t0
    cat_pred = [clf.categories[i] for i in c.argmax(1)]
    pri_pred = [clf.priorities[i] for i in p.argmax(1)]
    out["test_category"] = metrics.report([e.category for e in test], cat_pred, CATEGORY_LIST)
    out["test_priority"] = metrics.report([e.priority for e in test], pri_pred, PRIORITIES)
    out["test_priority_within_one"] = metrics.within_one(
        [e.priority for e in test], pri_pred, PRIORITIES
    )
    if external:
        ce, _ = clf.probabilities([e.text for e in external])
        out["external_category"] = metrics.report(
            [e.category for e in external], [clf.categories[i] for i in ce.argmax(1)], CATEGORY_LIST
        )
    samples = [e.text for e in test[:200]]
    for t in samples[:10]:
        clf.predict(t)
    lat = [clf.predict(t).latency_ms for t in samples]
    out["latency_ms"] = {
        "p50": float(np.percentile(lat, 50)),
        "p90": float(np.percentile(lat, 90)),
        "p99": float(np.percentile(lat, 99)),
    }
    out["size_mb"] = clf.path.stat().st_size / 1024**2
    return out


def score_baseline(heads, test: list[Example], external: list[Example]) -> dict[str, Any]:
    texts = [e.text for e in test]
    out = {
        "test_category": metrics.report(
            [e.category for e in test], list(heads["category"].predict(texts)), CATEGORY_LIST
        ),
        "test_priority": metrics.report(
            [e.priority for e in test], list(heads["priority"].predict(texts)), PRIORITIES
        ),
    }
    pri = list(heads["priority"].predict(texts))
    out["test_priority_within_one"] = metrics.within_one(
        [e.priority for e in test], pri, PRIORITIES
    )
    if external:
        out["external_category"] = metrics.report(
            [e.category for e in external],
            list(heads["category"].predict([e.text for e in external])),
            CATEGORY_LIST,
        )
    t = time.perf_counter()
    for s in texts[:200]:
        heads["category"].predict([s])
        heads["priority"].predict([s])
    out["latency_ms"] = {"mean": (time.perf_counter() - t) / min(200, len(texts)) * 1000}
    return out
