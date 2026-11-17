"""FastAPI service: classify tickets, serve the triage dashboard and evaluation metrics."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise SystemExit('the API needs FastAPI: pip install -e ".[server]"') from exc

from . import __version__
from .classifier import TicketClassifier
from .taxonomy import PRIORITY_NAMES


class Ticket(BaseModel):
    text: str = Field(min_length=5, max_length=5000)


class Batch(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


def create_app(
    clf: TicketClassifier, metrics_path: Path | None = None, queue_path: Path | None = None
) -> FastAPI:
    app = FastAPI(
        title="IT Ticket Triage",
        version=__version__,
        description="On-device ticket classification (category + ITIL priority).",
    )
    page = files("ticket_triage").joinpath("web/index.html").read_text(encoding="utf-8")
    queue: list[dict[str, Any]] = []
    if queue_path and queue_path.exists():
        for line in queue_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                queue.append({**item, "prediction": asdict(clf.predict(item["text"]))})

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return page

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "variant": clf.variant,
            "model": clf.path.name,
            "providers": clf.session.get_providers(),
            "version": __version__,
        }

    @app.post("/classify")
    def classify(ticket: Ticket) -> dict[str, Any]:
        return asdict(clf.predict(ticket.text))

    @app.post("/classify/batch")
    def classify_batch(batch: Batch) -> list[dict[str, Any]]:
        if any(len(t) > 5000 for t in batch.texts):
            raise HTTPException(422, "each text must be at most 5000 characters")
        c, p = clf.probabilities(batch.texts)
        return [
            {
                "category": clf.categories[int(ci.argmax())],
                "category_confidence": float(ci.max()),
                "priority": clf.priorities[int(pi.argmax())],
                "priority_name": PRIORITY_NAMES[clf.priorities[int(pi.argmax())]],
                "priority_confidence": float(pi.max()),
            }
            for ci, pi in zip(c, p, strict=True)
        ]

    @app.get("/queue")
    def get_queue() -> list[dict[str, Any]]:
        return queue

    @app.get("/metrics")
    def get_metrics() -> dict[str, Any]:
        if not metrics_path or not metrics_path.exists():
            raise HTTPException(404, "no evaluation results; run `triage evaluate --out ...`")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    return app
