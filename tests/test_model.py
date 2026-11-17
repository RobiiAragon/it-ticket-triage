from __future__ import annotations

import json

import numpy as np
import pytest

from ticket_triage import metrics
from ticket_triage.classifier import TicketClassifier
from ticket_triage.data import Example
from ticket_triage.evaluate import score
from ticket_triage.taxonomy import CATEGORY_LIST, PRIORITIES


def test_metrics_match_sklearn() -> None:
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = list(rng.choice(CATEGORY_LIST, 300))
    p = [t if rng.random() < 0.7 else rng.choice(CATEGORY_LIST) for t in y]
    r = metrics.report(y, p, CATEGORY_LIST)
    assert r["accuracy"] == pytest.approx(sk.accuracy_score(y, p))
    assert r["macro_f1"] == pytest.approx(sk.f1_score(y, p, average="macro"))
    assert np.array_equal(r["confusion"], sk.confusion_matrix(y, p, labels=CATEGORY_LIST))
    assert metrics.within_one(["P1", "P1", "P4"], ["P2", "P3", "P4"], PRIORITIES) == pytest.approx(
        2 / 3
    )


def test_classifier_contract(tiny_models) -> None:
    clf = TicketClassifier(tiny_models, "fp32")
    p = clf.predict("The printer on line 4 is down for all users, urgent!")
    assert p.category in CATEGORY_LIST and p.priority in PRIORITIES
    assert sum(p.category_probs.values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(p.priority_probs.values()) == pytest.approx(1.0, abs=1e-3)
    c, pr = clf.probabilities(["vpn down"] * 40, batch=16)
    assert c.shape == (40, len(CATEGORY_LIST)) and pr.shape == (40, len(PRIORITIES))


def test_score_produces_reports(tiny_models) -> None:
    clf = TicketClassifier(tiny_models, "fp32")
    test = [
        Example(str(i), "my password is locked", CATEGORY_LIST[i % 7], PRIORITIES[i % 4])
        for i in range(30)
    ]
    r = score(clf, test, test[:5])
    assert r["test_category"]["n"] == 30 and r["external_category"]["n"] == 5
    assert r["latency_ms"]["p50"] > 0 and r["size_mb"] > 0


def test_baseline_learns_keywords() -> None:
    pytest.importorskip("sklearn")
    from ticket_triage.baseline import train_baseline

    train = [
        Example(f"{i}", f"{w} problem number {i}", c, p)
        for i in range(60)
        for w, c, p in [
            ("printer jam toner", "printing", "P4"),
            ("phishing link clicked", "security", "P1"),
        ]
    ]
    heads = train_baseline(train)
    assert heads["category"].predict(["toner and printer jam again"])[0] == "printing"
    assert heads["priority"].predict(["phishing link clicked"])[0] == "P1"


def test_api(tiny_models, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from ticket_triage.api import create_app

    queue = tmp_path / "q.jsonl"
    queue.write_text(
        json.dumps({"id": "1", "text": "vpn down", "category": "network", "priority": "P2"}) + "\n"
    )
    client = TestClient(create_app(TicketClassifier(tiny_models), tmp_path / "none.json", queue))
    assert "IT Ticket Triage" in client.get("/").text
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/classify", json={"text": "printer jam on line 2"}).json()
    assert r["priority"] in PRIORITIES and r["latency_ms"] >= 0
    assert (
        len(client.post("/classify/batch", json={"texts": ["a b c d e", "vpn down"]}).json()) == 2
    )
    assert client.post("/classify", json={"text": "hi"}).status_code == 422
    assert client.get("/queue").json()[0]["prediction"]["category"] in CATEGORY_LIST
    assert client.get("/metrics").status_code == 404


def test_cli_classify(tiny_models, capsys) -> None:
    from ticket_triage.cli import main

    assert (
        main(
            [
                "classify",
                "--models-dir",
                str(tiny_models),
                "--variant",
                "fp32",
                "--json",
                "printer",
                "jam",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["category"] in CATEGORY_LIST
