from __future__ import annotations

import itertools
import json
import socketserver
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ticket_triage import data, synth
from ticket_triage.taxonomy import IMPACT, PRIORITY_MATRIX, URGENCY, priority


def test_itil_matrix_is_complete_and_monotonic() -> None:
    assert set(PRIORITY_MATRIX) == {(i, u) for i in IMPACT for u in URGENCY}
    assert priority("high", "high") == "P1" and priority("low", "low") == "P4"
    order = ["low", "medium", "high"]
    rank = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    for i in order:  # raising impact or urgency never lowers the priority
        for a, b in itertools.pairwise(order):
            assert rank[priority(i, b)] <= rank[priority(i, a)]
            assert rank[priority(b, i)] <= rank[priority(a, i)]


def test_specs_are_balanced_and_deterministic() -> None:
    specs = synth.make_specs(700, seed=1)
    assert Counter(s.category for s in specs) == Counter({c: 100 for c in synth.CATEGORIES})
    assert [s.issue for s in specs] == [s.issue for s in synth.make_specs(700, seed=1)]
    assert all(s.priority == priority(s.impact, s.urgency) for s in specs)


def test_parse_and_leakage_filters() -> None:
    assert synth.parse(
        "Subject: VPN broken\n\nI cannot connect to the VPN from home since "
        "this morning and my whole team is waiting on me."
    )
    assert synth.parse("no subject line here") is None
    assert (
        synth.parse(
            "Subject: x\n\nThis is P1 and very urgent for the whole factory "
            "floor right now please help"
        )
        is None
    )
    spec = synth.make_specs(1)[0]
    impact, _ = synth.hints(spec)
    assert synth.copies_prompt(spec, f"Hello, {impact} and that is all I can say today.")
    assert not synth.copies_prompt(spec, "Totally unrelated words about printers and toner.")


class _FakeLLM(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        _FakeLLM.calls += 1
        self.rfile.read(int(self.headers["Content-Length"]))
        # every other call returns an invalid reply to exercise the retry path
        content = (
            "Subject: Printer on line 4 prints blank labels\n\nThe Zebra printer next to "
            "packing station 4 has printed blank labels since 7am and we ship at noon."
            if _FakeLLM.calls % 2
            else "sorry I cannot"
        )
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server(HTTPServer):
    def server_bind(self) -> None:  # skip slow reverse DNS (getfqdn) on macOS
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def test_generate_is_resumable(tmp_path) -> None:
    srv = _Server(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    out = tmp_path / "t.jsonl"
    try:
        s1 = synth.generate(out, 6, url, workers=1, retries=3)
        assert s1["written"] == 6 and s1["rejected"] >= 1
        s2 = synth.generate(out, 6, url, workers=1)
        assert s2["written"] == 0  # nothing left to do
    finally:
        srv.shutdown()
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert {r["id"] for r in recs} == {f"syn-{i:05d}" for i in range(6)}
    assert all(r["text"].startswith("Printer on line 4") for r in recs)


def test_load_dedupes_and_split_is_stratified(tmp_path) -> None:
    lines = []
    for i, s in enumerate(synth.make_specs(280, seed=3)):
        lines.append(
            json.dumps(
                {
                    "id": s.id,
                    "text": f"ticket {i} about {s.issue}",
                    "category": s.category,
                    "priority": s.priority,
                }
            )
        )
    lines.append(lines[0].replace('"syn-00000"', '"dup"'))  # same text, new id
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(lines))
    ex = data.load_synthetic(path)
    assert len(ex) == 280
    split = data.stratified_split(ex)
    ids = [set(e.id for e in split[k]) for k in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(map(len, ids)) == 280
    assert set(e.category for e in split["test"]) == set(e.category for e in ex)


def test_external_mapping(tmp_path) -> None:
    csv_path = tmp_path / "k.csv"
    csv_path.write_text(
        "number,short_description,content,category,role\n"
        "INC1,Printer offline,Floor 2 printer offline,PRINTER,customer\n"
        "INC2,Fixed printer,Replaced fuser,PRINTER,agent\n"
        "INC3,Badge broken,My PIV card fails,PIV CARD,customer\n"
        "INC4,Locked out,Account locked,ACCOUNT,customer\n"
    )
    ext = data.load_external(csv_path)
    assert sorted((e.category, e.id) for e in ext) == [
        ("access", "ext-INC4"),
        ("printing", "ext-INC1"),
    ]
    assert all(e.priority is None for e in ext)


def test_roundtrip_jsonl(tmp_path) -> None:
    ex = [data.Example("a", "text", "email", "P2"), data.Example("b", "t2", "access", None)]
    data.save_jsonl(tmp_path / "x.jsonl", ex)
    assert data.load_jsonl(tmp_path / "x.jsonl") == ex


@pytest.mark.parametrize("cat", ["access", "security"])
def test_every_category_has_issues(cat) -> None:
    assert len(synth.CATEGORIES[cat]) >= 5
