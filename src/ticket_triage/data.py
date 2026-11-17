"""Datasets: the generated synthetic set (train/val/test) and an external public test set.

External set: KameronB/synthetic-it-callcenter-tickets (Apache-2.0, Hugging Face), a
different generator and taxonomy. Its categories are mapped to ours where the meaning is
unambiguous; it has no priority label, so it is used for *category* generalisation only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .taxonomy import CATEGORY_LIST, PRIORITIES

KAMERONB_URL = (
    "https://huggingface.co/datasets/KameronB/synthetic-it-callcenter-tickets/"
    "resolve/11db8c3d64ffca2027afc49da9db9516a3254bbc/synthetic-it-call-center-tickets.csv"
)
KAMERONB_SHA256 = "0c5b8427ca1981c43d123abddbff7ed601d2e92cdc3cd3ba543c8b92c44ee2a3"
# PIV CARD (smart-card badges) is left out: it spans access, hardware and security.
KAMERONB_MAP = {
    "ACCOUNT": "access",
    "EMAIL": "email",
    "NETWORK": "network",
    "PRINTER": "printing",
    "SECURITY": "security",
    "HARDWARE": "hardware",
    "SOFTWARE": "software",
}


@dataclass
class Example:
    id: str
    text: str
    category: str
    priority: str | None


def load_synthetic(
    path: Path, leak_filter: bool = True, stats: dict[str, int] | None = None
) -> list[Example]:
    """Load generated tickets, dropping exact duplicates and (by default) tickets that
    reuse the generator's impact/urgency hint wording (see ``synth.leaks_label``)."""
    from .synth import leaks_label, spec_from_record

    seen: set[str] = set()
    out = []
    counts = {"read": 0, "duplicates": 0, "label_leaks": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        counts["read"] += 1
        key = " ".join(r["text"].lower().split())
        if key in seen:  # exact duplicates after whitespace/case normalisation
            counts["duplicates"] += 1
            continue
        seen.add(key)
        if leak_filter and "seed" in r and leaks_label(spec_from_record(r), r["text"]):
            counts["label_leaks"] += 1
            continue
        out.append(Example(r["id"], r["text"], r["category"], r["priority"]))
    if stats is not None:
        stats.update(counts, kept=len(out))
    return sorted(out, key=lambda e: e.id)


def stratified_split(
    examples: list[Example], val: float = 0.15, test: float = 0.15, seed: int = 0
) -> dict[str, list[Example]]:
    groups: dict[tuple[str, str | None], list[Example]] = defaultdict(list)
    for e in examples:
        groups[(e.category, e.priority)].append(e)
    rng = random.Random(seed)
    split: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for key in sorted(groups, key=str):
        items = groups[key][:]
        rng.shuffle(items)
        n_val, n_test = round(len(items) * val), round(len(items) * test)
        split["test"] += items[:n_test]
        split["val"] += items[n_test : n_test + n_val]
        split["train"] += items[n_test + n_val :]
    return split


def download_kameronb(cache: Path) -> Path:
    target = cache / "kameronb.csv"
    if not (target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == KAMERONB_SHA256):
        cache.mkdir(parents=True, exist_ok=True)
        print("downloading KameronB/synthetic-it-callcenter-tickets (36 MB, Apache-2.0)")
        with urllib.request.urlopen(KAMERONB_URL, timeout=120) as resp:
            data = resp.read()
        if hashlib.sha256(data).hexdigest() != KAMERONB_SHA256:
            raise RuntimeError("KameronB checksum mismatch")
        target.write_bytes(data)
    return target


def load_external(csv_path: Path, per_class: int = 300, seed: int = 0) -> list[Example]:
    """Customer-written tickets only (agent-written ones describe the fix, not the problem)."""
    by_cat: dict[str, list[Example]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cat = KAMERONB_MAP.get(r["category"])
            if cat is None or r["role"] != "customer":
                continue
            text = f"{r['short_description'].strip()}\n{r['content'].strip()}"
            by_cat[cat].append(Example(f"ext-{r['number']}", text, cat, None))
    rng = random.Random(seed)
    out = []
    for cat in CATEGORY_LIST:
        items = by_cat.get(cat, [])
        rng.shuffle(items)
        out += items[:per_class]
    return out


def save_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e.__dict__, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[Example]:
    return [Example(**json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines() if x]


def describe(examples: list[Example]) -> dict[str, dict[str, int]]:
    return {
        "category": dict(Counter(e.category for e in examples)),
        "priority": {p: sum(e.priority == p for e in examples) for p in PRIORITIES},
    }
