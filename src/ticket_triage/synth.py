"""Generate a documented synthetic ticket dataset with a local LLM (llama.cpp / Ollama).

Every ticket is generated from explicit attributes (category, issue, impact, urgency,
channel, role, tone); labels are therefore known by construction, and the priority label
comes from the ITIL matrix in ``taxonomy``. Generation is seeded and resumable.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .taxonomy import CATEGORIES, IMPACT, URGENCY, priority

# Several phrasings per level so the generator cannot learn one fixed cue to copy.
IMPACT_HINTS = {
    "low": [
        IMPACT["low"],
        "it is just the writer's own computer or account",
        "nobody else seems to have the problem",
    ],
    "medium": [
        IMPACT["medium"],
        "the writer's whole team or shift has the same problem",
        "a department of roughly a dozen colleagues is stuck",
    ],
    "high": [
        IMPACT["high"],
        "a full production line or the whole plant is affected",
        "customers or shipments are affected, not just internal staff",
    ],
}
URGENCY_HINTS = {
    "low": [
        URGENCY["low"],
        "there is a workaround and it can be handled later this week",
        "it is annoying but not stopping anything",
    ],
    "medium": [
        URGENCY["medium"],
        "the writer cannot finish today's tasks without a fix",
        "work is blocked for the rest of the day",
    ],
    "high": [
        URGENCY["high"],
        "everything has stopped and every minute costs money",
        "there may be a security breach or a safety risk happening right now",
    ],
}

CHANNELS = ["email", "self-service portal form", "chat message", "phone call summary"]
ROLES = [
    "machine operator",
    "quality engineer",
    "accountant",
    "HR specialist",
    "plant manager",
    "software developer",
    "warehouse supervisor",
    "sales representative",
    "intern",
    "executive assistant",
    "maintenance technician",
    "purchasing agent",
]
TONES = ["polite", "frustrated", "terse", "anxious", "very detailed", "casual"]
SYSTEM = (
    "You write realistic IT helpdesk tickets submitted by employees of a manufacturing "
    "company. Write in English. Output exactly two parts: a line starting with "
    "'Subject:' and then the ticket body. Never use the words impact, urgency, "
    "priority, severity, P1, P2, P3 or P4."
)
FORBIDDEN = re.compile(
    r"\b(impact|urgency|priority|severity|P[1-4])\b|who is affected|time pressure|^problem:",
    re.I | re.M,
)


@dataclass
class Spec:
    id: str
    category: str
    issue: str
    impact: str
    urgency: str
    priority: str
    channel: str
    role: str
    tone: str
    sentences: int
    seed: int


def make_specs(n: int, seed: int = 7) -> list[Spec]:
    """Balanced over categories; impact and urgency uniform, so priority follows the matrix."""
    rng = random.Random(seed)
    cats = list(CATEGORIES)
    specs = []
    for i in range(n):
        cat = cats[i % len(cats)]
        imp, urg = rng.choice(list(IMPACT)), rng.choice(list(URGENCY))
        specs.append(
            Spec(
                id=f"syn-{i:05d}",
                category=cat,
                issue=rng.choice(CATEGORIES[cat]),
                impact=imp,
                urgency=urg,
                priority=priority(imp, urg),
                channel=rng.choice(CHANNELS),
                role=rng.choice(ROLES),
                tone=rng.choice(TONES),
                sentences=rng.randint(2, 6),
                seed=rng.randint(0, 2**31 - 1),
            )
        )
    return specs


def hints(spec: Spec) -> tuple[str, str]:
    rng = random.Random(spec.seed)
    return rng.choice(IMPACT_HINTS[spec.impact]), rng.choice(URGENCY_HINTS[spec.urgency])


def prompt(spec: Spec) -> str:
    impact, urgency = hints(spec)
    return (
        f"Write a ticket sent as a {spec.channel} by a {spec.role} who sounds {spec.tone}.\n"
        f"Situation (do not copy this wording): {spec.issue}; {impact}; {urgency}.\n"
        "Write it the way a real employee would, in your own words, as one natural message "
        "(no headings or labels). Make it clear from concrete details how many people are "
        "affected and how much time pressure there is. Invent realistic specifics (application "
        f"names, error messages, building or line numbers). Body: about {spec.sentences} "
        "sentences."
    )


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "is",
        "it",
        "in",
        "on",
        "for",
        "be",
        "can",
        "not",
        "no",
        "just",
        "that",
        "this",
        "with",
        "its",
        "has",
        "have",
        "are",
        "was",
    ]
)
_NUMERIC_HINT = re.compile(r"5\s*(?:-|\u2013|to)\s*30")  # 5-30, 5\u201330, 5 to 30


def leaks_label(spec: Spec, text: str) -> bool:
    """Stricter anti-shortcut check used when building the splits: the ticket reuses any
    3-word run (with >= 2 content words) of the impact/urgency hints, or the "5-30 people"
    range. Such phrases map 1:1 to a label, so a model would learn the phrase, not the
    situation."""
    impact, urgency = hints(spec)
    hint = {g for g in _ngrams(f"{impact} ; {urgency}", 3) if sum(w not in _STOP for w in g) >= 2}
    return bool(hint & _ngrams(text, 3)) or bool(_NUMERIC_HINT.search(text))


def spec_from_record(rec: dict) -> Spec:
    return Spec(**{k: rec[k] for k in Spec.__dataclass_fields__})


def copies_prompt(spec: Spec, text: str) -> bool:
    """True if the output reuses any 5-word run of the hidden instructions (label leakage)."""
    impact, urgency = hints(spec)
    return bool(_ngrams(f"{spec.issue} {impact} {urgency}") & _ngrams(text))


def parse(text: str) -> tuple[str, str] | None:
    text = text.strip()
    m = re.match(r"(?is)^\**subject:\**\s*(.+?)\n+(.+)$", text)
    if not m:
        return None
    subject, body = m.group(1).strip().strip("*"), m.group(2).strip()
    if len(body) < 60 or len(subject) < 5 or FORBIDDEN.search(subject + " " + body):
        return None
    return subject, body


def _call(url: str, model: str, spec: Spec, attempt: int, timeout: float) -> str:
    payload = {
        "model": model,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 320,
        "seed": spec.seed + attempt,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt(spec)},
        ],
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def generate(
    out: Path,
    n: int,
    url: str,
    model: str = "local",
    workers: int = 8,
    seed: int = 7,
    retries: int = 3,
    timeout: float = 120,
) -> dict[str, int]:
    """Append tickets to ``out`` (JSONL). Already generated ids are skipped (resumable)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
    todo = [s for s in make_specs(n, seed) if s.id not in done]
    lock = threading.Lock()
    stats = {"written": 0, "rejected": 0, "failed": 0}
    t0 = time.perf_counter()

    def work(spec: Spec) -> None:
        for attempt in range(retries):
            parsed = parse(_call(url, model, spec, attempt, timeout))
            if parsed and not copies_prompt(spec, " ".join(parsed)):
                subject, body = parsed
                rec = {
                    **asdict(spec),
                    "subject": subject,
                    "body": body,
                    "text": f"{subject}\n{body}",
                    "generator": f"llama.cpp:{model}",
                }
                with lock, out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                return
            with lock:
                stats["rejected"] += 1
        with lock:
            stats["failed"] += 1

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(work, s) for s in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 200 == 0:
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(todo)} ({rate:.1f}/s) {stats}", flush=True)
    return stats
