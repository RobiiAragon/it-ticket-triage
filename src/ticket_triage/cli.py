"""``triage`` command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from . import __version__

MODELS = Path("models")
SPLITS = Path("data/splits")
SYNTH = Path("data/synthetic/tickets.jsonl")
RELEASE = "https://github.com/RobiiAragon/it-ticket-triage/releases/download/v{version}/{name}"


def _cache() -> Path:
    import os

    return Path(os.environ.get("TRIAGE_CACHE", Path.home() / ".cache" / "it-ticket-triage"))


def cmd_generate(a: argparse.Namespace) -> int:
    from .synth import generate

    stats = generate(a.out, a.n, a.url, a.model, a.workers)
    print(stats)
    return 0


def cmd_prepare(a: argparse.Namespace) -> int:
    from . import data

    stats: dict[str, int] = {}
    examples = data.load_synthetic(a.synthetic, stats=stats)
    print(f"synthetic: {stats}")
    a.splits.mkdir(parents=True, exist_ok=True)
    (a.splits / "prepare_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    splits = data.stratified_split(examples)
    for name, items in splits.items():
        data.save_jsonl(a.splits / f"{name}.jsonl", items)
        print(f"{name}: {len(items)} {data.describe(items)}")
    ext = data.load_external(data.download_kameronb(_cache()))
    data.save_jsonl(a.splits / "external.jsonl", ext)
    print(f"external: {len(ext)} {data.describe(ext)['category']}")
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from huggingface_hub import hf_hub_download

    from . import data
    from .train import BASE_MODEL, BASE_REVISION, fine_tune, quantize_variants

    vocab = Path(hf_hub_download(BASE_MODEL, "vocab.txt", revision=BASE_REVISION))
    a.models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(vocab, a.models_dir / "vocab.txt")
    train, val = data.load_jsonl(a.splits / "train.jsonl"), data.load_jsonl(a.splits / "val.jsonl")
    meta = fine_tune(train, val, a.models_dir, a.models_dir / "vocab.txt", epochs=a.epochs)
    quantize_variants(
        a.models_dir / "triage.fp32.onnx",
        [e.text for e in train[:200]],
        a.models_dir / "vocab.txt",
        meta["max_len"],
    )
    manifest = {
        p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in sorted(a.models_dir.glob("*.onnx"))
    }
    (a.models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_download(a: argparse.Namespace) -> int:
    manifest = json.loads((a.models_dir / "manifest.json").read_text())
    wanted = list(manifest) if a.variant == "all" else [f"triage.{a.variant}.onnx"]
    for name in wanted:
        target = a.models_dir / name
        info = manifest[name]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == info["sha256"]:
            print(f"ok   {name}")
            continue
        url = RELEASE.format(version=__version__, name=name)
        print(f"get  {name} ({info['bytes'] / 1024**2:.1f} MB) from {url}")
        with urllib.request.urlopen(url, timeout=300) as resp:
            blob = resp.read()
        if hashlib.sha256(blob).hexdigest() != info["sha256"]:
            raise SystemExit(f"checksum mismatch for {name}")
        target.write_bytes(blob)
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    from . import data, device_info, providers
    from .classifier import VARIANTS, TicketClassifier
    from .evaluate import score, score_baseline

    test = data.load_jsonl(a.splits / "test.jsonl")
    ext_path = a.splits / "external.jsonl"
    external = data.load_jsonl(ext_path) if ext_path.exists() and not a.no_external else []
    usable, skipped = providers.resolve(a.providers)
    for alias, why in skipped.items():
        print(f"[skip] {alias}: {why}")
    variants = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    result: dict = {
        "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "version": __version__,
        "device": device_info.collect(),
        "n_test": len(test),
        "n_external": len(external),
        "variants": {},
    }
    for v in variants:
        if not (a.models_dir / VARIANTS[v]).exists():
            print(f"[skip] {v}: {VARIANTS[v]} missing (run `triage download-models`)")
            continue
        for ep in usable:
            r = score(TicketClassifier(a.models_dir, v, ep), test, external)
            result["variants"].setdefault(v, {})[ep] = r
            ext = r.get("external_category", {}).get("accuracy")
            print(
                f"{v:<6} {ep:<7} cat {r['test_category']['accuracy']:.3f} "
                f"(F1 {r['test_category']['macro_f1']:.3f})  "
                f"pri {r['test_priority']['accuracy']:.3f} "
                f"(±1 {r['test_priority_within_one']:.3f})  ext cat "
                f"{'n/a' if ext is None else f'{ext:.3f}'}  p50 {r['latency_ms']['p50']:.2f} ms"
            )
    if a.baseline:
        from .baseline import train_baseline

        heads = train_baseline(data.load_jsonl(a.splits / "train.jsonl"))
        b = score_baseline(heads, test, external)
        result["baseline"] = b
        print(
            f"tfidf  sklearn cat {b['test_category']['accuracy']:.3f}  pri "
            f"{b['test_priority']['accuracy']:.3f}  ext cat "
            f"{b.get('external_category', {}).get('accuracy', float('nan')):.3f}"
        )
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


def cmd_classify(a: argparse.Namespace) -> int:
    from .classifier import TicketClassifier

    text = " ".join(a.text) if a.text else sys.stdin.read()
    p = TicketClassifier(a.models_dir, a.variant, a.provider).predict(text)
    if a.json:
        print(json.dumps(asdict(p), indent=2))
    else:
        print(
            f"{p.priority} ({p.priority_name}, {p.priority_confidence:.0%})  "
            f"{p.category} ({p.category_confidence:.0%})  [{p.latency_ms:.1f} ms]"
        )
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app
    from .classifier import TicketClassifier

    clf = TicketClassifier(a.models_dir, a.variant, a.provider)
    app = create_app(clf, a.metrics, a.queue)
    print(f"IT Ticket Triage on http://{a.host}:{a.port}  (API docs at /docs)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_providers(_: argparse.Namespace) -> int:
    from . import providers

    rt = providers.ort()
    print(f"onnxruntime {rt.__version__}: {', '.join(rt.get_available_providers())}")
    _, skipped = providers.resolve(",".join(providers.EP_ALIASES))
    for alias, why in skipped.items():
        print(f"  [skip] {alias}: {why}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="triage", description="IT ticket triage (category + priority)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic tickets with a local LLM server")
    g.add_argument("--n", type=int, default=6000)
    g.add_argument("--url", default="http://127.0.0.1:8080/v1")
    g.add_argument("--model", default="local")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--out", type=Path, default=SYNTH)
    g.set_defaults(func=cmd_generate)

    pr = sub.add_parser("prepare", help="dedupe + stratified split; download the external set")
    pr.add_argument("--synthetic", type=Path, default=SYNTH)
    pr.add_argument("--splits", type=Path, default=SPLITS)
    pr.set_defaults(func=cmd_prepare)

    t = sub.add_parser("train", help="fine-tune MiniLM, export ONNX, quantize ([train] extra)")
    t.add_argument("--splits", type=Path, default=SPLITS)
    t.add_argument("--models-dir", type=Path, default=MODELS)
    t.add_argument("--epochs", type=int, default=10)
    t.set_defaults(func=cmd_train)

    d = sub.add_parser("download-models", help="fetch the released ONNX models (SHA-256 checked)")
    d.add_argument("--variant", default="int8", choices=["all", "fp32", "int8", "w8a16"])
    d.add_argument("--models-dir", type=Path, default=MODELS)
    d.set_defaults(func=cmd_download)

    e = sub.add_parser("evaluate", help="metrics + latency per variant and execution provider")
    e.add_argument("--splits", type=Path, default=SPLITS)
    e.add_argument("--models-dir", type=Path, default=MODELS)
    e.add_argument("--variants", default="all")
    e.add_argument("--providers", default="cpu,coreml,qnn,cuda,nnapi,xnnpack")
    e.add_argument("--baseline", action="store_true", help="also train/score TF-IDF + LogReg")
    e.add_argument("--no-external", action="store_true")
    e.add_argument("--out", type=Path)
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("classify", help="classify one ticket (argument or stdin)")
    c.add_argument("text", nargs="*")
    c.add_argument("--models-dir", type=Path, default=MODELS)
    c.add_argument("--variant", default="int8", choices=["fp32", "int8", "w8a16"])
    c.add_argument("--provider", default="cpu")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_classify)

    s = sub.add_parser("serve", help="FastAPI + triage dashboard")
    s.add_argument("--models-dir", type=Path, default=MODELS)
    s.add_argument("--variant", default="int8", choices=["fp32", "int8", "w8a16"])
    s.add_argument("--provider", default="cpu")
    s.add_argument("--metrics", type=Path, default=Path("results/mac-m5max.json"))
    s.add_argument("--queue", type=Path, default=Path("data/demo_queue.jsonl"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    sub.add_parser("providers", help="list ONNX Runtime execution providers").set_defaults(
        func=cmd_providers
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
