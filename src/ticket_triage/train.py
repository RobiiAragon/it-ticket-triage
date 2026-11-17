"""Fine-tune all-MiniLM-L6-v2 with two heads (category, priority) and export to ONNX.

Needs the ``[train]`` extra (torch + transformers). Inference never imports this module.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np

from .data import Example
from .taxonomy import CATEGORY_LIST, PRIORITIES

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit('training needs PyTorch: pip install -e ".[train]"') from exc


class TriageModel(nn.Module):
    """MiniLM encoder + mean pooling + two linear heads. Outputs softmax probabilities."""

    def __init__(
        self, encoder: nn.Module, hidden: int, n_cat: int, n_pri: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.category = nn.Linear(hidden, n_cat)
        self.priority = nn.Linear(hidden, n_pri)

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = self.dropout((out * mask).sum(1) / mask.sum(1).clamp(min=1e-6))
        return self.category(pooled), self.priority(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        c, p = self.logits(input_ids, attention_mask)
        return c.softmax(-1), p.softmax(-1)


def _encode(tokenizer, texts: list[str], max_len: int) -> dict[str, torch.Tensor]:
    enc = tokenizer.encode_batch(texts, max_len)
    return {
        "input_ids": torch.from_numpy(enc["input_ids"]),
        "attention_mask": torch.from_numpy(enc["attention_mask"]),
    }


def fine_tune(
    train: list[Example],
    val: list[Example],
    out_dir: Path,
    vocab: Path,
    epochs: int = 4,
    batch: int = 32,
    lr: float = 5e-5,
    max_len: int = 160,
    seed: int = 0,
    device: str | None = None,
) -> dict:
    from transformers import AutoModel

    from .tokenizer import WordPieceTokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = WordPieceTokenizer.from_file(vocab)
    encoder = AutoModel.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = TriageModel(
        encoder, encoder.config.hidden_size, len(CATEGORY_LIST), len(PRIORITIES)
    ).to(dev)
    cat_idx = {c: i for i, c in enumerate(CATEGORY_LIST)}
    pri_idx = {p: i for i, p in enumerate(PRIORITIES)}
    y_cat = torch.tensor([cat_idx[e.category] for e in train])
    y_pri = torch.tensor([pri_idx[e.priority] for e in train])
    # Priority classes are imbalanced by construction (the ITIL matrix): weight the loss.
    counts = torch.bincount(y_pri, minlength=len(PRIORITIES)).float()
    pri_w = (counts.sum() / (len(PRIORITIES) * counts.clamp(min=1))).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(train) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps)
    )
    ce = nn.CrossEntropyLoss()
    ce_pri = nn.CrossEntropyLoss(weight=pri_w)
    history = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train))
        total = 0.0
        for i in range(0, len(train), batch):
            idx = perm[i : i + batch]
            feats = _encode(tok, [train[j].text for j in idx.tolist()], max_len)
            feats = {k: v.to(dev) for k, v in feats.items()}
            lc, lp = model.logits(**feats)
            loss = ce(lc, y_cat[idx].to(dev)) + ce_pri(lp, y_pri[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        acc_c, acc_p = _val_accuracy(model, tok, val, max_len, dev, cat_idx, pri_idx)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total / len(train),
                "val_category_acc": acc_c,
                "val_priority_acc": acc_p,
            }
        )
        print(
            f"  epoch {epoch + 1}: loss {total / len(train):.4f}  val cat {acc_c:.3f}  "
            f"val pri {acc_p:.3f}"
        )
    seconds = time.perf_counter() - t0
    model.eval().cpu()
    out_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(model, out_dir / "triage.fp32.onnx", max_len)
    meta = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "max_len": max_len,
        "categories": CATEGORY_LIST,
        "priorities": PRIORITIES,
        "epochs": epochs,
        "lr": lr,
        "batch": batch,
        "device": dev,
        "train_seconds": seconds,
        "history": history,
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


@torch.no_grad()
def _val_accuracy(model, tok, val, max_len, dev, cat_idx, pri_idx):
    model.eval()
    hits_c = hits_p = 0
    for i in range(0, len(val), 64):
        chunk = val[i : i + 64]
        feats = {k: v.to(dev) for k, v in _encode(tok, [e.text for e in chunk], max_len).items()}
        pc, pp = model(**feats)
        hits_c += int(
            (pc.argmax(-1).cpu() == torch.tensor([cat_idx[e.category] for e in chunk])).sum()
        )
        hits_p += int(
            (pp.argmax(-1).cpu() == torch.tensor([pri_idx[e.priority] for e in chunk])).sum()
        )
    return hits_c / len(val), hits_p / len(val)


def export_onnx(model: TriageModel, path: Path, max_len: int) -> None:
    import onnx

    ids = torch.ones(1, 16, dtype=torch.long)
    mask = torch.ones(1, 16, dtype=torch.long)
    torch.onnx.export(
        model,
        (ids, mask),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["category_probs", "priority_probs"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "category_probs": {0: "batch"},
            "priority_probs": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    proto = onnx.load(str(path), load_external_data=True)
    onnx.save_model(proto, str(path), save_as_external_data=False)
    for leftover in path.parent.glob(path.name + ".data"):
        leftover.unlink()


def quantize_variants(
    fp32: Path, calib_texts: list[str], vocab: Path, max_len: int
) -> dict[str, Path]:
    """INT8 dynamic (CPU-friendly) and static QDQ W8A16 (the NPU-style recipe)."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from .tokenizer import WordPieceTokenizer

    out = {}
    dyn = fp32.with_name("triage.int8.onnx")
    quantize_dynamic(
        str(fp32),
        str(dyn),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],
    )
    out["int8"] = dyn

    tok = WordPieceTokenizer.from_file(vocab)
    batches = [tok.encode_batch([t], max_len) for t in calib_texts]

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter(
                [
                    {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
                    for b in batches
                ]
            )

        def get_next(self):
            return next(self.it, None)

    pre = fp32.with_name("triage.pre.onnx")
    # Symbolic shape inference cannot resolve the attention reshapes; ONNX inference suffices.
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    qdq = fp32.with_name("triage.w8a16.onnx")
    quantize_static(
        str(pre),
        str(qdq),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],
        extra_options={"UseQDQContribOps": True},
    )
    pre.unlink(missing_ok=True)
    out["w8a16"] = qdq
    return out
