from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ticket_triage.taxonomy import CATEGORY_LIST, PRIORITIES

REPO = Path(__file__).resolve().parents[1]
VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "printer",
    "vpn",
    "password",
    "down",
    "urgent",
    "the",
    "is",
    "my",
    "line",
    "jam",
    "locked",
    "all",
    "users",
]


def _tiny_model(path: Path, vocab_size: int) -> None:
    """Embedding-bag classifier with the same I/O contract as the exported MiniLM model."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    d = 8
    inits = [
        numpy_helper.from_array(rng.normal(size=(vocab_size, d)).astype(np.float32), "emb"),
        numpy_helper.from_array(rng.normal(size=(d, len(CATEGORY_LIST))).astype(np.float32), "wc"),
        numpy_helper.from_array(rng.normal(size=(d, len(PRIORITIES))).astype(np.float32), "wp"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "ax1"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "axl"),
    ]
    nodes = [
        helper.make_node("Gather", ["emb", "input_ids"], ["e"]),
        helper.make_node("Cast", ["attention_mask"], ["m"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["m", "axl"], ["m3"]),
        helper.make_node("Mul", ["e", "m3"], ["em"]),
        helper.make_node("ReduceSum", ["em", "ax1"], ["s"], keepdims=0),
        helper.make_node("ReduceSum", ["m3", "ax1"], ["n"], keepdims=0),
        helper.make_node("Div", ["s", "n"], ["pooled"]),
        helper.make_node("MatMul", ["pooled", "wc"], ["lc"]),
        helper.make_node("Softmax", ["lc"], ["category_probs"], axis=-1),
        helper.make_node("MatMul", ["pooled", "wp"], ["lp"]),
        helper.make_node("Softmax", ["lp"], ["priority_probs"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_triage",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "t"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "t"]),
        ],
        [
            helper.make_tensor_value_info(
                "category_probs", TensorProto.FLOAT, ["b", len(CATEGORY_LIST)]
            ),
            helper.make_tensor_value_info(
                "priority_probs", TensorProto.FLOAT, ["b", len(PRIORITIES)]
            ),
        ],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture()
def tiny_models(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    (d / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    _tiny_model(d / "triage.fp32.onnx", len(VOCAB))
    (d / "model_meta.json").write_text(
        json.dumps({"categories": CATEGORY_LIST, "priorities": PRIORITIES, "max_len": 32})
    )
    return d
