"""Fixtures : un vrai mini-modèle ONNX (mêmes entrées/sorties que le modèle réel).

Pas besoin de PyTorch ni des vrais poids : le graphe est construit avec onnx.helper.
  logits = GlobalAveragePool(input) @ W + b      (N, 7)
  cam    = Sigmoid(ReduceMean(input, axe canal)) (N, S, S)
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
INPUT_SIZE = 64


def build_tiny_onnx(path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(size=(3, 7)).astype(np.float32), "W")
    b = numpy_helper.from_array(np.zeros(7, np.float32), "b")
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"]),
        helper.make_node("MatMul", ["flat", "W"], ["mm"]),
        helper.make_node("Add", ["mm", "b"], ["logits"]),
        helper.make_node("ReduceMean", ["input"], ["rm"], axes=[1], keepdims=0),
        helper.make_node("Sigmoid", ["rm"], ["cam"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, INPUT_SIZE, INPUT_SIZE])],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 7]),
            helper.make_tensor_value_info("cam", TensorProto.FLOAT, ["N", INPUT_SIZE, INPUT_SIZE]),
        ],
        initializer=[w, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    build_tiny_onnx(d / "model.onnx")
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "architecture": "tiny",
                "version": "test-1",
                "framework": "ONNX",
                "input_size": INPUT_SIZE,
                "n_parameters": 28,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "classes": CLASSES,
                "class_names": {c: c.upper() for c in CLASSES},
                "malignant_classes": ["akiec", "bcc", "mel"],
                "metrics": {"test": {"balanced_accuracy": 0.8, "f1_macro": 0.75}},
            }
        )
    )
    return d


@pytest.fixture()
def client(model_dir):
    app = create_app(Settings(model_dir=model_dir, max_upload_mb=1))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_without_model(tmp_path):
    app = create_app(Settings(model_dir=tmp_path))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def jpeg_bytes():
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
