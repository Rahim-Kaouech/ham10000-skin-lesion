"""Tests des endpoints FastAPI (succès, erreurs 400 / 413 / 503)."""

import base64
import io

import numpy as np
from PIL import Image

from app.inference import resize_center_crop, softmax
from app.visualization import jet_colormap


def upload(data: bytes, name="lesion.jpg", ctype="image/jpeg"):
    return {"file": (name, data, ctype)}


# ---------------------------------------------------------------- système
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-1"
    assert isinstance(body["gpu_available"], bool)
    assert "X-Process-Time-Ms" in r.headers


def test_health_degraded_without_model(client_without_model):
    r = client_without_model.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


def test_metadata(client):
    r = client.get("/model/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["input_size"] == 64
    assert [c["code"] for c in body["classes"]][4] == "mel"
    assert body["classes"][4]["malignant"] is True
    assert body["metrics"]["test"]["balanced_accuracy"] == 0.8


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/model/metadata", "/predict", "/predict/visualize"):
        assert p in paths


# ---------------------------------------------------------------- inférence
def test_predict_returns_valid_distribution(client, jpeg_bytes):
    r = client.post("/predict", files=upload(jpeg_bytes))
    assert r.status_code == 200, r.text
    body = r.json()
    probs = [p["probability"] for p in body["probabilities"]]
    assert len(probs) == 7
    assert abs(sum(probs) - 1) < 1e-4
    assert probs == sorted(probs, reverse=True)
    assert body["prediction"] == body["probabilities"][0]
    assert body["risk_alert"] == body["prediction"]["malignant"]
    assert body["inference_ms"] >= 0 and body["server_total_ms"] >= body["inference_ms"]
    assert body["filename"] == "lesion.jpg"


def test_predict_png_image(client):
    buf = io.BytesIO()
    Image.new("RGBA", (80, 80), (200, 100, 50, 255)).save(buf, format="PNG")
    r = client.post("/predict", files=upload(buf.getvalue(), "a.png", "image/png"))
    assert r.status_code == 200


def test_visualize_png(client, jpeg_bytes):
    r = client.post("/predict/visualize", files=upload(jpeg_bytes))
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["X-Predicted-Class"] in {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (64, 64)


def test_visualize_json(client, jpeg_bytes):
    r = client.post("/predict/visualize?format=json&alpha=0.6", files=upload(jpeg_bytes))
    assert r.status_code == 200
    body = r.json()
    for key in ("image_png_base64", "heatmap_png_base64", "overlay_png_base64"):
        img = Image.open(io.BytesIO(base64.b64decode(body[key])))
        assert img.size == (64, 64)
    assert Image.open(io.BytesIO(base64.b64decode(body["heatmap_png_base64"]))).mode == "RGBA"


def test_monitoring_counts_predictions(client, jpeg_bytes):
    client.post("/predict", files=upload(jpeg_bytes))
    stats = client.get("/monitoring/stats").json()
    assert stats["total_predictions"] >= 1
    assert abs(sum(stats["predicted_class_distribution"].values()) - 1) < 1e-6


# ---------------------------------------------------------------- erreurs
def test_invalid_image_returns_400(client):
    r = client.post("/predict", files=upload(b"not an image at all", "x.jpg"))
    assert r.status_code == 400
    assert "image" in r.json()["detail"].lower()


def test_wrong_content_type_returns_400(client, jpeg_bytes):
    r = client.post("/predict", files=upload(jpeg_bytes, "x.txt", "text/plain"))
    assert r.status_code == 400


def test_empty_file_returns_400(client):
    assert client.post("/predict", files=upload(b"")).status_code == 400


def test_missing_file_returns_422(client):
    assert client.post("/predict").status_code == 422


def test_too_large_returns_413(client):
    big = b"\xff" * (1024 * 1024 + 10)  # limite à 1 Mo dans la fixture
    assert client.post("/predict", files=upload(big)).status_code == 413


def test_tiny_image_returns_400(client):
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="JPEG")
    assert client.post("/predict", files=upload(buf.getvalue())).status_code == 400


def test_model_not_loaded_returns_503(client_without_model, jpeg_bytes):
    assert client_without_model.post("/predict", files=upload(jpeg_bytes)).status_code == 503
    assert client_without_model.post("/predict/visualize", files=upload(jpeg_bytes)).status_code == 503
    assert client_without_model.get("/model/metadata").status_code == 503


# ---------------------------------------------------------------- fonctions pures
def test_resize_center_crop_is_square():
    out = resize_center_crop(Image.new("RGB", (600, 450)), 300)
    assert out.size == (300, 300)


def test_softmax_sums_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(p.sum(1), 1)
    assert np.isfinite(p).all()


def test_jet_colormap_extremes():
    c = jet_colormap(np.array([0.0, 1.0]))
    assert c.dtype == np.uint8
    assert c[0, 2] > c[0, 0]  # froid -> bleu
    assert c[1, 0] > c[1, 2]  # chaud -> rouge
