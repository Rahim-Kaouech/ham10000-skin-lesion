"""Wrapper de chargement et d'inférence du modèle ONNX (ONNX Runtime).

Le modèle exporté par `src/models/evaluate.py` possède deux sorties :
  - `logits` (N, K)
  - `cam`    (N, h, w) : Grad-CAM de la classe prédite, normalisée dans [0, 1]
Le prétraitement reproduit exactement celui de la validation :
Resize(côté court = input_size) -> CenterCrop(input_size) -> Normalize(ImageNet).
"""

from __future__ import annotations

import io
import json
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)


class InvalidImageError(ValueError):
    """Le fichier reçu n'est pas une image exploitable (-> HTTP 400)."""


class ModelNotLoadedError(RuntimeError):
    """Le modèle n'est pas (encore) disponible (-> HTTP 503)."""


@dataclass
class InferenceResult:
    probabilities: np.ndarray  # (K,)
    cam: np.ndarray  # (h, w) dans [0, 1]
    image: Image.Image  # image recadrée vue par le modèle
    preprocessing_ms: float
    inference_ms: float


def decode_image(data: bytes, min_side: int = 32) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # détecte les fichiers tronqués/corrompus
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError("Le fichier n'est pas une image valide.") from exc
    if min(img.size) < min_side:
        raise InvalidImageError(f"Image trop petite (côté minimal {min_side}px).")
    return img


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class ModelService:
    def __init__(self, model_path: Path, metadata_path: Path, variant: str = "fp32"):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.variant = variant
        self.session: ort.InferenceSession | None = None
        self.metadata: dict = {}
        self.stats = PredictionStats()

    # ------------------------------------------------------------------ chargement
    @staticmethod
    def available_providers() -> list[str]:
        return ort.get_available_providers()

    @classmethod
    def gpu_available(cls) -> bool:
        return "CUDAExecutionProvider" in cls.available_providers()

    def load(self) -> None:
        if not self.model_path.exists() or not self.metadata_path.exists():
            raise ModelNotLoadedError(
                f"Modèle ou métadonnées introuvables ({self.model_path}, {self.metadata_path}). "
                "Lancez `dvc pull` ou `python -m src.models.evaluate`."
            )
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in self.available_providers()
        ]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=providers)
        self._warmup()
        logger.info(
            "Modèle chargé : %s (%s) providers=%s",
            self.model_path.name,
            self.version,
            self.session.get_providers(),
        )

    def _warmup(self) -> None:
        s = self.input_size
        self.session.run(None, {"input": np.zeros((1, 3, s, s), np.float32)})

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unknown"))

    @property
    def input_size(self) -> int:
        return int(self.metadata.get("input_size", 224))

    @property
    def classes(self) -> list[str]:
        return list(self.metadata["classes"])

    def class_name(self, code: str) -> str:
        return self.metadata.get("class_names", {}).get(code, code)

    def is_malignant(self, code: str) -> bool:
        return code in set(self.metadata.get("malignant_classes", []))

    # ------------------------------------------------------------------ inférence
    def preprocess(self, img: Image.Image) -> tuple[np.ndarray, Image.Image]:
        crop = resize_center_crop(img, self.input_size)
        x = np.asarray(crop, dtype=np.float32) / 255.0
        mean = np.asarray(self.metadata.get("mean", (0.485, 0.456, 0.406)), np.float32)
        std = np.asarray(self.metadata.get("std", (0.229, 0.224, 0.225)), np.float32)
        x = (x - mean) / std
        return x.transpose(2, 0, 1)[None].astype(np.float32), crop

    def predict(self, data: bytes) -> InferenceResult:
        if not self.is_loaded:
            raise ModelNotLoadedError("Le modèle n'est pas encore chargé.")
        t0 = time.perf_counter()
        img = decode_image(data)
        x, crop = self.preprocess(img)
        t1 = time.perf_counter()
        logits, cam = self.session.run(["logits", "cam"], {"input": x})
        t2 = time.perf_counter()
        probs = softmax(logits[0].astype(np.float32))
        result = InferenceResult(
            probabilities=probs,
            cam=np.asarray(cam[0], dtype=np.float32),
            image=crop,
            preprocessing_ms=(t1 - t0) * 1000,
            inference_ms=(t2 - t1) * 1000,
        )
        self.stats.record(self.classes[int(probs.argmax())], float(probs.max()), result.inference_ms)
        return result


class PredictionStats:
    """Monitoring léger en mémoire : latence, confiance et distribution des prédictions.

    Une dérive de la distribution des classes prédites ou une hausse des
    prédictions peu confiantes est un signal précoce de *data drift*.
    """

    def __init__(self, window: int = 1000):
        self.latencies: deque[float] = deque(maxlen=window)
        self.confidences: deque[float] = deque(maxlen=window)
        self.classes: Counter[str] = Counter()
        self.total = 0

    def record(self, cls: str, confidence: float, latency_ms: float) -> None:
        self.total += 1
        self.classes[cls] += 1
        self.confidences.append(confidence)
        self.latencies.append(latency_ms)

    def summary(self) -> dict:
        lat = np.asarray(self.latencies or [0.0])
        conf = np.asarray(self.confidences or [0.0])
        n = max(1, sum(self.classes.values()))
        return {
            "total_predictions": self.total,
            "mean_inference_ms": float(lat.mean()),
            "p95_inference_ms": float(np.percentile(lat, 95)),
            "mean_confidence": float(conf.mean()),
            "low_confidence_rate": float((conf < 0.5).mean()) if self.confidences else 0.0,
            "predicted_class_distribution": {k: v / n for k, v in self.classes.items()},
        }
