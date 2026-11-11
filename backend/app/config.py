"""Configuration de l'API par variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "/app/models")))
    # fp32 -> model.onnx ; int8 -> model_int8.onnx ; fp16 -> model_fp16.onnx (GPU)
    model_variant: str = field(default_factory=lambda: os.getenv("MODEL_VARIANT", "fp32"))
    max_upload_mb: float = field(default_factory=lambda: float(os.getenv("MAX_UPLOAD_MB", "10")))
    cors_origins: list[str] = field(
        default_factory=lambda: _list(
            os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
        )
    )
    heatmap_alpha: float = 0.45

    @property
    def model_path(self) -> Path:
        name = {"fp32": "model.onnx"}.get(self.model_variant, f"model_{self.model_variant}.onnx")
        return self.model_dir / name

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "metadata.json"
