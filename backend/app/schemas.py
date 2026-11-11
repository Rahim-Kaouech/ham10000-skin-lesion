"""Schémas Pydantic (v2) stricts pour toutes les réponses de l'API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = (
    "Outil pédagogique d'aide à l'analyse : ne remplace en aucun cas l'avis d'un "
    "dermatologue ni un examen histopathologique."
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    model_variant: str
    gpu_available: bool
    execution_providers: list[str]
    uptime_s: float = Field(ge=0)


class ClassInfo(StrictModel):
    index: int = Field(ge=0)
    code: str
    name: str
    malignant: bool


class ModelMetadataResponse(StrictModel):
    model_name: str
    architecture: str
    version: str
    framework: str
    variant: str
    input_size: int = Field(gt=0)
    n_parameters: int | None = None
    classes: list[ClassInfo]
    metrics: dict[str, dict[str, float]]
    onnx_variants: dict[str, dict[str, float]] = {}
    trained_epochs: int | None = None
    mlflow_run_id: str | None = None
    created_at: str | None = None


class ClassProbability(StrictModel):
    code: str
    name: str
    probability: float = Field(ge=0.0, le=1.0)
    malignant: bool


class PredictionResponse(StrictModel):
    filename: str | None = None
    prediction: ClassProbability
    probabilities: list[ClassProbability] = Field(description="Triées par probabilité décroissante")
    risk_alert: bool = Field(description="Vrai si la classe prédite est maligne/pré-maligne")
    preprocessing_ms: float = Field(ge=0)
    inference_ms: float = Field(ge=0)
    server_total_ms: float = Field(ge=0)
    model_version: str
    disclaimer: str = DISCLAIMER


class VisualizationResponse(PredictionResponse):
    image_png_base64: str = Field(description="Image recadrée vue par le modèle")
    heatmap_png_base64: str = Field(description="Carte Grad-CAM colorée (RGBA)")
    overlay_png_base64: str = Field(description="Superposition image + Grad-CAM")


class MonitoringStats(StrictModel):
    total_predictions: int
    mean_inference_ms: float
    p95_inference_ms: float
    mean_confidence: float
    low_confidence_rate: float = Field(description="Part des prédictions avec confiance < 0.5")
    predicted_class_distribution: dict[str, float]


class ErrorResponse(StrictModel):
    detail: str
