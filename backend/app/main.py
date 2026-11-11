"""API REST FastAPI : classification de lésions cutanées HAM10000 + Grad-CAM.

Documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.config import Settings
from app.inference import InferenceResult, InvalidImageError, ModelNotLoadedError, ModelService
from app.schemas import (
    ClassInfo,
    ClassProbability,
    ErrorResponse,
    HealthResponse,
    ModelMetadataResponse,
    MonitoringStats,
    PredictionResponse,
    VisualizationResponse,
)
from app.visualization import heatmap_rgba, overlay, to_base64_png, to_png_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

ERRORS = {
    400: {"model": ErrorResponse, "description": "Fichier absent ou image invalide"},
    413: {"model": ErrorResponse, "description": "Fichier trop volumineux"},
    503: {"model": ErrorResponse, "description": "Modèle non chargé"},
}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = ModelService(settings.model_path, settings.metadata_path, settings.model_variant)
        try:
            service.load()
        except Exception as exc:  # noqa: BLE001 - l'API démarre en mode dégradé (503)
            logger.error("Chargement du modèle impossible : %s", exc)
        app.state.service = service
        app.state.started_at = time.time()
        yield

    app = FastAPI(
        title="HAM10000 Skin Lesion Classifier API",
        description=(
            "Classification de 7 types de lésions pigmentées (dermatoscopie, HAM10000) "
            "par CNN fine-tuné, servi via ONNX Runtime, avec explicabilité Grad-CAM."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-Ms", "X-Predicted-Class", "X-Confidence", "X-Inference-Time-Ms"],
    )

    @app.middleware("http")
    async def timing_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
        return response

    # ------------------------------------------------------------------ helpers
    def get_service(request: Request) -> ModelService:
        service: ModelService = request.app.state.service
        if not service.is_loaded:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "Le modèle n'est pas encore chargé. Réessayez plus tard."
            )
        return service

    async def read_upload(file: UploadFile) -> bytes:
        if file.content_type and not (
            file.content_type.startswith("image/") or file.content_type == "application/octet-stream"
        ):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"Type de fichier non supporté : {file.content_type}"
            )
        limit = int(settings.max_upload_mb * 1024 * 1024)
        data = await file.read(limit + 1)
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Fichier vide.")
        if len(data) > limit:
            raise HTTPException(413, f"Fichier > {settings.max_upload_mb} Mo.")
        return data

    async def run_inference(service: ModelService, data: bytes) -> InferenceResult:
        try:
            # ONNX Runtime est CPU-bound : on libère la boucle asyncio
            return await run_in_threadpool(service.predict, data)
        except InvalidImageError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except ModelNotLoadedError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    def build_prediction(service: ModelService, res: InferenceResult, filename, t0) -> dict:
        probs = [
            ClassProbability(
                code=c, name=service.class_name(c), probability=float(p), malignant=service.is_malignant(c)
            )
            for c, p in zip(service.classes, res.probabilities)
        ]
        probs.sort(key=lambda cp: cp.probability, reverse=True)
        return dict(
            filename=filename,
            prediction=probs[0],
            probabilities=probs,
            risk_alert=probs[0].malignant,
            preprocessing_ms=round(res.preprocessing_ms, 3),
            inference_ms=round(res.inference_ms, 3),
            server_total_ms=round((time.perf_counter() - t0) * 1000, 3),
            model_version=service.version,
        )

    # ------------------------------------------------------------------ routes
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "HAM10000 API", "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse, tags=["Système"])
    async def health(request: Request) -> HealthResponse:
        """État du serveur, présence d'un GPU et version du modèle chargé."""
        service: ModelService = request.app.state.service
        return HealthResponse(
            status="ok" if service.is_loaded else "degraded",
            model_loaded=service.is_loaded,
            model_version=service.version if service.is_loaded else None,
            model_variant=service.variant,
            gpu_available=service.gpu_available(),
            execution_providers=(
                service.session.get_providers() if service.is_loaded else service.available_providers()
            ),
            uptime_s=round(time.time() - request.app.state.started_at, 2),
        )

    @app.get(
        "/model/metadata", response_model=ModelMetadataResponse, tags=["Modèle"], responses={503: ERRORS[503]}
    )
    async def model_metadata(request: Request) -> ModelMetadataResponse:
        """Nom, architecture, métriques de validation/test et classes du modèle."""
        service = get_service(request)
        md = service.metadata
        return ModelMetadataResponse(
            model_name=md.get("model_name", "ham10000-classifier"),
            architecture=md.get("architecture", "unknown"),
            version=service.version,
            framework=md.get("framework", "ONNX Runtime"),
            variant=service.variant,
            input_size=service.input_size,
            n_parameters=md.get("n_parameters"),
            classes=[
                ClassInfo(index=i, code=c, name=service.class_name(c), malignant=service.is_malignant(c))
                for i, c in enumerate(service.classes)
            ],
            metrics=md.get("metrics", {}),
            onnx_variants=md.get("onnx_variants", {}),
            trained_epochs=md.get("trained_epochs"),
            mlflow_run_id=md.get("mlflow_run_id"),
            created_at=md.get("created_at"),
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["Inférence"], responses=ERRORS)
    async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
        """Classifie une image dermatoscopique et renvoie les probabilités des 7 classes."""
        t0 = time.perf_counter()
        service = get_service(request)
        res = await run_inference(service, await read_upload(file))
        return PredictionResponse(**build_prediction(service, res, file.filename, t0))

    @app.post(
        "/predict/visualize",
        tags=["Inférence"],
        responses={
            200: {
                "content": {
                    "image/png": {},
                    "application/json": {"schema": VisualizationResponse.model_json_schema()},
                },
                "description": "PNG annoté (défaut) ou JSON avec images base64",
            },
            **ERRORS,
        },
    )
    async def predict_visualize(
        request: Request,
        file: UploadFile = File(...),
        format: Literal["png", "json"] = Query("png", description="png | json"),
        alpha: float = Query(0.45, ge=0.0, le=1.0, description="Opacité de la carte"),
    ):
        """Renvoie l'image annotée par la carte Grad-CAM de la classe prédite."""
        t0 = time.perf_counter()
        service = get_service(request)
        res = await run_inference(service, await read_upload(file))
        pred = build_prediction(service, res, file.filename, t0)
        blended = overlay(res.image, res.cam, alpha)

        if format == "json":
            return VisualizationResponse(
                **pred,
                image_png_base64=to_base64_png(res.image),
                heatmap_png_base64=to_base64_png(heatmap_rgba(res.cam, res.image.size)),
                overlay_png_base64=to_base64_png(blended),
            )
        top = pred["prediction"]
        return Response(
            content=to_png_bytes(blended),
            media_type="image/png",
            headers={
                "X-Predicted-Class": top.code,
                "X-Confidence": f"{top.probability:.4f}",
                "X-Inference-Time-Ms": f"{res.inference_ms:.2f}",
            },
        )

    @app.get("/monitoring/stats", response_model=MonitoringStats, tags=["Système"])
    async def monitoring(request: Request) -> MonitoringStats:
        """Statistiques glissantes (latence, confiance, distribution des prédictions)."""
        return MonitoringStats(**request.app.state.service.stats.summary())

    return app


app = create_app()
