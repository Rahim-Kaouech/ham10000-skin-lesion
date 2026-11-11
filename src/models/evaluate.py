"""Étape DVC `evaluate` : évaluation finale sur le test, export ONNX, quantification,
benchmark de latence et enregistrement dans le MLflow Model Registry.

Sorties (models/) :
  model.onnx         FP32 avec 2 sorties : `logits` et `cam` (Grad-CAM intégré)
  model_fp16.onnx    FP16 (pour GPU ; peu utile sur CPU)
  model_int8.onnx    INT8 statique QDQ, calibré sur la validation
  metadata.json      lu par l'API FastAPI (/model/metadata)
  test_metrics.json  métriques DVC

Usage : python -m src.models.evaluate [--promote]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.config import (
    CLASS_NAMES,
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MALIGNANT,
    ROOT,
    load_params,
)
from src.data.dataset import build_loaders
from src.models.architecture import count_parameters, load_checkpoint
from src.models.train import predict
from src.utils.gradcam import GradCAMExportWrapper
from src.utils.metrics import compute_metrics, plot_confusion_matrix, report

MODELS_DIR = ROOT / "models"


def export_onnx(model, img_size: int, path, opset: int) -> None:
    wrapper = GradCAMExportWrapper(model).eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    kwargs = dict(
        input_names=["input"],
        output_names=["logits", "cam"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "cam": {0: "batch"}},
        opset_version=opset,
    )
    try:  # exporteur TorchScript (stable avec dynamic_axes)
        torch.onnx.export(wrapper, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # anciennes versions de torch sans l'argument `dynamo`
        torch.onnx.export(wrapper, dummy, str(path), **kwargs)
    onnx.checker.check_model(str(path))


def onnx_predict(session: ort.InferenceSession, loader) -> tuple[np.ndarray, np.ndarray]:
    probs, ys = [], []
    for x, y in loader:
        logits, _ = session.run(None, {"input": x.numpy()})
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs.append(e / e.sum(1, keepdims=True))
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def benchmark(session: ort.InferenceSession, img_size: int, runs: int) -> dict[str, float]:
    x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(5):
        session.run(None, {"input": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "latency_ms_p50": float(np.percentile(times, 50)),
        "latency_ms_p95": float(np.percentile(times, 95)),
    }


def to_fp16(src, dst) -> bool:
    try:
        from onnxconverter_common import float16

        m = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
        onnx.save(m, str(dst))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FP16] ignoré : {exc}")
        return False


def to_int8(src, dst, loader, n_samples: int) -> bool:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.batches = []
            seen = 0
            for x, _ in loader:
                for xi in x.numpy():
                    self.batches.append({"input": xi[None]})
                    seen += 1
                    if seen >= n_samples:
                        break
                if seen >= n_samples:
                    break
            self.it = iter(self.batches)

        def get_next(self):
            return next(self.it, None)

    try:
        pre = MODELS_DIR / "model_pre.onnx"
        quant_pre_process(str(src), str(pre))
        # On ne quantifie que les convolutions et la couche dense (l'essentiel du calcul) ;
        # la branche Grad-CAM (Gather/Mul/Reduce) reste en FP32 pour garder une carte précise.
        quantize_static(
            str(pre),
            str(dst),
            Reader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm"],
        )
        # Vérification : le modèle quantifié doit s'exécuter
        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        sess.run(None, Reader().batches[0])
        pre.unlink(missing_ok=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[INT8] ignoré : {exc}")
        dst.unlink(missing_ok=True)
        return False


def register(onnx_path, test_metrics: dict, registered_name: str, promote: bool) -> str | None:
    """Enregistre le modèle ONNX et gère les alias `staging` / `production`."""
    client = mlflow.MlflowClient()
    info = mlflow.onnx.log_model(
        onnx.load(str(onnx_path)), "onnx_model", registered_model_name=registered_name
    )
    version = str(info.registered_model_version)
    for k in ("balanced_accuracy", "f1_macro", "recall_mel"):
        client.set_model_version_tag(registered_name, version, k, f"{test_metrics[k]:.4f}")
    client.set_registered_model_alias(registered_name, "staging", version)

    current = None
    try:
        current = client.get_model_version_by_alias(registered_name, "production")
    except Exception:  # noqa: BLE001 - pas encore de modèle en production
        pass
    champion = float(current.tags.get("balanced_accuracy", -1)) if current else -1.0
    if promote or test_metrics["balanced_accuracy"] > champion:
        client.set_registered_model_alias(registered_name, "production", version)
        print(
            f"Version {version} promue en production (bal_acc {champion:.4f} -> "
            f"{test_metrics['balanced_accuracy']:.4f})"
        )
    return version


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true", help="forcer l'alias production")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()

    params = load_params()
    ecfg = params["export"]
    model, ckpt = load_checkpoint(MODELS_DIR / "best.pt")
    cfg = ckpt["config"]
    img_size = cfg["train"]["img_size"]
    cfg["train"]["num_workers"] = params["train"]["num_workers"]
    loaders, _ = build_loaders(ROOT / params["data"]["processed_dir"] / "splits.csv", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_test, p_test, _ = predict(model.to(device), loaders["test"], device, amp=False)
    test_metrics = compute_metrics(y_test, p_test)
    print(report(y_test, p_test))

    # --- Export ONNX + variantes optimisées
    model.cpu()
    variants = {"fp32": MODELS_DIR / "model.onnx"}
    export_onnx(model, img_size, variants["fp32"], ecfg["opset"])
    if to_fp16(variants["fp32"], MODELS_DIR / "model_fp16.onnx"):
        variants["fp16"] = MODELS_DIR / "model_fp16.onnx"
    if ecfg["quantize_int8"] and to_int8(
        variants["fp32"], MODELS_DIR / "model_int8.onnx", loaders["val"], ecfg["calibration_samples"]
    ):
        variants["int8"] = MODELS_DIR / "model_int8.onnx"

    # Parité PyTorch / ONNX
    x = next(iter(loaders["test"]))[0][:4]
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(variants["fp32"]), providers=["CPUExecutionProvider"])
    parity = float(np.abs(sess.run(["logits"], {"input": x.numpy()})[0] - ref).max())
    print(f"Écart max logits PyTorch vs ONNX : {parity:.2e}")

    bench = {}
    for name, path in variants.items():
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        yv, pv = onnx_predict(s, loaders["test"])
        bench[name] = {
            "size_mb": path.stat().st_size / 2**20,
            **benchmark(s, img_size, ecfg["benchmark_runs"]),
            "balanced_accuracy": compute_metrics(yv, pv)["balanced_accuracy"],
        }
        print(f"[{name}] {bench[name]}")

    metadata = {
        "model_name": cfg["mlflow"]["registered_model"],
        "architecture": cfg["model"]["name"],
        "version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "framework": "PyTorch -> ONNX Runtime",
        "n_parameters": count_parameters(model),
        "input_size": img_size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "classes": CLASSES,
        "class_names": CLASS_NAMES,
        "malignant_classes": sorted(MALIGNANT),
        "metrics": {"test": test_metrics, "validation": ckpt["val_metrics"]},
        "onnx_variants": bench,
        "trained_epochs": ckpt["epoch"] + 1,
        "mlflow_run_id": ckpt.get("mlflow_run_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (MODELS_DIR / "test_metrics.json").write_text(
        json.dumps(
            {
                **test_metrics,
                "onnx_parity_max_abs": parity,
                **{f"{v}_{k}": val for v, b in bench.items() for k, val in b.items()},
            },
            indent=2,
        )
    )

    # --- MLflow : on complète le run d'entraînement
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_id=ckpt.get("mlflow_run_id")):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("onnx_parity_max_abs", parity)
        for v, b in bench.items():
            mlflow.log_metrics({f"onnx_{v}_{k}": val for k, val in b.items()})
        mlflow.log_figure(plot_confusion_matrix(y_test, p_test, "Test"), "plots/cm_test.png")
        mlflow.log_text(report(y_test, p_test), "reports/test_classification_report.txt")
        mlflow.log_artifact(str(MODELS_DIR / "metadata.json"), "serving")
        for path in variants.values():
            mlflow.log_artifact(str(path), "serving")
        if not args.no_register:
            version = register(
                variants["fp32"], test_metrics, cfg["mlflow"]["registered_model"], args.promote
            )
            mlflow.set_tag("registered_version", version)
    # Le backend lit models/model.onnx + models/metadata.json (volume Docker)


if __name__ == "__main__":
    main()
