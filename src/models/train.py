"""Étape DVC `train` : fine-tuning en 2 phases + suivi MLflow complet.

Phase 1 (`freeze_epochs`) : backbone gelé, seule la tête apprend.
Phase 2 : tout le réseau, LR discriminants (backbone = lr * backbone_lr_mult).
Sélection du meilleur modèle sur la *balanced accuracy* de validation.

Usage :
    python -m src.models.train                       # params.yaml
    python -m src.models.train --set model.name=resnet50 --set train.loss=ce   # baseline
"""

from __future__ import annotations

import argparse
import json
import math
import time

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from timm.data import Mixup

from src.config import CLASSES, ROOT, load_params, seed_everything
from src.data.dataset import build_loaders, class_counts, class_weights
from src.models.architecture import (
    build_model,
    count_parameters,
    param_groups,
    set_backbone_trainable,
)
from src.models.losses import build_loss
from src.utils.metrics import compute_metrics, plot_confusion_matrix, report

MODELS_DIR = ROOT / "models"


def parse_overrides(params: dict, overrides: list[str]) -> dict:
    """`--set train.lr=1e-4` -> params['train']['lr'] = 1e-4 (typage YAML)."""
    import yaml

    for item in overrides:
        key, value = item.split("=", 1)
        node = params
        *parents, leaf = key.split(".")
        for k in parents:
            node = node[k]
        node[leaf] = yaml.safe_load(value)
    return params


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def predict(model, loader, device, amp: bool):
    model.eval()
    all_probs, all_y, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        losses.append(F.cross_entropy(logits.float(), y, reduction="sum").item())
        all_probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        all_y.append(y.cpu().numpy())
    y_true = np.concatenate(all_y)
    return y_true, np.concatenate(all_probs), sum(losses) / len(y_true)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, mixup, device, amp):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if mixup is not None:
            x, y = mixup(x, y)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="override params (a.b=val)")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    params = parse_overrides(load_params(), args.set)
    tcfg, mcfg = params["train"], params["model"]
    seed_everything(params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS_DIR.mkdir(exist_ok=True)

    splits_csv = ROOT / params["data"]["processed_dir"] / "splits.csv"
    loaders, parts = build_loaders(splits_csv, params)
    weights = class_weights(parts["train"]).to(device)

    model = build_model(mcfg["name"], mcfg["pretrained"], mcfg["drop_rate"], mcfg["drop_path_rate"])
    model.to(device)
    criterion = build_loss(tcfg, weights).to(device)
    mixup = None
    if tcfg["mixup_alpha"] > 0 or tcfg["cutmix_alpha"] > 0:
        mixup = Mixup(
            mixup_alpha=tcfg["mixup_alpha"],
            cutmix_alpha=tcfg["cutmix_alpha"],
            prob=tcfg["mix_prob"],
            label_smoothing=0.0,
            num_classes=len(CLASSES),
        )

    optimizer = torch.optim.AdamW(
        param_groups(model, tcfg["lr"], tcfg["backbone_lr_mult"], tcfg["weight_decay"])
    )
    steps = len(loaders["train"])
    scheduler = make_scheduler(optimizer, tcfg["epochs"] * steps, tcfg["warmup_epochs"] * steps)
    scaler = torch.amp.GradScaler(device.type, enabled=tcfg["amp"] and device.type == "cuda")

    mlflow.set_experiment(params["mlflow"]["experiment"])
    run_name = args.run_name or f"{mcfg['name']}-{tcfg['loss']}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flatten(params))
        mlflow.set_tags(
            {
                "axe": "CNN + Transfer Learning",
                "dataset": "HAM10000",
                "split": "StratifiedGroupKFold(lesion_id)",
                "device": str(device),
            }
        )
        mlflow.log_metric("n_parameters", count_parameters(model))
        mlflow.log_dict({s: class_counts(parts[s]).tolist() for s in parts}, "data/class_counts.json")
        stats = ROOT / params["data"]["processed_dir"] / "stats.json"
        if stats.exists():
            mlflow.log_artifact(str(stats), "data")

        best, best_epoch, patience = -1.0, -1, 0
        for epoch in range(tcfg["epochs"]):
            frozen = epoch < tcfg["freeze_epochs"]
            set_backbone_trainable(model, not frozen)
            t0 = time.time()
            train_loss = train_one_epoch(
                model,
                loaders["train"],
                criterion,
                optimizer,
                scheduler,
                scaler,
                mixup,
                device,
                tcfg["amp"],
            )
            y_val, p_val, val_loss = predict(model, loaders["val"], device, tcfg["amp"])
            m = compute_metrics(y_val, p_val)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr_head": optimizer.param_groups[0]["lr"],
                    "epoch_time_s": time.time() - t0,
                    **{f"val_{k}": v for k, v in m.items()},
                },
                step=epoch,
            )
            print(
                f"[{epoch + 1:02d}/{tcfg['epochs']}] {'(gelé) ' if frozen else ''}"
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"bal_acc={m['balanced_accuracy']:.4f} f1={m['f1_macro']:.4f} "
                f"rec_mel={m['recall_mel']:.3f}"
            )

            if m["balanced_accuracy"] > best:
                best, best_epoch, patience = m["balanced_accuracy"], epoch, 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": params,
                        "classes": CLASSES,
                        "epoch": epoch,
                        "val_metrics": m,
                        "mlflow_run_id": run.info.run_id,
                    },
                    MODELS_DIR / "best.pt",
                )
            elif not frozen:
                patience += 1
                if patience >= tcfg["early_stopping_patience"]:
                    print(f"Early stopping (meilleure époque : {best_epoch + 1})")
                    break

        # Rechargement du meilleur checkpoint et artefacts finaux
        ckpt = torch.load(MODELS_DIR / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        y_val, p_val, _ = predict(model, loaders["val"], device, tcfg["amp"])
        best_metrics = compute_metrics(y_val, p_val)
        mlflow.log_metrics({f"best_val_{k}": v for k, v in best_metrics.items()})
        mlflow.log_metric("best_epoch", best_epoch + 1)
        mlflow.log_figure(plot_confusion_matrix(y_val, p_val, "Validation"), "plots/cm_val.png")
        mlflow.log_text(report(y_val, p_val), "reports/val_classification_report.txt")
        mlflow.log_artifact(str(MODELS_DIR / "best.pt"), "checkpoint")

        (MODELS_DIR / "train_metrics.json").write_text(
            json.dumps({"best_epoch": best_epoch + 1, **best_metrics}, indent=2)
        )
        (MODELS_DIR / "run_id.txt").write_text(run.info.run_id)
        print(report(y_val, p_val))


if __name__ == "__main__":
    main()
