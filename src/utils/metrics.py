"""Métriques adaptées au déséquilibre (l'accuracy seule est trompeuse : ~67 % de `nv`)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from src.config import CLASSES


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_pred = probs.argmax(1)
    labels = list(range(len(CLASSES)))
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        # Métrique officielle du challenge ISIC 2018 (tâche 3)
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
    }
    try:
        m["auc_ovr_macro"] = roc_auc_score(y_true, probs, multi_class="ovr", labels=labels)
    except ValueError:
        m["auc_ovr_macro"] = float("nan")
    recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    for c, r in zip(CLASSES, recalls):
        m[f"recall_{c}"] = r
    return {k: float(v) for k, v in m.items()}


def report(y_true: np.ndarray, probs: np.ndarray) -> str:
    return classification_report(
        y_true,
        probs.argmax(1),
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )


def plot_confusion_matrix(y_true: np.ndarray, probs: np.ndarray, title: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, probs.argmax(1), labels=list(range(len(CLASSES))))
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title or "Matrice de confusion (normalisée par ligne)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
