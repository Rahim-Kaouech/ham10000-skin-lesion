"""Fonctions de perte pour le déséquilibre de classes.

`FocalLoss` accepte des cibles dures (indices) ou souples (one-hot mélangés par
MixUp/CutMix) : FL = - sum_c alpha_c * y_c * (1 - p_c)^gamma * log(p_c).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _to_soft(target: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    if target.ndim == 1:
        target = F.one_hot(target, num_classes).float()
    if smoothing > 0:
        target = target * (1 - smoothing) + smoothing / num_classes
    return target


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=1)
        y = _to_soft(target, logits.shape[1], self.smoothing)
        loss = -y * (1 - logp.exp()).pow(self.gamma) * logp
        if self.alpha is not None:
            loss = loss * self.alpha
        return loss.sum(1).mean()


class SoftTargetCE(nn.Module):
    """Entropie croisée (éventuellement pondérée) compatible cibles souples."""

    def __init__(self, weight: torch.Tensor | None = None, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=1)
        y = _to_soft(target, logits.shape[1], self.smoothing)
        loss = -y * logp
        if self.weight is not None:
            loss = loss * self.weight
        return loss.sum(1).mean()


def build_loss(cfg: dict, class_weights: torch.Tensor) -> nn.Module:
    kind = cfg["loss"]
    ls = cfg.get("label_smoothing", 0.0)
    if kind == "ce":
        return SoftTargetCE(None, ls)
    if kind == "weighted_ce":
        return SoftTargetCE(class_weights, ls)
    if kind == "focal":
        alpha = class_weights if cfg["sampler"] != "weighted" else None
        return FocalLoss(cfg["focal_gamma"], alpha, ls)
    raise ValueError(f"Perte inconnue : {kind}")
