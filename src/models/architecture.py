"""Définition des réseaux : backbones timm pré-entraînés ImageNet + tête 7 classes.

Baseline : resnet50. Modèles avancés : efficientnetv2_rw_s, convnext_tiny.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from src.config import CLASSES


def build_model(
    name: str,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    num_classes: int = len(CLASSES),
) -> nn.Module:
    kwargs = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate and not name.startswith("resnet"):
        kwargs["drop_path_rate"] = drop_path_rate
    return timm.create_model(name, **kwargs)


def head_modules(model: nn.Module) -> list[nn.Module]:
    """Modules de la tête de classification (restent entraînables en phase 1)."""
    mods = [model.get_classifier()]
    head = getattr(model, "head", None)
    if isinstance(head, nn.Module):
        mods.append(head)
    return mods


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.parameters():
        p.requires_grad = trainable
    for m in head_modules(model):
        for p in m.parameters():
            p.requires_grad = True


def param_groups(model: nn.Module, lr: float, backbone_lr_mult: float, weight_decay: float):
    """LR discriminants : tête à `lr`, backbone à `lr * backbone_lr_mult`.

    Pas de weight decay sur les biais et les paramètres de normalisation.
    """
    head_ids = {id(p) for m in head_modules(model) for p in m.parameters()}
    groups = {k: [] for k in ("head_decay", "head_no", "bb_decay", "bb_no")}
    for p in model.parameters():
        part = "head" if id(p) in head_ids else "bb"
        decay = "decay" if p.ndim > 1 else "no"
        groups[f"{part}_{decay}"].append(p)
    return [
        {"params": groups["head_decay"], "lr": lr, "weight_decay": weight_decay},
        {"params": groups["head_no"], "lr": lr, "weight_decay": 0.0},
        {"params": groups["bb_decay"], "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": groups["bb_no"], "lr": lr * backbone_lr_mult, "weight_decay": 0.0},
    ]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def load_checkpoint(path, map_location="cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"]["name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
