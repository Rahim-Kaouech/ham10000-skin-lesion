"""Explicabilité : Grad-CAM.

Deux implémentations équivalentes :

1. `GradCAM` : version classique par autograd (hooks), pour l'analyse et le rapport.
2. `GradCAMExportWrapper` : version **exportable en ONNX**. ONNX ne fait que le
   forward, donc on calcule le gradient analytiquement. Pour une tête
   `pool moyen -> [LayerNorm] -> Linear`, le gradient du logit y_c par rapport à
   la carte A_k(i,j) vaut (1/HW) * dy_c/dp_k, où p est le vecteur poolé. Les
   poids Grad-CAM alpha_k (moyenne spatiale des gradients) sont donc
   proportionnels à dy_c/dp_k, connu en forme fermée :
     - tête linéaire (ResNet, EfficientNet) : dy_c/dp = W_c
     - tête LayerNorm + linéaire (ConvNeXt) :
       g = gamma * W_c ; dy_c/dp = (g - mean(g) - x_hat * mean(g * x_hat)) / sigma
   Le résultat est identique à Grad-CAM par autograd (vérifié dans
   tests/test_gradcam.py) et le serveur n'a besoin que d'ONNX Runtime.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    cam = F.relu(cam)
    flat = cam.flatten(1)
    mx = flat.max(dim=1).values.clamp_min(1e-8)
    return cam / mx[:, None, None]


class GradCAM:
    """Grad-CAM par autograd sur la sortie de `model.forward_features`."""

    def __init__(self, model: nn.Module):
        self.model = model.eval()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        feats = self.model.forward_features(x)
        feats.retain_grad()
        logits = self.model.forward_head(feats)
        idx = logits.argmax(1) if class_idx is None else torch.full_like(logits[:, 0], class_idx).long()
        score = logits.gather(1, idx[:, None]).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()
        alpha = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = (alpha * feats).sum(1)
        return logits.detach(), _normalize_cam(cam.detach())


class GradCAMExportWrapper(nn.Module):
    """Sorties : `logits` (N, K) et `cam` (N, h, w) dans [0, 1] pour la classe prédite."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        self.fc = model.get_classifier()
        head = getattr(model, "head", None)
        norm = getattr(head, "norm", None) if isinstance(head, nn.Module) else None
        self.norm = None if norm is None or isinstance(norm, nn.Identity) else norm

    def forward(self, x: torch.Tensor):
        feats = self.model.forward_features(x)  # (N, C, h, w)
        logits = self.model.forward_head(feats)  # (N, K)
        idx = logits.argmax(dim=1)
        g = self.fc.weight[idx]  # (N, C) = dy_c/d(entrée de la couche linéaire)

        if self.norm is not None:  # dérivée analytique de la LayerNorm
            p = feats.mean(dim=(2, 3))
            mu = p.mean(dim=1, keepdim=True)
            var = p.var(dim=1, unbiased=False, keepdim=True)
            sigma = torch.sqrt(var + self.norm.eps)
            x_hat = (p - mu) / sigma
            g = g * self.norm.weight
            g = (g - g.mean(1, keepdim=True) - x_hat * (g * x_hat).mean(1, keepdim=True)) / sigma

        cam = (g[:, :, None, None] * feats).sum(dim=1)
        return logits, _normalize_cam(cam)


def overlay_heatmap(image, cam, alpha: float = 0.45):
    """Superpose une carte [0,1] (h, w) sur une image PIL (utile pour le rapport)."""
    import matplotlib.cm as cm
    import numpy as np
    from PIL import Image

    heat = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
    heat = cm.jet(np.asarray(heat) / 255.0)[..., :3]
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    out = (1 - alpha) * base + alpha * heat
    return Image.fromarray(np.uint8(out.clip(0, 1) * 255))
