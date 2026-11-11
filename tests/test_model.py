"""Tests du code de modélisation (CPU, poids aléatoires, pas de téléchargement)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from src.models.architecture import build_model, param_groups, set_backbone_trainable  # noqa: E402
from src.models.losses import FocalLoss, SoftTargetCE  # noqa: E402
from src.utils.gradcam import GradCAM, GradCAMExportWrapper  # noqa: E402


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0", "convnext_atto"])
def test_export_wrapper_matches_autograd_gradcam(name):
    """Le Grad-CAM analytique (exportable ONNX) doit égaler Grad-CAM par autograd."""
    torch.manual_seed(0)
    model = build_model(name, pretrained=False).eval()
    if hasattr(model, "head") and hasattr(model.head, "norm"):
        # poids LayerNorm non triviaux pour vraiment tester la dérivée
        torch.nn.init.normal_(model.head.norm.weight, 1.0, 0.3)
    x = torch.randn(2, 3, 96, 96)
    logits_ref, cam_ref = GradCAM(model)(x)
    with torch.no_grad():
        logits, cam = GradCAMExportWrapper(model)(x)
    assert torch.allclose(logits, logits_ref, atol=1e-5)
    assert cam.shape == cam_ref.shape
    assert torch.allclose(cam, cam_ref, atol=1e-4)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_freeze_keeps_head_trainable():
    model = build_model("resnet18", pretrained=False)
    set_backbone_trainable(model, False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}
    groups = param_groups(model, 1e-3, 0.1, 1e-4)
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))


def test_focal_loss_reduces_to_ce_when_gamma_zero():
    logits = torch.randn(8, 7)
    y = torch.randint(0, 7, (8,))
    fl = FocalLoss(gamma=0.0)(logits, y)
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_losses_accept_soft_targets():
    logits = torch.randn(4, 7)
    soft = torch.softmax(torch.randn(4, 7), 1)
    assert FocalLoss(2.0)(logits, soft).item() > 0
    assert SoftTargetCE()(logits, soft).item() > 0
