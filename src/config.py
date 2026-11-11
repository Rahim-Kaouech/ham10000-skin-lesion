"""Constantes et chargement des paramètres partagés par tout le pipeline."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Ordre canonique des classes (index = sortie du réseau)
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Kératose actinique / carcinome intraépithélial (Bowen)",
    "bcc": "Carcinome basocellulaire",
    "bkl": "Lésion kératosique bénigne",
    "df": "Dermatofibrome",
    "mel": "Mélanome",
    "nv": "Nævus mélanocytaire",
    "vasc": "Lésion vasculaire",
}
# Classes malignes ou pré-malignes -> alerte dans l'interface
MALIGNANT = {"akiec", "bcc", "mel"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
