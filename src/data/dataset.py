"""PyTorch Dataset, transformations et DataLoaders pour HAM10000.

Anti-fuite : la normalisation utilise les statistiques ImageNet (fixes, pas
calculées sur nos données) et l'augmentation n'est appliquée QU'AU train.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from src.config import CLASSES, IMAGENET_MEAN, IMAGENET_STD


class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


def build_transforms(img_size: int, train: bool, randaugment: bool = True):
    normalize = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if not train:
        return T.Compose([T.Resize(img_size), T.CenterCrop(img_size), *normalize])
    aug = [
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        # Les lésions dermatoscopiques n'ont pas d'orientation canonique
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
    ]
    if randaugment:
        aug.append(T.RandAugment(num_ops=2, magnitude=7))
    return T.Compose([*aug, *normalize])


def class_counts(df: pd.DataFrame) -> np.ndarray:
    return np.bincount(df["label"].to_numpy(), minlength=len(CLASSES))


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Poids inverses de la fréquence, normalisés pour une moyenne de 1."""
    counts = class_counts(df).astype(np.float64)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def build_loaders(splits_csv: Path, params: dict):
    tcfg = params["train"]
    df = pd.read_csv(splits_csv)
    root = Path(splits_csv).parent
    parts = {s: df[df.split == s] for s in ("train", "val", "test")}

    ds = {
        "train": HAM10000Dataset(
            parts["train"], root, build_transforms(tcfg["img_size"], True, tcfg["randaugment"])
        ),
        "val": HAM10000Dataset(parts["val"], root, build_transforms(tcfg["img_size"], False)),
        "test": HAM10000Dataset(parts["test"], root, build_transforms(tcfg["img_size"], False)),
    }

    sampler = None
    if tcfg["sampler"] == "weighted":
        w_class = 1.0 / np.maximum(class_counts(parts["train"]), 1)
        w_sample = w_class[parts["train"]["label"].to_numpy()]
        sampler = WeightedRandomSampler(
            torch.as_tensor(w_sample, dtype=torch.double), len(w_sample), replacement=True
        )

    common = dict(
        batch_size=tcfg["batch_size"],
        num_workers=tcfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    loaders = {
        "train": DataLoader(ds["train"], sampler=sampler, shuffle=sampler is None, drop_last=True, **common),
        "val": DataLoader(ds["val"], shuffle=False, **common),
        "test": DataLoader(ds["test"], shuffle=False, **common),
    }
    return loaders, parts
