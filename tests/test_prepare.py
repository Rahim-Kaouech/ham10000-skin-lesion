"""Le split doit être stratifié ET sans lésion partagée entre train/val/test."""

import numpy as np
import pandas as pd

from src.config import CLASSES
from src.data.prepare import check_no_leakage, grouped_split


def make_fake_metadata(n_lesions: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = np.array([3.3, 5.1, 11.0, 1.1, 11.1, 67.0, 1.4])
    probs /= probs.sum()
    rows = []
    for i in range(n_lesions):
        dx = rng.choice(CLASSES, p=probs)
        for j in range(rng.integers(1, 4)):  # 1 à 3 images par lésion
            rows.append({"lesion_id": f"HAM_{i:05d}", "image_id": f"ISIC_{i:05d}_{j}", "dx": dx})
    return pd.DataFrame(rows)


def test_split_has_no_leakage():
    df = grouped_split(make_fake_metadata(), test_size=0.15, val_size=0.15, seed=42)
    check_no_leakage(df)
    assert set(df["split"]) == {"train", "val", "test"}


def test_split_proportions_and_stratification():
    df = grouped_split(make_fake_metadata(), test_size=0.15, val_size=0.15, seed=42)
    frac = df["split"].value_counts(normalize=True)
    assert 0.10 < frac["test"] < 0.20
    assert 0.10 < frac["val"] < 0.20
    # la classe majoritaire garde une proportion proche dans chaque split
    for s in ("train", "val", "test"):
        p_nv = (df.loc[df.split == s, "dx"] == "nv").mean()
        assert 0.55 < p_nv < 0.80
