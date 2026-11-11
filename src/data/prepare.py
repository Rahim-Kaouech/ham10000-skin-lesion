"""Étape DVC `prepare` : index des images + split train/val/test SANS fuite.

HAM10000 contient plusieurs images d'une même lésion (colonne `lesion_id`).
Un split image par image placerait des quasi-doublons dans train ET test
(fuite de données). On découpe donc PAR LÉSION avec StratifiedGroupKFold,
tout en conservant la distribution des classes.

Usage : python -m src.data.prepare
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.config import CLASSES, ROOT, load_params


def index_images(raw_dir: Path) -> dict[str, Path]:
    """Retrouve chaque image quelle que soit l'arborescence Kaggle (part_1/part_2)."""
    paths = {p.stem: p for p in raw_dir.rglob("*.jpg")}
    if not paths:
        raise FileNotFoundError(
            f"Aucune image .jpg trouvée sous {raw_dir}. " "Téléchargez HAM10000 (voir README) dans data/raw/."
        )
    return paths


def grouped_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> pd.DataFrame:
    """Assigne une colonne `split` en groupant par lesion_id, stratifié par dx."""
    df = df.copy()
    n_test = max(2, round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
    df["split"] = "train"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    trainval = df.iloc[trainval_idx]
    n_val = max(2, round((1 - test_size) / val_size))
    sgkf = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    _, val_rel = next(sgkf.split(trainval, trainval["dx"], groups=trainval["lesion_id"]))
    df.loc[trainval.index[val_rel], "split"] = "val"
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    groups = {s: set(df.loc[df.split == s, "lesion_id"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = groups[a] & groups[b]
        assert not inter, f"Fuite : {len(inter)} lésions communes entre {a} et {b}"


def resize_copy(src: Path, dst: Path, short_side: int) -> None:
    if dst.exists():
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = short_side / min(w, h)
        if scale < 1:
            im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        im.save(dst, quality=95)


def main() -> None:
    params = load_params()
    dcfg = params["data"]
    raw_dir = ROOT / dcfg["raw_dir"]
    out_dir = ROOT / dcfg["processed_dir"]
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    meta_path = next(raw_dir.rglob(dcfg["metadata_csv"]), None)
    if meta_path is None:
        raise FileNotFoundError(f"{dcfg['metadata_csv']} introuvable sous {raw_dir}")
    df = pd.read_csv(meta_path)
    paths = index_images(raw_dir)
    df = df[df["image_id"].isin(paths)].reset_index(drop=True)
    df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASSES)})

    df = grouped_split(df, dcfg["test_size"], dcfg["val_size"], params["seed"])
    check_no_leakage(df)

    for image_id in df["image_id"]:
        resize_copy(paths[image_id], img_out / f"{image_id}.jpg", dcfg["resize"])
    df["path"] = df["image_id"].map(lambda i: f"images/{i}.jpg")

    df.to_csv(out_dir / "splits.csv", index=False)
    stats = {
        "n_images": int(len(df)),
        "n_lesions": int(df["lesion_id"].nunique()),
        "per_split": df["split"].value_counts().to_dict(),
        "per_class": {
            s: df.loc[df.split == s, "dx"].value_counts().reindex(CLASSES, fill_value=0).to_dict()
            for s in ("train", "val", "test")
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
