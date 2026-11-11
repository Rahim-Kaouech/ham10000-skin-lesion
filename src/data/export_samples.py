"""Copie quelques images du jeu de TEST dans le frontend (galerie « 1 clic »).

Usage : python -m src.data.export_samples --per-class 2
"""

from __future__ import annotations

import argparse
import json
import shutil

import pandas as pd

from src.config import CLASS_NAMES, CLASSES, ROOT, load_params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()

    params = load_params()
    proc = ROOT / params["data"]["processed_dir"]
    df = pd.read_csv(proc / "splits.csv")
    df = df[df.split == "test"]
    out = ROOT / "frontend" / "public" / "samples"
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for c in CLASSES:
        rows = df[df.dx == c].sample(min(args.per_class, (df.dx == c).sum()), random_state=params["seed"])
        for _, r in rows.iterrows():
            name = f"{c}_{r.image_id}.jpg"
            shutil.copy(proc / r.path, out / name)
            manifest.append({"file": name, "label": c, "label_name": CLASS_NAMES[c]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"{len(manifest)} images exportées dans {out}")


if __name__ == "__main__":
    main()
