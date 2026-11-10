"""Cell-count share versus library-count share for the primary labels.

Uses obs n_counts only. FACS n_counts are read totals. Droplet n_counts are UMI totals.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tms"
FILES = {
    ("Bladder", "facs"): DATA / "facs_bladder.h5ad",
    ("Bladder", "droplet"): DATA / "droplet_bladder.h5ad",
    ("Tongue", "facs"): DATA / "facs_tongue.h5ad",
    ("Tongue", "droplet"): DATA / "droplet_tongue.h5ad",
}
LABEL = {"Bladder": "bladder urothelial cell", "Tongue": "basal cell of epidermis"}


def categories(handle, name):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in handle["uns"][f"{name}_categories"][:]]


def mouse_key(mouse_id):
    parts = mouse_id.replace("-", "_").split("_")
    age, second, third = parts
    if second in {"M", "F"}:
        return int(age), ("male" if second == "M" else "female"), int(third)
    return int(age), ("male" if third == "M" else "female"), int(second)


def main():
    rows = []
    for (tissue, platform), path in FILES.items():
        label = LABEL[tissue]
        with h5py.File(path, "r") as handle:
            ages = np.array(categories(handle, "age"))[handle["obs"]["age"][:]]
            mice = np.array(categories(handle, "mouse.id"))[handle["obs"]["mouse.id"][:]]
            labels = np.array(categories(handle, "cell_ontology_class"))[handle["obs"]["cell_ontology_class"][:]]
            counts = np.asarray(handle["obs"]["n_counts"][:], dtype=float)
        keep = np.isin(ages, ["3m", "24m"])
        ages, mice, labels, counts = ages[keep], mice[keep], labels[keep], counts[keep]
        for mouse_id in sorted(set(mice)):
            member = mice == mouse_id
            if member.sum() < 1:
                continue
            age, sex, number = mouse_key(mouse_id)
            in_label = member & (labels == label)
            rows.append({
                "tissue": tissue,
                "platform": platform,
                "age": age,
                "sex": sex,
                "number": number,
                "n_cells": int(member.sum()),
                "cell_fraction": float(in_label.sum() / member.sum()),
                "count_fraction": float(counts[in_label].sum() / counts[member].sum()),
            })

    frame = pd.DataFrame(rows)
    out_rows = []
    for (tissue, platform), block in frame.groupby(["tissue", "platform"]):
        for kind, column in (("cell", "cell_fraction"), ("count", "count_fraction")):
            young = block.loc[block["age"] == 3, column]
            aged = block.loc[block["age"] == 24, column]
            out_rows.append({
                "tissue": tissue,
                "platform": platform,
                "weight": kind,
                "n_young": int(len(young)),
                "n_aged": int(len(aged)),
                "mean_young": float(young.mean()),
                "mean_aged": float(aged.mean()),
                "difference": float(aged.mean() - young.mean()),
            })
    pd.DataFrame(out_rows).to_csv(DATA / "cell_vs_count_share.csv", index=False)
    frame.to_csv(DATA / "cell_vs_count_share_by_mouse.csv", index=False)
    print("wrote", DATA / "cell_vs_count_share.csv")


if __name__ == "__main__":
    main()
