"""Donor-level keratin means for the two tongue class names.

Describes whether basal cell of epidermis and keratinocyte point the same
direction on both platforms. Does not rewrite correlations.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from decompose_rates import FILES, load_platform  # noqa: E402

MARKERS = ("Krt5", "Krt14", "Krt15", "Krt10", "Sbsn")
BASAL = "basal cell of epidermis"
KERATIN = "keratinocyte"
MIN_CELLS = 20


def donor_means(loaded) -> list[dict]:
    genes = {gene: index for index, gene in enumerate(loaded["genes"])}
    missing = [gene for gene in MARKERS if gene not in genes]
    if missing:
        raise RuntimeError(f"{loaded['platform']} missing {missing}")
    columns = np.array([genes[gene] for gene in MARKERS])
    rows = []
    for mouse_id in sorted(set(loaded["mice"])):
        member = loaded["mice"] == mouse_id
        if int(member.sum()) < MIN_CELLS:
            continue
        for label in (BASAL, KERATIN):
            in_label = member & (loaded["labels"] == label)
            n_label = int(in_label.sum())
            if n_label < MIN_CELLS:
                continue
            counts = np.asarray(loaded["matrix"][in_label][:, columns].todense(), dtype=np.float64)
            umi = loaded["umi"][in_label]
            values = np.log1p(counts / umi[:, None] * 10000.0).mean(axis=0)
            for gene, value in zip(MARKERS, values):
                rows.append(
                    {
                        "platform": loaded["platform"],
                        "mouse_id": mouse_id,
                        "label": label,
                        "n_cells": n_label,
                        "gene": gene,
                        "mean_log1p": float(value),
                    }
                )
    return rows


def main() -> None:
    rows = []
    for platform in ("facs", "droplet"):
        loaded = load_platform(FILES[("Tongue", platform)], "Tongue", platform)
        rows.extend(donor_means(loaded))
    out = ROOT / "data" / "tms" / "tongue_label_markers.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out, "rows", len(rows))
    for platform in ("facs", "droplet"):
        print("PLATFORM", platform)
        for gene in MARKERS:
            basal = [
                row for row in rows
                if row["platform"] == platform and row["gene"] == gene and row["label"] == BASAL
            ]
            keratin = [
                row for row in rows
                if row["platform"] == platform and row["gene"] == gene and row["label"] == KERATIN
            ]
            both = []
            by_mouse_k = {row["mouse_id"]: row["mean_log1p"] for row in keratin}
            for row in basal:
                if row["mouse_id"] in by_mouse_k:
                    both.append(row["mean_log1p"] - by_mouse_k[row["mouse_id"]])
            signs = sum(value > 0 for value in both)
            print(
                gene,
                "mice",
                len(both),
                "basal_minus_keratinocyte_mean",
                round(float(np.mean(both)), 3),
                "min",
                round(float(np.min(both)), 3),
                "max",
                round(float(np.max(both)), 3),
                "positive",
                f"{signs}/{len(both)}",
            )


if __name__ == "__main__":
    main()
