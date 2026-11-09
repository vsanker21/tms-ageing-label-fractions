"""Mouse-level label fractions for the Tabula Muris Senis bladder and tongue cohort.

Reads the released cell-metadata tables. Expression columns are not used.
"""

from __future__ import annotations

import csv
import gzip
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tms"
OUT = ROOT / "data" / "tms"

FACS_RE = re.compile(r"^(\d+)_(\d+)_([MF])$")
DROP_RE = re.compile(r"^(\d+)-([MF])-(\d+)$")

PRIMARY = {
    "Bladder": ("bladder urothelial cell",),
    "Tongue": ("basal cell of epidermis", "keratinocyte"),
}
SECONDARY = ("Limb_Muscle", "Marrow")
PRIMARY_AGES = (3, 24)
SECONDARY_AGES = (18, 24)
MIN_CELLS = 20
MIN_MICE = 3


def canon(platform: str, mouse_id: str):
    if platform == "facs":
        match = FACS_RE.match(mouse_id)
        if not match:
            return None
        age, number, sex = int(match.group(1)), int(match.group(2)), match.group(3)
    else:
        match = DROP_RE.match(mouse_id)
        if not match:
            return None
        age, sex, number = int(match.group(1)), match.group(2), int(match.group(3))
    return age, ("male" if sex == "M" else "female"), number


def load(name: str, platform: str):
    counts = defaultdict(lambda: defaultdict(int))
    with gzip.open(DATA / name, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mouse = canon(platform, row["mouse.id"])
            if mouse is None:
                continue
            counts[(platform, row["tissue"], mouse)][row["cell_ontology_class"]] += 1
    return counts


def exact_p(values: list[tuple[int, float]]) -> tuple[float, int, int, float]:
    """Two-sided exact p for the aged-minus-younger mean difference."""
    aged_n = sum(age == max(age for age, _ in values) for age, _ in values)
    observed_ages = {age for age, _ in values}
    younger = min(observed_ages)
    older = max(observed_ages)
    fractions = [fraction for _, fraction in values]

    def difference(aged_index: set[int]) -> float:
        aged = [fractions[i] for i in aged_index]
        young = [fractions[i] for i in range(len(fractions)) if i not in aged_index]
        return sum(aged) / len(aged) - sum(young) / len(young)

    observed_index = {i for i, (age, _) in enumerate(values) if age == older}
    observed = difference(observed_index)
    n_as_extreme = 0
    n_perm = 0
    for choice in combinations(range(len(fractions)), aged_n):
        n_perm += 1
        if abs(difference(set(choice))) >= abs(observed) - 1e-12:
            n_as_extreme += 1
    return observed, n_as_extreme, n_perm, n_as_extreme / n_perm


def eligible_labels(counts, tissue: str, ages: tuple[int, int]) -> tuple[str, ...]:
    if tissue in PRIMARY and ages == PRIMARY_AGES:
        return PRIMARY[tissue]
    passing = []
    labels = set()
    for (platform, this_tissue, mouse), class_counts in counts.items():
        if this_tissue == tissue and mouse[0] in ages:
            labels.update(class_counts)
    for label in sorted(labels):
        ok = True
        for platform in ("facs", "droplet"):
            for age in ages:
                n_mice = 0
                for (this_platform, this_tissue, mouse), class_counts in counts.items():
                    if this_platform == platform and this_tissue == tissue and mouse[0] == age:
                        if class_counts.get(label, 0) >= MIN_CELLS:
                            n_mice += 1
                if n_mice < MIN_MICE:
                    ok = False
        if ok:
            passing.append(label)
    return tuple(passing)


def mouse_rows(counts, tissue: str, ages: tuple[int, int], labels: tuple[str, ...]):
    rows = []
    mice = sorted(
        {
            (platform, mouse)
            for (platform, this_tissue, mouse) in counts
            if this_tissue == tissue and mouse[0] in ages
        }
    )
    for platform, mouse in mice:
        class_counts = counts[(platform, tissue, mouse)]
        total = sum(class_counts.values())
        age, sex, number = mouse
        record = {
            "tissue": tissue,
            "platform": platform,
            "age_months": age,
            "sex": sex,
            "mouse_number": number,
            "n_cells": total,
        }
        for label in labels:
            n_label = class_counts.get(label, 0)
            record[f"n__{label}"] = n_label
            record[f"fraction__{label}"] = n_label / total if total else float("nan")
        n_other = total - sum(class_counts.get(label, 0) for label in labels)
        record["n__other"] = n_other
        record["fraction__other"] = n_other / total if total else float("nan")
        rows.append(record)
    return rows


def summarize(rows, tissue: str, labels: tuple[str, ...], ages: tuple[int, int]):
    summaries = []
    for platform in ("facs", "droplet"):
        block = [row for row in rows if row["platform"] == platform]
        for label in list(labels) + ["other"]:
            values = [(row["age_months"], row[f"fraction__{label}"]) for row in block]
            difference, n_extreme, n_perm, p = exact_p(values)
            younger = [fraction for age, fraction in values if age == ages[0]]
            older = [fraction for age, fraction in values if age == ages[1]]
            summaries.append({
                "tissue": tissue,
                "platform": platform,
                "label": label,
                "n_younger": len(younger),
                "n_older": len(older),
                "mean_younger": sum(younger) / len(younger),
                "mean_older": sum(older) / len(older),
                "difference_older_minus_younger": difference,
                "min_younger": min(younger),
                "max_younger": max(younger),
                "min_older": min(older),
                "max_older": max(older),
                "n_permutations_as_extreme": n_extreme,
                "n_permutations": n_perm,
                "exact_p": p,
            })
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    counts = {}
    counts.update(load("facs.csv.gz", "facs"))
    counts.update(load("droplet.csv.gz", "droplet"))

    mouse_out = []
    summary_out = []
    for tissue, ages in (
        ("Bladder", PRIMARY_AGES),
        ("Tongue", PRIMARY_AGES),
        ("Limb_Muscle", SECONDARY_AGES),
        ("Marrow", SECONDARY_AGES),
    ):
        labels = eligible_labels(counts, tissue, ages)
        rows = mouse_rows(counts, tissue, ages, labels)
        mouse_out.extend(rows)
        summary_out.extend(summarize(rows, tissue, labels, ages))
        print(tissue, ages, "labels", labels, flush=True)

    long_rows = []
    for record in mouse_out:
        labels = [key[len("fraction__"):] for key in record if key.startswith("fraction__")]
        for label in labels:
            long_rows.append({
                "tissue": record["tissue"],
                "platform": record["platform"],
                "age_months": record["age_months"],
                "sex": record["sex"],
                "mouse_number": record["mouse_number"],
                "n_cells": record["n_cells"],
                "label": label,
                "n_label": record[f"n__{label}"],
                "fraction": f"{record[f'fraction__{label}']:.6f}",
            })
    write_csv(OUT / "mouse_label_fractions.csv", long_rows)
    write_csv(OUT / "label_fraction_contrasts.csv", summary_out)
    for row in summary_out:
            print(
                f"{row['tissue']:8} {row['platform']:8} {row['label'][:32]:32} "
                f"{row['mean_younger']:.3f} -> {row['mean_older']:.3f} "
                f"diff {row['difference_older_minus_younger']:+.3f} "
                f"exact {row['n_permutations_as_extreme']}/{row['n_permutations']} "
                f"p {row['exact_p']:.3f} "
                f"young {row['min_younger']:.3f}-{row['max_younger']:.3f} "
                f"old {row['min_older']:.3f}-{row['max_older']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
