"""Summaries that use the fraction table and the same age enumeration.

Does not rewrite correlations.csv. The age-enumeration counts are checked
against that file before the percentiles are written.
"""

from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from decompose_rates import (  # noqa: E402
    analyze_tissue,
    compositional_fraction,
    mean_vectors,
    spearman,
    split,
)

DATA = ROOT / "data" / "tms"
EXPR = DATA / "expression"
LABELS = {
    "Bladder": "bladder urothelial cell",
    "Tongue": "basal cell of epidermis",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_difference(records: list[dict]) -> dict:
    """Same absolute-difference enumeration as label_fractions.py."""
    aged_n = sum(row["age_months"] == 24 for row in records)
    fractions = [row["fraction"] for row in records]

    def difference(aged_index: set[int]) -> float:
        aged = [fractions[i] for i in aged_index]
        young = [fractions[i] for i in range(len(fractions)) if i not in aged_index]
        return sum(aged) / len(aged) - sum(young) / len(young)

    observed_index = {i for i, row in enumerate(records) if row["age_months"] == 24}
    observed = difference(observed_index)
    n_extreme = 0
    n_perm = 0
    for choice in combinations(range(len(records)), aged_n):
        n_perm += 1
        if abs(difference(set(choice))) >= abs(observed) - 1e-12:
            n_extreme += 1
    young = [row["fraction"] for row in records if row["age_months"] == 3]
    aged = [row["fraction"] for row in records if row["age_months"] == 24]
    return {
        "n_younger": len(young),
        "n_older": len(aged),
        "mean_younger": sum(young) / len(young),
        "mean_older": sum(aged) / len(aged),
        "difference": observed,
        "n_extreme": n_extreme,
        "n_perm": n_perm,
        "exact_p": n_extreme / n_perm,
    }


def fraction_records() -> list[dict]:
    rows = []
    for row in read_csv(DATA / "mouse_label_fractions.csv"):
        if row["tissue"] not in LABELS or row["label"] != LABELS[row["tissue"]]:
            continue
        if row["age_months"] not in {"3", "24"}:
            continue
        rows.append(
            {
                "tissue": row["tissue"],
                "platform": row["platform"],
                "age_months": int(row["age_months"]),
                "sex": row["sex"],
                "mouse_number": int(row["mouse_number"]),
                "n_cells": int(row["n_cells"]),
                "fraction": float(row["fraction"]),
            }
        )
    return rows


def male_only_table(records: list[dict]) -> list[dict]:
    out = []
    for tissue in ("Bladder", "Tongue"):
        for platform in ("facs", "droplet"):
            base = [row for row in records if row["tissue"] == tissue and row["platform"] == platform]
            subsets = {
                "all_mice": base,
                "males": [row for row in base if row["sex"] == "male"],
                "males_excluding_libraries_under_20": [
                    row for row in base if row["sex"] == "male" and row["n_cells"] >= 20
                ],
            }
            for name, subset in subsets.items():
                summary = exact_difference(subset)
                n_young_males = sum(row["age_months"] == 3 for row in subset)
                below = n_young_males < 3 or summary["n_older"] < 3
                out.append(
                    {
                        "tissue": tissue,
                        "platform": platform,
                        "label": LABELS[tissue],
                        "subset": name,
                        "below_three_mouse_bar": "yes" if below else "no",
                        **summary,
                    }
                )
    return out


def paired_table(records: list[dict]) -> list[dict]:
    index = {
        (row["tissue"], row["platform"], row["age_months"], row["sex"], row["mouse_number"]): row
        for row in records
    }
    pairs = []
    seen = set()
    for row in records:
        if row["platform"] != "facs":
            continue
        key = (row["tissue"], row["age_months"], row["sex"], row["mouse_number"])
        if key in seen:
            continue
        seen.add(key)
        other = index.get((row["tissue"], "droplet", row["age_months"], row["sex"], row["mouse_number"]))
        if other is None:
            continue
        signed = other["fraction"] - row["fraction"]
        pairs.append(
            {
                "tissue": row["tissue"],
                "label": LABELS[row["tissue"]],
                "age_months": row["age_months"],
                "sex": row["sex"],
                "mouse_number": row["mouse_number"],
                "facs_n_cells": row["n_cells"],
                "facs_fraction": row["fraction"],
                "droplet_n_cells": other["n_cells"],
                "droplet_fraction": other["fraction"],
                "droplet_minus_facs": signed,
                "absolute_difference": abs(signed),
                "either_library_under_20": "yes" if row["n_cells"] < 20 or other["n_cells"] < 20 else "no",
            }
        )
    pairs.sort(key=lambda item: (item["tissue"], item["age_months"], item["sex"], item["mouse_number"]))
    return pairs


def pair_tests(pairs: list[dict]) -> list[dict]:
    out = []
    for tissue in ("Bladder", "Tongue"):
        for name, subset in (
            ("all_pairs", [row for row in pairs if row["tissue"] == tissue]),
            (
                "libraries_at_least_20",
                [row for row in pairs if row["tissue"] == tissue and row["either_library_under_20"] == "no"],
            ),
        ):
            signed = np.array([row["droplet_minus_facs"] for row in subset], dtype=float)
            nonzero = signed[np.abs(signed) > 1e-12]
            n_pos = int(np.sum(nonzero > 0))
            n_neg = int(np.sum(nonzero < 0))
            sign = binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided")
            wilcox = wilcoxon(nonzero, alternative="two-sided", method="exact", zero_method="wilcox")
            out.append(
                {
                    "tissue": tissue,
                    "subset": name,
                    "n_pairs": len(subset),
                    "median_droplet_minus_facs": float(np.median(signed)),
                    "mean_droplet_minus_facs": float(np.mean(signed)),
                    "median_absolute_difference": float(np.median(np.abs(signed))),
                    "mean_absolute_difference": float(np.mean(np.abs(signed))),
                    "min_absolute_difference": float(np.min(np.abs(signed))),
                    "max_absolute_difference": float(np.max(np.abs(signed))),
                    "n_droplet_higher": n_pos,
                    "n_facs_higher": n_neg,
                    "sign_test_p": float(sign.pvalue),
                    "wilcoxon_statistic": float(wilcox.statistic),
                    "wilcoxon_p": float(wilcox.pvalue),
                }
            )
    return out


def critical_abs_rho(values: np.ndarray) -> tuple[float, int]:
    """Smallest enumerated |ρ| whose exact tail probability is at most 0.05."""
    ordered = np.sort(values)
    n = len(ordered)
    critical = float(ordered[-1])
    tail_n = int(np.sum(ordered >= critical - 1e-12))
    for candidate in ordered[::-1]:
        count = int(n - np.searchsorted(ordered, candidate - 1e-12, side="left"))
        if count / n <= 0.05 + 1e-15:
            critical = float(candidate)
            tail_n = count
        else:
            break
    return critical, tail_n


def age_enumerations(result) -> dict[str, np.ndarray]:
    prepared = result["prepared"]
    used = {name: prepared[name]["used"] for name in ("facs", "droplet")}
    choices = {}
    for name in ("facs", "droplet"):
        donors = prepared[name]["ready"]
        n_aged = sum(donor["age"] == 24 for donor in donors)
        choices[name] = list(combinations(range(len(donors)), n_aged))
    store = {name: [] for name in ("facs", "droplet")}
    for facs_aged in choices["facs"]:
        facs_set = set(facs_aged)
        for drop_aged in choices["droplet"]:
            drop_set = set(drop_aged)
            fractions = {}
            betas = []
            for name, aged_index in (("facs", facs_set), ("droplet", drop_set)):
                relabeled = []
                for index, donor in enumerate(prepared[name]["ready"]):
                    copy = dict(donor)
                    copy["age"] = 24 if index in aged_index else 3
                    relabeled.append(copy)
                _, means = mean_vectors(relabeled, used[name])
                abundance, within = split(means, used[name], "p")
                fractions[name] = compositional_fraction(abundance, within)
                betas.append(means[24]["tissue"] - means[3]["tissue"])
            disagreement = np.abs(betas[0] - betas[1])
            for name in ("facs", "droplet"):
                rho, _ = spearman(fractions[name], disagreement)
                store[name].append(abs(rho) if np.isfinite(rho) else np.nan)
    return {name: np.asarray(values, dtype=float) for name, values in store.items()}


def null_dispersion() -> list[dict]:
    locked = read_csv(EXPR / "correlations.csv")
    locked_index = {(row["tissue"], row["fraction_platform"]): row for row in locked}
    rows = []
    for tissue in ("Bladder", "Tongue"):
        print("TISSUE", tissue, flush=True)
        result = analyze_tissue(tissue)
        if result["status"] != "tested":
            raise RuntimeError(f"{tissue} is not tested")
        enumerated = age_enumerations(result)
        for platform, values in enumerated.items():
            locked_row = locked_index[(tissue, platform)]
            observed = float(locked_row["spearman_cell_fraction"])
            finite = values[np.isfinite(values)]
            n_extreme = int(np.sum(finite >= abs(observed) - 1e-12))
            expected = int(locked_row["age_perm_as_extreme"])
            expected_n = int(locked_row["age_perm_n"])
            if n_extreme != expected or len(values) != expected_n:
                raise RuntimeError(
                    f"{tissue} {platform} enumeration {n_extreme}/{len(values)} "
                    f"does not match locked {expected}/{expected_n}"
                )
            critical, tail_n = critical_abs_rho(finite)
            ordered = np.sort(finite)
            rank = int(np.ceil(0.95 * len(ordered))) - 1
            rows.append(
                {
                    "tissue": tissue,
                    "fraction_platform": platform,
                    "observed_rho": observed,
                    "observed_abs_rho": abs(observed),
                    "n_assignments": len(values),
                    "n_nonfinite": int(np.sum(~np.isfinite(values))),
                    "null_median_abs_rho": float(np.median(finite)),
                    "null_p95_nearest_rank": float(ordered[rank]),
                    "critical_abs_rho_5pct": critical,
                    "critical_tail_n": tail_n,
                    "critical_tail_p": tail_n / len(finite),
                    "age_perm_as_extreme": n_extreme,
                }
            )
            print(rows[-1], flush=True)
    return rows


def main() -> None:
    records = fraction_records()
    males = male_only_table(records)
    pairs = paired_table(records)
    tests = pair_tests(pairs)
    write_csv(DATA / "male_only_fractions.csv", males)
    write_csv(DATA / "paired_platform_fractions.csv", pairs)
    write_csv(DATA / "paired_platform_tests.csv", tests)
    for row in males:
        print(
            "MALE",
            row["tissue"],
            row["platform"],
            row["subset"],
            f"{row['difference']:+.6f}",
            f"{row['n_younger']} vs {row['n_older']}",
            f"{row['n_extreme']}/{row['n_perm']}",
            row["below_three_mouse_bar"],
        )
    for row in pairs:
        print(
            "PAIR",
            row["tissue"],
            row["age_months"],
            row["sex"],
            row["mouse_number"],
            f"{row['facs_fraction']:.6f}",
            row["facs_n_cells"],
            f"{row['droplet_fraction']:.6f}",
            row["droplet_n_cells"],
            f"{row['droplet_minus_facs']:+.6f}",
        )
    for row in tests:
        print("TEST", row)
    dispersion = null_dispersion()
    write_csv(EXPR / "age_null_dispersion.csv", dispersion)


if __name__ == "__main__":
    main()
