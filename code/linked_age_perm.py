"""Linked age reassignment for the four compositional correlations.

An animal measured on both platforms receives one reassigned age.
The number of 24-month labels on each platform stays at the observed number.
Does not rewrite correlations.csv.
"""

from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from decompose_rates import (  # noqa: E402
    analyze_tissue,
    compositional_fraction,
    mean_vectors,
    spearman,
    split,
)

OUT = ROOT / "data" / "tms" / "expression" / "linked_age_perm.csv"


def animal(donor):
    return (donor["age"], donor["sex"], donor["number"])


def enumerate_linked(prepared):
    facs = prepared["facs"]["ready"]
    drop = prepared["droplet"]["ready"]
    facs_ids = {animal(donor): i for i, donor in enumerate(facs)}
    drop_ids = {animal(donor): i for i, donor in enumerate(drop)}
    both = sorted(set(facs_ids) & set(drop_ids))
    facs_only = sorted(set(facs_ids) - set(drop_ids))
    drop_only = sorted(set(drop_ids) - set(both))
    n_facs_aged = sum(donor["age"] == 24 for donor in facs)
    n_drop_aged = sum(donor["age"] == 24 for donor in drop)
    plans = []
    for k in range(len(both) + 1):
        n_f = n_facs_aged - k
        n_d = n_drop_aged - k
        if n_f < 0 or n_d < 0 or n_f > len(facs_only) or n_d > len(drop_only):
            continue
        for both_aged in combinations(both, k):
            both_set = set(both_aged)
            for facs_aged in combinations(facs_only, n_f):
                for drop_aged in combinations(drop_only, n_d):
                    plans.append((both_set, set(facs_aged), set(drop_aged)))
    return plans, both, facs_only, drop_only


def rho_for(prepared, plan, platform):
    both_set, facs_extra, drop_extra = plan
    aged_ids = both_set | facs_extra | drop_extra
    donors = prepared[platform]["ready"]
    relabeled = []
    for donor in donors:
        copy = dict(donor)
        copy["age"] = 24 if animal(donor) in aged_ids else 3
        relabeled.append(copy)
    used = prepared[platform]["used"]
    _, means = mean_vectors(relabeled, used)
    abundance, within = split(means, used, "p")
    fraction = compositional_fraction(abundance, within)
    beta = means[24]["tissue"] - means[3]["tissue"]
    return fraction, beta


def main():
    rows = []
    for tissue in ("Bladder", "Tongue"):
        print("LOAD", tissue, flush=True)
        result = analyze_tissue(tissue)
        prepared = result["prepared"]
        for platform in ("facs", "droplet"):
            ready = prepared[platform]["ready"]
            print(platform, [(donor["mouse_id"], donor["sex"], donor["age"], donor["n_cells"]) for donor in ready], flush=True)
        plans, both, facs_only, drop_only = enumerate_linked(prepared)
        print("plans", len(plans), "both", len(both), "facs_only", len(facs_only), "drop_only", len(drop_only), flush=True)
        observed = {
            row["fraction_platform"]: float(row["spearman_cell_fraction"])
            for row in result["rows"]
        }
        extreme = {name: 0 for name in ("facs", "droplet")}
        stored = {name: [] for name in ("facs", "droplet")}
        for plan in plans:
            fractions = {}
            betas = []
            for platform in ("facs", "droplet"):
                fraction, beta = rho_for(prepared, plan, platform)
                fractions[platform] = fraction
                betas.append(beta)
            disagreement = np.abs(betas[0] - betas[1])
            for platform in ("facs", "droplet"):
                rho, _ = spearman(fractions[platform], disagreement)
                stored[platform].append(abs(rho) if np.isfinite(rho) else np.nan)
                if np.isfinite(rho) and abs(rho) >= abs(observed[platform]) - 1e-12:
                    extreme[platform] += 1
        n_plans = len(plans)
        for platform in ("facs", "droplet"):
            finite = np.asarray(stored[platform], dtype=float)
            finite = finite[np.isfinite(finite)]
            ordered = np.sort(finite)
            critical = float(ordered[-1])
            for candidate in ordered[::-1]:
                count = int(np.sum(ordered >= candidate - 1e-12))
                if count / len(ordered) <= 0.05 + 1e-15:
                    critical = float(candidate)
                else:
                    break
            rows.append({
                "tissue": tissue,
                "fraction_platform": platform,
                "subset": "linked_animals",
                "n_both": len(both),
                "n_facs_only": len(facs_only),
                "n_drop_only": len(drop_only),
                "n_assignments": n_plans,
                "n_extreme": extreme[platform],
                "exact_p": extreme[platform] / n_plans,
                "observed_rho": observed[platform],
                "null_median_abs_rho": float(np.median(finite)),
                "critical_abs_rho_5pct": critical,
            })
            print(rows[-1], flush=True)
    pvals = [row["exact_p"] for row in rows]
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    for row, qv in zip(rows, qvals):
        row["bh_q_four"] = float(qv)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
