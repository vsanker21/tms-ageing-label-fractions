"""Adult-fixed and aged-fixed Kitagawa weights for the midpoint split.

The midpoint split in decompose_rates.py remains the primary result.
These two standardizations were named before the coefficients were read.
They are not used to replace it. Limb muscle and marrow counts are not opened.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from decompose_rates import ROOT, analyze_tissue, compositional_fraction, spearman

OUT = ROOT / "data" / "tms" / "expression"
LOCKED = OUT / "correlations.csv"


def product_difference(means, labels, weight):
    total = np.zeros(means[3]["tissue"].shape, dtype=np.float64)
    for label in labels:
        e24 = np.asarray(means[24]["e"][label], dtype=np.float64)
        e3 = np.asarray(means[3]["e"][label], dtype=np.float64)
        total += float(means[24][weight][label]) * e24
        total -= float(means[3][weight][label]) * e3
    return total


def standardized(means, labels, weight, standard):
    abundance = np.zeros(means[3]["tissue"].shape, dtype=np.float64)
    within = np.zeros_like(abundance)
    interaction = np.zeros_like(abundance)
    for label in labels:
        p3 = float(means[3][weight][label])
        p24 = float(means[24][weight][label])
        e3 = np.asarray(means[3]["e"][label], dtype=np.float64)
        e24 = np.asarray(means[24]["e"][label], dtype=np.float64)
        delta_p = p24 - p3
        delta_e = e24 - e3
        interaction += delta_p * delta_e
        if standard == "adult":
            abundance += delta_p * e3
            within += p3 * delta_e
        elif standard == "aged":
            abundance += delta_p * e24
            within += p24 * delta_e
        else:
            raise ValueError(standard)
    return abundance, within, interaction


def deciles(share, disagreement):
    mask = np.isfinite(share) & np.isfinite(disagreement)
    order = np.argsort(share[mask])
    share = share[mask][order]
    disagreement = disagreement[mask][order]
    edges = np.linspace(0, len(share), 11, dtype=int)
    rows = []
    for decile in range(10):
        block_s = share[edges[decile] : edges[decile + 1]]
        block_d = disagreement[edges[decile] : edges[decile + 1]]
        rows.append(
            {
                "decile": decile + 1,
                "n_genes": int(len(block_s)),
                "median_share": float(np.median(block_s)),
                "median_abs_difference": float(np.median(block_d)),
            }
        )
    return rows


def main():
    locked = {}
    with LOCKED.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            locked[(row["tissue"], row["fraction_platform"])] = float(row["spearman_cell_fraction"])

    sensitivity_rows = []
    decile_rows = []
    for tissue in ("Bladder", "Tongue"):
        result = analyze_tissue(tissue)
        if result["status"] != "tested":
            raise RuntimeError(f"{tissue} was not identified")
        disagreement = result["disagreement"]
        for platform in ("facs", "droplet"):
            means = result["prepared"][platform]["means"]
            used = result["prepared"][platform]["used"]
            primary = result["prepared"][platform]["fraction"]
            rho, _n_genes = spearman(primary, disagreement)
            expected = locked[(tissue, platform)]
            if abs(rho - expected) > 1e-9:
                raise RuntimeError(f"{tissue} {platform} midpoint rho {rho} != locked {expected}")
            for standard in ("adult", "aged"):
                abundance, within, interaction = standardized(means, used, "p", standard)
                share = compositional_fraction(abundance, within)
                rho_s, n_s = spearman(share, disagreement)
                product = product_difference(means, used, "p")
                balance = abundance + within + (interaction if standard == "adult" else -interaction)
                gap = np.max(np.abs(balance - product))
                scale = np.max(np.abs(product))
                if not np.allclose(balance, product, rtol=1e-8, atol=1e-6):
                    raise RuntimeError(
                        f"{tissue} {platform} {standard} does not balance "
                        f"max_abs={gap} max_product={scale} shapes={balance.shape},{product.shape}"
                    )
                finite = np.isfinite(share)
                denom = np.abs(abundance) + np.abs(within) + np.abs(interaction)
                interaction_share = np.full(denom.shape, np.nan)
                usable = denom > 0
                interaction_share[usable] = np.abs(interaction[usable]) / denom[usable]
                sensitivity_rows.append(
                    {
                        "tissue": tissue,
                        "fraction_platform": platform,
                        "standard": standard,
                        "n_genes": n_s,
                        "spearman": rho_s,
                        "median_abundance_share": float(np.nanmedian(share)),
                        "median_interaction_share": float(np.nanmedian(interaction_share)),
                    }
                )
            for row in deciles(primary, disagreement):
                row["tissue"] = tissue
                row["fraction_platform"] = platform
                decile_rows.append(row)
        del result

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "weight_sensitivities.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    with (OUT / "share_deciles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tissue", "fraction_platform", "decile", "n_genes", "median_share", "median_abs_difference"],
        )
        writer.writeheader()
        writer.writerows(decile_rows)
    for row in sensitivity_rows:
        print(row)


if __name__ == "__main__":
    main()
