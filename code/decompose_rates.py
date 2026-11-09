"""Split bladder and tongue ageing contrasts into abundance and within-label terms.

Uses the raw count layer.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tms"
OUT = DATA / "expression"
MIN_CELLS = 20
SEED = 20261108
N_RANDOM = 1000

FILES = {
    ("Bladder", "facs"): DATA / "facs_bladder.h5ad",
    ("Bladder", "droplet"): DATA / "droplet_bladder.h5ad",
    ("Tongue", "facs"): DATA / "facs_tongue.h5ad",
    ("Tongue", "droplet"): DATA / "droplet_tongue.h5ad",
}
SHARED = {
    "Bladder": ("bladder urothelial cell",),
    "Tongue": ("basal cell of epidermis", "keratinocyte"),
}


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def categories(handle, name: str) -> list[str]:
    return [decode(value) for value in handle["uns"][f"{name}_categories"][:]]


def load_platform(path: Path, tissue: str, platform: str):
    handle = h5py.File(path, "r")
    if "raw.X" not in handle:
        raise RuntimeError(f"{path.name} has no raw counts")
    obs = handle["obs"]
    n_cells = len(obs)
    raw = handle["raw.X"]
    matrix = sp.csr_matrix(
        (raw["data"][:], raw["indices"][:], raw["indptr"][:]),
        shape=(n_cells, len(handle["raw.var"])),
    )
    genes = np.array([decode(value) for value in handle["raw.var"]["index"][:]])
    ages = np.array(categories(handle, "age"))[obs["age"][:]]
    mice = np.array(categories(handle, "mouse.id"))[obs["mouse.id"][:]]
    labels = np.array(categories(handle, "cell_ontology_class"))[obs["cell_ontology_class"][:]]
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    n_counts = np.asarray(obs["n_counts"][:], dtype=np.float64)
    if not np.allclose(row_sums, n_counts, rtol=0, atol=1e-3):
        raise RuntimeError(f"{path.name} raw row sums do not match n_counts")
    keep = np.isin(ages, ["3m", "24m"])
    return {
        "tissue": tissue,
        "platform": platform,
        "matrix": matrix[keep],
        "genes": genes,
        "ages": np.array([int(age[:-1]) for age in ages[keep]]),
        "mice": mice[keep],
        "labels": labels[keep],
        "umi": row_sums[keep],
    }


def mouse_key(mouse_id: str):
    parts = mouse_id.replace("-", "_").split("_")
    if len(parts) != 3:
        return None
    age, second, third = parts
    if second in {"M", "F"}:
        sex = "male" if second == "M" else "female"
        number = int(third)
        return int(age), sex, number
    if third in {"M", "F"}:
        sex = "male" if third == "M" else "female"
        number = int(second)
        return int(age), sex, number
    return None


def aggregate(loaded, shared: tuple[str, ...]):
    matrix = loaded["matrix"]
    ages = loaded["ages"]
    mice = loaded["mice"]
    labels = loaded["labels"]
    umi = loaded["umi"]
    donors = []
    for mouse_id in sorted(set(mice)):
        key = mouse_key(mouse_id)
        if key is None:
            continue
        member = mice == mouse_id
        if int(member.sum()) < MIN_CELLS:
            continue
        age = int(ages[member][0])
        record = {
            "mouse_id": mouse_id,
            "age": age,
            "sex": key[1],
            "number": key[2],
            "n_cells": int(member.sum()),
            "p": {},
            "w": {},
            "e": {},
            "counts": {},
            "umi_label": {},
        }
        tissue_counts = np.asarray(matrix[member].sum(axis=0)).ravel()
        tissue_umi = float(umi[member].sum())
        record["tissue_rate"] = tissue_counts / tissue_umi * 10000.0
        record["tissue_detected"] = tissue_counts > 0
        for label in list(shared) + ["other"]:
            if label == "other":
                in_label = member & ~np.isin(labels, shared)
            else:
                in_label = member & (labels == label)
            n_label = int(in_label.sum())
            record["p"][label] = n_label / record["n_cells"]
            if n_label == 0:
                record["w"][label] = 0.0
                continue
            label_umi = float(umi[in_label].sum())
            record["w"][label] = label_umi / tissue_umi
            record["umi_label"][label] = label_umi
            if n_label >= MIN_CELLS:
                label_counts = np.asarray(matrix[in_label].sum(axis=0)).ravel()
                record["e"][label] = label_counts / label_umi * 10000.0
                record["counts"][label] = label_counts
        donors.append(record)
    return donors


def eligible_labels(donors, labels: tuple[str, ...]) -> tuple[str, ...]:
    kept = []
    for label in labels:
        ok = True
        for age in (3, 24):
            n_ready = sum(1 for donor in donors if donor["age"] == age and label in donor["e"])
            if n_ready < 3:
                ok = False
        if ok:
            kept.append(label)
    return tuple(kept)


def mean_vectors(donors, labels: tuple[str, ...]):
    """Age means of p, w, and e on donors who qualify for every used label."""
    ready = [donor for donor in donors if all(label in donor["e"] for label in labels)]
    out = {}
    for age in (3, 24):
        block = [donor for donor in ready if donor["age"] == age]
        out[age] = {
            "n": len(block),
            "p": {label: float(np.mean([donor["p"][label] for donor in block])) for label in labels},
            "w": {label: float(np.mean([donor["w"][label] for donor in block])) for label in labels},
            "e": {label: np.mean([donor["e"][label] for donor in block], axis=0) for label in labels},
            "tissue": np.mean([donor["tissue_rate"] for donor in block], axis=0),
            "log_tissue": np.mean([np.log1p(donor["tissue_rate"]) for donor in block], axis=0),
            "detected": np.mean([donor["tissue_detected"] for donor in block], axis=0),
        }
    return ready, out


def split(age_means, labels: tuple[str, ...], weight: str):
    abundance = np.zeros_like(age_means[3]["tissue"])
    within = np.zeros_like(abundance)
    for label in labels:
        delta_weight = age_means[24][weight][label] - age_means[3][weight][label]
        delta_e = age_means[24]["e"][label] - age_means[3]["e"][label]
        mid_e = 0.5 * (age_means[3]["e"][label] + age_means[24]["e"][label])
        mid_w = 0.5 * (age_means[3][weight][label] + age_means[24][weight][label])
        abundance += delta_weight * mid_e
        within += mid_w * delta_e
    return abundance, within


def compositional_fraction(abundance, within):
    total = np.abs(abundance) + np.abs(within)
    fraction = np.full(total.shape, np.nan)
    usable = total > 0
    fraction[usable] = np.abs(abundance[usable]) / total[usable]
    return fraction


def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, int(mask.sum())
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return np.nan, int(mask.sum())
    return float((rx * ry).sum() / denom), int(mask.sum())


def exact_age_null(prepared, labels, weight, observed, disagreement):
    """Reassign ages inside each platform, preserving the number aged."""
    platform_choices = []
    for platform in ("facs", "droplet"):
        donors = prepared[platform]["ready"]
        ages = [donor["age"] for donor in donors]
        n_aged = sum(age == 24 for age in ages)
        platform_choices.append(list(combinations(range(len(donors)), n_aged)))
    as_extreme = 0
    n_perm = 0
    for facs_aged in platform_choices[0]:
        for drop_aged in platform_choices[1]:
            n_perm += 1
            fractions = []
            betas = []
            for platform, aged_index in (("facs", set(facs_aged)), ("droplet", set(drop_aged))):
                donors = prepared[platform]["ready"]
                relabeled = []
                for index, donor in enumerate(donors):
                    copy = dict(donor)
                    copy["age"] = 24 if index in aged_index else 3
                    relabeled.append(copy)
                _, means = mean_vectors(relabeled, labels[platform])
                abundance, within = split(means, labels[platform], weight)
                fractions.append(compositional_fraction(abundance, within))
                betas.append(means[24]["tissue"] - means[3]["tissue"])
            rho, _ = spearman(fractions[0] if False else fractions[prepared["which"]], np.abs(betas[0] - betas[1]))
            if np.isfinite(rho) and abs(rho) >= abs(observed) - 1e-12:
                as_extreme += 1
    return as_extreme, n_perm


def analyze_tissue(tissue: str):
    shared = SHARED[tissue]
    loaded = {platform: load_platform(FILES[(tissue, platform)], tissue, platform) for platform in ("facs", "droplet")}
    genes = sorted(set(loaded["facs"]["genes"]) & set(loaded["droplet"]["genes"]))
    gene_index = {platform: {gene: i for i, gene in enumerate(loaded[platform]["genes"])} for platform in loaded}
    columns = {platform: np.array([gene_index[platform][gene] for gene in genes]) for platform in loaded}

    prepared = {}
    for platform in ("facs", "droplet"):
        donors = aggregate(loaded[platform], shared)
        for donor in donors:
            donor["tissue_rate"] = donor["tissue_rate"][columns[platform]]
            donor["tissue_detected"] = donor["tissue_detected"][columns[platform]]
            for label in list(donor["e"]):
                donor["e"][label] = donor["e"][label][columns[platform]]
        used = eligible_labels(donors, shared + ("other",))
        ready, means = mean_vectors(donors, used) if used else ([], None)
        prepared[platform] = {"donors": donors, "used": used, "ready": ready, "means": means}

    common = tuple(label for label in shared + ("other",) if label in prepared["facs"]["used"] and label in prepared["droplet"]["used"])
    # Fractions are computed per platform on the labels identified on that platform.
    results = {"genes": genes, "prepared": prepared, "rows": []}
    betas = {}
    log_betas = {}
    fractions = {}
    residuals = {}
    for platform in ("facs", "droplet"):
        means = prepared[platform]["means"]
        used = prepared[platform]["used"]
        if means is None or any(means[age]["n"] < 3 for age in (3, 24)):
            betas[platform] = None
            continue
        betas[platform] = means[24]["tissue"] - means[3]["tissue"]
        log_betas[platform] = means[24]["log_tissue"] - means[3]["log_tissue"]
        abundance, within = split(means, used, "p")
        fractions[platform] = compositional_fraction(abundance, within)
        residuals[platform] = betas[platform] - (abundance + within)
        abundance_w, within_w = split(means, used, "w")
        prepared[platform]["fraction_w"] = compositional_fraction(abundance_w, within_w)
        prepared[platform]["residual_w"] = betas[platform] - (abundance_w + within_w)
    if any(value is None for value in betas.values()):
        results["status"] = "not_identified"
        return results
    disagreement = np.abs(betas["facs"] - betas["droplet"])
    log_disagreement = np.abs(log_betas["facs"] - log_betas["droplet"])
    for platform in ("facs", "droplet"):
        rho, n_genes = spearman(fractions[platform], disagreement)
        rho_w, _ = spearman(prepared[platform]["fraction_w"], disagreement)
        rho_log, _ = spearman(fractions[platform], log_disagreement)
        finite = np.isfinite(fractions[platform])
        residual_share = np.abs(residuals[platform][finite]) / (
            np.abs(fractions[platform][finite] * 0 + residuals[platform][finite]) + np.abs(prepared[platform]["means"][24]["tissue"][finite] * 0)
            + 1e-12
        )
        # residual share of |A|+|B|+|R|
        abundance, within = split(prepared[platform]["means"], prepared[platform]["used"], "p")
        share = np.abs(residuals[platform]) / (np.abs(abundance) + np.abs(within) + np.abs(residuals[platform]) + 1e-12)
        results["rows"].append({
            "tissue": tissue,
            "fraction_platform": platform,
            "labels": "|".join(prepared[platform]["used"]),
            "n_facs_3": prepared["facs"]["means"][3]["n"],
            "n_facs_24": prepared["facs"]["means"][24]["n"],
            "n_droplet_3": prepared["droplet"]["means"][3]["n"],
            "n_droplet_24": prepared["droplet"]["means"][24]["n"],
            "n_genes": n_genes,
            "spearman_cell_fraction": rho,
            "spearman_umi_fraction": rho_w,
            "spearman_log1p_disagreement": rho_log,
            "median_cell_fraction": float(np.nanmedian(fractions[platform])),
            "median_residual_share": float(np.nanmedian(share)),
        })
        prepared[platform]["fraction"] = fractions[platform]
        prepared[platform]["share"] = share
    results["disagreement"] = disagreement
    results["common_labels_on_both"] = common
    results["status"] = "tested"
    return results


def nulls(result):
    """Exact age reassignment, proportion reassignment, and expression-decile reassignment."""
    if result["status"] != "tested":
        return
    tissue_prepared = result["prepared"]
    for row in result["rows"]:
        platform = row["fraction_platform"]
        observed = row["spearman_cell_fraction"]
        used = {name: tissue_prepared[name]["used"] for name in ("facs", "droplet")}
        choices = {}
        for name in ("facs", "droplet"):
            donors = tissue_prepared[name]["ready"]
            n_aged = sum(donor["age"] == 24 for donor in donors)
            choices[name] = list(combinations(range(len(donors)), n_aged))
        as_extreme = 0
        n_perm = 0
        for facs_aged in choices["facs"]:
            for drop_aged in choices["droplet"]:
                n_perm += 1
                fractions = None
                betas = []
                for name, aged_index in (("facs", set(facs_aged)), ("droplet", set(drop_aged))):
                    relabeled = []
                    for index, donor in enumerate(tissue_prepared[name]["ready"]):
                        copy = dict(donor)
                        copy["age"] = 24 if index in aged_index else 3
                        relabeled.append(copy)
                    _, means = mean_vectors(relabeled, used[name])
                    abundance, within = split(means, used[name], "p")
                    fraction = compositional_fraction(abundance, within)
                    if name == platform:
                        fractions = fraction
                    betas.append(means[24]["tissue"] - means[3]["tissue"])
                rho, _ = spearman(fractions, np.abs(betas[0] - betas[1]))
                if np.isfinite(rho) and abs(rho) >= abs(observed) - 1e-12:
                    as_extreme += 1
        row["age_perm_as_extreme"] = as_extreme
        row["age_perm_n"] = n_perm
        row["age_perm_p"] = as_extreme / n_perm

        rng = np.random.default_rng(SEED)
        fraction = tissue_prepared[platform]["fraction"]
        disagreement = result["disagreement"]
        donors = tissue_prepared[platform]["ready"]
        order = np.arange(len(donors))
        prop_extreme = 0
        for _ in range(N_RANDOM):
            shuffled = rng.permutation(order)
            relabeled = []
            for index, donor in enumerate(donors):
                source = donors[shuffled[index]]
                copy = dict(donor)
                copy["p"] = source["p"]
                copy["w"] = source["w"]
                relabeled.append(copy)
            _, means = mean_vectors(relabeled, tissue_prepared[platform]["used"])
            abundance, within = split(means, tissue_prepared[platform]["used"], "p")
            rho, _ = spearman(compositional_fraction(abundance, within), disagreement)
            if np.isfinite(rho) and abs(rho) >= abs(observed) - 1e-12:
                prop_extreme += 1
        row["proportion_null_p"] = (prop_extreme + 1) / (N_RANDOM + 1)

        mean_rate = 0.5 * (
            tissue_prepared["facs"]["means"][3]["tissue"] + tissue_prepared["facs"]["means"][24]["tissue"]
            + tissue_prepared["droplet"]["means"][3]["tissue"] + tissue_prepared["droplet"]["means"][24]["tissue"]
        ) / 2
        detection = 0.5 * (
            tissue_prepared["facs"]["means"][3]["detected"] + tissue_prepared["facs"]["means"][24]["detected"]
            + tissue_prepared["droplet"]["means"][3]["detected"] + tissue_prepared["droplet"]["means"][24]["detected"]
        ) / 2
        finite = np.isfinite(fraction)
        rate_bin = np.full(fraction.shape, -1)
        detect_bin = np.full(fraction.shape, -1)
        rate_bin[finite] = _decile(mean_rate[finite])
        detect_bin[finite] = _decile(detection[finite])
        groups = {}
        for index in np.flatnonzero(finite):
            groups.setdefault((int(rate_bin[index]), int(detect_bin[index])), []).append(index)
        match_extreme = 0
        for _ in range(N_RANDOM):
            permuted = fraction.copy()
            for indexes in groups.values():
                drawn = np.array(indexes)
                permuted[drawn] = fraction[rng.permutation(drawn)]
            rho, _ = spearman(permuted, disagreement)
            if np.isfinite(rho) and abs(rho) >= abs(observed) - 1e-12:
                match_extreme += 1
        row["matched_null_p"] = (match_extreme + 1) / (N_RANDOM + 1)


def _decile(values: np.ndarray) -> np.ndarray:
    order = rankdata(values, method="average")
    bins = np.floor((order - 1) / len(values) * 10).astype(int)
    return np.clip(bins, 0, 9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tissue in ("Bladder", "Tongue"):
        print("TISSUE", tissue, flush=True)
        result = analyze_tissue(tissue)
        print(tissue, result["status"], {platform: result["prepared"][platform]["used"] for platform in ("facs", "droplet")}, flush=True)
        for platform in ("facs", "droplet"):
            ready = result["prepared"][platform]["ready"]
            print(platform, "donors", [(donor["mouse_id"], donor["n_cells"], round(donor["p"].get("other", 0), 3)) for donor in ready], flush=True)
        if result["status"] != "tested":
            continue
        print("nulls", tissue, flush=True)
        nulls(result)
        rows.extend(result["rows"])
        for row in result["rows"]:
            print(row, flush=True)
    with (OUT / "correlations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
