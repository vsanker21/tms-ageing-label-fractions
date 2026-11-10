"""Age-by-platform contrast for the primary label fractions.

Reads the saved mouse table. Does not rewrite it.
The matched contrast is the ageing difference on droplet minus the ageing
difference on FACS, among mice measured on both platforms.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tms"
SRC = DATA / "mouse_label_fractions.csv"
OUT = DATA / "platform_age_interaction.csv"

LABEL = {
    "Bladder": "bladder urothelial cell",
    "Tongue": "basal cell of epidermis",
}


def fit_row(frame, formula, term, cluster=None):
    model = smf.ols(formula, data=frame)
    if cluster is None:
        fit = model.fit(cov_type="HC3", use_t=True)
        df = int(fit.df_resid)
    else:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": frame[cluster], "use_t": True})
        df = int(fit.df_resid)
    ci = fit.conf_int().loc[term]
    return {
        "coefficient": float(fit.params[term]),
        "se": float(fit.bse[term]),
        "df_resid": df,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p": float(fit.pvalues[term]),
        "n": int(len(frame)),
    }


def main():
    raw = pd.read_csv(SRC)
    rows = []
    for tissue, label in LABEL.items():
        block = raw[(raw["tissue"] == tissue) & (raw["label"] == label)].copy()
        block["aged"] = (block["age_months"] == 24).astype(int)
        block["droplet"] = (block["platform"] == "droplet").astype(int)
        block["animal"] = block["age_months"].astype(str) + "_" + block["sex"] + "_" + block["mouse_number"].astype(str)
        both = set(block.groupby("animal")["platform"].nunique().loc[lambda s: s == 2].index)
        block["matched"] = block["animal"].isin(both)
        wide = (
            block[block["matched"]]
            .pivot_table(index=["animal", "age_months", "sex", "mouse_number", "aged"], columns="platform", values="fraction")
            .reset_index()
        )
        wide["gap"] = wide["droplet"] - wide["facs"]
        for subset, frame in (
            ("all_mice", block),
            ("matched", block[block["matched"]].copy()),
            ("males", block[block["sex"] == "male"].copy()),
            ("matched_males", block[(block["matched"]) & (block["sex"] == "male")].copy()),
            ("at_least_20_cells", block[block["n_cells"] >= 20].copy()),
        ):
            if frame["aged"].nunique() < 2 or frame["droplet"].nunique() < 2:
                continue
            # platform reference is FACS, so aged is the FACS ageing difference
            # and aged:droplet is droplet minus FACS.
            use = frame.copy()
            use["animal"] = use["age_months"].astype(str) + "_" + use["sex"] + "_" + use["mouse_number"].astype(str)
            info = fit_row(use, "fraction ~ aged + droplet + aged:droplet", "aged:droplet", cluster="animal")
            info.update({"tissue": tissue, "label": label, "subset": subset, "estimand": "interaction_cluster"})
            facs = fit_row(use[use["droplet"] == 0], "fraction ~ aged", "aged")
            drop = fit_row(use[use["droplet"] == 1], "fraction ~ aged", "aged") if use.loc[use["droplet"] == 1, "aged"].nunique() == 2 else None
            info["facs_age"] = facs["coefficient"]
            info["facs_ci_low"] = facs["ci_low"]
            info["facs_ci_high"] = facs["ci_high"]
            info["facs_n"] = facs["n"]
            info["facs_df"] = facs["df_resid"]
            if drop:
                info["droplet_age"] = drop["coefficient"]
                info["droplet_ci_low"] = drop["ci_low"]
                info["droplet_ci_high"] = drop["ci_high"]
                info["droplet_n"] = drop["n"]
                info["droplet_df"] = drop["df_resid"]
            rows.append(info)
        if len(wide):
            matched = fit_row(wide, "gap ~ aged", "aged")
            matched.update({
                "tissue": tissue,
                "label": label,
                "subset": "matched_gap",
                "estimand": "droplet_minus_facs_age_gap",
                "n_young": int((wide["aged"] == 0).sum()),
                "n_aged": int((wide["aged"] == 1).sum()),
                "n_male_young": int(((wide["aged"] == 0) & (wide["sex"] == "male")).sum()),
                "mean_gap_young": float(wide.loc[wide["aged"] == 0, "gap"].mean()),
                "mean_gap_aged": float(wide.loc[wide["aged"] == 1, "gap"].mean()),
            })
            rows.append(matched)
            males = wide[wide["sex"] == "male"]
            if males["aged"].nunique() == 2 and (males["aged"] == 0).sum() >= 1:
                male_fit = fit_row(males, "gap ~ aged", "aged")
                male_fit.update({
                    "tissue": tissue,
                    "label": label,
                    "subset": "matched_male_gap",
                    "estimand": "droplet_minus_facs_age_gap",
                    "n_young": int((males["aged"] == 0).sum()),
                    "n_aged": int((males["aged"] == 1).sum()),
                })
                rows.append(male_fit)
            lines = [f"{tissue} matched mice"]
            for _, row in wide.sort_values(["aged", "mouse_number"]).iterrows():
                lines.append(
                    f"  {int(row['age_months'])} {row['sex']} {int(row['mouse_number'])} "
                    f"facs {row['facs']:.4f} drop {row['droplet']:.4f} gap {row['gap']:+.4f}"
                )
            note = OUT.with_name(f"platform_age_interaction_{tissue.lower()}_mice.txt")
            note.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(OUT)


if __name__ == "__main__":
    main()
