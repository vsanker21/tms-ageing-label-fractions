# Ageing label fractions in bladder and tongue

Donor-level comparison of ageing from 3 to 24 months in bladder and tongue, on plate sorting and on droplet capture. The mouse is the replicate.

The scripts read local files:

- `data/tms/facs.csv.gz` and `data/tms/droplet.csv.gz`, the released cell-metadata tables
- `data/tms/facs_bladder.h5ad`, `data/tms/droplet_bladder.h5ad`, `data/tms/facs_tongue.h5ad`, `data/tms/droplet_tongue.h5ad`

Counts are the `raw.X` layer.

```
python code/label_fractions.py
python code/decompose_rates.py
python code/weight_sensitivities.py
python code/paired_and_null_summary.py
python code/tongue_label_markers.py
```

`label_fractions.py` writes the mouse fractions and the age contrasts. `decompose_rates.py` writes the abundance-share correlations and the age reassignment. `weight_sensitivities.py` checks that the midpoint correlation is unchanged, then writes the adult-fixed and aged-fixed weights. `paired_and_null_summary.py` writes the male-only contrasts, the within-mouse platform gaps, and the age-reassignment percentiles. `tongue_label_markers.py` writes the within-platform keratin contrasts for the two tongue class names.

Python packages used are `numpy`, `scipy`, and `h5py`.
