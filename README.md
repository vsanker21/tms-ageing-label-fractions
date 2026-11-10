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
python code/linked_age_perm.py
python code/platform_age_interaction.py
python code/cell_vs_count_share.py
```

`label_fractions.py` writes the mouse fractions and the age contrasts. `decompose_rates.py` writes the abundance-share correlations. `weight_sensitivities.py` checks that the midpoint correlation is unchanged, then writes the adult-fixed and aged-fixed weights. `paired_and_null_summary.py` writes the male-only contrasts and the within-mouse platform gaps. `tongue_label_markers.py` writes the within-platform keratin contrasts for the two tongue class names. `linked_age_perm.py` writes the age reassignment in which an animal measured on both platforms receives one age and each platform keeps its observed number of 24-month labels. `platform_age_interaction.py` writes the within-mouse difference between the droplet and plate-sorting ageing contrasts for the urothelial and basal fractions. `cell_vs_count_share.py` compares the share of cells with the share of the library total. On the plate-sorted objects that total is the read count. On droplet it is the UMI count.

Python packages used are `numpy`, `scipy`, `h5py`, `pandas`, and `statsmodels`.
