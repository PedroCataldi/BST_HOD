# Paper figures

Scripts that make Figs. 7, 9, 10, 11 and 12 of Cataldi et al. (2026),
*HOD estimation performance for LSST data* (arXiv:2603.01978), from the BST
results already computed for the paper. Nothing is recomputed: each script reads
the per-halo `.npz` tables and reproduces the corresponding plotting notebook
(`Articulo1/scripts/Draft/` in the analysis folder) with the same binning,
jackknife (50 random subsamples, seed 42) and layout.

```bash
export BST_HOD_TABS=/path/to/Tabs        # folder with the .npz tables
python paper/make_figures.py             # all figures -> paper/figures/*.pdf
python paper/make_figures.py --figs 7 9 --ext png
python paper/fig10_photoz.py --out fig10.pdf     # a single figure
```

`make_figures.py` also prints the median BST/true ratio of each panel next to the
value in Table 3 of the paper.

| Fig. | script | notebook (cell) | tables |
|---|---|---|---|
| 7 | `fig07_bst_r200.py` | `Fig6_BST_r200` (3) | `BST_1.0rmax_1.25r200_Mlim_{m}.npz` |
| 9 | `fig09_bst_rlum.py` | `Fig8_BST_rlum` (5) | `BST_1.0rmax_1.25rlum_Mlim_{m}.npz`, `BST_1.0rmax_1.25r200_Mlim_{m}.npz` |
| 10 | `fig10_photoz.py` | `Fig11_BST_Odds` (9) | `BST_{FZ,BPZ}_0.0Odds_r200_Mlim_{m}.npz` |
| 11 | `fig11_mass_scatter.py` | `Fig13_BST_M200_dex_Apx` (13) | `BST_{FZ,BPZ}_0.0Odds_r200_Mlim_{m}.npz` |
| 12 | `fig12_cgf.py` | `Fig12_BST_CGF_Odds` (10) | `BST_{FZ,BPZ}_CGF_0.8Odds_r200_Mlim_{m}.npz` |

All figures also need `NTrue_DC2{m}.npz` (true HOD), for m = -17, -18, -19, -20.

## Agreement with Table 3

* Figs. 7, 9 and 12, and the BPZ panels of Fig. 10: identical to Table 3.
* Fig. 10, FlexZBoost: identical to the notebook output (0.99, 0.93, 0.96, 0.96),
  but Table 3 quotes 0.98, 0.90, 0.93, 0.97. The table was probably filled from a
  different version of the FZ tables.
* Fig. 11: the notebook perturbed the masses with an unseeded random generator,
  so the "Mass error" values change from run to run. The script uses a fixed seed
  (`SEED` in `fig11_mass_scatter.py`); the values agree with Table 3 within that
  run-to-run scatter, except for the most noisy bins.
