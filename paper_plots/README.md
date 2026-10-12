# Paper figures

Scripts that make the figures of Cataldi et al. (2026), *HOD estimation
performance for LSST data* (arXiv:2603.01978), from the results already computed
for the paper. Nothing is recomputed: each script reads the per-halo BST tables
(`.npz`) and/or the mock galaxy tables and reproduces the corresponding plotting
notebook (`Articulo1/scripts/Draft/` in Pedro Cataldi's analysis folder, named
in the header of each script) with the same selections, binning, jackknife
(50 random subsamples, seed 42) and layout.

The scripts only need numpy, scipy, pandas and matplotlib (not the `bst_hod`
package).

**`figures/` has the figures already made with these scripts** (PDF, ~0.8 MB in
total), kept as the reference copy. Running the scripts does not overwrite them:
by default the output goes to `output/`, which is not in git.

## Data

| folder (Drive copy) | contents | figures | option |
|---|---|---|---|
| `Articulo1/tabs/` | `.npz` results of the BST, the CGF and `NTrue_DC2*` (true HOD), and `norm_purity_completeness_vs_M200.csv` | 6, 7, 9-15 | `--tabs` |
| `mocks/` | mock galaxy tables, **heavy files**: `mock_21_large_z05_cosmoDC2.mpeg` (0.2 GB) and `mock_21_large_FZ_z05_cosmoDC2.mpg` (0.5 GB), plain text (never the `.dat` or `.bin` files) | 1, 2, 3, 4, 6, 8, 14, 16 | `--mocks` |

None of these files is produced by these scripts or by the `bst_hod` package:
they are the results Pedro computed for the paper, copied as they are from the
Drive, and the scripts only read and plot them. That is why the numbers match
the paper. If the tables are ever recomputed with `bst_hod`, compare again,
since some parameters differ (see the notes at the end).

In Pedro's analysis folder both kinds of files were in `Tabs/`, so `--tabs` alone
is enough there: without `--mocks` the mock tables are looked for in the `--tabs`
folder. Both options can also be set with the environment variables
`BST_HOD_TABS` and `BST_HOD_MOCKS`.

```bash
export BST_HOD_TABS=/path/to/Tabs
export BST_HOD_MOCKS=/path/to/mocks
python paper_plots/make_figures.py                     # all figures -> paper_plots/output/*.pdf
python paper_plots/make_figures.py --light             # skip the figures that read the heavy mock tables
python paper_plots/make_figures.py --figs 7 9 --ext png
python paper_plots/fig10_photoz.py --out fig10.pdf     # a single figure
```

`make_figures.py` reads each mock table only once (about 10 s in total) and
prints the median BST/true ratio of each panel next to the value in Table 3.

## Figures

| Fig. | script | heavy mock table | notebook (cell) | input |
|---|---|---|---|---|
| 1 | `fig01_volume_limited.py` | **yes** | `Fig2_volume_limited` (6) | `mock_21_large_z05_cosmoDC2.mpeg` |
| 2 | `fig02_central_satellite.py` | **yes** | `Fig3_Central_vs_Satellite` (1) | `mock_21_large_z05_cosmoDC2.mpeg` |
| 3 | `fig03_photoz.py` | **yes** (both) | `Fig5_z_photo_spec` (10) | `mock_21_large_FZ_z05_cosmoDC2.mpg`, `mock_21_large_z05_cosmoDC2.mpeg` |
| 4 | `fig04_bst_method.py` | **yes** | `Fig1_BST_Method` (0-2) | `mock_21_large_z05_cosmoDC2.mpeg` |
| 6 | `fig06_cgf_purity.py` | **yes** | `Fig10_purity` (1) | `CGF_fr_2.5_fz_0.25Mlim_{m}.0.npz` + mock table |
| 7 | `fig07_bst_r200.py` | no | `Fig6_BST_r200` (3) | `BST_1.0rmax_1.25r200_Mlim_{m}.npz` |
| 8 | `fig08_lum_r200_fit.py` | **yes** | `Fig7_Lum_r200_Fit` (0) | `mock_21_large_z05_cosmoDC2.mpeg` |
| 9 | `fig09_bst_rlum.py` | no | `Fig8_BST_rlum` (5) | `BST_1.0rmax_1.25rlum_Mlim_{m}.npz`, `BST_1.0rmax_1.25r200_Mlim_{m}.npz` |
| 10 | `fig10_photoz.py` | no | `Fig11_BST_Odds` (9) | `BST_{FZ,BPZ}_0.0Odds_r200_Mlim_{m}.npz` |
| 11 | `fig11_mass_scatter.py` | no | `Fig13_BST_M200_dex_Apx` (13) | `BST_{FZ,BPZ}_0.0Odds_r200_Mlim_{m}.npz` |
| 12 | `fig12_cgf.py` | no | `Fig12_BST_CGF_Odds` (10) | `BST_{FZ,BPZ}_CGF_0.8Odds_r200_Mlim_{m}.npz` |
| 13 | `fig13_cgf_fr_fz.py` | no | `FigApx1_Fr_Fz` (4) | `norm_purity_completeness_vs_M200.csv` |
| 14 | `fig14_cgf_vs_groups.py` | **yes** | `FigApx2_BST_CGF` (2) | `CGF_fr_2.5_fz_0.25Mlim_{m}.0.npz` + mock table |
| 15 | `fig15_bst_cgf_fr_fz.py` | no | `FigApx3_BST_CGF_fr_x_fz_0.25` (10) | `Back_r200_CGF_fr_{f_r}_fz_{f_z}_Mlim_{m}.npz` |
| 16 | `fig16_mass_scatter_hist.py` | **yes** | `Fig13_BST_M200_dex_Apx` (1) | `mock_21_large_z05_cosmoDC2.mpeg` |

Figs. 7, 9-12 and 15 also need `NTrue_DC2{m}.npz` (true HOD), for m = -17, -18,
-19, -20. Fig. 5 is a schematic without a notebook.

The mock table `mock_21_large_z05_cosmoDC2.mpeg` is the one the notebooks use for
the true-redshift analysis; its photo-z columns are BPZ (Fig. 3 and the photo-z
BST runs use it as the BPZ table).

## Agreement with the paper

* Fig. 1: N_gal and n_g of the four samples identical to Table 1.
* Figs. 2, 3, 4, 8, 14: identical to the numbers printed by the notebooks
  (sample sizes, photo-z scatter and Odds fractions, fit coefficients, KS values).
* Figs. 7, 9 and 12, and the BPZ panels of Fig. 10: identical to Table 3.
* Fig. 10, FlexZBoost: identical to the notebook output (0.99, 0.93, 0.96, 0.96),
  but Table 3 quotes 0.98, 0.90, 0.93, 0.97. The table was probably filled from a
  different version of the FZ tables.
* Fig. 11: the notebook perturbed the masses with an unseeded random generator,
  so the "Mass error" values change from run to run. The script uses a fixed seed
  (`SEED` in `fig11_mass_scatter.py`).
* Fig. 4: the notebook selected pixels with unseeded random points; the script
  uses a fixed seed, so individual pixels can differ but the figure is the same.
* Fig. 16: as in Fig. 11, the mass scatter is drawn with a fixed seed, and M_r is
  computed with H0 = 70, Om0 = 0.3 as in the notebook.
* Fig. 13 plots the precomputed `norm_purity_completeness_vs_M200.csv`. The
  notebook cell that writes this kind of table uses H0 = 70, Om0 = 0.3 and writes
  `norm_all_mass_...csv`, so the plotted table comes from another version of that
  cell (e.g. purity 15.46 vs 15.66 for M_lim = -17, f_r = 1, f_z = 0.05).
* Fig. 14: the KS test compares the 20 histogram bin heights, not the two samples
  of galaxies (as in the notebook), which is why p >> 0.05.
* Figs. 6, 13 and 14 use the `CGF_fr_*` runs, as the notebooks did. Table 2 and the BST runs
  of Fig. 12 use the `CGFnew_fr_*` runs, which have fewer centres.
* Fig. 8 fits log L vs log r_c (a power law), while the r_c^lum used in the BST
  runs (Fig. 9) comes from a linear L - r_c relation (Eq. 3).
* If the tables are recomputed with `bst_hod` (branch `bst_function`), note that
  its defaults are not those of every paper run: `r_max=1.25` Mpc (the r200 and
  r_c^lum tables of Figs. 7 and 9 used 1.0 Mpc, the photo-z and CGF tables 1.25)
  and the CGF `dz=0.05` (the paper uses f_z = 0.25).
