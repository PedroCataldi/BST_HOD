# bst_hod — HOD from projected galaxy counts

`bst_hod` measures the **Halo Occupation Distribution (HOD)**, the mean number of
galaxies per halo as a function of halo mass, from projected galaxy counts. It
works for spectroscopic and purely photometric surveys, because galaxies that
are not group members are removed statistically with the **Background
Subtraction Technique (BST)** instead of requiring 3D membership.

For each group centre the number of members brighter than an absolute-magnitude
limit `M_lim` is (eq. 2 of Rodriguez et al. 2015)

```
N = N_aperture − N_ring × A_aperture / A_ring
```

where `N_aperture` counts the galaxies projected within an aperture of physical
radius `r_ap` (by default R200) and `N_ring` counts them in a control annulus
around it. Magnitudes are computed at the distance of the centre.

The package also includes the **Central Galaxy Finder (CGF)**, which identifies
candidate central galaxies without membership information, so centres and
satellite counts can be treated consistently in group, cluster and protocluster
studies.

The original BST was written in C by Facundo Rodriguez. This Python framework,
its extension to photometric catalogues and the CGF were developed for the article HOD estimation performance for LSST data (Cataldi et al. 2026, https://arxiv.org/abs/2603.01978).


---

## Installation

```bash
git clone https://github.com/PedroCataldi/bst-hod.git
cd bst-hod
pip install -e ".[plot]"        # numpy, scipy, pandas (+ matplotlib for plots)
```

At NERSC, install it in your own environment or kernel with `pip install --user -e .`.

## Quick start (Jupyter or Python)

```python
from bst_hod import compute_hod, hod_curve, plot_hod

# 1. BST counts for every group centre of a volume-limited sample
res = compute_hod("Tabs/mock_21_large_FZ_z05_cosmoDC2.mpeg", mlim=-19)

# 2. <N|M> in halo-mass bins, with errors
hod = hod_curve(res, errors="bootstrap")       # or "jackknife"

# 3. plot
plot_hod(hod)
```

`examples/quickstart.ipynb` walks through the same steps. It runs on a synthetic
catalogue if your table is not available.

## The input table

`compute_hod` needs the path to a galaxy table, or a pandas DataFrame. It must
contain these columns:

| what | default column name | required |
|---|---|---|
| galaxy id | `galid` | yes |
| RA, Dec [deg] | `ra`, `dec` | yes |
| redshift | `z` | yes |
| apparent r magnitude | `mr_lsst` | yes |
| host halo id | `halo_id` | mocks (for `N_true`) |
| log10 halo mass [Msun] | `loghalo_mass` | mocks, or give your own masses |
| central flag (1/0) | `is_central` | mocks with `centres="is_central"` |

These are the column names of the tables made by the original notebooks
(`#galid, ra, dec, z, mr_lsst, mg_lsst, halo_id, loghalo_mass, ...`), so those
files work as they are. Supported formats are whitespace text with a `#` header
(any extension: `.dat`, `.mpeg`, `.txt`), `.csv`, `.parquet`, `.npz` and `.fits`
(the last needs astropy).

If your columns have other names, or you want photometric redshifts and
magnitudes, map them:

```python
res = compute_hod("mock_21_large_FZ_z05_cosmoDC2.mpeg", mlim=-19,
                  columns={"z": "photoz_mode", "mag": "mag_r_photoz"})
```

## `compute_hod`: options

```python
compute_hod(catalog, mlim, z_max=None, *,
            columns=None, cosmology=None, m_app_lim=21.0,
            centres="is_central", mabs_central_max=-19.5, apply_centre_cuts=True,
            factor_r=1.0, r_min=0.05, r_max=1.25,
            ring="classic", ring_radius=(1.5, 3.5),
            mag_cut=True, area="pixel", clip_cos=False,
            n_jobs=-1, verbose=True, save=None)
```

| option | default | meaning |
|---|---|---|
| `catalog` | **required** | path to the galaxy table, or a DataFrame |
| `mlim` | **required** | absolute-magnitude limit of the sample (e.g. −19) |
| `z_max` | see below | maximum redshift of the centres |
| `centres` | `"is_central"` | `"is_central"` (mock truth), `"cgf"` (run the CGF), an array of galaxy ids, or a DataFrame with `galaxy_id` and optionally `log_halo_mass` |
| `mabs_central_max` | −19.5 | centres must also be brighter than this |
| `factor_r` | 1.0 | aperture radius in units of R200; a scalar or one value per centre |
| `r_min`, `r_max` | 0.05, 1.25 | the aperture is clipped to this range [physical Mpc]; `r_max=None` means no upper clip |
| `ring` | `"classic"` | `"classic"`: ring at `ring_radius` × aperture; `"fix"`: from aperture + 1 Mpc to aperture + 2 Mpc |
| `ring_radius` | (1.5, 3.5) | inner and outer ring radius, in units of the aperture |
| `mag_cut` | True | count only galaxies brighter than `mlim` at the centre's distance (see notes) |
| `area` | `"pixel"` | `"pixel"`: original pixel-grid estimate of A_aperture/A_ring; `"analytic"`: exact ratio r_ap² / (r_out² − r_in²) |
| `clip_cos` | False | fix the rounding issue described in the notes |
| `cosmology` | cosmoDC2 | flat ΛCDM with H0 = 71, Ωm = 0.2648; an astropy cosmology also works |
| `save` | None | write the result to `.csv`, `.npz` or `.parquet` |

When `z_max` is not given, the cuts of the original scripts are used for
M_lim = −16 … −21 (m_r ≤ 21): 0.0563, 0.0875, 0.1345, 0.2055, 0.3038 and 0.4547.
For other limits, `z_max` is the redshift where `M_lim` reaches `m_app_lim`.

**Output:** a DataFrame with one row per centre, most massive first. Columns:
`galaxy_id, halo_id, ra, dec, z, log_M200, r200, r_ap, r_in, r_out, N_bst, N_true, n_circle, n_ring, area_ratio`.
`N_bst` is the background-subtracted count and `N_true` is the true number of
members brighter than `mlim` (mocks only). The parameters used are stored in
`res.attrs`.

## Several samples, or the command line

```python
from bst_hod import compute_hod_samples
results = compute_hod_samples("Tabs/mock_21_large_FZ_z05_cosmoDC2.mpeg",
                              mlims=[-17, -18, -19, -20], save_dir="Tabs/")
```

This replaces `BST_example_for_DC2.py`. The table is read and indexed only once.
The same run from a terminal:

```bash
python examples/run_dc2.py Tabs/mock_21_large_FZ_z05_cosmoDC2.mpeg --mlims -17 -18 -19 -20 --out Tabs/
```

## Data without membership: CGF and your own centres

```python
from bst_hod import find_central_galaxies, compute_hod

centres = find_central_galaxies("my_catalogue.dat", mlim=-19)   # CGF
res = compute_hod("my_catalogue.dat", mlim=-19, centres=centres)

# or, in one step
res = compute_hod("my_catalogue.dat", mlim=-19, centres="cgf")

# real data: your own centres and group masses
import pandas as pd
groups = pd.DataFrame({"galaxy_id": ids, "log_halo_mass": log_masses})
res = compute_hod("my_catalogue.dat", mlim=-19, centres=groups)
```

CGF options: `factor_radius=2.5` (exclusion radius in units of the
luminosity-dependent radius), `dz=0.05` (redshift window), and `lum_radius`
(coefficients of the luminosity–radius relation).

## Preparing the LSST DESC mock tables (NERSC)

This needs `GCRCatalogs`, e.g. in the `desc-stack-weekly-latest` kernel. One
call goes from the GCR catalogue directly to the BST table with m_r ≤ 21:

```python
from bst_hod.prepare import export_gcr_catalog
export_gcr_catalog("flexzboost", "Tabs/mock_21_large_FZ_z05_cosmoDC2.mpeg", z_max=0.4)
export_gcr_catalog("bpz",        "Tabs/mock_21_large_BPZ_z05_cosmoDC2.mpeg", z_max=0.4)
export_gcr_catalog("skysim",     "Tabs/mock_SkySim_z_lt_0_35_mag_lt_21.dat", z_max=0.35)
```

If you already have `.bin` files from the old notebooks, convert them in seconds:

```python
from bst_hod.prepare import bin_to_table
bin_to_table("photoz_lsst_large_FZ_mock_z_lt_0_5.bin", "Tabs/mock_21_large_FZ_z05_cosmoDC2.mpeg")
```

See `examples/prepare_dc2_tables.py`.

## Where each part of the old repository went

| old file | now |
|---|---|
| `BST_method.py` (`background_method_new`, `_rmax`) | `bst_hod.core.bst_count`, used by `compute_hod` (original code kept in `bst_hod/legacy.py`) |
| `BST_example_for_DC2.py` | `compute_hod` / `compute_hod_samples`, `examples/run_dc2.py` |
| `CGF_Mlim.py` | `find_central_galaxies` |
| `lsst-COSMODC2_FZ.ipynb`, `lsst-COSMODC2_BPZ.ipynb`, `lsst_SkySim5000.ipynb` | `bst_hod.prepare.export_gcr_catalog` |
| `read_bin-Pedro.ipynb` | `bst_hod.prepare.read_legacy_bin`, `bin_to_table` (no longer needed for new tables) |

The following were dropped because they did not affect the results: the exploratory
cells (catalogue listings, redshift-range checks, sky plots, histograms, the
conditional-luminosity-function test in the SkySim notebook), `background_method`,
`background_method_rgroup`, the unused 2D histogram computed for every halo, and the
binary write/read round trip.

## Notes and differences from the original code

* **Same numbers.** With the default options, `compute_hod` gives exactly the counts
  of `background_method_rmax` (`mag_cut=True`, as `BST_example_for_DC2.py` used) or
  of `background_method_new` (`mag_cut=False`). The tests in `tests/` check this.
  It is much faster, because each centre only looks at nearby galaxies and pixels
  (a KD-tree) instead of the whole catalogue and a 3136 × 3136 pixel grid.
* **`mag_cut`.** `background_method_new` counts every galaxy in the aperture and
  ring without a magnitude cut, while `background_method_rmax` keeps only galaxies
  brighter than `M_lim` at the centre's distance. The DC2 driver used `_rmax`, so
  that is the default.
* **Pixel areas.** The original grid spans whole radians (`floor`/`ceil` of the
  bounds) and uses `cos(RA)` where `cos(Dec)` would be expected for the box
  height. `area="pixel"` keeps this to reproduce old results; `area="analytic"`
  uses the exact area ratio (0.1 for the default ring).
* **`clip_cos`.** In the angular-distance formula, rounding can make cos θ
  slightly larger than 1 at zero separation, which gives a NaN distance. For about
  2% of centres the central galaxy is then not counted in its own aperture, so N
  is low by 1. `clip_cos=True` fixes it.
* Galaxies more than 90° from a centre are no longer counted as inside the
  aperture. This was a sign issue in the original formula and has no effect on
  footprints like DC2.
* The DC2 tables named `*_z05_*` were cut at true z < 0.4 in the export
  notebooks.
* The SkySim table written by `lsst_SkySim5000.ipynb` stores the linear halo mass
  in the `loghalo_mass` column. `load_catalog` detects this and converts it, with a
  warning.
* The redshift cuts of the original scripts differ by up to ~0.003 from the
  value computed with the cosmoDC2 cosmology. They are kept as defaults (see
  `bst_hod.ORIGINAL_Z_CUTS`).
* The CGF luminosity–radius relation uses the "new" fit of `CGF_Mlim.py`
  (`bst_hod.cgf.LUM_RADIUS_NEW`). The old one is available as `LUM_RADIUS_OLD`.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests build a synthetic light-cone (`bst_hod.synthetic.make_mock_catalog`) and
compare the new code with the original `BST_method.py` functions and the original
CGF loop.

## Reference

Rodriguez F., Merchán M., Sgró M. A., 2015, *Taking advantage of photometric galaxy
catalogues to determine the halo occupation distribution*, A&A, 580, A86
([doi:10.1051/0004-6361/201525798](https://doi.org/10.1051/0004-6361/201525798)).
