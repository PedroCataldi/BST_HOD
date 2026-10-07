# bst_hod — notes for Claude

Python package that measures the Halo Occupation Distribution (HOD) with the
Background Subtraction Technique (BST) and finds group centres with the Central
Galaxy Finder (CGF). Author: Pedro Cataldi (original BST in C by Facundo Rodriguez).

## Layout
- `bst_hod/core.py`: per-centre BST count (`bst_count`, `aperture_radii`, pixel-grid area).
- `bst_hod/hod.py`: public API `compute_hod`, `compute_hod_samples`, `hod_curve`.
- `bst_hod/cgf.py`: `find_central_galaxies`.
- `bst_hod/catalog.py`: table reading and column mapping (`load_catalog`).
- `bst_hod/prepare.py`: GCRCatalogs export and legacy `.bin` reader (DESC mocks).
- `bst_hod/legacy.py`: the original `background_method_new/_rmax`, unchanged; only used by tests.
- `examples/`: quick-start notebook and scripts.

## Rules
- The default results must stay identical to the original code; `tests/test_regression.py`
  compares against `legacy.py` and a copy of the original CGF loop. Run `pytest` after any
  change to `core.py`, `hod.py` or `cgf.py`.
- New behaviour goes behind an option whose default keeps the old numbers (see `clip_cos`, `area`).
- Distances are physical Mpc at the centre's redshift; masses in Msun; RA/Dec in degrees in
  tables and radians inside `core.py`.
- Cosmology defaults to cosmoDC2 (flat LCDM, H0=71, Om0=0.2648); astropy is optional.
- Data files (`Tabs/`, `*.mpeg`, `*.bin`) are large and not in git.
