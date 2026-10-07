"""Reading galaxy tables into the format the BST functions expect.

The pipeline works on a pandas DataFrame with these *canonical* column names:

=================  ========================================================  =========
canonical name     meaning                                                   required
=================  ========================================================  =========
``ra``             right ascension [deg]                                     yes
``dec``            declination [deg]                                         yes
``z``              redshift (true, spectroscopic or photometric)             yes
``mag``            apparent magnitude (r band), used for the M_lim cuts      yes
``galaxy_id``      galaxy identifier                                         yes
``halo_id``        host-halo id (mocks): gives ``N_true``                    no
``log_halo_mass``  log10 host-halo mass of each galaxy [Msun] (mocks)        no*
``is_central``     1 = central, 0 = satellite (mocks)                        no*
=================  ========================================================  =========

(*) needed when the centres are taken from the mock truth
(``centres="is_central"``). For real data you give your own centres and
masses, see :func:`bst_hod.compute_hod`.

The default ``columns`` mapping matches the tables written by the original
notebooks (``mock_21_large_FZ_z05_cosmoDC2.mpeg``, ``mock_21_large_BPZ_...``,
``mock_SkySim_z_lt_0_35_mag_lt_21.dat`` and the tables written by
:mod:`bst_hod.prepare`)::

    #galid, ra, dec, z, mr_lsst, mg_lsst, halo_id, loghalo_mass, ...

To use photometric redshifts and magnitudes instead, override the mapping,
e.g. ``columns={"z": "photoz_mode", "mag": "mag_r_photoz"}``.
"""

import os
import warnings

import numpy as np
import pandas as pd

#: canonical name -> column name in the original DC2/SkySim tables
DEFAULT_COLUMNS = {
    "galaxy_id": "galid",
    "ra": "ra",
    "dec": "dec",
    "z": "z",
    "mag": "mr_lsst",
    "halo_id": "halo_id",
    "log_halo_mass": "loghalo_mass",
    "is_central": "is_central",
}

REQUIRED = ("galaxy_id", "ra", "dec", "z", "mag")


def _clean_name(name):
    return str(name).strip().lstrip("#").strip().rstrip(",").strip()


def read_table(path):
    """Read a galaxy table from disk into a DataFrame (no renaming).

    Supported: whitespace-separated text with a ``#``-prefixed header (any
    extension, e.g. ``.dat``, ``.txt``, ``.mpeg``), ``.csv``, ``.parquet``,
    ``.fits``/``.hdf5`` (through astropy) and ``.npz``. Header names are
    cleaned of the ``#`` and trailing commas used by the original tables.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Galaxy table not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".fits", ".fit", ".hdf5", ".h5"):
        from astropy.table import Table  # optional dependency
        df = Table.read(path).to_pandas()
    elif ext == ".npz":
        with np.load(path) as f:
            df = pd.DataFrame({k: f[k] for k in f.files})
    else:
        df = pd.read_csv(path, sep=r"\s+")
    df.columns = [_clean_name(c) for c in df.columns]
    # header lines like "#galid, ..., last \n" leave an empty trailing column
    drop = [c for c in df.columns if (c == "" or c.startswith("Unnamed")) and df[c].isna().all()]
    return df.drop(columns=drop)


def load_catalog(catalog, columns=None, z_positive=True):
    """Return a DataFrame with the canonical column names used by the pipeline.

    Parameters
    ----------
    catalog : str or pandas.DataFrame
        Path to the galaxy table, or a DataFrame already in memory.
    columns : dict, optional
        Mapping ``{canonical_name: name_in_your_table}`` that overrides
        :data:`DEFAULT_COLUMNS`. Columns that already carry the canonical name
        are used directly.
    z_positive : bool
        Drop galaxies with z <= 0 (as the original scripts did).
    """
    df = read_table(catalog) if isinstance(catalog, (str, os.PathLike)) else catalog.copy()
    df.columns = [_clean_name(c) for c in df.columns]

    user = {k: _clean_name(v) for k, v in (columns or {}).items()}
    unknown = set(user) - set(DEFAULT_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown canonical column(s) {sorted(unknown)}; "
                       f"valid names are {list(DEFAULT_COLUMNS)}")

    out = df
    for canon, default_src in DEFAULT_COLUMNS.items():
        if canon in user:                       # explicit choice always wins
            src = user[canon]
            if src not in out.columns:
                raise KeyError(f"Column '{src}' (given for '{canon}') is not in the "
                               f"table. Available columns: {list(df.columns)}")
            if src != canon:
                if canon in out.columns:        # keep the old one under another name
                    out = out.rename(columns={canon: f"{canon}_original"})
                out = out.rename(columns={src: canon})
        elif canon not in out.columns and default_src in out.columns:
            out = out.rename(columns={default_src: canon})

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise KeyError(
            f"The galaxy table is missing required column(s) {missing}. "
            f"Available columns: {list(df.columns)}. Pass e.g. "
            f"columns={{'z': 'photoz_mode', 'mag': 'mag_r_photoz'}} to map them."
        )
    if "log_halo_mass" in out:
        lm = out["log_halo_mass"].to_numpy(dtype=float)
        if np.nanmedian(lm) > 25:
            # e.g. mock_SkySim_z_lt_0_35_mag_lt_21.dat from lsst_SkySim5000.ipynb,
            # where the linear mass was written in the "loghalo_mass" column
            warnings.warn("'log_halo_mass' looks like a linear mass (median "
                          f"{np.nanmedian(lm):.3g}); converting with log10.", stacklevel=2)
            out = out.assign(log_halo_mass=np.log10(lm))
    if z_positive:
        out = out[out["z"] > 0]
    return out.reset_index(drop=True)
