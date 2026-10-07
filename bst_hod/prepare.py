"""Build the galaxy tables used by the BST from the LSST DESC mocks.

This replaces the notebooks ``lsst-COSMODC2_FZ.ipynb``,
``lsst-COSMODC2_BPZ.ipynb``, ``lsst_SkySim5000.ipynb`` and
``read_bin-Pedro.ipynb``. In the original workflow the mock was read with
GCRCatalogs, written to a binary file, read back and written as a text table
with m_r <= 21. Here :func:`export_gcr_catalog` goes straight from the GCR
catalogue to the table. :func:`read_legacy_bin` and :func:`bin_to_table` are
kept to convert ``.bin`` files you already have.

:func:`export_gcr_catalog` needs ``GCRCatalogs`` and access to the DESC data
(e.g. a NERSC Jupyter kernel such as ``desc-stack-weekly-latest``).
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Output table format
# --------------------------------------------------------------------------- #
#: GCR quantity -> column name in the output table (same names as the
#: original ``mock_21_large_*_cosmoDC2`` tables, so old scripts still work)
TABLE_COLUMNS = {
    "galaxy_id": "galid",
    "ra": "ra",
    "dec": "dec",
    "redshift": "z",
    "mag_r_lsst": "mr_lsst",
    "mag_g_lsst": "mg_lsst",
    "halo_id": "halo_id",
    "halo_mass": "loghalo_mass",          # written as log10
    "stellar_mass": "logstellar_mass",    # written as log10
    "position_x": "position_x",
    "position_y": "position_y",
    "position_z": "position_z",
    "is_central": "is_central",
    "photoz_mean": "photoz_mean",
    "photoz_mode": "photoz_mode",
    "photoz_odds": "photoz_odds",
    "mag_r_photoz": "mag_r_photoz",
    "mag_g_photoz": "mag_g_photoz",
    "mag_err_r_photoz": "mag_err_r_photoz",
    "mag_err_g_photoz": "mag_err_g_photoz",
}
_LOG_COLUMNS = ("halo_mass", "stellar_mass")
_INT_COLUMNS = ("galaxy_id", "halo_id", "is_central")

BASE_QUANTITIES = ["galaxy_id", "halo_id", "ra", "dec", "redshift", "mag_r_lsst", "mag_g_lsst",
                   "stellar_mass", "halo_mass", "is_central",
                   "position_x", "position_y", "position_z"]
PHOTOZ_QUANTITIES = ["photoz_mean", "photoz_mode", "photoz_odds",
                     "mag_r_photoz", "mag_g_photoz", "mag_err_r_photoz", "mag_err_g_photoz"]

#: catalogues used in the original notebooks
CATALOGS = {
    "flexzboost": "cosmoDC2_v1.1.4_image_with_photozs_flexzboost_v1",  # -> *_FZ_* tables
    "bpz": "cosmoDC2_v1.1.4_image_with_photozs_v1",                    # -> *_BPZ_* tables
    "cosmodc2": "cosmoDC2_v1.1.4_image",
    "skysim": "skysim5000_v1.1.2",
}


def write_table(df, path, mag_col="mag_r_lsst", mag_max=21.0):
    """Write a GCR-like DataFrame as the whitespace text table read by
    :func:`bst_hod.load_catalog` (masses converted to log10, m_r <= mag_max)."""
    keep = df[mag_col] <= mag_max if mag_max is not None else np.ones(len(df), bool)
    out = pd.DataFrame()
    for q, name in TABLE_COLUMNS.items():
        if q not in df:
            continue
        v = df.loc[keep, q].to_numpy()
        if q in _LOG_COLUMNS:
            v = np.log10(v.astype(float))
        elif q in _INT_COLUMNS:
            v = v.astype(np.int64)
        out[name] = v
    with open(path, "w") as f:
        f.write("#" + " ".join(out.columns) + "\n")
        out.to_csv(f, sep=" ", header=False, index=False, float_format="%.5f")
    print(f"[bst_hod] wrote {len(out)} galaxies to {path}")
    return out


# --------------------------------------------------------------------------- #
# GCR export (run where GCRCatalogs is available, e.g. NERSC)
# --------------------------------------------------------------------------- #
def export_gcr_catalog(catalog="flexzboost", out_path=None, z_max=0.4, mag_max=21.0,
                       photoz=None, healpix=None, verbose=True):
    """Extract a light-cone mock with GCRCatalogs and write the BST input table.

    Parameters
    ----------
    catalog : str
        GCR catalogue name, or one of the shortcuts in :data:`CATALOGS`
        ("flexzboost", "bpz", "cosmodc2", "skysim").
    out_path : str, optional
        Output text table. If None the DataFrame is only returned.
    z_max : float
        Keep galaxies with true redshift < z_max. The original notebooks used
        0.4 (the tables are named ``*_z05_*`` but were cut at z < 0.4).
    mag_max : float or None
        Keep m_r(LSST) <= mag_max in the written table (21 in the original).
    photoz : bool, optional
        Also read the photo-z quantities and keep only galaxies with
        ``photoz_mask``. Default: True for the photo-z catalogues.
    healpix : list of int, optional
        Restrict to these healpix pixels (useful for a quick test).

    Returns
    -------
    DataFrame with the GCR quantities (before the magnitude cut).
    """
    import GCRCatalogs  # only available on DESC systems

    name = CATALOGS.get(catalog, catalog)
    cat = GCRCatalogs.load_catalog(name)
    if photoz is None:
        photoz = "photoz" in name
    quantities = BASE_QUANTITIES + (PHOTOZ_QUANTITIES + ["photoz_mask"] if photoz else [])
    missing = [q for q in quantities if not cat.has_quantity(q)]
    if missing:
        raise KeyError(f"{name} does not provide {missing}")

    pixels = healpix if healpix is not None else cat.available_healpix_pixels
    frames = []
    for i, pix in enumerate(pixels):
        if verbose:
            print(f"[bst_hod] healpix {i + 1}/{len(pixels)} ({pix})")
        for data in cat.get_quantities(quantities, return_iterator=True,
                                       native_filters=[f"healpix_pixel == {pix}"]):
            z = data["redshift"]
            zcut = z < z_max
            if photoz:
                # photo-z quantities only exist for galaxies with photoz_mask
                pm = data["photoz_mask"]
                full_mask = pm & zcut
                sub_mask = z[pm] < z_max
            else:
                full_mask = zcut
            if not full_mask.any():
                continue
            cols = {}
            for q in quantities:
                if q == "photoz_mask":
                    continue
                v = data[q]
                cols[q] = v[full_mask] if len(v) == len(z) else v[sub_mask]
            frames.append(pd.DataFrame(cols))
    df = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"[bst_hod] {len(df)} galaxies with z < {z_max}")
    if out_path:
        write_table(df, out_path, mag_max=mag_max)
    return df


# --------------------------------------------------------------------------- #
# Binary files written by the original notebooks
# --------------------------------------------------------------------------- #
_q, _f, _I = "<i8", "<f4", "<u4"
#: record layouts of the .bin files written by the original notebooks
BIN_LAYOUTS = {
    # lsst-COSMODC2_FZ/BPZ.ipynb: photoz_lsst_large_{FZ,BPZ}_mock_z_lt_0_5.bin
    "cosmodc2_photoz": [("galaxy_id", _q), ("halo_id", _q), ("ra", _f), ("dec", _f),
                        ("redshift", _f), ("mag_r_lsst", _f), ("mag_g_lsst", _f),
                        ("halo_mass", _f), ("stellar_mass", _f), ("is_central", _I),
                        ("position_x", _f), ("position_y", _f), ("position_z", _f),
                        ("photoz_mean", _f), ("photoz_mode", _f), ("photoz_odds", _f),
                        ("mag_r_photoz", _f), ("mag_g_photoz", _f),
                        ("mag_err_r_photoz", _f), ("mag_err_g_photoz", _f)],
    # lsst_SkySim5000.ipynb: lsst_mock_z_lt_0_35.bin
    "skysim": [("galaxy_id", _q), ("halo_id", _q), ("ra", _f), ("dec", _f), ("redshift", _f),
               ("mag_r_lsst", _f), ("mag_g_lsst", _f), ("halo_mass", _f), ("stellar_mass", _f),
               ("is_central", _I), ("position_x", _f), ("position_y", _f), ("position_z", _f)],
    # lsst_SkySim5000.ipynb: lsst_mock_z_lt_0_2.bin (mag_r_sdss instead of g band)
    "skysim_z02": [("galaxy_id", _q), ("halo_id", _q), ("ra", _f), ("dec", _f),
                   ("redshift", _f), ("mag_r_lsst", _f), ("mag_r_sdss", _f), ("halo_mass", _f),
                   ("stellar_mass", _f), ("is_central", _I)],
}


def read_legacy_bin(path, layout="cosmodc2_photoz"):
    """Read a ``.bin`` file written by the original notebooks into a DataFrame.

    Vectorised replacement of ``read_bin-Pedro.ipynb`` (seconds instead of
    minutes). The file is a little-endian uint32 count followed by packed
    records; ``layout`` is one of :data:`BIN_LAYOUTS`.
    """
    dtype = np.dtype(BIN_LAYOUTS[layout])
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        rec = np.fromfile(f, dtype=dtype, count=n)
    if rec.size != n:
        raise ValueError(f"{path}: expected {n} records, read {rec.size} "
                         f"(wrong layout? options: {list(BIN_LAYOUTS)})")
    return pd.DataFrame({name: rec[name].astype(rec.dtype[name].newbyteorder("="))
                         for name in rec.dtype.names})


def bin_to_table(bin_path, out_path, layout="cosmodc2_photoz", mag_max=21.0):
    """``.bin`` file -> BST input table (what read_bin-Pedro.ipynb did)."""
    return write_table(read_legacy_bin(bin_path, layout), out_path, mag_max=mag_max)
