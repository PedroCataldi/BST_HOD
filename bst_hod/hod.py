"""High-level functions: measure the HOD of a galaxy catalogue with the BST.

Typical use (in a notebook or a script)::

    from bst_hod import compute_hod, hod_curve

    res = compute_hod("mock_21_large_FZ_z05_cosmoDC2.mpeg", mlim=-19)
    hod = hod_curve(res, errors="bootstrap")
"""

import os
import time

import numpy as np
import pandas as pd

from .catalog import load_catalog
from .core import GalaxyField, N_GRID_ORIGINAL, aperture_radii, bst_count, to_radians
from .cosmology import COSMODC2, absolute_magnitude, comoving_distance, default_z_max, r200_mpc


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _Prepared:
    """Galaxy-level quantities that do not depend on M_lim (computed once)."""

    def __init__(self, cat, cosmo):
        self.cat = cat
        self.cosmo = cosmo
        self.z = cat["z"].to_numpy(dtype=float)
        self.d_com = comoving_distance(self.z, cosmo)
        self.mabs = absolute_magnitude(cat["mag"].to_numpy(dtype=float), self.z, d_com=self.d_com)
        self._field = None

    @property
    def field(self):
        if self._field is None:
            self._field = GalaxyField(self.cat["ra"].to_numpy(), self.cat["dec"].to_numpy(),
                                      self.cat["mag"].to_numpy())
        return self._field


def _prepare(catalog, columns, cosmology):
    if isinstance(catalog, _Prepared):
        return catalog
    cat = load_catalog(catalog, columns=columns)
    return _Prepared(cat, cosmology or COSMODC2)


def _select_centres(prep, mlim, z_max, centres, mabs_central_max, apply_centre_cuts, cgf_kwargs):
    cat = prep.cat
    cuts = (prep.z < z_max) & (prep.mabs < mlim)
    if mabs_central_max is not None:
        cuts &= prep.mabs < mabs_central_max

    log_mass = None
    if isinstance(centres, str) and centres == "is_central":
        if "is_central" not in cat:
            raise KeyError("centres='is_central' needs an 'is_central' column in the table. "
                           "For data without it use centres='cgf' or pass your own centre ids.")
        mask = (cat["is_central"].to_numpy() == 1) & cuts
    elif isinstance(centres, str) and centres == "cgf":
        from .cgf import find_central_galaxies
        found = find_central_galaxies(prep, mlim=mlim, z_max=z_max,
                                      mabs_central_max=mabs_central_max, **(cgf_kwargs or {}))
        mask = cat["galaxy_id"].isin(found["galaxy_id"]).to_numpy()
    else:
        if isinstance(centres, pd.DataFrame):
            if "galaxy_id" not in centres:
                raise KeyError("A centres DataFrame needs a 'galaxy_id' column")
            ids = centres["galaxy_id"].to_numpy()
            if "log_halo_mass" in centres:
                log_mass = pd.Series(centres["log_halo_mass"].to_numpy(), index=ids)
        else:
            ids = np.asarray(centres)
        mask = cat["galaxy_id"].isin(ids).to_numpy()
        if apply_centre_cuts:
            mask = mask & cuts

    sel = np.flatnonzero(mask)
    if log_mass is not None:
        lm = log_mass.reindex(cat["galaxy_id"].to_numpy()[sel]).to_numpy(dtype=float)
    elif "log_halo_mass" in cat:
        lm = cat["log_halo_mass"].to_numpy(dtype=float)[sel]
    else:
        raise KeyError("No halo masses: the table has no 'log_halo_mass' column. Pass the centres "
                       "as a DataFrame with columns 'galaxy_id' and 'log_halo_mass'.")
    m200 = 10 ** lm
    order = m200.argsort()[::-1]          # most massive first, as the original driver
    return sel[order], lm[order], m200[order]


def _true_counts(prep, mlim, halo_ids):
    cat = prep.cat
    if "halo_id" not in cat:
        return np.full(len(halo_ids), np.nan)
    bright = cat["halo_id"].to_numpy()[prep.mabs < mlim]
    counts = pd.Series(bright).value_counts()
    return counts.reindex(halo_ids).fillna(0).to_numpy(dtype=int)


def _save(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    elif ext == ".npz":
        np.savez(path, **{c: df[c].to_numpy() for c in df.columns})
    else:
        raise ValueError("save must end in .csv, .npz or .parquet")


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def compute_hod(catalog, mlim, z_max=None, *, columns=None, cosmology=None, m_app_lim=21.0,
                centres="is_central", mabs_central_max=-19.5, apply_centre_cuts=True,
                factor_r=1.0, r_min=0.05, r_max=1.25, ring="classic", ring_radius=(1.5, 3.5),
                mag_cut=True, area="pixel", clip_cos=False, n_grid=N_GRID_ORIGINAL, cgf_kwargs=None,
                n_jobs=-1, chunk_size=5000, verbose=True, save=None):
    """Estimate the number of galaxies brighter than ``mlim`` in each group/halo
    with the background subtraction technique (BST).

    Parameters
    ----------
    catalog : str or pandas.DataFrame
        **Required.** Path to the galaxy table (e.g.
        ``"Tabs/mock_21_large_FZ_z05_cosmoDC2.mpeg"``) or a DataFrame. It must
        contain RA, Dec, redshift, apparent magnitude and galaxy id; see
        :mod:`bst_hod.catalog` for the column names and ``columns`` to map yours.
    mlim : float
        Absolute-magnitude limit of the volume-limited sample (e.g. -19).
    z_max : float, optional
        Maximum redshift of the centres. Default: the value used in the
        original scripts for M_lim = -16 ... -21 (m_r <= 21), otherwise the
        redshift where ``mlim`` reaches ``m_app_lim``.
    columns : dict, optional
        Column-name mapping, e.g. ``{"z": "photoz_mode", "mag": "mag_r_photoz"}``.
    cosmology : object, optional
        Default: flat LCDM with H0 = 71, Om0 = 0.2648 (cosmoDC2). An astropy
        cosmology also works.
    centres : "is_central" | "cgf" | array of galaxy ids | DataFrame
        Which galaxies are the group centres.
        ``"is_central"``: mock truth (``is_central == 1``).
        ``"cgf"``: run the Central Galaxy Finder (:func:`find_central_galaxies`).
        array: galaxy ids of your centres.
        DataFrame: columns ``galaxy_id`` and optionally ``log_halo_mass`` (use
        this for real data, with your own group masses).
    mabs_central_max : float or None
        Centres must also be brighter than this absolute magnitude (-19.5).
    apply_centre_cuts : bool
        Apply z < z_max, M < mlim and M < mabs_central_max to user-given centres.
    factor_r : float or array
        Aperture radius in units of R200 (array: one value per centre, in the
        order of the output, i.e. sorted by decreasing mass).
    r_min, r_max : float
        Aperture radius is clipped to [r_min, r_max] physical Mpc
        (0.05 and 1.25 as in the DC2 run; ``r_max=None`` for no upper limit).
    ring : "classic" | "fix"
        Control ring at ``ring_radius * r_ap`` ("classic") or at
        r_ap + 1 Mpc ... r_ap + 2 Mpc ("fix").
    ring_radius : (float, float)
        Inner and outer ring radii in units of the aperture radius.
    mag_cut : bool
        Count only galaxies with M (computed at the centre's distance) < mlim.
        True reproduces ``background_method_rmax`` (used by
        BST_example_for_DC2.py); False reproduces ``background_method_new``.
    area : "pixel" | "analytic"
        How A_circle / A_ring is estimated. "pixel" = original pixel grid
        (default, reproduces old results); "analytic" = exact ratio
        r_ap^2 / (r_out^2 - r_in^2).
    clip_cos : bool
        Fix a rounding issue of the original distance formula that leaves the
        central galaxy out of its own aperture for ~2% of centres (N low by 1).
        Default False reproduces the original numbers.
    n_jobs : int
        Threads for the neighbour search (-1 = all cores).
    save : str, optional
        Write the result to ``.csv``, ``.npz`` or ``.parquet``.

    Returns
    -------
    pandas.DataFrame
        One row per centre (most massive first) with ``galaxy_id``,
        ``halo_id``, ``ra``, ``dec``, ``z``, ``log_M200``, ``r200``, ``r_ap``,
        ``r_in``, ``r_out``, ``N_bst`` (background-subtracted count),
        ``N_true`` (true number of members brighter than mlim, mocks only),
        ``n_circle``, ``n_ring`` and ``area_ratio``. The parameters used are
        stored in ``df.attrs``.
    """
    t0 = time.time()
    cosmo = cosmology or COSMODC2
    prep = _prepare(catalog, columns, cosmo)
    cat = prep.cat
    if z_max is None:
        z_max = default_z_max(mlim, m_app_lim, cosmo)

    sel, log_m, m200 = _select_centres(prep, mlim, z_max, centres, mabs_central_max,
                                       apply_centre_cuts, cgf_kwargs)
    n = sel.size
    if verbose:
        print(f"[bst_hod] M_lim = {mlim}, z < {z_max}: {n} centres, {len(cat)} galaxies")

    z0 = prep.z[sel]
    d_com0 = prep.d_com[sel]
    alfa0, delta0 = to_radians(cat["ra"].to_numpy()[sel], cat["dec"].to_numpy()[sel])
    r200 = r200_mpc(m200, z0, cosmo)
    factor = np.asarray(factor_r, dtype=float)
    if factor.ndim and factor.size != n:
        raise ValueError(f"factor_r has {factor.size} values but there are {n} centres")
    r_ap, r_in, r_out = aperture_radii(r200, factor, r_min, r_max, ring, ring_radius)
    r_ap, r_in, r_out = (np.broadcast_to(x, (n,)).astype(float) for x in (r_ap, r_in, r_out))

    N = np.zeros(n)
    n_circ = np.zeros(n, dtype=int)
    n_ring = np.zeros(n, dtype=int)
    ratio = np.full(n, np.nan)
    field = prep.field
    d_phys = d_com0 / (1 + z0)
    theta = np.arctan(r_out / d_phys)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        neigh = field.neighbours(alfa0[start:stop], delta0[start:stop], theta[start:stop],
                                 workers=n_jobs)
        for k, idx in enumerate(neigh):
            j = start + k
            N[j], n_circ[j], n_ring[j], ratio[j] = bst_count(
                field, idx, alfa0[j], delta0[j], z0[j], d_com0[j], r_ap[j], r_in[j], r_out[j],
                mlim, mag_cut=mag_cut, area=area, n_grid=n_grid, clip_cos=clip_cos)
        if verbose and n > chunk_size:
            print(f"[bst_hod]   {stop}/{n} centres done ({time.time() - t0:.0f} s)")

    halo_ids = cat["halo_id"].to_numpy()[sel] if "halo_id" in cat else np.full(n, -1)
    out = pd.DataFrame({
        "galaxy_id": cat["galaxy_id"].to_numpy()[sel],
        "halo_id": halo_ids,
        "ra": cat["ra"].to_numpy()[sel],
        "dec": cat["dec"].to_numpy()[sel],
        "z": z0,
        "log_M200": log_m,
        "r200": r200,
        "r_ap": r_ap,
        "r_in": r_in,
        "r_out": r_out,
        "N_bst": N,
        "N_true": _true_counts(prep, mlim, halo_ids),
        "n_circle": n_circ,
        "n_ring": n_ring,
        "area_ratio": ratio,
    })
    out.attrs.update(dict(mlim=mlim, z_max=z_max, m_app_lim=m_app_lim,
                          centres=centres if isinstance(centres, str) else "user",
                          mabs_central_max=mabs_central_max, factor_r=factor_r if np.ndim(factor_r) == 0 else "array",
                          r_min=r_min, r_max=r_max, ring=ring, ring_radius=tuple(ring_radius),
                          mag_cut=mag_cut, area=area, clip_cos=clip_cos,
                          cosmology=repr(cosmo)))
    if verbose:
        print(f"[bst_hod] done in {time.time() - t0:.1f} s")
    if save:
        _save(out, save)
        if verbose:
            print(f"[bst_hod] saved {save}")
    return out


def compute_hod_samples(catalog, mlims=(-17, -18, -19, -20), z_max=None, *, columns=None,
                        cosmology=None, save_dir=None, prefix="BST", **kwargs):
    """Run :func:`compute_hod` for several volume-limited samples.

    This replaces the loop of ``BST_example_for_DC2.py``. The catalogue is read
    and indexed only once.

    Parameters
    ----------
    catalog : str or DataFrame
        **Required.** The galaxy table.
    mlims : sequence of float
        Absolute-magnitude limits.
    z_max : None, float, sequence or dict
        Redshift limit per sample (None: defaults, see :func:`compute_hod`).
    save_dir : str, optional
        If given, each result is written to ``{save_dir}/{prefix}_Mlim_{mlim}.csv``.
    **kwargs
        Any other option of :func:`compute_hod`.

    Returns
    -------
    dict {mlim: DataFrame}
    """
    prep = _prepare(catalog, columns, cosmology or COSMODC2)
    if z_max is None or np.isscalar(z_max):
        zs = {m: z_max for m in mlims}
    elif isinstance(z_max, dict):
        zs = dict(z_max)
    else:
        zs = dict(zip(mlims, z_max))
    results = {}
    for m in mlims:
        save = os.path.join(save_dir, f"{prefix}_Mlim_{m}.csv") if save_dir else None
        results[m] = compute_hod(prep, m, zs.get(m), cosmology=prep.cosmo, save=save, **kwargs)
    return results


def _jackknife_labels(ra, dec, n_jack):
    """Split the sky into ~n_jack regions with equal numbers of centres
    (RA strips, each split in Dec)."""
    n_ra = max(1, int(round(np.sqrt(n_jack))))
    n_dec = max(1, int(np.ceil(n_jack / n_ra)))
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    labels = np.zeros(ra.size, dtype=int)
    ra_q = np.quantile(ra, np.linspace(0, 1, n_ra + 1)[1:-1]) if n_ra > 1 else []
    strip = np.searchsorted(ra_q, ra, side="right")
    for s in range(n_ra):
        m = strip == s
        if not m.any():
            continue
        dec_q = np.quantile(dec[m], np.linspace(0, 1, n_dec + 1)[1:-1]) if n_dec > 1 else []
        labels[m] = s * n_dec + np.searchsorted(dec_q, dec[m], side="right")
    return labels


def _mean_and_error(values, method, n_boot, rng, labels):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)] if labels is None else v
    n = v.size
    if n == 0:
        return np.nan, np.nan
    mean = np.nanmean(v)
    if n == 1:
        return mean, np.nan
    if method == "std":
        return mean, np.nanstd(v, ddof=1) / np.sqrt(n)
    if method == "bootstrap":
        idx = rng.integers(0, n, size=(n_boot, n))
        return mean, np.nanstd(np.nanmean(v[idx], axis=1), ddof=1)
    if method == "jackknife":
        regions = np.unique(labels)
        if regions.size < 2:
            return mean, np.nan
        k = regions.size
        means = np.array([np.nanmean(v[labels != r]) if np.any(labels != r) else np.nan
                          for r in regions])
        means = means[np.isfinite(means)]
        return mean, np.sqrt((k - 1) / k * np.sum((means - means.mean()) ** 2))
    raise ValueError("errors must be 'bootstrap', 'jackknife' or 'std'")


def hod_curve(result, mass_bins=None, *, errors="bootstrap", n_boot=1000, n_jack=16,
              jack_labels=None, min_halos=1, seed=42):
    """Bin the per-halo counts in halo mass: <N|M> with uncertainties.

    Parameters
    ----------
    result : DataFrame
        Output of :func:`compute_hod`.
    mass_bins : array, optional
        Edges in log10(M200). Default: 0.25 dex bins covering the data.
    errors : "bootstrap" | "jackknife" | "std"
        Bootstrap over halos in each bin, delete-one jackknife over sky regions,
        or standard error of the mean.
    n_jack : int
        Number of sky regions for the jackknife (ignored if ``jack_labels``).
    jack_labels : array, optional
        Your own region label for each row of ``result``.
    min_halos : int
        Bins with fewer halos are dropped.

    Returns
    -------
    DataFrame with columns ``logM_lo``, ``logM_hi``, ``logM_mid``,
    ``logM_mean``, ``n_halos``, ``N_bst``, ``N_bst_err`` and, for mocks,
    ``N_true``, ``N_true_err``.
    """
    logm = result["log_M200"].to_numpy(dtype=float)
    if mass_bins is None:
        lo = np.floor(np.nanmin(logm) * 4) / 4
        hi = np.ceil(np.nanmax(logm) * 4) / 4 + 1e-9
        mass_bins = np.arange(lo, hi + 0.25, 0.25)
    mass_bins = np.asarray(mass_bins, dtype=float)
    rng = np.random.default_rng(seed)
    if errors == "jackknife":
        jack_labels = (_jackknife_labels(result["ra"], result["dec"], n_jack)
                       if jack_labels is None else np.asarray(jack_labels))
    has_true = "N_true" in result and np.isfinite(result["N_true"].to_numpy(dtype=float)).any()

    rows = []
    which = np.digitize(logm, mass_bins) - 1
    for b in range(mass_bins.size - 1):
        m = which == b
        if m.sum() < min_halos:
            continue
        lab = jack_labels[m] if errors == "jackknife" else None
        row = {"logM_lo": mass_bins[b], "logM_hi": mass_bins[b + 1],
               "logM_mid": 0.5 * (mass_bins[b] + mass_bins[b + 1]),
               "logM_mean": logm[m].mean(), "n_halos": int(m.sum())}
        row["N_bst"], row["N_bst_err"] = _mean_and_error(
            result["N_bst"].to_numpy(dtype=float)[m], errors, n_boot, rng, lab)
        if has_true:
            row["N_true"], row["N_true_err"] = _mean_and_error(
                result["N_true"].to_numpy(dtype=float)[m], errors, n_boot, rng, lab)
        rows.append(row)
    return pd.DataFrame(rows)
