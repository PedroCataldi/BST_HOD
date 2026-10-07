"""Central Galaxy Finder (CGF), rewritten from the original ``CGF_Mlim.py``.

Galaxies brighter than ``mabs_central_max`` (and than ``mlim``, with
z < z_max) are candidate centres. Going from the brightest to the faintest,
each surviving candidate is kept as a centre and every other candidate within
``factor_radius * r_lum`` (projected physical distance) and ``|dz| < dz`` is
discarded. ``r_lum`` is a luminosity-dependent radius

    r_lum = (L - intercept) / slope,     L = 10**(-0.4 M)

with the "new" linear fit of the original script.

The result is the same as the original loop (see ``tests/``), but a KD-tree
finds the neighbours instead of scanning all candidates for every target.
"""

import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .core import _unit_vectors, to_radians
from .cosmology import COSMODC2, default_z_max

#: (slope, intercept) of the luminosity-radius relation ("New" in CGF_Mlim.py)
LUM_RADIUS_NEW = (1634917221.9876266, -43596515.20330819)
#: the older fit, commented out in CGF_Mlim.py
LUM_RADIUS_OLD = (1868175147.2854986, -42424487.794043064)


def find_central_galaxies(catalog, mlim, z_max=None, *, columns=None, cosmology=None,
                          m_app_lim=21.0, mabs_central_max=-19.5, factor_radius=2.5, dz=0.05,
                          lum_radius=LUM_RADIUS_NEW, n_jobs=-1, verbose=True, save=None):
    """Identify central galaxies without using halo membership.

    Parameters
    ----------
    catalog : str or DataFrame
        **Required.** The galaxy table (see :func:`bst_hod.load_catalog`).
    mlim : float
        Absolute-magnitude limit of the sample.
    z_max : float, optional
        Redshift limit (default as in :func:`bst_hod.compute_hod`).
    mabs_central_max : float
        Candidates must be brighter than this (-19.5).
    factor_radius : float
        Exclusion radius in units of r_lum (2.5 in the original run).
    dz : float
        Redshift window for the exclusion (0.05).
    lum_radius : (slope, intercept)
        Coefficients of the luminosity-radius relation.
    save : str, optional
        Write the result to .csv.

    Returns
    -------
    DataFrame with one row per central galaxy found (brightest first):
    ``galaxy_id``, ``ra``, ``dec``, ``z``, ``mabs``, ``r_lum`` and, if present
    in the table, ``halo_id``, ``is_central`` (mock truth) and ``log_halo_mass``.
    Pass it to ``compute_hod(..., centres=result)`` or use ``centres="cgf"``.
    """
    from .hod import _prepare  # local import to avoid a cycle

    t0 = time.time()
    cosmo = cosmology or COSMODC2
    prep = _prepare(catalog, columns, cosmo)
    cat = prep.cat
    if z_max is None:
        z_max = default_z_max(mlim, m_app_lim, cosmo)

    sel = (prep.z < z_max) & (prep.mabs < mlim)
    if mabs_central_max is not None:
        sel &= prep.mabs < mabs_central_max
    cand = np.flatnonzero(sel)
    mabs = prep.mabs[cand]
    order = mabs.argsort()                    # brightest first, as the original
    cand = cand[order]
    mabs = mabs[order]
    z = prep.z[cand]
    alfa, delta = to_radians(cat["ra"].to_numpy()[cand], cat["dec"].to_numpy()[cand])
    d_phys = prep.d_com[cand] / (1 + z)
    slope, intercept = lum_radius
    r_lum = (10 ** (-0.4 * mabs) - intercept) / slope
    radius = factor_radius * r_lum

    n = cand.size
    removed = np.zeros(n, dtype=bool)
    keep = []
    if n:
        xyz = _unit_vectors(alfa, delta)
        tree = cKDTree(xyz)
        theta = np.arctan(np.clip(radius, 0, None) / d_phys)
        chord = 2 * np.sin(np.minimum(theta * (1 + 1e-6) + 1e-9, np.pi) / 2)
        neigh = tree.query_ball_point(xyz, chord, workers=n_jobs)
        for i in range(n):
            if removed[i]:
                continue
            keep.append(i)
            nb = np.asarray(neigh[i], dtype=np.intp)
            nb = nb[~removed[nb]]
            if nb.size == 0:
                continue
            radio = (np.sin(delta[i]) * np.sin(delta[nb])
                     + np.cos(delta[i]) * np.cos(delta[nb]) * np.cos(alfa[i] - alfa[nb]))
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.sqrt(1.0 - radio * radio) / radio * d_phys[i]
            hit = (s < radius[i]) & (s > 0) & (np.abs(z[nb] - z[i]) < dz)
            removed[nb[hit]] = True

    keep = np.asarray(keep, dtype=int)
    rows = cand[keep]
    out = pd.DataFrame({
        "galaxy_id": cat["galaxy_id"].to_numpy()[rows],
        "ra": cat["ra"].to_numpy()[rows],
        "dec": cat["dec"].to_numpy()[rows],
        "z": z[keep],
        "mabs": mabs[keep],
        "r_lum": r_lum[keep],
    })
    for col in ("halo_id", "is_central", "log_halo_mass"):
        if col in cat:
            out[col] = cat[col].to_numpy()[rows]
    out.attrs.update(dict(mlim=mlim, z_max=z_max, mabs_central_max=mabs_central_max,
                          factor_radius=factor_radius, dz=dz, lum_radius=tuple(lum_radius)))
    if verbose:
        msg = f"[bst_hod] CGF M_lim = {mlim}: {len(out)} centres from {n} candidates"
        if "is_central" in out:
            msg += f" (purity vs mock truth: {np.mean(out['is_central'] == 1):.3f})"
        print(msg + f" in {time.time() - t0:.1f} s")
    if save:
        out.to_csv(save, index=False)
    return out
