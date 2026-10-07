"""A small synthetic light-cone for testing and for trying the package without
the DESC mocks. It is *not* a realistic galaxy model: halos get a central plus
Poisson satellites spread around them, on top of a uniform field population.
"""

import numpy as np
import pandas as pd

from .cosmology import COSMODC2, comoving_distance, r200_mpc


def make_mock_catalog(n_halos=3000, n_field=200000, ra_range=(50.0, 70.0),
                      dec_range=(-40.0, -25.0), z_max=0.4, mag_max=21.0, seed=1,
                      cosmo=COSMODC2):
    """Return a DataFrame with the columns of the original DC2 tables
    (``galid, ra, dec, z, mr_lsst, halo_id, loghalo_mass, is_central``)."""
    rng = np.random.default_rng(seed)

    def sky(n):
        ra = rng.uniform(*ra_range, n)
        s0, s1 = np.sin(np.radians(dec_range))
        dec = np.degrees(np.arcsin(rng.uniform(s0, s1, n)))
        return ra, dec

    def redshifts(n):  # roughly uniform in comoving volume
        zg = np.linspace(1e-3, z_max, 2000)
        w = comoving_distance(zg, cosmo) ** 2
        cdf = np.cumsum(w) / w.sum()
        return np.interp(rng.uniform(0, 1, n), cdf, zg)

    # halos and centrals
    logm = 11.5 + rng.exponential(0.6, n_halos)
    logm = logm[logm < 15.0][:n_halos]
    nh = logm.size
    ra_h, dec_h = sky(nh)
    z_h = redshifts(nh)
    m_cen = -20.0 - 1.2 * (logm - 12.0) + rng.normal(0, 0.3, nh)
    r200 = r200_mpc(10 ** logm, z_h, cosmo)
    d_phys = comoving_distance(z_h, cosmo) / (1 + z_h)

    rows = [dict(ra=ra_h, dec=dec_h, z=z_h, M=m_cen, halo=np.arange(nh), cen=np.ones(nh, int),
                 logm=logm)]
    n_sat = rng.poisson(3.0 * 10 ** (logm - 13.0))
    host = np.repeat(np.arange(nh), n_sat)
    if host.size:
        r = r200[host] * np.sqrt(rng.uniform(0, 1, host.size)) ** 1.5
        phi = rng.uniform(0, 2 * np.pi, host.size)
        ang = np.degrees(r / d_phys[host])
        dec_s = dec_h[host] + ang * np.sin(phi)
        ra_s = ra_h[host] + ang * np.cos(phi) / np.cos(np.radians(dec_h[host]))
        m_sat = m_cen[host] + 0.5 + rng.exponential(1.5, host.size)
        rows.append(dict(ra=ra_s, dec=dec_s, z=z_h[host] + rng.normal(0, 3e-4, host.size),
                         M=m_sat, halo=host, cen=np.zeros(host.size, int), logm=logm[host]))
    # field galaxies, each in its own small halo
    ra_f, dec_f = sky(n_field)
    z_f = redshifts(n_field)
    m_f = -17.0 - np.minimum(rng.exponential(1.3, n_field), 5.5)
    rows.append(dict(ra=ra_f, dec=dec_f, z=z_f, M=m_f, halo=nh + np.arange(n_field),
                     cen=np.ones(n_field, int), logm=np.full(n_field, 11.0)))

    d = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    dc = comoving_distance(d["z"], cosmo)
    mag = d["M"] + 25 + 5 * np.log10(dc * (1 + d["z"]))
    df = pd.DataFrame({
        "galid": np.arange(d["ra"].size, dtype=np.int64) + 10_000_000,
        "ra": d["ra"], "dec": d["dec"], "z": d["z"], "mr_lsst": mag,
        "halo_id": d["halo"].astype(np.int64) + 1_000_000, "loghalo_mass": d["logm"],
        "is_central": d["cen"],
    })
    return df[(df["mr_lsst"] <= mag_max) & (df["z"] > 0)].reset_index(drop=True)
