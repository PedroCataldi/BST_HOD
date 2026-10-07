"""Cosmology helpers used by the BST pipeline.

By default the package uses a flat LambdaCDM cosmology with the cosmoDC2 /
SkySim5000 parameters (WMAP-7: H0 = 71 km/s/Mpc, Om0 = 0.2648, no radiation),
which is what the original scripts used through
``astropy.cosmology.FlatLambdaCDM(H0=71, Om0=0.2648)``.

astropy is *not* required: distances are integrated numerically here. If you
prefer, you can pass any astropy cosmology object instead (anything with
``comoving_distance`` and ``efunc`` methods and an ``H0`` attribute).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

C_KM_S = 299792.458          # speed of light [km/s]
G_KPC = 4.30091e-6           # gravitational constant [kpc (km/s)^2 / Msun]


class FlatLCDM:
    """Minimal flat LambdaCDM cosmology (no radiation).

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    Om0 : float
        Matter density parameter today. Ode0 = 1 - Om0.
    """

    def __init__(self, H0=71.0, Om0=0.2648):
        self.H0 = float(H0)
        self.Om0 = float(Om0)
        self.Ode0 = 1.0 - self.Om0
        self._zgrid = None
        self._dcgrid = None

    def __repr__(self):
        return f"FlatLCDM(H0={self.H0}, Om0={self.Om0})"

    def efunc(self, z):
        """E(z) = H(z)/H0."""
        z = np.asarray(z, dtype=float)
        return np.sqrt(self.Om0 * (1.0 + z) ** 3 + self.Ode0)

    def _build_grid(self, zmax):
        zmax = max(float(zmax), 1.0) * 1.2
        z = np.linspace(0.0, zmax, int(40000 * zmax) + 1)
        integrand = 1.0 / self.efunc(z)
        dc = cumulative_trapezoid(integrand, z, initial=0.0) * C_KM_S / self.H0
        self._zgrid, self._dcgrid = z, dc

    def comoving_distance(self, z):
        """Line-of-sight comoving distance in Mpc (plain float/array, no units)."""
        z = np.asarray(z, dtype=float)
        finite = np.isfinite(z)
        zmax = np.nanmax(z) if finite.any() else 1.0
        if self._zgrid is None or zmax > self._zgrid[-1]:
            self._build_grid(zmax)
        out = np.interp(z, self._zgrid, self._dcgrid)
        return np.where(finite, out, np.nan)


#: Default cosmology: cosmoDC2 / SkySim5000 (WMAP-7).
COSMODC2 = FlatLCDM(H0=71.0, Om0=0.2648)


def _h0(cosmo):
    h0 = cosmo.H0
    return float(getattr(h0, "value", h0))


def comoving_distance(z, cosmo=None):
    """Comoving distance in Mpc as a plain numpy array (works with astropy too)."""
    cosmo = cosmo or COSMODC2
    d = cosmo.comoving_distance(z)
    return np.asarray(getattr(d, "value", d), dtype=float)


def absolute_magnitude(mag, z, cosmo=None, d_com=None):
    """Absolute magnitude M = m - 25 - 5 log10(D_L / Mpc), with D_L = D_C (1+z).

    No k-correction is applied (same as the original scripts).
    """
    if d_com is None:
        d_com = comoving_distance(z, cosmo)
    return np.asarray(mag) - 25.0 - 5.0 * np.log10(d_com * (1.0 + np.asarray(z)))


def r200_mpc(m200, z, cosmo=None):
    """Physical R200 (radius enclosing 200 x rho_crit(z)) in Mpc.

    Same formula as BST_example_for_DC2.py: rho_crit = 3 H(z)^2 / (8 pi G) with
    H in km/s/kpc and G in kpc (km/s)^2/Msun, so M200 is in the mass units of
    the catalogue (Msun for cosmoDC2).
    """
    cosmo = cosmo or COSMODC2
    e = np.asarray(cosmo.efunc(z), dtype=float)
    h_km_s_kpc = _h0(cosmo) * e * 1e-3
    rho_crit = 3.0 * h_km_s_kpc ** 2 / (8.0 * np.pi * G_KPC)
    return 1e-3 * (3.0 * np.asarray(m200) / (4.0 * np.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)


#: Redshift limits of the volume-limited samples used in the original
#: BST_example_for_DC2.py / CGF_Mlim.py (apparent limit m_r <= 21).
ORIGINAL_Z_CUTS = {
    -16: 0.0563,
    -17: 0.0875,
    -18: 0.1345,
    -19: 0.2055,
    -20: 0.3038,
    -21: 0.4547,
}


def z_max_for_mlim(mlim, m_app_lim=21.0, cosmo=None):
    """Redshift at which a galaxy of absolute magnitude ``mlim`` has apparent
    magnitude ``m_app_lim``. Below this redshift the sample is volume-limited.

    Note: the values hard-coded in the original scripts (``ORIGINAL_Z_CUTS``)
    differ from this calculation by up to ~0.003 in z; ``default_z_max`` uses
    them for the standard limits so old results are reproduced exactly.
    """
    def f(z):
        return absolute_magnitude(m_app_lim, z, cosmo) - mlim

    return float(brentq(f, 1e-5, 10.0, xtol=1e-10))


def default_z_max(mlim, m_app_lim=21.0, cosmo=None):
    """z_max used when the user does not give one: the original table value for
    M_lim = -16 ... -21 with m_app_lim = 21, otherwise ``z_max_for_mlim``."""
    key = int(mlim) if float(mlim).is_integer() else None
    if m_app_lim == 21.0 and key in ORIGINAL_Z_CUTS:
        return ORIGINAL_Z_CUTS[key]
    return z_max_for_mlim(mlim, m_app_lim, cosmo)
