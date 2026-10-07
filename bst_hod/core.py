"""Core of the Background Subtraction Technique (BST).

For one group/halo centre the number of member galaxies is estimated as
(eq. 2 of Rodriguez et al. 2015)

    N = N_circle - N_ring * A_circle / A_ring

where ``N_circle`` counts the galaxies projected inside the aperture of
physical radius ``r_ap`` around the centre and ``N_ring`` counts the galaxies
in the control annulus ``r_in < r < r_out``. Distances are projected physical
distances at the redshift of the centre, and (when ``mag_cut=True``) galaxies
are counted only if their absolute magnitude *computed at the distance of the
centre* is brighter than ``mlim``.

This module is a rewrite of ``background_method_new`` /
``background_method_rmax`` from the original ``BST_method.py`` that gives the
same numbers (see ``tests/``) but only looks at galaxies and pixels near each
centre, so it is orders of magnitude faster. The original functions are kept
untouched in :mod:`bst_hod.legacy`.
"""

import numpy as np
from scipy.spatial import cKDTree

#: Number of grid edges per axis in the original code (resolucion * 56 = 56 * 56).
N_GRID_ORIGINAL = 56 * 56


def to_radians(ra_deg, dec_deg):
    """RA/Dec in degrees -> radians, RA wrapped to (-pi, pi] (as the original)."""
    alfa = np.asarray(ra_deg, dtype=float) * np.pi / 180.0
    delta = np.asarray(dec_deg, dtype=float) * np.pi / 180.0
    alfa = np.where(alfa > np.pi, alfa - 2.0 * np.pi, alfa)
    return alfa, delta


def _unit_vectors(alfa, delta):
    cd = np.cos(delta)
    return np.column_stack((cd * np.cos(alfa), cd * np.sin(alfa), np.sin(delta)))


class GalaxyField:
    """All galaxies that can fall inside an aperture or a control ring.

    Builds a KD-tree once so each centre only looks at its neighbours.

    Parameters
    ----------
    ra, dec : array, degrees
    mag : array
        Apparent magnitudes (used for the magnitude cut in the centre frame).
    """

    def __init__(self, ra, dec, mag):
        self.alfa, self.delta = to_radians(ra, dec)
        self.mag = np.asarray(mag, dtype=float)
        ok = np.isfinite(self.alfa) & np.isfinite(self.delta)
        self._index = np.flatnonzero(ok)
        self.tree = cKDTree(_unit_vectors(self.alfa[ok], self.delta[ok]))

    def __len__(self):
        return self.alfa.size

    def neighbours(self, alfa0, delta0, theta, workers=1):
        """Indices of galaxies within angle ``theta`` (array) of each centre."""
        alfa0, delta0, theta = map(np.atleast_1d, (alfa0, delta0, theta))
        # small safety margin: candidates are a superset, exact cuts come later
        th = np.minimum(theta * (1.0 + 1e-6) + 1e-9, np.pi)
        chord = 2.0 * np.sin(th / 2.0)
        lists = self.tree.query_ball_point(_unit_vectors(alfa0, delta0), chord,
                                           workers=workers)
        return [self._index[np.asarray(l, dtype=np.intp)] for l in lists]


def aperture_radii(r200, factor_r=1.0, r_min=0.05, r_max=1.25,
                   ring="classic", ring_radius=(1.5, 3.5)):
    """Aperture and control-ring radii (physical Mpc), as in background_method_new.

    r_ap = factor_r * r200, clipped to [r_min, r_max] (r_max=None: no upper clip).
    ring="classic": r_in, r_out = ring_radius[0] * r_ap, ring_radius[1] * r_ap
    ring="fix":     r_in, r_out = r_ap + 1 Mpc, r_ap + 2 Mpc
    """
    r200 = np.asarray(r200, dtype=float)
    r_ap = np.asarray(factor_r, dtype=float) * r200
    r_ap = np.where(r_ap < r_min, r_min, r_ap)
    if r_max is not None:
        r_ap = np.where(r_ap > r_max, r_max, r_ap)
    if ring == "fix":
        r_out = r_ap + 2.0
        r_in = r_ap + 1.0
    elif ring == "classic":
        r_out = ring_radius[1] * r_ap
        r_in = ring_radius[0] * r_ap
    else:
        raise ValueError("ring must be 'classic' or 'fix'")
    return r_ap, r_in, r_out


def _projected_distance(alfa0, delta0, alfa, delta, d_phys, clip_cos=False):
    # identical expressions to the original code
    radio = np.sin(delta0) * np.sin(delta) + np.cos(delta0) * np.cos(delta) * np.cos(alfa0 - alfa)
    if clip_cos:  # rounding can give cos > 1 at zero separation -> NaN distance
        radio = np.minimum(radio, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tang_tita = np.sqrt(1.0 - radio * radio) / radio
    return tang_tita * d_phys, radio


def pixel_counts(alfa0, delta0, d_phys, r_ap, r_in, r_out, n_grid=N_GRID_ORIGINAL):
    """Number of grid pixels inside the aperture and inside the ring.

    Reproduces exactly the pixel grid of the original code (same edges, same
    pixel centres, same distance formula) but only evaluates the pixels that
    can be within ``r_out`` of the centre. Returns ``None`` where the original
    code raised ``ValueError`` (and therefore returned N = 0).
    """
    radioext_rad = np.arctan(r_out / d_phys)
    xmin, xmax = alfa0 - radioext_rad, alfa0 + radioext_rad
    ymin = delta0 - radioext_rad / np.cos(alfa0)   # sic: cos(alfa) as in the original
    ymax = delta0 + radioext_rad / np.cos(alfa0)
    bounds = np.array([xmin, xmax, ymin, ymax])
    if not np.all(np.isfinite(bounds)):
        return None
    xi = np.linspace(np.floor(xmin), np.ceil(xmax), n_grid)
    yi = np.linspace(np.floor(ymin), np.ceil(ymax), n_grid)
    if xi[0] > xi[-1] or yi[0] > yi[-1]:   # numpy.histogram2d raised ValueError here
        return None
    cx = xi[:-1] + (xi[1:] - xi[:-1]) / 2
    cy = yi[:-1] + (yi[1:] - yi[:-1]) / 2

    def count(xs, ys):
        xx, yy = np.meshgrid(xs, ys)
        s, _ = _projected_distance(alfa0, delta0, xx, yy, d_phys)
        return int(np.count_nonzero(s < r_ap)), int(np.count_nonzero((s > r_in) & (s < r_out)))

    # Can we safely restrict to a window around the centre? Only if no pixel of
    # the grid is more than 90 deg away (those have cos < 0 and the original
    # formula counts them as "inside" the aperture).
    small = (cx[-1] - cx[0] <= np.pi) and (cy[0] >= -np.pi / 2) and (cy[-1] <= np.pi / 2)
    if small:
        _, r_first = _projected_distance(alfa0, delta0, cx[0], cy, d_phys)
        _, r_last = _projected_distance(alfa0, delta0, cx[-1], cy, d_phys)
        small = bool(np.all(r_first > 0) and np.all(r_last > 0))
    if not small:
        return count(cx, cy)

    dx = abs(cx[1] - cx[0]) if cx.size > 1 else 0.0
    dy = abs(cy[1] - cy[0]) if cy.size > 1 else 0.0
    if abs(delta0) + radioext_rad < np.pi / 2 - 1e-6:
        half_x = np.arcsin(min(1.0, np.sin(radioext_rad) / np.cos(delta0)))
    else:
        half_x = np.pi
    dxs = np.abs((cx - alfa0 + np.pi) % (2 * np.pi) - np.pi)
    sel_x = cx[dxs <= half_x * (1 + 1e-6) + 2 * dx]
    sel_y = cy[np.abs(cy - delta0) <= radioext_rad * (1 + 1e-6) + 2 * dy]
    if sel_x.size == 0 or sel_y.size == 0:
        return 0, 0
    return count(sel_x, sel_y)


def bst_count(field, idx, alfa0, delta0, z0, d_com0, r_ap, r_in, r_out, mlim,
              mag_cut=True, area="pixel", n_grid=N_GRID_ORIGINAL, clip_cos=False):
    """BST estimate for a single centre.

    Parameters
    ----------
    field : GalaxyField
    idx : array of int
        Indices (into ``field``) of the candidate galaxies around this centre,
        usually from ``field.neighbours(...)`` with angle atan(r_out/d_phys).
    alfa0, delta0 : float
        Centre position in radians (RA wrapped to (-pi, pi]).
    z0, d_com0 : float
        Redshift and comoving distance [Mpc] of the centre.
    r_ap, r_in, r_out : float
        Aperture and ring radii in physical Mpc (see :func:`aperture_radii`).
    mlim : float
        Absolute-magnitude limit of the sample.
    mag_cut : bool
        True: count only galaxies with M (at the centre's distance) < mlim
        (as ``background_method_rmax``, used in BST_example_for_DC2.py).
        False: count every galaxy given (as ``background_method_new``).
    area : {"pixel", "analytic"}
        "pixel" reproduces the original pixel-grid estimate of A_circle/A_ring.
        "analytic" uses r_ap^2 / (r_out^2 - r_in^2) (exact flat-sky ratio).
    clip_cos : bool
        In the original formula, rounding makes cos(theta) slightly > 1 for
        ~2% of galaxies at zero separation, so the central galaxy is not
        counted in its own aperture (N low by 1). True fixes this; False
        (default) reproduces the original numbers.

    Returns
    -------
    N, n_circle, n_ring, area_ratio : float, int, int, float
    """
    d_phys = d_com0 / (1.0 + z0)
    a = field.alfa[idx]
    d = field.delta[idx]
    s, _ = _projected_distance(alfa0, delta0, a, d, d_phys, clip_cos)
    in_circ = s < r_ap
    in_ring = (s > r_in) & (s < r_out)
    if mag_cut:
        mabs_hod = field.mag[idx] - 25.0 - 5 * np.log10(d_com0 * (1 + z0))
        bright = mabs_hod < mlim
        in_circ &= bright
        in_ring &= bright
    n_circ = int(np.count_nonzero(in_circ))
    n_ring = int(np.count_nonzero(in_ring))

    if area == "analytic":
        denom = r_out ** 2 - r_in ** 2
        ratio = r_ap ** 2 / denom if denom > 0 else np.nan
        N = n_circ - n_ring * ratio if denom > 0 else 0.0
        return float(N), n_circ, n_ring, ratio
    if area != "pixel":
        raise ValueError("area must be 'pixel' or 'analytic'")

    pix = pixel_counts(alfa0, delta0, d_phys, r_ap, r_in, r_out, n_grid)
    if pix is None:
        return 0.0, n_circ, n_ring, np.nan
    n_pix_circ, n_pix_ring = pix
    if n_pix_ring > 0:
        N = n_circ - n_ring * n_pix_circ / n_pix_ring
        ratio = n_pix_circ / n_pix_ring
    else:
        N, ratio = 0, np.nan
    return float(N), n_circ, n_ring, ratio
