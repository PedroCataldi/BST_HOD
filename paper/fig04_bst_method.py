r"""Source notebook: Articulo1/scripts/Draft/Fig1_BST_Method.ipynb, cells 0-2
(saved there as Fig1_BST_Method.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 4: the BST for one group. Galaxies in the aperture r < r_c (blue) and in
the control ring r_in < r < r_out (orange), with the pixels used for the area
ratio (red, green). The group is the 50th most massive of the centrals with
M_r < -21 at z < 0.1345; r_c = 1.5 r200, r_in = 1.5 r_c, r_out = 3.5 r_c.

As in the notebook, this figure uses a different cosmology (H0 = 70,
Om0 = 0.3) and comoving (not physical) projected distances. The notebook kept
only pixels hit by ~1e8 random points drawn (unseeded) uniformly in
RA in [0, 90] deg, Dec in [-90, 0] deg. Here the same random points are drawn
only inside the plotted region (same distribution, much less memory), with a
fixed seed.

Reads the mock galaxy table (heavy file, ~0.2 GB).

    python paper/fig04_bst_method.py --mocks /path/to/mocks [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper/fig04_bst_method.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig04_bst_method.pdf

    # copy of that folder downloaded from Drive
    python paper/fig04_bst_method.py --mocks /home/tavernam/projects/lsst_hod/drive/mocks

Instead of --mocks you can set the environment variable BST_HOD_MOCKS; without
it the table is looked for in the --tabs folder.
Without --out the figure goes to paper/figures/fig04_bst_method.pdf.

Input: ``mock_21_large_z05_cosmoDC2.mpeg``.
"""

import math

import numpy as np

from common import comoving_distance, load_mock, parse_args, save

NEEDS = ("mocks",)
H0, OM0, OML = 70.0, 0.3, 0.71        # as in the notebook (Om + OL = 1.01 in rho_crit)
G = 43007.1 / 1.0e10
Z_MAX = 0.1345
GROUP_RANK = 50                        # ind_sort[-50]: 50th most massive
N_GRID = 56 * 56
SEED = 2026


def _projected(alfa0, delta0, alfa, delta, d):
    radio = np.sin(delta0) * np.sin(delta) + np.cos(delta0) * np.cos(delta) * np.cos(alfa0 - alfa)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(1.0 - radio * radio) / radio * d


def make_figure(mocks, seed=SEED):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"axes.linewidth": 0.8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "font.size": 8, "font.family": "serif", "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    gal = load_mock(mocks, "z05")
    gal = gal[(gal["loghalo_mass"] > 11.0) & (gal["z"] > 0.0)]
    alfa = np.radians(gal["ra"].to_numpy())
    alfa[alfa > np.pi] -= 2 * np.pi
    delta = np.radians(gal["dec"].to_numpy())
    z = gal["z"].to_numpy()
    d_com = comoving_distance(z, H0, OM0)
    mabs = gal["mr_lsst"].to_numpy() - 25.0 - 5 * np.log10(d_com * (1 + z))

    sel = (gal["is_central"].to_numpy() == 1) & (mabs < -21) & (z < Z_MAX)
    print("n_sample:", sel.sum())
    log_m = gal["loghalo_mass"].to_numpy()[sel]
    j = np.argsort(log_m)[-GROUP_RANK]
    alfa0, delta0, z0 = alfa[sel][j], delta[sel][j], z[sel][j]
    a = 1.0 / (1 + z0)
    adot = a * H0 * (OML + OM0 * a ** -3 - (OM0 + OML - 1) * a ** -2) ** 0.5
    rho_crit = 3 * (1.0e-3 * adot / (H0 / 100 * a)) ** 2 / (8.0 * np.pi * G)
    r200 = max((3 * 10 ** log_m[j] / (800.0 * np.pi * rho_crit)) ** (1 / 3.0) / 1000.0, 0.2)
    d0 = comoving_distance(z0, H0, OM0)
    r_ap = 1.5 * r200
    r_out, r_in = 3.5 * r_ap, 1.5 * r_ap

    s = _projected(alfa0, delta0, alfa, delta, d0)
    in_circ = s < r_ap
    in_ring = (s > r_in) & (s < r_out)
    outside = (s > r_out) | ((s > r_ap) & (s < r_in))

    # pixel grid of the original BST code (edges span whole radians)
    ext = np.arctan(r_out / d0)
    xi = np.linspace(np.floor(alfa0 - ext), np.ceil(alfa0 + ext), N_GRID)
    yi = np.linspace(np.floor(delta0 - ext / np.cos(alfa0)), np.ceil(delta0 + ext / np.cos(alfa0)),
                     N_GRID)
    # only the pixels around the group matter; keep a margin of 2 pixels
    ix = np.flatnonzero((xi > alfa0 - 2 * ext) & (xi < alfa0 + 2 * ext))
    iy = np.flatnonzero((yi > delta0 - 2 * ext) & (yi < delta0 + 2 * ext))
    xe = xi[ix[0] - 1:ix[-1] + 2]
    ye = yi[iy[0] - 1:iy[-1] + 2]
    # random points of the notebook (N = 100 x galaxies, uniform on the sphere in
    # RA [0, pi/2], Dec [-pi/2, 0]) that fall in this sub-grid
    rng = np.random.default_rng(seed)
    n_rand = 100 * alfa.size
    xa, xb = max(xe[0], 0.0), min(xe[-1], np.pi / 2)
    ya, yb = max(ye[0], -np.pi / 2), min(ye[-1], 0.0)
    p_box = (xb - xa) / (np.pi / 2) * (np.sin(yb) - np.sin(ya))
    n_box = rng.binomial(n_rand, p_box)
    h_rand, _, _ = np.histogram2d(rng.uniform(xa, xb, n_box),
                                  np.arcsin(rng.uniform(np.sin(ya), np.sin(yb), n_box)),
                                  bins=(xe, ye))
    cx = xe[:-1] + np.diff(xe) / 2
    cy = ye[:-1] + np.diff(ye) / 2
    xx, yy = np.meshgrid(cx, cy)
    sp = _projected(alfa0, delta0, xx, yy, d0)
    pix_circ = (sp < r_ap) & (h_rand.T > 0)
    pix_ring = (sp > r_in) & (sp < r_out) & (h_rand.T > 0)

    deg = 180.0 / np.pi
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    ax.plot(alfa[in_circ] * deg, delta[in_circ] * deg, ".", label="Galaxies in circle", ms=0.6)
    ax.plot(alfa[in_ring] * deg, delta[in_ring] * deg, ".", color="orange",
            label="Galaxies in ring", ms=0.6)
    ax.plot(xx[pix_circ] * deg, yy[pix_circ] * deg, ".", ms=1, color="r", alpha=0.5, zorder=-10)
    ax.plot(xx[pix_ring] * deg, yy[pix_ring] * deg, ".", color="green", alpha=0.5, ms=1,
            zorder=-10)
    ax.plot(alfa[outside] * deg, delta[outside] * deg, ".", color="gray", ms=0.3)

    ax.minorticks_on()
    ax.tick_params("both", length=3, width=0.8, which="major", direction="in", right=True,
                   top=True)
    ax.tick_params("both", length=2, width=0.5, which="minor", direction="in", right=True,
                   top=True)
    ax.set_xlabel("Right Ascension", fontsize=8)
    ax.set_ylabel("Declination", fontsize=8)
    ring_x, ring_y = xx[pix_ring] * deg, yy[pix_ring] * deg
    ax.set_xlim(ring_x.max() + 0.1, ring_x.min() - 0.1)
    ax.set_ylim(ring_y.min() - 0.1, ring_y.max() + 0.1)
    ax.xaxis.set_inverted(True)

    head = 0.05
    for angle, length in ((np.pi, r_out / 3.5), (np.pi / 6.0, r_in / 3.8),
                          (1.7 * np.pi, r_in / 5.8)):
        ax.arrow(alfa0 * deg, delta0 * deg, (length - head) * math.cos(angle),
                 (length - head) * math.sin(angle), head_width=0.05, head_length=head, fc="k",
                 ec="k", zorder=100)
    box = dict(facecolor="w", edgecolor="k", linewidth=0.5)
    ax.text(61.0, -33.0, r"$r_{\mathrm{out}}$", ha="right", va="top", bbox=box, fontsize=8)
    ax.text(62.5, -33.0, r"$r_{\mathrm{in}}$", ha="right", va="top", bbox=box, fontsize=8)
    ax.text(61.7, -33.25, r"$r_{\mathrm{c}}$", ha="right", va="top", bbox=box, fontsize=8)
    return fig, {}


if __name__ == "__main__":
    args = parse_args(__doc__, "paper/figures/fig04_bst_method.pdf", NEEDS)
    fig, _ = make_figure(args.mocks)
    save(fig, args.out)
