r"""Source notebook: Articulo1/scripts/Draft/Fig10_purity.ipynb, cell 1
(saved there as Fig10_purity.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 6: purity (top) and completeness (bottom) of the centrals found by the CGF
with (f_r, f_z) = (2.5, 0.25), vs M_r, M_star and M200, for each M_lim sample.
Purity = fraction of CGF centres that are true centrals; completeness =
fraction of the true centrals (candidates with M_r < -19.5, z < z_max) found.

Note: as in the notebook, this uses the CGF_fr_* runs. Table 2 of the paper and
the BST runs of Fig. 12 use the CGFnew_fr_* runs, which have fewer centres
(e.g. 13179 vs 19063 for M_lim = -19).

Reads the mock galaxy table (heavy file, ~0.2 GB) and the CGF .npz files.

    python paper_plots/fig06_cgf_purity.py --tabs /path/to/Tabs --mocks /path/to/mocks [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig06_cgf_purity.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig06_cgf_purity.pdf

    # copy of that folder downloaded from Drive
    python paper_plots/fig06_cgf_purity.py --tabs /home/tavernam/projects/lsst_hod/drive/Articulo1/tabs \
        --mocks /home/tavernam/projects/lsst_hod/drive/mocks

Instead of --tabs / --mocks you can set BST_HOD_TABS / BST_HOD_MOCKS; without
--mocks the mock table is looked for in the --tabs folder.
Without --out the figure goes to paper_plots/output/fig06_cgf_purity.pdf.

Input: ``CGF_fr_2.5_fz_0.25Mlim_{m}.0.npz`` and ``mock_21_large_z05_cosmoDC2.mpeg``.
"""

import numpy as np
from scipy.stats import binned_statistic

from common import COLORS, MLIMS, abs_mag, load_mock, parse_args, save, tabs_path

NEEDS = ("tabs", "mocks")
CGF_FILE = "CGF_fr_2.5_fz_0.25Mlim_{m:.1f}.npz"
Z_CUTS = (0.0875, 0.1345, 0.2055, 0.3038)


def make_figure(tabs, mocks):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 6, "axes.labelsize": 6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                         "axes.linewidth": 0.7, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "text.usetex": False})
    fig, ax = plt.subplots(2, 3, figsize=(3.46, 3.6), sharex="col", sharey="row", dpi=300)

    table = load_mock(mocks, "z05")
    gal = table[table["z"] > 0.0]
    z = gal["z"].to_numpy()
    mabs = abs_mag(gal["mr_lsst"].to_numpy(), z)
    stellar = 10 ** gal["logstellar_mass"].to_numpy()
    m200 = 10 ** gal["loghalo_mass"].to_numpy()
    is_central = gal["is_central"].to_numpy()
    bins_star = np.logspace(9.0, 12.5, 20)
    bins_m200 = np.logspace(10.5, 15, 20)

    for m, (mlim, zc) in enumerate(zip(MLIMS, Z_CUTS)):
        bins_mag = np.linspace(-24, -19, 20) if m < 3 else np.linspace(-24, -20, 18)
        # true centrals among the candidates (denominator of the completeness)
        cand = (mabs < -19.5) & (z < zc)
        n_cand = [binned_statistic(x[cand], is_central[cand], bins=b, statistic=np.sum)[0]
                  for x, b in ((mabs, bins_mag), (stellar, bins_star), (m200, bins_m200))]

        with np.load(tabs_path(tabs, CGF_FILE.format(m=mlim))) as d:
            gal_id = d["Gal_id"]
        gal_id = gal_id[~np.isnan(gal_id)]
        sel = table[table["galid"].isin(gal_id)]
        z_s = sel["z"].to_numpy()
        quantities = (abs_mag(sel["mr_lsst"].to_numpy(), z_s),
                      10 ** sel["logstellar_mass"].to_numpy(),
                      10 ** sel["loghalo_mass"].to_numpy())
        cen_s = sel["is_central"].to_numpy()
        label = f"$M_{{\\rm lim}}<{mlim}$"
        for col, (x, b) in enumerate(zip(quantities, (bins_mag, bins_star, bins_m200))):
            centres = 0.5 * (b[:-1] + b[1:])
            purity = binned_statistic(x, cen_s, bins=b, statistic=np.average)[0]
            found = binned_statistic(x, cen_s, bins=b, statistic=np.sum)[0]
            ax[0, col].plot(centres, 100 * purity, color=COLORS[m], lw=1.0,
                            label=label if col == 0 else None)
            with np.errstate(invalid="ignore", divide="ignore"):
                ax[1, col].plot(centres, 100 * found / n_cand[col], color=COLORS[m], lw=1.0)

    for j in range(2):
        ax[j, 0].invert_xaxis()
        ax[j, 1].set_xscale("log")
        ax[j, 2].set_xscale("log")
    ax[0, 0].set_ylabel(r"$\%$ Purity", fontsize=6)
    ax[1, 0].set_ylabel(r"$\%$ Completeness", fontsize=6)
    ax[1, 0].set_xlabel(r"$M_r$", fontsize=6)
    ax[1, 1].set_xlabel(r"$M_\star \, \mathrm{[M_\odot]}$", fontsize=6)
    ax[1, 2].set_xlabel(r"$M\mathrm{_{200} \, [M_\odot]}$", fontsize=6)
    for a in ax.flat:
        a.minorticks_on()
        a.tick_params(axis="both", which="both", width=0.8)
        a.tick_params(axis="both", which="major", length=2)
        a.tick_params(axis="both", which="minor", length=1)
    for j in range(3):
        ax[0, j].set_ylim(60, 100)
        ax[1, j].set_ylim(0, 100)
    ax[0, 0].legend(frameon=False, loc="lower right", fontsize=6, handlelength=2)
    ax[1, 2].set_xticks([1e11, 1e13, 1e15])
    ax[1, 1].set_xticks([1e10, 1e11, 1e12])
    fig.subplots_adjust(wspace=0.05, hspace=0.07)
    return fig, {}


if __name__ == "__main__":
    args = parse_args(__doc__, "fig06_cgf_purity.pdf", NEEDS)
    fig, _ = make_figure(args.tabs, args.mocks)
    save(fig, args.out)
