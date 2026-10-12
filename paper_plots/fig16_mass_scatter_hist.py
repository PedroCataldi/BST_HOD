r"""Source notebook: Articulo1/scripts/Draft/Fig13_BST_M200_dex_Apx.ipynb, cell 1
(saved there as FigApx4_Mass_CosmoDC2.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 16 (appendix B): distribution of the halo masses of the group sample
(centrals with M_r < -19.5): original (green) and with a Gaussian scatter of
0.1 (blue) and 0.2 dex (red) in log10(M200), as used for Fig. 11.

As in the notebook, M_r is computed here with H0 = 70, Om0 = 0.3 (not the
cosmoDC2 cosmology). The notebook drew the scatter with an unseeded random
generator; here the draws are seeded (``SEED``).

Reads the mock galaxy table (heavy file, ~0.2 GB).

    python paper_plots/fig16_mass_scatter_hist.py --mocks /path/to/mocks [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig16_mass_scatter_hist.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig16_mass_scatter_hist.pdf

    # copy of that folder downloaded from Drive
    python paper_plots/fig16_mass_scatter_hist.py --mocks /home/tavernam/projects/lsst_hod/drive/mocks

Instead of --mocks you can set the environment variable BST_HOD_MOCKS; without
it the table is looked for in the --tabs folder.
Without --out the figure goes to paper_plots/output/fig16_mass_scatter_hist.pdf.

Input: ``mock_21_large_z05_cosmoDC2.mpeg``.
"""

import numpy as np

from common import abs_mag, load_mock, parse_args, save

NEEDS = ("mocks",)
SCATTER_DEX = (0.0, 0.1, 0.2)
COLORS = ("g", "b", "r")
SEED = 2026


def make_figure(mocks, seed=SEED):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
                         "axes.linewidth": 0.8, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.major.size": 4,
                         "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
                         "text.usetex": False})
    gal = load_mock(mocks, "z05")
    gal = gal[gal["z"] > 0.0]
    mabs = abs_mag(gal["mr_lsst"].to_numpy(), gal["z"].to_numpy(), H0=70.0, Om0=0.3)
    mask = (gal["is_central"].to_numpy() == 1) & (mabs < -19.5)
    log_m = gal["loghalo_mass"].to_numpy()[mask]

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(1, 1, figsize=(3.46, 3.0), dpi=300)
    bins = np.logspace(10.5, 15.3, 10)
    for dex, color in zip(SCATTER_DEX, COLORS):
        m = 10 ** (log_m + rng.normal(0, dex, size=log_m.size))
        ax.hist(m, bins=bins, histtype="step", linewidth=1.0, color=color, label=f"{dex:.1f} dex")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\mathrm{200}}$ [$\mathrm{M_\odot}$]")
    ax.set_ylabel("Sample count")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.legend(frameon=False, handlelength=1.5)
    fig.tight_layout()
    return fig, {}


if __name__ == "__main__":
    args = parse_args(__doc__, "fig16_mass_scatter_hist.pdf", NEEDS)
    fig, _ = make_figure(args.mocks)
    save(fig, args.out)
