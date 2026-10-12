r"""Source notebook: Articulo1/scripts/Draft/FigApx3_BST_CGF_fr_x_fz_0.25.ipynb, cell 10
(saved there as FigApx3_fr_fz_BST.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 15 (appendix A): BST HOD with centres found by the CGF (true redshifts),
vs the true HOD (black). Top: f_z = 0.25 fixed, f_r = 1.0 ... 3.0. Bottom:
f_r = 2.5 fixed, f_z = 0.05 ... 0.25. Lower panels of each pair: BST/true ratio.

    python paper_plots/fig15_bst_cgf_fr_fz.py --tabs /path/to/Tabs [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig15_bst_cgf_fr_fz.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig15_bst_cgf_fr_fz.pdf

    # copy of that folder downloaded from Drive (Articulo1/tabs)
    python paper_plots/fig15_bst_cgf_fr_fz.py --tabs /home/tavernam/projects/lsst_hod/drive/Articulo1/tabs

Instead of --tabs you can set the environment variable BST_HOD_TABS.
Without --out the figure goes to paper_plots/output/fig15_bst_cgf_fr_fz.pdf.

Input: ``Back_r200_CGF_fr_{f_r}_fz_{f_z}_Mlim_{m}.npz`` and ``NTrue_DC2{m}.npz``.
"""

import numpy as np

from common import (MLIMS, jackknife_mean_bins_arr, load_bst, load_true, parse_args,
                    plot_binned, ratio_with_error, save)
from photoz_grid import finish, make_grid

NEEDS = ("tabs",)
BST_FILE = "Back_r200_CGF_fr_{fr}_fz_{fz}_Mlim_{m}.npz"
MASS_BINS = np.logspace(10.8, 15.4, 9)
F_R = (1.0, 1.5, 2.0, 2.5, 3.0)
F_Z = (0.05, 0.1, 0.15, 0.2, 0.25)
FIXED_FZ, FIXED_FR = 0.25, 2.5


def make_figure(tabs):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = [plt.cm.jet(i) for i in np.linspace(0, 1, 5)]
    cmap = ListedColormap(colors)
    sm_r = ScalarMappable(cmap=cmap, norm=BoundaryNorm([0.75, 1.25, 1.75, 2.25, 2.75, 3.25], 5))
    sm_z = ScalarMappable(cmap=cmap,
                          norm=BoundaryNorm([0.025, 0.075, 0.125, 0.175, 0.225, 0.275], 5))

    fig, ax = make_grid(row_labels=None, title_y=0.90, xlabel_size=14)
    runs = (  # (row, list of (f_r, f_z))
        (0, [(fr, FIXED_FZ) for fr in F_R]),
        (2, [(FIXED_FR, fz) for fz in F_Z]),
    )
    for col, mlim in enumerate(MLIMS):
        m_true, n_true_halo = load_true(tabs, mlim)
        n_true, r_e, err_true = jackknife_mean_bins_arr(m_true, n_true_halo, MASS_BINS)
        for row, params in runs:
            ax[row, col].plot(r_e, n_true, ls="-", lw=1, color="k")
            ax[row, col].axhline(1, color="black", linewidth=0.8)
            for j, (fr, fz) in enumerate(params):
                M, N, _ = load_bst(tabs, BST_FILE.format(fr=fr, fz=fz, m=mlim))
                n_bst, _, err = jackknife_mean_bins_arr(M, N, MASS_BINS)
                plot_binned(ax[row, col], r_e, n_bst, err, colors[j], lw=1, dash_last=False)
                ratio, yerr = ratio_with_error(n_bst, err, n_true, err_true)
                plot_binned(ax[row + 1, col], r_e, ratio, yerr, colors[j], lw=1,
                            dash_last=False)
    finish(fig, ax, wspace=0.1)
    cb1 = fig.colorbar(sm_r, ax=ax[0:2, 3], orientation="vertical", fraction=0.12, pad=0.03,
                       aspect=18, ticks=list(F_R))
    cb1.set_label(r"$f_r \,(f_z=0.25)$", fontsize=14)
    cb2 = fig.colorbar(sm_z, ax=ax[2:4, 3], orientation="vertical", fraction=0.12, pad=0.03,
                       aspect=18, ticks=list(F_Z))
    cb2.set_label(r"$f_z \,(f_r=2.5)$", fontsize=14)
    cb1.ax.tick_params(labelsize=13)
    cb2.ax.tick_params(labelsize=13)
    return fig, {}


if __name__ == "__main__":
    args = parse_args(__doc__, "fig15_bst_cgf_fr_fz.pdf", NEEDS)
    fig, _ = make_figure(args.tabs)
    save(fig, args.out)
