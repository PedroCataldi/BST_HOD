r"""Source notebook: Articulo1/scripts/Draft/Fig13_BST_M200_dex_Apx.ipynb, cell 13
(saved there as Fig13_BST_M200_dex.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 11: effect of halo-mass errors on the photo-z BST HOD. The halo masses
of the centres are perturbed with a Gaussian of 0, 0.1 and 0.2 dex in
log10(M200) before binning. FlexZBoost (top), BPZ (bottom).

The notebook drew the perturbations with an unseeded ``np.random.normal``, so
the paper numbers (Table 3, "Mass error" rows) cannot be reproduced exactly;
here the draws are seeded (``SEED``) so the figure is reproducible.

    python paper_plots/fig11_mass_scatter.py --tabs /path/to/Tabs [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig11_mass_scatter.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig11_mass_scatter.pdf

    # copy of that folder downloaded from Drive (Articulo1/tabs)
    python paper_plots/fig11_mass_scatter.py --tabs /home/tavernam/projects/lsst_hod/drive/Articulo1/tabs

Instead of --tabs you can set the environment variable BST_HOD_TABS.
Without --out the figure goes to paper_plots/figures/fig11_mass_scatter.pdf.

Input: ``BST_{FZ,BPZ}_0.0Odds_r200_Mlim_{m}.npz`` and ``NTrue_DC2{m}.npz``.
"""

import numpy as np

from common import (COLORS, MLIMS, jackknife_mean_bins_arr, load_bst, parse_args, plot_binned,
                    ratio_summary, ratio_with_error, save)
from photoz_grid import CODES, MASS_BINS, finish, make_grid, true_hod

NEEDS = ("tabs",)
BST_FILE = "BST_{code}_0.0Odds_r200_Mlim_{m}.npz"
SCATTER_DEX = (0, 0.1, 0.2)
SEED = 2026


def make_figure(tabs, seed=SEED):
    rng = np.random.default_rng(seed)
    fig, ax = make_grid(ylim=(3e-1, 5e2), ratio_ylim=(0.4, 1.8))
    summary = {}
    for col, mlim in enumerate(MLIMS):
        n_true, err_true = true_hod(ax, col, tabs, mlim)
        data = {code: load_bst(tabs, BST_FILE.format(code=code, m=mlim)) for code in CODES}
        for j, dex in enumerate(SCATTER_DEX):
            for row, code in zip((0, 2), CODES):      # same draw order as the notebook
                M, N, _ = data[code]
                M = 10 ** (np.log10(M) + rng.normal(0, dex, size=M.size))
                n_bst, r_e, err = jackknife_mean_bins_arr(M, N, MASS_BINS)
                plot_binned(ax[row, col], r_e, n_bst, err, COLORS[j], lw=1, dash_last=False,
                            label=f"{dex} dex" if row == 0 else None)
                ratio, yerr = ratio_with_error(n_bst, err, n_true, err_true)
                plot_binned(ax[row + 1, col], r_e, ratio, yerr, COLORS[j], lw=1,
                            dash_last=False)
                if j == 0:
                    continue
                med, std = ratio_summary(ratio, yerr)
                summary.setdefault(f"Mass error {dex} dex {code}", {})[mlim] = (med, std)
                ax[row + 1, col].annotate(f"{code} : {med:.2f} $\\pm$ {std:.2f}",
                                          xy=(0.06 if j == 1 else 0.52, 0.85),
                                          xycoords="axes fraction", ha="left", va="top",
                                          fontsize=9, color=COLORS[j])
    ax[0, 2].legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.65), fontsize=12)
    finish(fig, ax)
    return fig, summary


if __name__ == "__main__":
    args = parse_args(__doc__, "paper_plots/figures/fig11_mass_scatter.pdf")
    fig, summary = make_figure(args.tabs)
    save(fig, args.out)
    for name, s in summary.items():
        print(name, "  ".join(f"{m}: {v[0]:.2f}±{v[1]:.2f}" for m, v in s.items()))
