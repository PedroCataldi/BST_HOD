r"""Source notebook: Articulo1/scripts/Draft/Fig11_BST_Odds.ipynb, cell 9
(saved there as Fig12_BST_r200_Odds.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 10: BST HOD with photometric redshifts, FlexZBoost (top) and BPZ
(bottom), true centrals, no Odds cut, vs the true HOD.

    python paper_plots/fig10_photoz.py --tabs /path/to/Tabs [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig10_photoz.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig10_photoz.pdf

    # copy of that folder downloaded from Drive (Articulo1/tabs)
    python paper_plots/fig10_photoz.py --tabs /home/tavernam/projects/lsst_hod/drive/Articulo1/tabs

Instead of --tabs you can set the environment variable BST_HOD_TABS.
Without --out the figure goes to paper_plots/figures/fig10_photoz.pdf.

Input: ``BST_{FZ,BPZ}_0.0Odds_r200_Mlim_{m}.npz`` and ``NTrue_DC2{m}.npz``.
"""

from common import (COLORS, MLIMS, jackknife_mean_bins_arr, load_bst, parse_args, plot_binned,
                    ratio_summary, ratio_with_error, save)
from photoz_grid import CODES, MASS_BINS, finish, make_grid, true_hod

NEEDS = ("tabs",)
BST_FILE = "BST_{code}_0.0Odds_r200_Mlim_{m}.npz"


def plot_photoz(tabs, bst_file, label):
    """FZ/BPZ grid with one BST curve per panel (used by Figs. 10 and 12)."""
    fig, ax = make_grid()
    summary = {code: {} for code in CODES}
    for col, mlim in enumerate(MLIMS):
        n_true, err_true = true_hod(ax, col, tabs, mlim)
        for row, code in zip((0, 2), CODES):
            M, N, _ = load_bst(tabs, bst_file.format(code=code, m=mlim))
            n_bst, r_e, err = jackknife_mean_bins_arr(M, N, MASS_BINS)
            plot_binned(ax[row, col], r_e, n_bst, err, COLORS[col], lw=1, dash_last=False)
            ratio, yerr = ratio_with_error(n_bst, err, n_true, err_true)
            plot_binned(ax[row + 1, col], r_e, ratio, yerr, COLORS[col], lw=1,
                        dash_last=False)
            med, std = ratio_summary(ratio, yerr)
            summary[code][mlim] = (med, std)
            ax[row + 1, col].annotate(f"{code} : {med:.2f} $\\pm$ {std:.2f}", xy=(0.1, 0.85),
                                      xycoords="axes fraction", ha="left", va="top",
                                      fontsize=12)
    finish(fig, ax)
    return fig, {f"{label} {code}": s for code, s in summary.items()}


def make_figure(tabs):
    return plot_photoz(tabs, BST_FILE, "Photo-z")


if __name__ == "__main__":
    args = parse_args(__doc__, "paper_plots/figures/fig10_photoz.pdf")
    fig, summary = make_figure(args.tabs)
    save(fig, args.out)
    for name, s in summary.items():
        print(name, "  ".join(f"{m}: {v[0]:.2f}±{v[1]:.2f}" for m, v in s.items()))
