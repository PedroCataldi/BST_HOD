r"""Source notebook: Articulo1/scripts/Draft/Fig8_BST_rlum.ipynb, cell 5
(saved there as Fig8_BST_1.0rlum_M200.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 9: HOD recovered by the BST with the luminosity-based aperture
r_c^lum (colour) and with r_c = r200 (grey), vs the true HOD (black).

The ratio panels quote the median and scatter of the r_c^lum ratio (Table 3).

    python paper/fig09_bst_rlum.py --tabs /path/to/Tabs [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper/fig09_bst_rlum.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig09_bst_rlum.pdf

    # copy of that folder downloaded from Drive (Articulo1/tabs)
    python paper/fig09_bst_rlum.py --tabs /home/tavernam/projects/lsst_hod/drive/Articulo1/tabs

Instead of --tabs you can set the environment variable BST_HOD_TABS.
Without --out the figure goes to paper/figures/fig09_bst_rlum.pdf.

Input: ``BST_1.0rmax_1.25rlum_Mlim_{m}.npz``, ``BST_1.0rmax_1.25r200_Mlim_{m}.npz``
and ``NTrue_DC2{m}.npz``.
"""

import numpy as np

from common import (COLORS, MLIMS, jackknife_mean_bins_arr, load_bst, load_true, parse_args,
                    plot_binned, ratio_summary, ratio_with_error, save, set_style, style_ticks)

NEEDS = ("tabs",)
RLUM_FILE = "BST_1.0rmax_1.25rlum_Mlim_{m}.npz"
R200_FILE = "BST_1.0rmax_1.25r200_Mlim_{m}.npz"
MASS_BINS = np.logspace(10.8, 15.4, 9)


def make_figure(tabs):
    import matplotlib.pyplot as plt

    set_style(15)
    fig, ax = plt.subplots(2, 4, figsize=(13.5, 5.2), sharex="col", sharey="row",
                           gridspec_kw={"height_ratios": [4, 1]})
    style_ticks(ax, 10)
    for col in range(4):
        ax[0, col].set_xscale("log")
        ax[0, col].set_yscale("log")
        ax[1, col].set_xscale("log")
        ax[0, col].set_xlim(5e10, 3e15)
        ax[0, col].set_ylim(6e-1, 4e2)
        ax[1, col].set_ylim(0.4, 1.7)
        ax[1, col].set_xlabel(r"$M_{\mathrm{200}}$ [M$_{\odot}$]", fontsize=12)
        ax[0, col].tick_params(labelbottom=False)
    ax[0, 0].set_ylabel(r"$N(M_{\mathrm{200}})$", fontsize=12)
    ax[1, 0].set_ylabel("Ratio", fontsize=10)

    summary = {}
    for col, mlim in enumerate(MLIMS):
        ax[0, col].annotate(f"M$_{{\\mathrm{{r}}}}$ < {mlim}", xy=(0.5, 0.87),
                            xycoords="axes fraction", ha="center", fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.5))

        m_true, n_true_halo = load_true(tabs, mlim)
        n_true, r_e, err_true = jackknife_mean_bins_arr(m_true, n_true_halo, MASS_BINS)
        plot_binned(ax[0, col], r_e, n_true, err_true, "k", label="True")

        M, N, _ = load_bst(tabs, RLUM_FILE.format(m=mlim))
        n_lum, _, err_lum = jackknife_mean_bins_arr(M, N, MASS_BINS)
        plot_binned(ax[0, col], r_e, n_lum, err_lum, COLORS[col],
                    label=r"$r_{\mathrm{c}}^{\mathrm{lum}}$", zorder=5)

        M, N, _ = load_bst(tabs, R200_FILE.format(m=mlim))
        n_rc, _, err_rc = jackknife_mean_bins_arr(M, N, MASS_BINS)
        plot_binned(ax[0, col], r_e, n_rc, err_rc, "gray", label=r"$r_{\mathrm{c}}$")
        ax[0, col].axhline(1, color="k", lw=0.8)

        ratio, yerr = ratio_with_error(n_lum, err_lum, n_true, err_true)
        plot_binned(ax[1, col], r_e, ratio, yerr, COLORS[col], zorder=5)
        med, std = ratio_summary(ratio, yerr)
        summary[mlim] = (med, std)
        ax[1, col].annotate(f"{med:.2f} $\\pm$ {std:.2f}", xy=(0.1, 0.88),
                            xycoords="axes fraction", ha="left", va="top", fontsize=10)
        ratio, yerr = ratio_with_error(n_rc, err_rc, n_true, err_true)
        plot_binned(ax[1, col], r_e, ratio, yerr, "gray")
        ax[1, col].axhline(1, color="k", ls="--", lw=0.7)
        ax[1, col].set_xticks([1e11, 1e13, 1e15])

    ax[0, 3].legend(frameon=False, fontsize=10, loc="center left")
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig, {"BST r_c^lum": summary}


if __name__ == "__main__":
    args = parse_args(__doc__, "paper/figures/fig09_bst_rlum.pdf")
    fig, summary = make_figure(args.tabs)
    save(fig, args.out)
    for name, s in summary.items():
        print(name, "  ".join(f"{m}: {v[0]:.2f}±{v[1]:.2f}" for m, v in s.items()))
