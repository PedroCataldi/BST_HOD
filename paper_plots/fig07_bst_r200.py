r"""Source notebook: Articulo1/scripts/Draft/Fig6_BST_r200.ipynb, cell 3
(saved there as Fig6_BST_1.0r200_M200_new.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 7: HOD recovered by the BST with r_c = r200 vs the true HOD (cosmoDC2,
true centrals, true redshifts and masses).

Top: N(M200) of every centre (2D histogram), BST binned mean (colour) and
true HOD (black), jackknife errors. Bottom: BST/true ratio with its median and
scatter.

    python paper_plots/fig07_bst_r200.py --tabs /path/to/Tabs [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig07_bst_r200.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig07_bst_r200.pdf

    # copy of that folder downloaded from Drive (Articulo1/tabs)
    python paper_plots/fig07_bst_r200.py --tabs /home/tavernam/projects/lsst_hod/drive/Articulo1/tabs

Instead of --tabs you can set the environment variable BST_HOD_TABS.
Without --out the figure goes to paper_plots/output/fig07_bst_r200.pdf.

Input: ``BST_1.0rmax_1.25r200_Mlim_{m}.npz`` and ``NTrue_DC2{m}.npz``.
"""

import numpy as np
import numpy.ma as ma

from common import (COLORS, COLOR_MAPS, MLIMS, jackknife_mean_bins_arr, load_bst, load_true,
                    parse_args, plot_binned, ratio_summary, ratio_with_error, save, set_style,
                    style_ticks)

NEEDS = ("tabs",)
BST_FILE = "BST_1.0rmax_1.25r200_Mlim_{m}.npz"
MASS_BINS = np.logspace(10.8, 15.4, 9)
COLORBAR_TICKS = ([1, 5, 40], [1, 10, 100], [1, 30, 700], [1, 50, 1000])


def make_figure(tabs):
    import matplotlib.pyplot as plt

    set_style(15)
    fig, ax = plt.subplots(2, 4, figsize=(13.5, 5.2), sharex="col", sharey="row",
                           gridspec_kw={"height_ratios": [4, 1]})
    style_ticks(ax, 10)
    for col in range(4):
        ax[0, col].set_xscale("log")
        ax[0, col].set_yscale("log")
        ax[0, col].set_xlim(5e10, 3e15)
        ax[0, col].set_ylim(1e-2, 5e2)
        ax[1, col].set_xscale("log")
        ax[1, col].set_ylim(0.4, 1.7)
        ax[1, col].set_xlabel(r"$M_{\mathrm{200}}$ [M$_{\odot}$]", fontsize=11)
    ax[0, 0].set_ylabel(r"$N(M_{\mathrm{200}})$", fontsize=11)
    ax[1, 0].set_ylabel("Ratio", fontsize=11)

    summary = {}
    for col, mlim in enumerate(MLIMS):
        ax[0, col].annotate(f"M$_{{\\mathrm{{r}}}}$ < {mlim}", xy=(0.5, 0.88),
                            xycoords="axes fraction", ha="center", va="bottom", fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8))
        M, N, _ = load_bst(tabs, BST_FILE.format(m=mlim))

        # 2D histogram of the individual centres
        m_edges = np.logspace(np.log10(M.min()), np.log10(M.max()), 64)
        n_edges = np.logspace(np.log10(N[N > 0].min()), np.log10(N[N > 0].max()), 64)
        hist, _, _ = np.histogram2d(M, N, bins=(m_edges, n_edges))
        with np.errstate(divide="ignore"):
            img = ma.masked_where(np.isnan(hist), np.log10(hist)).T
        xx, yy = np.meshgrid(m_edges, n_edges)
        pc = ax[0, col].pcolormesh(xx, yy, img, cmap=COLOR_MAPS[col], rasterized=True)
        cb = fig.colorbar(pc, cax=ax[0, col].inset_axes([0.59, 0.24, 0.35, 0.07]),
                          orientation="horizontal")
        ticks = COLORBAR_TICKS[col]
        cb.set_ticks(np.log10(ticks))
        cb.set_ticklabels([str(t) for t in ticks])
        cb.set_label("Counts", fontsize=8)
        cb.ax.tick_params(labelsize=9)

        n_bst, r_e, err_bst = jackknife_mean_bins_arr(M, N, MASS_BINS)
        plot_binned(ax[0, col], r_e, n_bst, err_bst, COLORS[col], label="BST", lw=1.4)
        m_true, n_true_halo = load_true(tabs, mlim)
        n_true, _, err_true = jackknife_mean_bins_arr(m_true, n_true_halo, MASS_BINS)
        plot_binned(ax[0, col], r_e, n_true, err_true, "k", label="True")
        ax[0, col].axhline(1, color="k", lw=0.8)

        ratio, yerr = ratio_with_error(n_bst, err_bst, n_true, err_true)
        plot_binned(ax[1, col], r_e, ratio, yerr, COLORS[col])
        ax[1, col].axhline(1, color="k", ls="--", lw=0.8)
        med, std = ratio_summary(ratio, yerr)
        summary[mlim] = (med, std)
        ax[1, col].annotate(f"$\\langle$Ratio$\\rangle$: {med:.2f}, $\\sigma$: {std:.2f}",
                            xy=(0.05, 0.7), xycoords="axes fraction", fontsize=10)

    ax[0, 3].legend(frameon=False, fontsize=10, loc="upper left")
    fig.subplots_adjust(wspace=0.05, hspace=0.06)
    return fig, {"BST r200": summary}


if __name__ == "__main__":
    args = parse_args(__doc__, "fig07_bst_r200.pdf")
    fig, summary = make_figure(args.tabs)
    save(fig, args.out)
    for name, s in summary.items():
        print(name, "  ".join(f"{m}: {v[0]:.2f}±{v[1]:.2f}" for m, v in s.items()))
