"""4 x 4 layout shared by Figs. 10-12: one column per M_lim, FlexZBoost HOD and
ratio on the top two rows, BPZ on the bottom two (as in the paper notebooks)."""

import numpy as np

from common import MLIMS, jackknife_mean_bins_arr, load_true, set_style, style_ticks

MASS_BINS = np.logspace(10.5, 15.5, 9)
CODES = ("FZ", "BPZ")      # row pair 0 -> FZ, row pair 1 -> BPZ


def make_grid(ylim=(6e-1, 4e2), ratio_ylim=(0.4, 1.7)):
    import matplotlib.pyplot as plt

    set_style(17)
    fig, ax = plt.subplots(4, 4, figsize=(13.5, 10.4), sharex="col", sharey="row",
                           gridspec_kw={"height_ratios": [4, 1, 4, 1]})
    style_ticks(ax, 12, major=7, minor=4)
    for col, mlim in enumerate(MLIMS):
        for row in (0, 2):
            ax[row, col].set_xscale("log")
            ax[row, col].set_yscale("log")
            ax[row, col].set_xlim(5e10, 3e15)
            ax[row, col].set_ylim(*ylim)
            ax[row, col].annotate(f"M$_{{\\mathrm{{r}}}}$ < {mlim}", xy=(0.5, 0.87),
                                  xycoords="axes fraction", ha="center", fontsize=12,
                                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black",
                                            lw=0.6))
        for row in (1, 3):
            ax[row, col].set_xscale("log")
            ax[row, col].set_ylim(*ratio_ylim)
        ax[3, col].set_xlabel(r"$M_{\mathrm{200}}$ [M$_{\odot}$]", fontsize=12)
    for row in (0, 2):
        ax[row, 0].set_ylabel(r"$N(M_{\mathrm{200}})$", fontsize=14)
        ax[row + 1, 0].set_ylabel("Ratio", fontsize=14)
    for row, code in zip((0, 2), CODES):
        ax[row, 0].text(0.2, 0.73, code, transform=ax[row, 0].transAxes, ha="center",
                        va="top", fontsize=16,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="black", linewidth=0.4))
    return fig, ax


def true_hod(ax, col, tabs, mlim):
    """Draw the true HOD on both HOD panels of a column and return it binned."""
    m_true, n_true_halo = load_true(tabs, mlim)
    n_true, r_e, err_true = jackknife_mean_bins_arr(m_true, n_true_halo, MASS_BINS)
    for row in (0, 2):
        ax[row, col].plot(r_e, n_true, ls="-", lw=1, color="k")
        ax[row, col].axhline(1, color="black", linewidth=0.8)
    return n_true, err_true


def finish(fig, ax):
    for col in range(4):
        ax[1, col].axhline(1, color="black", ls="--", lw=0.7)
        ax[3, col].axhline(1, color="black", ls="--", lw=0.7)
        for row in range(4):
            ax[row, col].set_xticks([1e11, 1e13, 1e15])
    fig.subplots_adjust(hspace=0.07, wspace=0.05, top=0.92)
