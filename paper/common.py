"""Shared helpers for the figures of Cataldi et al. (2026), arXiv:2603.01978.

The figure scripts read the per-halo BST results already computed for the
paper (``.npz`` files in the ``Tabs/`` folder of the original analysis) and
reproduce the plotting notebooks of ``Articulo1/scripts/Draft`` without
re-running the BST.

Every ``.npz`` written by the original drivers has the keys
``N_arr_18`` (BST count), ``Mhalo_arr_18`` (M200 [Msun]),
``N_arr_true_18`` (true count) and ``halo_id``. ``NTrue_DC2{mlim}.npz`` has
``M_200`` and ``N_true`` (true HOD of the cosmoDC2 sample).
"""

import argparse
import os
import warnings

import numpy as np
from scipy.stats import binned_statistic

MLIMS = (-17, -18, -19, -20)
COLORS = ("g", "b", "r", "m")
COLOR_MAPS = ("Greens", "Blues", "Reds", "Purples")

#: environment variable with the folder of the .npz tables
TABS_ENV = "BST_HOD_TABS"


def jackknife_mean_bins_arr(x, y, bins_arr, N_jk=50, seed=42):
    """Binned mean of ``y`` in bins of ``x`` and its jackknife error.

    Each point is assigned at random to one of ``N_jk`` subsamples; the error
    comes from the ``N_jk`` delete-one estimates. Copied unchanged from the
    paper notebooks (``N_jk=50``, ``seed=42``), so the numbers are identical.

    Returns ``means, bin_centers, errors``. ``bin_centers`` is the arithmetic
    centre of each bin, as in the notebooks.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    bins_arr = np.asarray(bins_arr)
    nbins = len(bins_arr) - 1
    rng = np.random.default_rng(seed)
    jk_id = rng.integers(0, N_jk, size=len(x))

    means, _, _ = binned_statistic(x, y, statistic="mean", bins=bins_arr)
    jk_vals = np.zeros((N_jk, nbins))
    for k in range(N_jk):
        mask = jk_id != k
        if np.sum(mask) == 0:
            jk_vals[k, :] = np.nan
            continue
        jk_vals[k, :], _, _ = binned_statistic(x[mask], y[mask], statistic="mean",
                                               bins=bins_arr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # empty bins -> NaN
        jk_mean = np.nanmean(jk_vals, axis=0)
        errors = np.sqrt((N_jk - 1) * np.nanmean((jk_vals - jk_mean) ** 2, axis=0))
    bin_centers = 0.5 * (bins_arr[:-1] + bins_arr[1:])
    return means, bin_centers, errors


def tabs_path(tabs, name):
    path = os.path.join(tabs, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Point --tabs (or ${TABS_ENV}) to the "
                                "folder with the paper .npz tables.")
    return path


def load_bst(tabs, name):
    """BST result: (M200, N_bst, N_true) of the halos with M200 > 0."""
    with np.load(tabs_path(tabs, name)) as d:
        M, N, N_true = d["Mhalo_arr_18"], d["N_arr_18"], d["N_arr_true_18"]
    ok = M > 0
    return M[ok], N[ok], N_true[ok]


def load_true(tabs, mlim):
    """True HOD sample of cosmoDC2: (M200, N_true)."""
    with np.load(tabs_path(tabs, f"NTrue_DC2{mlim}.npz")) as d:
        return d["M_200"], d["N_true"]


def ratio_with_error(n, err, n_true, err_true):
    """BST/true ratio per bin with errors added in quadrature."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = n / n_true
        yerr = ratio * np.sqrt((err / n) ** 2 + (err_true / n_true) ** 2)
    return ratio, yerr


def ratio_summary(ratio, yerr):
    """Median and standard deviation of the ratio over the finite bins
    (the "<Ratio> ± sigma" quoted in the panels and in Table 3)."""
    mask = np.isfinite(ratio) & np.isfinite(yerr)
    return float(np.median(ratio[mask])), float(np.std(ratio[mask]))


def plot_binned(ax, x, y, err, color, label=None, lw=1.2, zorder=None, dash_last=True):
    """Binned curve with error bars; the last (poorly sampled) segment dashed."""
    kw = {} if zorder is None else {"zorder": zorder}
    if dash_last:
        ax.plot(x[:-1], y[:-1], color=color, lw=lw, label=label, **kw)
        ax.plot(x[-2:], y[-2:], color=color, lw=lw, ls="--", **kw)
    else:
        ax.plot(x, y, color=color, lw=lw, label=label, **kw)
    ax.errorbar(x, y, yerr=err, color=color, capsize=2, ls="none",
                lw=None if dash_last else 0.5, **kw)


def set_style(ticksize):
    import matplotlib.pyplot as plt
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize


def style_ticks(axes, labelsize, major=6, minor=3):
    for axi in np.ravel(axes):
        axi.tick_params(axis="both", which="major", labelsize=labelsize, direction="in",
                        top=True, right=True, length=major, width=0.7)
        axi.tick_params(axis="both", which="minor", labelsize=labelsize, direction="in",
                        top=True, right=True, length=minor, width=0.5)
        axi.minorticks_on()


def parse_args(description, default_out):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--tabs", default=os.environ.get(TABS_ENV),
                   help=f"folder with the paper .npz tables (default: ${TABS_ENV})")
    p.add_argument("--out", default=default_out, help="output file (pdf/png)")
    args = p.parse_args()
    if not args.tabs:
        p.error(f"give --tabs or set ${TABS_ENV}")
    return args


def save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")
