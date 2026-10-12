"""Shared helpers for the figures of Cataldi et al. (2026), arXiv:2603.01978.

The figure scripts read the per-halo BST results already computed for the
paper (``.npz`` files in the ``Tabs/`` folder of the original analysis) and
reproduce the plotting notebooks of ``Articulo1/scripts/Draft`` without
re-running the BST.

Every ``.npz`` written by the original drivers has the keys
``N_arr_18`` (BST count), ``Mhalo_arr_18`` (M200 [Msun]),
``N_arr_true_18`` (true count) and ``halo_id``. ``NTrue_DC2{mlim}.npz`` has
``M_200`` and ``N_true`` (true HOD of the cosmoDC2 sample).

Some figures (1, 2, 3, 4, 6 and 8) also read the galaxy tables of the mock
(``mock_21_large_*_cosmoDC2``, 0.2-0.5 GB each, see :data:`MOCKS`), given with
``--mocks``. In Pedro's analysis folder they were in ``Tabs/`` with the .npz
files; in the Drive copy they are in ``mocks/``.
"""

import argparse
import os
import warnings

import numpy as np
from scipy.stats import binned_statistic

MLIMS = (-17, -18, -19, -20)
COLORS = ("g", "b", "r", "m")
COLOR_MAPS = ("Greens", "Blues", "Reds", "Purples")

#: environment variables with the folder of the .npz tables and of the mock tables
TABS_ENV = "BST_HOD_TABS"
MOCKS_ENV = "BST_HOD_MOCKS"

#: galaxy tables of the mock (m_r <= 21, true z < 0.4). "z05" is the table the
#: notebooks use for the true-redshift analysis; its photo-z columns are BPZ
#: (the photo-z scripts and Fig. 3 use it as the BPZ table).
MOCKS = {
    "z05": "mock_21_large_z05_cosmoDC2.mpeg",
    "FZ": "mock_21_large_FZ_z05_cosmoDC2.mpg",
    "BPZ": "mock_21_large_BPZ_z05_cosmoDC2.mpg",
}

#: cosmoDC2 cosmology (flat LCDM, no radiation), as FlatLambdaCDM(H0=71, Om0=0.2648)
H0_DC2, OM0_DC2 = 71.0, 0.2648
C_KM_S = 299792.458
G_KPC = 4.30091e-6           # kpc (km/s)^2 / Msun


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


_MOCK_CACHE = {}


def _clean(name):
    return str(name).strip().lstrip("#").strip(",").strip()


def load_mock(mocks, which="z05"):
    """Galaxy table of the mock as a DataFrame, read once and cached.

    Column names are cleaned of the ``#`` and commas of the original header
    (``'z,'`` -> ``'z'``, ``'#galid,'`` -> ``'galid'``). Rows with z <= 0 are
    kept, as each notebook applies its own cuts.
    """
    import pandas as pd

    path = os.path.join(mocks, MOCKS.get(which, which))
    if path not in _MOCK_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Point --mocks (or ${MOCKS_ENV}) to "
                                    "the folder with the mock galaxy tables.")
        print(f"reading {path} ...")
        df = pd.read_csv(path, sep=" ")
        df.columns = [_clean(c) for c in df.columns]
        df = df.drop(columns=[c for c in df.columns if c == "" or c.startswith("Unnamed")])
        _MOCK_CACHE[path] = df
    return _MOCK_CACHE[path]


def comoving_distance(z, H0=H0_DC2, Om0=OM0_DC2):
    """Comoving distance [Mpc] in flat LCDM without radiation (equal to astropy's
    FlatLambdaCDM(H0, Om0).comoving_distance to better than 1e-7)."""
    from scipy.integrate import cumulative_trapezoid

    z = np.asarray(z, dtype=float)
    zmax = max(float(np.nanmax(z)), 1.0)
    grid = np.linspace(0.0, zmax, int(20000 * zmax) + 1)
    dc = cumulative_trapezoid(1.0 / np.sqrt(Om0 * (1 + grid) ** 3 + 1 - Om0), grid, initial=0)
    return np.interp(z, grid, dc * C_KM_S / H0)


def comoving_volume(z, H0=H0_DC2, Om0=OM0_DC2):
    """Comoving volume [Mpc^3] of the whole sky out to z (flat)."""
    return 4.0 / 3.0 * np.pi * comoving_distance(z, H0, Om0) ** 3


def abs_mag(mag, z, H0=H0_DC2, Om0=OM0_DC2):
    """M = m - 25 - 5 log10(D_C (1+z)), no k-correction (as in the notebooks)."""
    z = np.asarray(z, dtype=float)
    return np.asarray(mag) - 25.0 - 5 * np.log10(comoving_distance(z, H0, Om0) * (1 + z))


def rho_crit_kpc(z, H0=H0_DC2, Om0=OM0_DC2):
    """Critical density [Msun / kpc^3] at z, as computed in the notebooks."""
    e = np.sqrt(Om0 * (1 + np.asarray(z, dtype=float)) ** 3 + 1 - Om0)
    return 3.0 * (H0 * e * 1e-3) ** 2 / (8.0 * np.pi * G_KPC)


def r200_kpc(log_m200, z):
    """r200 [kpc] from log10(M200 / Msun)."""
    return (3 * 10 ** np.asarray(log_m200) / (800.0 * np.pi * rho_crit_kpc(z))) ** (1 / 3.0)


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


def add_data_args(p):
    """--tabs and/or --mocks. Without --mocks the mock tables are looked for in
    the --tabs folder (as in Pedro's Tabs/)."""
    p.add_argument("--tabs", default=os.environ.get(TABS_ENV),
                   help=f"folder with the paper .npz tables (default: ${TABS_ENV})")
    p.add_argument("--mocks", default=os.environ.get(MOCKS_ENV),
                   help=f"folder with the mock galaxy tables (default: ${MOCKS_ENV}, "
                        "or the --tabs folder)")


def check_data_args(p, args, needs):
    if args.mocks is None:
        args.mocks = args.tabs
    if "tabs" in needs and not args.tabs:
        p.error(f"give --tabs or set ${TABS_ENV}")
    if "mocks" in needs and not args.mocks:
        p.error(f"give --mocks or set ${MOCKS_ENV}")


def parse_args(description, default_out, needs=("tabs",)):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--out", default=default_out, help="output file (pdf/png)")
    args = p.parse_args()
    check_data_args(p, args, needs)
    return args


def data_for(module, args):
    """Arguments of ``module.make_figure`` according to its NEEDS."""
    return [getattr(args, n) for n in module.NEEDS]


def save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")
