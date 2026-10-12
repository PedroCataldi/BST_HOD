r"""Source notebook: Articulo1/scripts/Draft/Fig7_Lum_r200_Fit.ipynb, cell 0
(saved there as Fig7_Lum_r200_Fit_CosmoDC2.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 8: left, luminosity of the group centres (centrals with M_r < -19.5) vs
r_c = r200, with the linear fit in log-log; right, r_c vs the radius predicted
from the luminosity, r_c^lum.

Reads the mock galaxy table (heavy file, ~0.2 GB).

    python paper_plots/fig08_lum_r200_fit.py --mocks /path/to/mocks [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig08_lum_r200_fit.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig08_lum_r200_fit.pdf

    # copy of that folder downloaded from Drive
    python paper_plots/fig08_lum_r200_fit.py --mocks /home/tavernam/projects/lsst_hod/drive/mocks

Instead of --mocks you can set the environment variable BST_HOD_MOCKS; without
it the table is looked for in the --tabs folder.
Without --out the figure goes to paper_plots/figures/fig08_lum_r200_fit.pdf.

Input: ``mock_21_large_z05_cosmoDC2.mpeg``.
"""

import numpy as np
import numpy.ma as ma

from common import abs_mag, load_mock, parse_args, r200_kpc, save

NEEDS = ("mocks",)


def make_figure(mocks):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 6,
                         "axes.linewidth": 0.6, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.major.size": 2,
                         "ytick.major.size": 2, "xtick.minor.size": 1, "ytick.minor.size": 1,
                         "xtick.major.width": 0.3, "ytick.major.width": 0.3,
                         "xtick.minor.width": 0.25, "ytick.minor.width": 0.25})
    gal = load_mock(mocks, "z05")
    gal = gal[gal["z"] > 0.0]
    z = gal["z"].to_numpy()
    mabs = abs_mag(gal["mr_lsst"].to_numpy(), z)
    mask = (gal["is_central"].to_numpy() == 1) & (mabs < -19.5)
    r200 = r200_kpc(gal["loghalo_mass"].to_numpy()[mask], z[mask]) / 1000.0     # Mpc
    lum = 10 ** (-0.4 * mabs[mask])

    width = 8.8 / 2.54
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(width, width * 0.7), dpi=300)
    cmap = plt.cm.cividis.copy()
    cmap.set_bad("white")

    # left: log L vs log r_c with the linear fit
    log_r, log_l = np.log10(r200), np.log10(lum)
    rbins = np.linspace(np.log10(0.1), np.log10(2.5), 128)
    lbins = np.linspace(log_l.min(), log_l.max(), 128)
    hist, _, _ = np.histogram2d(log_r, log_l, bins=(rbins, lbins))
    xx, yy = np.meshgrid(rbins, lbins)
    ax.pcolormesh(xx, yy, ma.log10(ma.masked_equal(hist.T, 0)), cmap=cmap,
                  rasterized=True)
    slope, intercept = np.polyfit(log_r, log_l, 1)
    print(f"fit: log r_c = {1 / slope:.4f} log L {-intercept / slope:+.4f}")
    xs = np.linspace(log_r.min(), log_r.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="purple", ls="--", lw=1.2,
            label="fit $r_{lum}$ [Mpc]")
    ax.set_xlabel(r"$\log r_{\mathrm{c}} \, [\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\log L \, [\mathrm{L_{\odot}}]$")
    ax.set_xlim(np.log10(1.1e-1), np.log10(2.0))
    ax.set_ylim(log_l.min(), log_l.max())
    ax.set_xticks([-0.6, -0.3, 0.0, 0.3])

    # right: r_c vs r_c^lum
    r_lum = 10 ** ((log_l - intercept) / slope)
    bins = np.linspace(0.1, 2.5, 128)
    hist2, _, _ = np.histogram2d(r200, r_lum, bins=(bins, bins))
    xx, yy = np.meshgrid(bins, bins)
    pc = ax2.pcolormesh(xx, yy, ma.log10(ma.masked_equal(hist2.T, 0)), cmap=cmap,
                        rasterized=True)
    ax2.plot([0, 0.2], [0.2, 0.2], "r--", lw=0.8)
    ax2.plot([0.2, 2.5], [0.2, 2.5], "k--", linewidth=0.8)
    ax2.set_xlabel(r"$r_{\mathrm{c}} \, [\mathrm{Mpc}]$")
    ax2.set_ylabel(r"$r_{\mathrm{c}}^{\mathrm{lum}} \, [\mathrm{Mpc}]$")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xlim(0.2, 2.0)
    ax2.set_ylim(0.2, 2.5)

    for axis in (ax, ax2):
        axis.minorticks_on()
        axis.tick_params(which="both", direction="in", top=True, right=True)
    cb = fig.colorbar(pc, cax=fig.add_axes([0.55, 0.8, 0.2, 0.05]), orientation="horizontal")
    cb.set_ticks(np.log10([10, 1000]))
    cb.set_ticklabels(["10", "1000"])
    cb.set_label("Counts")
    fig.subplots_adjust(wspace=0.05)
    return fig, {}


if __name__ == "__main__":
    args = parse_args(__doc__, "paper_plots/figures/fig08_lum_r200_fit.pdf", NEEDS)
    fig, _ = make_figure(args.mocks)
    save(fig, args.out)
