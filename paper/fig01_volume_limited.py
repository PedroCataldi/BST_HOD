r"""Source notebook: Articulo1/scripts/Draft/Fig2_volume_limited.ipynb, cell 6
(saved there as Fig2_volume_limited_with_inset.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 1: construction of the volume-limited samples in the true z - M_r plane
(cosmoDC2), with the M_lim / z_max boxes. Inset: number density n_g(z) of all
galaxies and of each sample, and the mean density of each sample (Table 1).

Reads the mock galaxy table (heavy file, ~0.2 GB).

    python paper/fig01_volume_limited.py --mocks /path/to/mocks [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper/fig01_volume_limited.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig01_volume_limited.pdf

    # copy of that folder downloaded from Drive
    python paper/fig01_volume_limited.py --mocks /home/tavernam/projects/lsst_hod/drive/mocks

Instead of --mocks you can set the environment variable BST_HOD_MOCKS; without
it the table is looked for in the --tabs folder.
Without --out the figure goes to paper/figures/fig01_volume_limited.pdf.

Input: ``mock_21_large_z05_cosmoDC2.mpeg``.
"""

import numpy as np

from common import COLORS, MLIMS, abs_mag, comoving_volume, load_mock, parse_args, save

NEEDS = ("mocks",)
Z_CUTS = (0.0875, 0.1345, 0.2055, 0.3038)
AREA_DEG2 = 440.0


def make_figure(mocks):
    import matplotlib as mpl
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    mpl.rcParams.update({"axes.linewidth": 0.8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "font.size": 8, "font.family": "serif", "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    gal = load_mock(mocks, "z05")
    gal = gal[gal["z"] > 0.0]
    z = gal["z"].to_numpy()
    mabs = abs_mag(gal["mr_lsst"].to_numpy(), z)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), constrained_layout=True)
    zbins = np.linspace(z.min(), z.max(), num=128)
    mbins = np.linspace(mabs.min(), mabs.max(), num=128)
    hist, xedges, yedges = np.histogram2d(z, mabs, bins=(zbins, mbins))
    with np.errstate(divide="ignore"):
        img = np.log10(hist.T)
    xx, yy = np.meshgrid(xedges, yedges)
    mesh = ax.pcolormesh(xx, yy, img, cmap="cividis", rasterized=True)

    for m, zc in enumerate(Z_CUTS):
        ax.add_patch(patches.Rectangle((0.0, -22.5), zc, MLIMS[m] + 22.5, linewidth=1.2,
                                       edgecolor=COLORS[m], facecolor="none"))
    ax.set_xlabel(r"$z$", fontsize=8)
    ax.set_ylabel(r"$M_r$", fontsize=8)
    ax.set_xlim(0.02, 0.4)
    ax.set_ylim(-21.0, -16)
    ax.invert_yaxis()
    ax.minorticks_on()

    cb = fig.colorbar(mesh, cax=fig.add_axes([0.69, 0.67, 0.21, 0.06]),
                      orientation="horizontal")
    ticks = [1, 10, 100, 1000]
    cb.set_ticks(np.log10(ticks))
    cb.set_ticklabels([str(t) for t in ticks])
    cb.set_label("Counts", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # inset: number density in redshift slices
    axins = inset_axes(ax, width="40%", height="40%", loc="lower right",
                       bbox_to_anchor=(+0.05, 0.05, 1, 1), bbox_transform=ax.transAxes,
                       borderpad=2)
    sky_fraction = AREA_DEG2 * (np.pi / 180.0) ** 2 / (4 * np.pi)

    def density(edges, sel):
        counts, _ = np.histogram(z[sel], bins=edges)
        return counts / (np.diff(comoving_volume(edges)) * sky_fraction)

    edges = np.linspace(0, 0.4, 50)
    axins.plot(0.5 * (edges[:-1] + edges[1:]), density(edges, np.ones(z.size, bool)), ls="-",
               color="k", label="Mean density", linewidth=1.0)
    offset = (0.12, 0.15, 0.13, 0.12)
    print("Table 1: M_lim, N_gal, n_g [1e-3 Mpc^-3]")
    for m, zc in enumerate(Z_CUTS):
        edges = np.linspace(0, zc, 50)
        sel = mabs < MLIMS[m]
        axins.plot(0.5 * (edges[:-1] + edges[1:]), density(edges, sel), color=COLORS[m],
                   ls="-", linewidth=0.5)
        n_tot = np.sum((z < zc) & sel)
        mean = n_tot / (comoving_volume(zc) * sky_fraction)
        print(f"  {MLIMS[m]}: {n_tot}  {mean * 1e3:.1f}")
        axins.hlines(mean, xmin=0.02, xmax=zc, colors=COLORS[m], linestyles="-", linewidth=0.5)
        axins.text(zc + offset[m], mean, str(MLIMS[m]), color=COLORS[m], va="center",
                   ha="right", fontsize=7)
    axins.set_xlabel("$z$", fontsize=8)
    axins.set_ylabel(r"$n_g$ [Mpc$^{-3}$]", fontsize=8)
    axins.set_yscale("log")
    axins.set_xlim(0.02, 0.45)
    axins.set_ylim(2e-3, 1e-1)
    axins.set_yticks([1e-2, 1e-1])
    axins.set_yticklabels(["0.01", "0.1"])
    axins.set_xticks([0.1, 0.2, 0.3, 0.4])
    axins.tick_params("both", length=3, width=0.8, which="major", direction="in", right=True,
                      top=True)
    axins.tick_params("both", length=2, width=0.5, which="minor", direction="in", right=True,
                      top=True)
    return fig, {}


if __name__ == "__main__":
    args = parse_args(__doc__, "paper/figures/fig01_volume_limited.pdf", NEEDS)
    fig, _ = make_figure(args.mocks)
    save(fig, args.out)
