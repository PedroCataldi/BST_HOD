r"""Source notebook: Articulo1/scripts/Draft/Fig5_z_photo_spec.ipynb, cell 10
(saved there as Fig5_z_photo_spec_1x2.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 3: photometric (mean) vs true redshift of the group centres (centrals with
M_r < -19.5, z < 0.4) for FlexZBoost (left) and BPZ (right); insets: cumulative
distribution of the photo-z Odds. As in the notebook, the BPZ panel uses the
photo-z columns of ``mock_21_large_z05_cosmoDC2.mpeg``.

Reads two mock galaxy tables (heavy files, ~0.2 and ~0.5 GB).

    python paper_plots/fig03_photoz.py --mocks /path/to/mocks [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig03_photoz.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig03_photoz.pdf

    # copy of that folder downloaded from Drive
    python paper_plots/fig03_photoz.py --mocks /home/tavernam/projects/lsst_hod/drive/mocks

Instead of --mocks you can set the environment variable BST_HOD_MOCKS; without
it the tables are looked for in the --tabs folder.
Without --out the figure goes to paper_plots/output/fig03_photoz.pdf.

Input: ``mock_21_large_FZ_z05_cosmoDC2.mpg`` and ``mock_21_large_z05_cosmoDC2.mpeg``.
"""

import numpy as np

from common import abs_mag, load_mock, parse_args, save

NEEDS = ("mocks",)
PANELS = (("FZ", "FZ"), ("z05", "BPZ"))      # (table, label)


def make_figure(mocks):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
                         "axes.linewidth": 0.6, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.major.size": 2,
                         "ytick.major.size": 2, "xtick.minor.size": 1, "ytick.minor.size": 1,
                         "xtick.major.width": 0.3, "ytick.major.width": 0.3,
                         "xtick.minor.width": 0.25, "ytick.minor.width": 0.25})
    width = 8.8 / 2.54
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.7), sharey=True, dpi=300)

    for ax, (table, label) in zip(axes, PANELS):
        gal = load_mock(mocks, table)
        gal = gal[(gal["z"] > 0.0) & (gal["z"] < 0.4) & (gal["is_central"] == 1)]
        gal = gal.drop_duplicates(subset="galid", keep="first").reset_index(drop=True)
        mask = (abs_mag(gal["mr_lsst"].to_numpy(), gal["z"].to_numpy()) < -19.5)
        z_true = gal["z"].to_numpy()[mask]
        z_photo = gal["photoz_mean"].to_numpy()[mask]
        odds = gal["photoz_odds"].to_numpy()[mask]

        dz = (z_photo - z_true) / (1.0 + z_true)
        good = np.isfinite(odds)
        print(f"{label}: N = {z_true.size}, sigma_std = {np.std(dz):.5f}, sigma_NMAD = "
              f"{1.48 * np.median(np.abs(dz - np.median(dz))):.5f}, "
              f"Odds > 0.8: {100 * np.mean(odds[good] > 0.8):.2f}%")

        zx = np.linspace(z_true.min(), z_true.max(), 128)
        zy = np.linspace(1e-2, z_photo.max(), 128)
        hist = np.histogram2d(z_true, z_photo, bins=(zx, zy))[0].T
        with np.errstate(divide="ignore"):
            img = np.log10(hist)
        xx, yy = np.meshgrid(zx, zy)
        pc = ax.pcolormesh(xx, yy, img, cmap="cividis", shading="auto", rasterized=True)
        ax.plot([0, 3], [0, 3], "k--", linewidth=0.8)
        ax.text(0.85, 0.95, label, transform=ax.transAxes, ha="center", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black",
                          linewidth=0.4))
        ax.set_xlabel(r"$\mathit{z}_{\mathrm{true}}$")
        ax.set_xlim(1e-2, 0.4)
        ax.set_ylim(1.7e-2, 2.1)
        ax.minorticks_on()
        ax.tick_params("both", length=3, width=0.8, which="major", direction="in", right=True,
                       top=True)
        ax.tick_params("both", length=2, width=0.5, which="minor", direction="in", right=True,
                       top=True)

        # inset: cumulative distribution of the Odds
        axins = inset_axes(ax, width="25%", height="25%", loc="upper left",
                           bbox_to_anchor=(0.15, -0.05, 1.1, 1.1), bbox_transform=ax.transAxes,
                           borderpad=2)
        x = np.sort(odds[good])
        axins.plot(x, np.arange(1, x.size + 1) / x.size, lw=1.2)
        axins.set_xlabel("Odds", fontsize=6)
        axins.set_ylabel("CDF", fontsize=6)
        axins.set_xlim(0, 1)
        axins.set_ylim(0, 1)
        axins.set_xticks([0, 0.5, 1])
        axins.set_yticks([0, 0.5, 1])
        axins.tick_params(axis="both", which="major", labelsize=5, direction="in", top=True,
                          right=True)
        axins.patch.set_facecolor("white")
        axins.patch.set_alpha(0.5)
        axins.patch.set_edgecolor("black")
        axins.patch.set_linewidth(0.6)

    axes[0].set_ylabel(r"$\mathit{z}_{\mathrm{photo}}$")
    formatter = FuncFormatter(lambda x, pos: f"{x:.3f}".rstrip("0").rstrip("."))
    for ax in axes:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    cb = fig.colorbar(pc, cax=fig.add_axes([0.36, 0.72, 0.12, 0.04]), orientation="horizontal")
    cb.set_ticks(np.log10([10, 1000]))
    cb.set_ticklabels(["10", "1000"])
    cb.set_label("Counts", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    fig.subplots_adjust(wspace=0.03)
    return fig, {}


if __name__ == "__main__":
    args = parse_args(__doc__, "fig03_photoz.pdf", NEEDS)
    fig, _ = make_figure(args.mocks)
    save(fig, args.out)
