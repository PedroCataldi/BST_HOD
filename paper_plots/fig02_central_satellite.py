r"""Source notebook: Articulo1/scripts/Draft/Fig3_Central_vs_Satellite.ipynb, cell 1
(saved there as Fig3_Central_vs_Satellite.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 2: normalised distributions of halo mass, stellar mass, redshift and r200
for central galaxies, group centres (centrals with M_r < -19.5) and satellites
of cosmoDC2 (true z < 0.4).

Reads the mock galaxy table (heavy file, ~0.2 GB).

    python paper_plots/fig02_central_satellite.py --mocks /path/to/mocks [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig02_central_satellite.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig02_central_satellite.pdf

    # copy of that folder downloaded from Drive
    python paper_plots/fig02_central_satellite.py --mocks /home/tavernam/projects/lsst_hod/drive/mocks

Instead of --mocks you can set the environment variable BST_HOD_MOCKS; without
it the table is looked for in the --tabs folder.
Without --out the figure goes to paper_plots/output/fig02_central_satellite.pdf.

Input: ``mock_21_large_z05_cosmoDC2.mpeg``.
"""

import numpy as np

from common import abs_mag, load_mock, parse_args, r200_kpc, save

NEEDS = ("mocks",)


def make_figure(mocks):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.linewidth": 0.8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "font.size": 8, "font.family": "serif", "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    gal = load_mock(mocks, "z05")
    gal = gal[(gal["z"] > 0.0) & (gal["z"] < 0.4)]
    z = gal["z"].to_numpy()
    halo_mass = gal["loghalo_mass"].to_numpy()
    stellar_mass = gal["logstellar_mass"].to_numpy()
    is_central = gal["is_central"].to_numpy()
    mabs = abs_mag(gal["mr_lsst"].to_numpy(), z)
    r200 = r200_kpc(halo_mass, z)

    samples = [  # (selection, colour, label)
        (is_central == 1, "black", "Central"),
        ((is_central == 1) & (mabs < -19.5), "red", "Group centre"),
        (is_central == 0, "orange", "Satellite"),
    ]
    print("N central, satellite, group centre:", np.sum(samples[0][0]), np.sum(samples[2][0]),
          np.sum(samples[1][0]))

    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.5))
    axli = axs.flatten()
    rbins = np.logspace(np.log10(0.01), np.log10(1500), 51)
    for sel, color, label in samples:
        kw = dict(histtype="step", linewidth=0.8, color=color, density=True)
        axli[0].hist(halo_mass[sel], bins=20, **kw)
        axli[1].hist(stellar_mass[sel], bins=20, label=label, **kw)
        axli[2].hist(z[sel], bins=20, **kw)
        if label != "Satellite":          # r200 is the host-halo radius: centrals only
            axli[3].hist(r200[sel], bins=rbins, **kw)

    labels_x = [r"$\log M_{200}\, [\mathrm{M_\odot}]$", r"$\log M_{\star}\, [\mathrm{M_\odot}]$",
                r"$z$", r"$r_{200}$ [kpc]"]
    axli[1].set_xlim(6, 12.5)
    axli[3].set_xlim(0, 1000)
    axli[1].legend(frameon=False, loc="upper left", fontsize=5)
    for i in (1, 3):
        axli[i].yaxis.set_label_position("right")
        axli[i].yaxis.tick_right()
    for i, ax in enumerate(axli):
        ax.set_xlabel(labels_x[i], fontsize=8)
        ax.set_ylabel("PDF", fontsize=8)
        ax.minorticks_on()
        ax.tick_params("both", length=2, width=0.8, which="major", direction="in", right=True,
                       top=True)
        ax.tick_params("both", length=1, width=0.5, which="minor", direction="in", right=True,
                       top=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    return fig, {}


if __name__ == "__main__":
    args = parse_args(__doc__, "fig02_central_satellite.pdf", NEEDS)
    fig, _ = make_figure(args.mocks)
    save(fig, args.out)
