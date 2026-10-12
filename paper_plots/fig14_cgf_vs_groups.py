r"""Source notebook: Articulo1/scripts/Draft/FigApx2_BST_CGF.ipynb, cell 2
(saved there as FigApx2_Hist_CGF_All.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 14 (appendix A): normalised distributions of M200, r200 and stellar mass of
the centres found by the CGF ((f_r, f_z) = (2.5, 0.25), the four M_lim samples
together) and of the original groups (centrals with M_r < -19.5), with the
Kolmogorov-Smirnov statistic and p-value in each panel.

As in the notebook, the KS test compares the 20 histogram bin heights of the two
distributions, not the two samples of galaxies, and the CGF_fr_* runs are used
(see Fig. 6). The notebook took the cosmology for M_r from an earlier session;
cosmoDC2 is used here, which reproduces the notebook's KS values.

Reads the mock galaxy table (heavy file, ~0.2 GB) and the CGF .npz files.

    python paper_plots/fig14_cgf_vs_groups.py --tabs /path/to/Tabs --mocks /path/to/mocks [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig14_cgf_vs_groups.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig14_cgf_vs_groups.pdf

    # copy of that folder downloaded from Drive
    python paper_plots/fig14_cgf_vs_groups.py --tabs /home/tavernam/projects/lsst_hod/drive/Articulo1/tabs \
        --mocks /home/tavernam/projects/lsst_hod/drive/mocks

Instead of --tabs / --mocks you can set BST_HOD_TABS / BST_HOD_MOCKS; without
--mocks the mock table is looked for in the --tabs folder.
Without --out the figure goes to paper_plots/output/fig14_cgf_vs_groups.pdf.

Input: ``CGF_fr_2.5_fz_0.25Mlim_{m}.0.npz`` and ``mock_21_large_z05_cosmoDC2.mpeg``.
"""

import numpy as np
from scipy import stats

from common import MLIMS, abs_mag, load_mock, parse_args, r200_kpc, save, tabs_path

NEEDS = ("tabs", "mocks")
CGF_FILE = "CGF_fr_2.5_fz_0.25Mlim_{m:.1f}.npz"


def make_figure(tabs, mocks):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.linewidth": 0.8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "font.size": 8, "font.family": "serif",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    table = load_mock(mocks, "z05")
    gal = table[table["z"] > 0.0]
    z = gal["z"].to_numpy()
    groups = (gal["is_central"].to_numpy() == 1) & (abs_mag(gal["mr_lsst"].to_numpy(), z) < -19.5)
    orig = {"M200": gal["loghalo_mass"].to_numpy()[groups],
            "r200": r200_kpc(gal["loghalo_mass"].to_numpy()[groups], z[groups]),
            "Mstar": gal["logstellar_mass"].to_numpy()[groups]}

    parts = []
    for mlim in MLIMS:
        with np.load(tabs_path(tabs, CGF_FILE.format(m=mlim))) as d:
            ids = d["Gal_id"]
        parts.append(table[table["galid"].isin(ids[~np.isnan(ids)])])
    log_m = np.concatenate([p["loghalo_mass"].to_numpy() for p in parts])
    z_cgf = np.concatenate([p["z"].to_numpy() for p in parts])
    cgf = {"M200": log_m, "r200": r200_kpc(log_m, z_cgf),
           "Mstar": np.concatenate([p["logstellar_mass"].to_numpy() for p in parts])}

    fig, axes = plt.subplots(1, 3, figsize=(5.3, 2.2))
    labels = {"M200": r"$\log M_{200}\,[M_{\odot}]$", "r200": r"$r_{200}$ [kpc]",
              "Mstar": r"$\log M_{\star}\,[M_{\odot}]$"}
    for a, key in zip(axes, ("M200", "r200", "Mstar")):
        a.hist(orig[key], bins=20, histtype="step", linewidth=0.8, color="red", density=True,
               label="Original groups")
        a.hist(cgf[key], bins=20, histtype="step", linewidth=0.8, color="black", density=True,
               label="CGF")
        h_orig, edges = np.histogram(orig[key], bins=20, density=True)
        h_cgf, _ = np.histogram(cgf[key], bins=edges, density=True)
        ks = stats.ks_2samp(h_cgf, h_orig)
        print(f"KS for {key}: D = {ks.statistic:.2f}, p = {ks.pvalue:.3f}")
        a.text(0.94, 0.50, r"$D^{KS}=%.2f$" "\n" r"$p=%.2f$" % (ks.statistic, ks.pvalue),
               transform=a.transAxes, fontsize=7, va="center", ha="right")
        a.set_xlabel(labels[key])
        a.minorticks_on()
        a.tick_params("both", length=3, width=0.8, which="major", direction="in", top=True,
                      right=True)
        a.tick_params("both", length=2, width=0.5, which="minor", direction="in", top=True,
                      right=True)
    axes[0].set_ylabel("PDF")
    axes[1].legend(frameon=False, loc="best", fontsize=6)
    axes[2].set_xlim(9, 13.5)
    fig.tight_layout()
    return fig, {}


if __name__ == "__main__":
    args = parse_args(__doc__, "fig14_cgf_vs_groups.pdf", NEEDS)
    fig, _ = make_figure(args.tabs, args.mocks)
    save(fig, args.out)
