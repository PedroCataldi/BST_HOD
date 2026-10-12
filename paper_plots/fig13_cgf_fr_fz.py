r"""Source notebook: Articulo1/scripts/Draft/FigApx1_Fr_Fz.ipynb, cell 4
(saved there as FigApx1_Fr_Fz.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 13 (appendix A): purity (left) and completeness (right) of the CGF for
each combination of its parameters f_r and f_z, one row per M_lim. The values
are the purity and completeness vs M200 integrated over 1e12 - 5e14 Msun and
normalised by that range, already computed in
``norm_purity_completeness_vs_M200.csv`` (cell 2 of the notebook, with the
CGF_fr_* runs; not recomputed here).

    python paper_plots/fig13_cgf_fr_fz.py --tabs /path/to/Tabs [--out file.pdf]

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/fig13_cgf_fr_fz.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
        --out /Users/pedrocataldi/Desktop/Background_LSST/Figs/fig13_cgf_fr_fz.pdf

    # copy of that folder downloaded from Drive (Articulo1/tabs)
    python paper_plots/fig13_cgf_fr_fz.py --tabs /home/tavernam/projects/lsst_hod/drive/Articulo1/tabs

Instead of --tabs you can set the environment variable BST_HOD_TABS.
Without --out the figure goes to paper_plots/output/fig13_cgf_fr_fz.pdf.

Input: ``norm_purity_completeness_vs_M200.csv``.
"""

import numpy as np

from common import MLIMS, parse_args, save, tabs_path

NEEDS = ("tabs",)
CSV_FILE = "norm_purity_completeness_vs_M200.csv"


def make_figure(tabs):
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
                         "axes.linewidth": 0.8, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": False, "ytick.right": False,
                         "ytick.left": False, "text.usetex": False})
    df = pd.read_csv(tabs_path(tabs, CSV_FILE))
    f_z = sorted(df["f_z"].unique())
    f_r = sorted(df["f_r"].unique())
    fz_mesh, fr_mesh = np.meshgrid(f_z, f_r)

    fig, axes = plt.subplots(4, 2, figsize=(7.08, 6.5), sharex=True, sharey=True)
    for row, mlim in enumerate(MLIMS):
        df_m = df[df["Mlim"] == mlim]
        for col, (key, title) in enumerate((("purity", r"Purity [$\%$]"),
                                            ("completeness", r"Completeness [$\%$]"))):
            grid = df_m.pivot(index="f_r", columns="f_z", values=key).reindex(index=f_r,
                                                                             columns=f_z)
            a = axes[row, col]
            a.pcolormesh(fz_mesh, fr_mesh, grid.values, shading="auto", cmap="jet", vmin=15,
                         vmax=100)
            a.invert_yaxis()
            if row == 0:
                a.set_title(title, fontsize=8)
            for i in range(len(f_r)):
                for j in range(len(f_z)):
                    v = grid.values[i, j]
                    if np.isfinite(v):
                        a.text(f_z[j], f_r[i], f"{v:.1f}", ha="center", va="center",
                               fontsize=7, color="black" if v > 30 else "white")
        axes[row, 1].annotate(f"$M_{{\\rm lim}} < {mlim}$", xy=(1.05, 0.5),
                              xycoords="axes fraction", ha="left", va="center", fontsize=11,
                              rotation=90)
    for col in range(2):
        axes[-1, col].set_xlabel("$f_z$", fontsize=11)
    for row in range(4):
        axes[row, 0].set_ylabel("$f_r$", fontsize=11)
        axes[row, 1].tick_params(labelleft=False)
    for a in axes.flat:
        a.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25])
    fig.subplots_adjust(wspace=0.07, hspace=0.15)
    return fig, {}


if __name__ == "__main__":
    args = parse_args(__doc__, "fig13_cgf_fr_fz.pdf", NEEDS)
    fig, _ = make_figure(args.tabs)
    save(fig, args.out)
