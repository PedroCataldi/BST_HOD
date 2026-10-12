r"""Make the paper figures from the precomputed BST tables (and, for Figs. 1, 2,
3, 4, 6, 8, 14 and 16, from the mock galaxy tables) and compare the median BST/true
ratios with Table 3 of Cataldi et al. (2026).

    python paper_plots/make_figures.py --tabs /path/to/Tabs [--mocks /path/to/mocks]
                                 [--out /path/to/Figs] [--figs 1 2 3 ...] [--ext pdf]
                                 [--light]

--light skips the figures that read the heavy mock galaxy tables (0.2-0.5 GB).

Examples of where the tables and figures were:

    # Pedro's analysis folder (paths used in the original notebooks)
    python paper_plots/make_figures.py --tabs /Users/pedrocataldi/Desktop/Background_LSST/Tabs \
                                 --out /Users/pedrocataldi/Desktop/Background_LSST/Figs

    # copy of that folder downloaded from Drive (.npz in Articulo1/tabs, mocks in mocks/)
    python paper_plots/make_figures.py --tabs /home/tavernam/projects/lsst_hod/drive/Articulo1/tabs \
                                 --mocks /home/tavernam/projects/lsst_hod/drive/mocks

Instead of --tabs / --mocks you can set BST_HOD_TABS / BST_HOD_MOCKS. Without
--mocks the mock tables are looked for in the --tabs folder (as in Pedro's Tabs/).
Without --out the figures go to paper_plots/output/ (not in git). paper_plots/figures/
has the reference copy of the figures, made with these scripts.
"""

import argparse
import os

import fig01_volume_limited
import fig02_central_satellite
import fig03_photoz
import fig04_bst_method
import fig06_cgf_purity
import fig07_bst_r200
import fig08_lum_r200_fit
import fig09_bst_rlum
import fig10_photoz
import fig11_mass_scatter
import fig12_cgf
import fig13_cgf_fr_fz
import fig14_cgf_vs_groups
import fig15_bst_cgf_fr_fz
import fig16_mass_scatter_hist
from common import MLIMS, OUTPUT_DIR, add_data_args, check_data_args, data_for, save

FIGURES = {
    1: ("fig01_volume_limited", fig01_volume_limited),
    2: ("fig02_central_satellite", fig02_central_satellite),
    3: ("fig03_photoz", fig03_photoz),
    4: ("fig04_bst_method", fig04_bst_method),
    6: ("fig06_cgf_purity", fig06_cgf_purity),
    8: ("fig08_lum_r200_fit", fig08_lum_r200_fit),
    7: ("fig07_bst_r200", fig07_bst_r200),
    9: ("fig09_bst_rlum", fig09_bst_rlum),
    10: ("fig10_photoz", fig10_photoz),
    11: ("fig11_mass_scatter", fig11_mass_scatter),
    12: ("fig12_cgf", fig12_cgf),
    13: ("fig13_cgf_fr_fz", fig13_cgf_fr_fz),
    14: ("fig14_cgf_vs_groups", fig14_cgf_vs_groups),
    15: ("fig15_bst_cgf_fr_fz", fig15_bst_cgf_fr_fz),
    16: ("fig16_mass_scatter_hist", fig16_mass_scatter_hist),
}

#: Table 3 of the paper: median ratio and scatter for M_r < -17, -18, -19, -20
TABLE3 = {
    "BST r200": [(1.02, 0.15), (0.97, 0.07), (0.97, 0.03), (0.97, 0.06)],
    "BST r_c^lum": [(0.98, 0.13), (0.93, 0.06), (0.96, 0.03), (0.98, 0.06)],
    "Photo-z FZ": [(0.98, 0.07), (0.90, 0.05), (0.93, 0.04), (0.97, 0.08)],
    "Photo-z BPZ": [(0.90, 0.13), (0.93, 0.09), (0.97, 0.03), (0.98, 0.05)],
    "Mass error 0.2 dex FZ": [(0.93, 0.11), (0.93, 0.21), (0.89, 0.19), (0.90, 0.08)],
    "Mass error 0.2 dex BPZ": [(0.93, 0.19), (0.89, 0.19), (0.94, 0.08), (0.92, 0.07)],
    "CGF FZ": [(0.74, 0.25), (0.86, 0.15), (0.93, 0.04), (0.94, 0.09)],
    "CGF BPZ": [(0.76, 0.25), (0.89, 0.11), (0.95, 0.05), (0.94, 0.09)],
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--out", default=OUTPUT_DIR)
    p.add_argument("--figs", type=int, nargs="+", default=sorted(FIGURES),
                   choices=sorted(FIGURES))
    p.add_argument("--ext", default="pdf")
    p.add_argument("--light", action="store_true",
                   help="skip the figures that read the heavy mock galaxy tables")
    args = p.parse_args()
    figs = [n for n in args.figs if not (args.light and "mocks" in FIGURES[n][1].NEEDS)]
    check_data_args(p, args, {need for n in figs for need in FIGURES[n][1].NEEDS})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {}
    for n in figs:
        name, module = FIGURES[n]
        fig, s = module.make_figure(*data_for(module, args))
        save(fig, os.path.join(args.out, f"{name}.{args.ext}"))
        plt.close(fig)
        summary.update(s)

    print("\nMedian ratio ± scatter: this run | Table 3")
    for name, s in summary.items():
        ref = TABLE3.get(name)
        cells = []
        for i, m in enumerate(MLIMS):
            here = f"{s[m][0]:.2f}±{s[m][1]:.2f}"
            if ref:
                same = (round(s[m][0], 2), round(s[m][1], 2)) == ref[i]
                here += f" | {ref[i][0]:.2f}±{ref[i][1]:.2f}" + ("" if same else " *")
            cells.append(f"{m}: {here}")
        print(f"  {name:24s} " + "   ".join(cells))
    print("(* = differs from Table 3)")


if __name__ == "__main__":
    main()
