"""Make the paper figures from the precomputed BST tables and compare the
median BST/true ratios with Table 3 of Cataldi et al. (2026).

    python paper/make_figures.py --tabs /path/to/Tabs [--figs 7 9 10 11 12] [--ext pdf]
"""

import argparse
import os

import fig07_bst_r200
import fig09_bst_rlum
import fig10_photoz
import fig11_mass_scatter
import fig12_cgf
from common import MLIMS, TABS_ENV, save

FIGURES = {
    7: ("fig07_bst_r200", fig07_bst_r200),
    9: ("fig09_bst_rlum", fig09_bst_rlum),
    10: ("fig10_photoz", fig10_photoz),
    11: ("fig11_mass_scatter", fig11_mass_scatter),
    12: ("fig12_cgf", fig12_cgf),
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
    p.add_argument("--tabs", default=os.environ.get(TABS_ENV),
                   help=f"folder with the paper .npz tables (default: ${TABS_ENV})")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figures"))
    p.add_argument("--figs", type=int, nargs="+", default=sorted(FIGURES))
    p.add_argument("--ext", default="pdf")
    args = p.parse_args()
    if not args.tabs:
        p.error(f"give --tabs or set ${TABS_ENV}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {}
    for n in args.figs:
        name, module = FIGURES[n]
        fig, s = module.make_figure(args.tabs)
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
