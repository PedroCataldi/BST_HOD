"""Source notebook: Articulo1/scripts/Draft/Fig12_BST_CGF_Odds.ipynb, cell 10
(saved there as Fig12_BST_CGF_r200_0.8Odds.pdf), in Pedro Cataldi's analysis folder on Drive.

Fig. 12: BST HOD with centres found by the CGF ((f_r, f_z) = (2.5, 0.25)) and
photometric redshifts, Odds > 0.8, FlexZBoost (top) and BPZ (bottom).

    python paper/fig12_cgf.py --tabs /path/to/Tabs --out paper/figures/fig12.pdf

Input: ``BST_{FZ,BPZ}_CGF_0.8Odds_r200_Mlim_{m}.npz`` and ``NTrue_DC2{m}.npz``.
"""

from common import parse_args, save
from fig10_photoz import plot_photoz

BST_FILE = "BST_{code}_CGF_0.8Odds_r200_Mlim_{m}.npz"


def make_figure(tabs):
    return plot_photoz(tabs, BST_FILE, "CGF")


if __name__ == "__main__":
    args = parse_args(__doc__, "paper/figures/fig12_cgf.pdf")
    fig, summary = make_figure(args.tabs)
    save(fig, args.out)
    for name, s in summary.items():
        print(name, "  ".join(f"{m}: {v[0]:.2f}±{v[1]:.2f}" for m, v in s.items()))
