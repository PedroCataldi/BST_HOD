"""Measure the HOD of the cosmoDC2 / SkySim mocks for several M_lim.

Replaces the original BST_example_for_DC2.py. Example:

    python examples/run_dc2.py Tabs/mock_21_large_FZ_z05_cosmoDC2.mpeg \
        --mlims -17 -18 -19 -20 --out Tabs/

Use photometric redshifts/magnitudes instead of the true ones with e.g.

    --z-col photoz_mode --mag-col mag_r_photoz
"""

import argparse

from bst_hod import compute_hod_samples, hod_curve


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("catalog", help="galaxy table (e.g. mock_21_large_FZ_z05_cosmoDC2.mpeg)")
    p.add_argument("--mlims", type=float, nargs="+", default=[-17, -18, -19, -20])
    p.add_argument("--out", default=".", help="output directory")
    p.add_argument("--prefix", default="BST_DC2")
    p.add_argument("--z-col", help="column to use as redshift (default: z)")
    p.add_argument("--mag-col", help="column to use as apparent r magnitude (default: mr_lsst)")
    p.add_argument("--centres", default="is_central", choices=["is_central", "cgf"])
    p.add_argument("--r-max", type=float, default=1.25)
    p.add_argument("--factor-r", type=float, default=1.0)
    p.add_argument("--ring", default="classic", choices=["classic", "fix"])
    p.add_argument("--ring-radius", type=float, nargs=2, default=[1.5, 3.5])
    p.add_argument("--area", default="pixel", choices=["pixel", "analytic"])
    p.add_argument("--no-mag-cut", action="store_true",
                   help="count all galaxies (background_method_new behaviour)")
    args = p.parse_args()

    columns = {}
    if args.z_col:
        columns["z"] = args.z_col
    if args.mag_col:
        columns["mag"] = args.mag_col

    mlims = [int(m) if float(m).is_integer() else m for m in args.mlims]
    results = compute_hod_samples(
        args.catalog, mlims=mlims, columns=columns or None, save_dir=args.out,
        prefix=args.prefix, centres=args.centres, r_max=args.r_max, factor_r=args.factor_r,
        ring=args.ring, ring_radius=tuple(args.ring_radius), area=args.area,
        mag_cut=not args.no_mag_cut)

    for m, res in results.items():
        curve = hod_curve(res, errors="bootstrap")
        path = f"{args.out}/{args.prefix}_HOD_Mlim_{m}.csv"
        curve.to_csv(path, index=False)
        print(f"M_lim = {m}: HOD curve written to {path}")


if __name__ == "__main__":
    main()
