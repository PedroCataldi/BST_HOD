"""Build the BST input tables from the LSST DESC mocks (run at NERSC).

Replaces lsst-COSMODC2_FZ.ipynb, lsst-COSMODC2_BPZ.ipynb, lsst_SkySim5000.ipynb
and read_bin-Pedro.ipynb. Needs GCRCatalogs (e.g. the desc-stack-weekly-latest
Jupyter kernel). Each call goes straight from the GCR catalogue to the text
table with m_r <= 21; no intermediate .bin file is needed.

The same lines work in a notebook cell.
"""

from bst_hod.prepare import bin_to_table, export_gcr_catalog

# cosmoDC2 with FlexZBoost photo-z  -> mock_21_large_FZ_z05_cosmoDC2.mpeg
export_gcr_catalog("flexzboost", "Tabs/mock_21_large_FZ_z05_cosmoDC2.mpeg", z_max=0.4)

# cosmoDC2 with BPZ photo-z         -> mock_21_large_BPZ_z05_cosmoDC2.mpeg
export_gcr_catalog("bpz", "Tabs/mock_21_large_BPZ_z05_cosmoDC2.mpeg", z_max=0.4)

# SkySim5000 (no photo-z)           -> mock_SkySim_z_lt_0_35_mag_lt_21.dat
export_gcr_catalog("skysim", "Tabs/mock_SkySim_z_lt_0_35_mag_lt_21.dat", z_max=0.35)

# Quick test on two healpix pixels only:
# export_gcr_catalog("flexzboost", "test.dat", healpix=[8786, 8787])

# Already have .bin files from the old notebooks? Convert them directly:
# bin_to_table("photoz_lsst_large_FZ_mock_z_lt_0_5.bin",
#              "Tabs/mock_21_large_FZ_z05_cosmoDC2.mpeg", layout="cosmodc2_photoz")
# bin_to_table("lsst_mock_z_lt_0_35.bin", "Tabs/mock_SkySim.dat", layout="skysim")
