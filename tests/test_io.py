"""Reading/writing tables and the legacy binary files."""

import struct

import numpy as np

from bst_hod import compute_hod_samples, hod_curve, load_catalog
from bst_hod.prepare import read_legacy_bin, write_table
from bst_hod.synthetic import make_mock_catalog


def test_original_table_format(tmp_path):
    """Header '#galid, ra, ..., \\n' and %.5d integers, as read_bin-Pedro.ipynb wrote."""
    df = make_mock_catalog(n_halos=300, n_field=5000, seed=2)
    path = tmp_path / "mock_21_large_FZ_z05_cosmoDC2.mpeg"
    with open(path, "w") as f:
        f.write("#galid, ra, dec, z, mr_lsst, halo_id, loghalo_mass, is_central \n")
        np.savetxt(f, np.column_stack([df.galid, df.ra, df.dec, df.z, df.mr_lsst, df.halo_id,
                                       df.loghalo_mass, df.is_central]),
                   fmt="%.14d %.5f %.5f %.5f %.5f %.14d %.5f %.5d")
    cat = load_catalog(str(path))
    assert list(cat.columns[:5]) == ["galaxy_id", "ra", "dec", "z", "mag"]
    assert len(cat) == len(df) and cat["is_central"].dtype.kind == "i"
    assert np.array_equal(cat["galaxy_id"].to_numpy(), df["galid"].to_numpy())


def test_column_mapping_override(tmp_path):
    df = make_mock_catalog(n_halos=100, n_field=2000, seed=4)
    df["photoz_mode"] = df["z"] + 0.01
    cat = load_catalog(df, columns={"z": "photoz_mode"})
    assert np.allclose(cat["z"], df["z"] + 0.01)
    assert np.allclose(cat["z_original"], df["z"])


def test_legacy_bin_roundtrip(tmp_path):
    """Write a .bin exactly like lsst-COSMODC2_FZ.ipynb and read it back."""
    rng = np.random.default_rng(0)
    n = 500
    cols = dict(galaxy_id=rng.integers(1, 10**12, n), halo_id=rng.integers(1, 10**12, n),
                ra=rng.uniform(50, 70, n), dec=rng.uniform(-40, -25, n),
                redshift=rng.uniform(0, 0.4, n), mag_r_lsst=rng.uniform(16, 24, n),
                mag_g_lsst=rng.uniform(16, 24, n), halo_mass=10 ** rng.uniform(11, 15, n),
                stellar_mass=10 ** rng.uniform(8, 12, n), is_central=rng.integers(0, 2, n))
    for k in ["position_x", "position_y", "position_z", "photoz_mean", "photoz_mode",
              "photoz_odds", "mag_r_photoz", "mag_g_photoz", "mag_err_r_photoz",
              "mag_err_g_photoz"]:
        cols[k] = rng.normal(size=n)
    path = tmp_path / "photoz_test.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<q", cols["galaxy_id"][i]))
            f.write(struct.pack("<q", cols["halo_id"][i]))
            for k in ["ra", "dec", "redshift", "mag_r_lsst", "mag_g_lsst", "halo_mass",
                      "stellar_mass"]:
                f.write(struct.pack("<f", cols[k][i]))
            f.write(struct.pack("<I", cols["is_central"][i]))
            for k in ["position_x", "position_y", "position_z", "photoz_mean", "photoz_mode",
                      "photoz_odds", "mag_r_photoz", "mag_g_photoz", "mag_err_r_photoz",
                      "mag_err_g_photoz"]:
                f.write(struct.pack("<f", cols[k][i]))
    df = read_legacy_bin(str(path), "cosmodc2_photoz")
    assert len(df) == n
    assert np.array_equal(df["galaxy_id"], cols["galaxy_id"])
    assert np.allclose(df["photoz_odds"], np.float32(cols["photoz_odds"]))
    out = tmp_path / "table.dat"
    write_table(df, str(out), mag_max=21.0)
    cat = load_catalog(str(out))
    keep = np.float32(cols["mag_r_lsst"]) <= 21
    assert len(cat) == keep.sum()
    assert np.allclose(cat["log_halo_mass"], np.log10(np.float32(cols["halo_mass"]))[keep],
                       atol=1e-5)


def test_samples_and_curve(tmp_path):
    cat = make_mock_catalog(seed=5)
    res = compute_hod_samples(cat, mlims=(-19, -20), save_dir=str(tmp_path), verbose=False)
    assert set(res) == {-19, -20}
    assert (tmp_path / "BST_Mlim_-19.csv").exists()
    for err in ("bootstrap", "jackknife", "std"):
        c = hod_curve(res[-19], errors=err)
        assert (c["n_halos"] > 0).all() and np.isfinite(c["N_bst"]).all()
