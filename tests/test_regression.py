"""The new code must reproduce the original BST_method.py / CGF_Mlim.py."""

import numpy as np
import pytest

from bst_hod import compute_hod, find_central_galaxies, load_catalog
from bst_hod.cosmology import COSMODC2, absolute_magnitude, comoving_distance, r200_mpc
from bst_hod.legacy import background_method_new, background_method_rmax
from bst_hod.synthetic import make_mock_catalog

MLIM, ZMAX = -19, 0.2055


@pytest.fixture(scope="module")
def cat():
    return load_catalog(make_mock_catalog(seed=3))


def legacy_inputs(cat, mlim, zmax):
    """Exactly what BST_example_for_DC2.py builds before calling the method."""
    alfa_gal = cat["ra"].to_numpy() * np.pi / 180.
    delta_gal = cat["dec"].to_numpy() * np.pi / 180.
    alfa_gal[alfa_gal > np.pi] -= 2 * np.pi
    z_gal = cat["z"].to_numpy()
    magr_gal = cat["mag"].to_numpy()
    d_com = comoving_distance(z_gal)
    mabs = absolute_magnitude(magr_gal, z_gal, d_com=d_com)
    sel = (cat["is_central"].to_numpy() == 1) & (mabs < -19.5) & (z_gal < zmax) & (mabs < mlim)
    M_200 = 10 ** cat["log_halo_mass"].to_numpy()[sel]
    o = M_200.argsort()[::-1]
    g = dict(halo_gr_id=cat["halo_id"].to_numpy()[sel][o], alfa_gr=alfa_gal[sel][o],
             delta_gr=delta_gal[sel][o], z_gr=z_gal[sel][o], M_200=M_200[o])
    g["r_200"] = r200_mpc(g["M_200"], g["z_gr"])
    g["d_com_gr"] = comoving_distance(g["z_gr"])
    gal = dict(halo_id=cat["halo_id"].to_numpy(), mabs=mabs, magr_gal=magr_gal,
               delta_gal=delta_gal, alfa_gal=alfa_gal, z_gal=z_gal)
    return g, gal


def run_legacy(fn, j, g, gal, ring="classic", ring_radius=(1.5, 3.5), r_max=1.25, factor_r=1.0):
    return fn(j, MLIM, ZMAX, g["halo_gr_id"], g["d_com_gr"], ring, list(ring_radius), r_max,
              factor_r, g["r_200"], g["delta_gr"], g["alfa_gr"], g["M_200"], gal["halo_id"],
              g["z_gr"], gal["mabs"], gal["magr_gal"], gal["delta_gal"], gal["alfa_gal"],
              gal["z_gal"])


def sample(n, k=40, seed=0):
    rng = np.random.default_rng(seed)
    return np.unique(np.r_[np.arange(min(15, n)), rng.choice(n, size=min(k, n), replace=False)])


@pytest.mark.parametrize("mag_cut,fn", [(True, background_method_rmax),
                                         (False, background_method_new)])
def test_core_matches_original(cat, mag_cut, fn):
    new = compute_hod(cat, MLIM, ZMAX, mag_cut=mag_cut, verbose=False)
    g, gal = legacy_inputs(cat, MLIM, ZMAX)
    assert len(new) == len(g["z_gr"])
    for j in sample(len(new)):
        n_true, n_bst, m200 = run_legacy(fn, j, g, gal)
        assert new["N_true"].iloc[j] == n_true
        assert np.isclose(10 ** new["log_M200"].iloc[j], m200)
        assert new["N_bst"].iloc[j] == pytest.approx(n_bst, abs=1e-9), j


@pytest.mark.parametrize("ring,ring_radius,r_max,factor_r", [
    ("fix", (1.5, 3.5), 1.25, 1.0),
    ("classic", (2.0, 4.0), None, 0.5),
    ("classic", (1.5, 3.5), 0.8, "array"),
])
def test_options_match_original(cat, ring, ring_radius, r_max, factor_r):
    g, gal = legacy_inputs(cat, MLIM, ZMAX)
    n = len(g["z_gr"])
    if factor_r == "array":
        factor_r = np.random.default_rng(5).uniform(0.5, 1.5, n)
    new = compute_hod(cat, MLIM, ZMAX, ring=ring, ring_radius=ring_radius, verbose=False,
                      r_max=r_max, factor_r=factor_r)
    for j in sample(n, k=20, seed=1):
        _, n_bst, _ = run_legacy(background_method_rmax, j, g, gal, ring, ring_radius,
                                 1e30 if r_max is None else r_max, factor_r)
        assert new["N_bst"].iloc[j] == pytest.approx(n_bst, abs=1e-9), j


def original_cgf(cat, mlim, zmax, factor_raddi=2.5, z_bin=0.05):
    """The loop of CGF_Mlim.py, unchanged except for reading from `cat`."""
    slope, ordenada = 1634917221.9876266, -43596515.20330819
    galaxy_id = cat["galaxy_id"].to_numpy()
    alfa_gal = cat["ra"].to_numpy() * np.pi / 180.
    delta_gal = cat["dec"].to_numpy() * np.pi / 180.
    alfa_gal[alfa_gal > np.pi] -= 2 * np.pi
    z_gal = cat["z"].to_numpy()
    d_com = comoving_distance(z_gal)
    mabs = absolute_magnitude(cat["mag"].to_numpy(), z_gal, d_com=d_com)
    r_Lum = (10 ** (-0.4 * mabs) - ordenada) / slope
    s = (mabs < -19.5) & (z_gal < zmax) & (mabs < mlim)
    galaxy_id_cand, alfa_cand, delta_cand = galaxy_id[s], alfa_gal[s], delta_gal[s]
    z_cand, r_Lum_cand, mabs_cand = z_gal[s], r_Lum[s], mabs[s]
    d_com_cand = d_com[s] / (1 + z_cand)
    o = mabs_cand.argsort()
    galaxy_id_cand, alfa_cand, delta_cand = galaxy_id_cand[o], alfa_cand[o], delta_cand[o]
    z_cand, r_Lum_cand, d_com_cand = z_cand[o], r_Lum_cand[o], d_com_cand[o]
    found = []
    for gal_target in list(galaxy_id_cand):
        m = galaxy_id_cand == gal_target
        if not m.any():
            continue
        found.append(gal_target)
        radio = (np.sin(delta_cand[m]) * np.sin(delta_cand)
                 + np.cos(delta_cand[m]) * np.cos(delta_cand) * np.cos(alfa_cand[m] - alfa_cand))
        with np.errstate(invalid="ignore", divide="ignore"):
            radio_sqr = np.sqrt(1. - radio * radio) / radio * d_com_cand[m]
        idx = np.where((radio_sqr < factor_raddi * r_Lum_cand[m]) & (radio_sqr > 0)
                       & (np.abs(z_cand - z_cand[m]) < z_bin))
        galaxy_id_cand = np.delete(galaxy_id_cand, idx)
        alfa_cand = np.delete(alfa_cand, idx)
        delta_cand = np.delete(delta_cand, idx)
        z_cand = np.delete(z_cand, idx)
        r_Lum_cand = np.delete(r_Lum_cand, idx)
        d_com_cand = np.delete(d_com_cand, idx)
    return np.array(found)


@pytest.mark.parametrize("mlim,zmax", [(-19, 0.2055), (-20, 0.3038)])
def test_cgf_matches_original(cat, mlim, zmax):
    new = find_central_galaxies(cat, mlim, zmax, verbose=False)
    old = original_cgf(cat, mlim, zmax)
    assert np.array_equal(new["galaxy_id"].to_numpy(), old)


def test_analytic_area_close_to_pixel(cat):
    pix = compute_hod(cat, MLIM, ZMAX, verbose=False)
    ana = compute_hod(cat, MLIM, ZMAX, area="analytic", verbose=False)
    assert np.allclose(ana["area_ratio"], 1 / (3.5 ** 2 - 1.5 ** 2))
    # same galaxies counted, only the area ratio differs
    assert np.array_equal(pix["n_circle"], ana["n_circle"])
    assert abs(pix["N_bst"].mean() - ana["N_bst"].mean()) < 0.05 * ana["N_bst"].mean()


def test_centres_from_cgf_and_user(cat):
    res = compute_hod(cat, MLIM, ZMAX, centres="cgf", verbose=False)
    cg = find_central_galaxies(cat, MLIM, ZMAX, verbose=False)
    assert set(res["galaxy_id"]) == set(cg["galaxy_id"])
    user = cg[["galaxy_id"]].assign(log_halo_mass=13.0)
    res2 = compute_hod(cat, MLIM, ZMAX, centres=user, verbose=False)
    assert np.allclose(res2["log_M200"], 13.0)


def test_clip_cos_only_adds_lost_centrals(cat):
    a = compute_hod(cat, MLIM, ZMAX, verbose=False)
    b = compute_hod(cat, MLIM, ZMAX, clip_cos=True, verbose=False)
    diff = b["n_circle"].to_numpy() - a["n_circle"].to_numpy()
    assert set(np.unique(diff)) <= {0, 1}
    assert 0 < diff.mean() < 0.1
