"""bst_hod: Halo Occupation Distribution from projected galaxy counts with the
Background Subtraction Technique (BST), plus the Central Galaxy Finder (CGF).

Main entry points
-----------------
compute_hod            per-halo background-subtracted counts for one M_lim
compute_hod_samples    the same for several M_lim (volume-limited samples)
hod_curve              <N|M> in halo-mass bins with bootstrap/jackknife errors
find_central_galaxies  Central Galaxy Finder (centres without membership)
load_catalog           read a galaxy table and map its columns
plot_hod               quick-look plot

Data preparation for the LSST DESC mocks is in :mod:`bst_hod.prepare`.
"""

from .catalog import DEFAULT_COLUMNS, load_catalog, read_table
from .cgf import find_central_galaxies
from .core import GalaxyField, aperture_radii, bst_count
from .cosmology import COSMODC2, FlatLCDM, ORIGINAL_Z_CUTS, default_z_max, z_max_for_mlim
from .hod import compute_hod, compute_hod_samples, hod_curve
from .plotting import plot_hod

__version__ = "1.0.0"

__all__ = [
    "compute_hod", "compute_hod_samples", "hod_curve", "find_central_galaxies",
    "load_catalog", "read_table", "plot_hod", "bst_count", "aperture_radii", "GalaxyField",
    "FlatLCDM", "COSMODC2", "ORIGINAL_Z_CUTS", "default_z_max", "z_max_for_mlim",
    "DEFAULT_COLUMNS",
]
