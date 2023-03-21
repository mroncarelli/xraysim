import pygadgetreader as pygr
import numpy as np
from proj2d import makemap

infile = 'snapshots/snap_128'

def almost_equal(v0: float, v1: float, tol = 1.e-6):
    return abs((v1-v0)/v0) <= tol

def test_total_mass(infile):
    """
    Tests that the total mass in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    part = pygr.readsnap(infile, 'mass', 'gas', units=0)  # [10^10 h^-1 M_Sun]
    val_snap = sum(part)
    map_str = makemap.makemap(infile, 'rho', npix=256, struct=True)
    val_map = np.sum(map_str['map']) * map_str['pixel_size'] ** 2

    assert almost_equal(val_snap, val_map)