import pygadgetreader as pygr
from gadgetutils.readspecial import readtemperature
import numpy as np
from sphprojection import makemap

def almost_equal(v0: float, v1: float, tol = 1e-6):
    return abs(v1/v0 - 1) <= tol

# Snapshot file on which the tests are performed
snapshot_file = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/snap_128'

# Reading particle information
mass_part = pygr.readsnap(snapshot_file, 'mass', 'gas', units=0)  # [10^10 h^-1 M_Sun]
rho_part = pygr.readsnap(snapshot_file, 'rho', 'gas', units=0)  # [10^10 h^-1 M_Sun]
temp_part = pygr.readsnap(snapshot_file, 'u', 'gas', units=1)  # [K]

def test_total_mass(infile = snapshot_file):
    """
    Tests that the total mass in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    mass_part = pygr.readsnap(infile, 'mass', 'gas', units=0)  # [10^10 h^-1 M_Sun]
    val_snap = sum(mass_part)
    map_str = makemap.makemap(infile, 'rho', npix=256, struct=True)
    val_map = np.sum(map_str['map']) * map_str['pixel_size'] ** 2

    assert almost_equal(val_snap, val_map)

def test_total_thermal_energy(infile = snapshot_file):
    """
    Tests that the average (mass-weighted) temperature in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    mass_part = pygr.readsnap(infile, 'mass', 'gas', units=0)  # [10^10 h^-1 M_Sun]
    temp_part = readtemperature(infile)  # [K]
    val_snap = np.sum(mass_part * temp_part) / np.sum(mass_part)
    map_str = makemap.makemap(infile, 'Tmw', npix=256, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm']) / np.sum(map_str['norm'])

    assert almost_equal(val_snap, val_map)