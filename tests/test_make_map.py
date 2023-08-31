import pytest
import numpy as np
import pygadgetreader as pygr
import os

from src.pkg.gadgetutils.readspecial import readtemperature, readvelocity
from src.pkg.sphprojection.mapping import make_map

# Snapshot file on which the tests are performed
snapshot_file = os.environ.get('XRAYSIM') + '/tests/data/snap_Gadget_sample'


def test_total_mass(infile=snapshot_file):
    """
    Tests that the total mass in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    mass = pygr.readsnap(infile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    val_snap = sum(mass)
    map_str = make_map(infile, 'rho', npix=128, struct=True)
    val_map = np.sum(map_str['map']) * map_str['pixel_size'] ** 2

    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_total_emission_measure(infile=snapshot_file):
    """
    Tests that the total emission measure Int(rho^2 dV) in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    mass = pygr.readsnap(infile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    rho = pygr.readsnap(infile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    val_snap = sum(mass * rho)
    map_str = make_map(infile, 'rho2', npix=128, struct=True)
    val_map = np.sum(map_str['map']) * map_str['pixel_size'] ** 2

    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_average_tmw(infile=snapshot_file):
    """
    Tests that the average (mass-weighted) temperature in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    mass = pygr.readsnap(infile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    temp = readtemperature(infile, suppress=1)  # [K]
    val_snap = np.sum(mass * temp) / np.sum(mass)
    map_str = make_map(infile, 'Tmw', npix=128, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm']) / np.sum(map_str['norm'])

    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_average_tew(infile=snapshot_file):
    """
    Tests that the average emission-weighted temperature in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    mass = pygr.readsnap(infile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    rho = pygr.readsnap(infile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    temp = readtemperature(infile, suppress=1)  # [K]
    val_snap = np.sum(mass * rho * temp) / np.sum(mass * rho)
    map_str = make_map(infile, 'Tew', npix=128, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm']) / np.sum(map_str['norm'])

    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_total_momentum(infile=snapshot_file):
    """
    Tests that the total momentum in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    redshift = pygr.readhead(infile, 'redshift')
    mass = pygr.readsnap(infile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    vel = readvelocity(infile, units='km/s', suppress=1)[:, 2]  # [km s^-1]
    val_snap = np.sum(mass * vel)
    map_str = make_map(infile, 'vmw', npix=128, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm']) * map_str['pixel_size'] ** 2

    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_total_ew_momentum(infile=snapshot_file):
    """
    Tests that the total emission-weighted momentum in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    redshift = pygr.readhead(infile, 'redshift')
    mass = pygr.readsnap(infile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    rho = pygr.readsnap(infile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    vel = readvelocity(infile, units='km/s', suppress=1)[:, 2]  # [km s^-1]
    val_snap = np.sum(mass * rho * vel)
    map_str = make_map(infile, 'vew', npix=128, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm']) * map_str['pixel_size'] ** 2

    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_average_velocity_dispersion(infile=snapshot_file):
    """
    Tests that the average (mass-weighted) velocity dispersion in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    redshift = pygr.readhead(infile, 'redshift')
    mass = pygr.readsnap(infile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    vel = readvelocity(infile, units='km/s', suppress=1)[:, 2]  # [km s^-1]
    val_snap = np.sqrt(
        np.sum(mass * vel ** 2) / np.sum(mass) - (np.sum(mass * vel) / np.sum(mass)) ** 2)
    map_str = make_map(infile, 'wmw', npix=128, struct=True)
    val1_map = np.sum((map_str['map'] ** 2 + map_str['map2'] ** 2) * map_str['norm']) / np.sum(map_str['norm'])
    val2_map = (np.sum(map_str['map2'] * map_str['norm']) / np.sum(map_str['norm'])) ** 2
    val_map = np.sqrt(val1_map - val2_map)

    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_average_ew_velocity_dispersion(infile=snapshot_file):
    """
    Tests that the average emission-weighted velocity dispersion in the projected map is the same as the snapshot one
    :param infile: (str) Snapshot file
    """
    redshift = pygr.readhead(infile, 'redshift')
    mass = pygr.readsnap(infile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    rho = pygr.readsnap(infile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    vel = readvelocity(infile, units='km/s', suppress=1)[:, 2]  # [km s^-1]
    val_snap = np.sqrt(
        np.sum(mass * rho * vel ** 2) / np.sum(mass * rho) - (np.sum(mass * rho * vel) / np.sum(mass * rho)) ** 2)
    map_str = make_map(infile, 'wew', npix=128, struct=True)
    val1_map = np.sum((map_str['map'] ** 2 + map_str['map2'] ** 2) * map_str['norm']) / np.sum(map_str['norm'])
    val2_map = (np.sum(map_str['map2'] * map_str['norm']) / np.sum(map_str['norm'])) ** 2
    val_map = np.sqrt(val1_map - val2_map)

    assert val_map == pytest.approx(val_snap, rel=1e-6)
