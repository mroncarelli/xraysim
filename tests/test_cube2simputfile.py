import pytest
from astropy.io import fits
import os

from src.pkg.specutils.sixte import cube2simputfile
from src.pkg.sphprojection.mapping import make_speccube
from src.pkg.specutils.tables import read_spectable, calc_spec
from src.pkg.gadgetutils.phys_const import keV2K

infile = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/snap_128'
spfile = os.environ.get('XRAYSIM') + '/tests/data/test_emission_table.fits'
npix, size, redshift, center, proj, flag_ene, nsample = 10, 1., 0.1, [500e3, 500e3], 'z', False, 10000
t_iso_keV = 6.3  # [keV]
t_iso = t_iso_keV * keV2K  # 73108018.313372612  # [K] (= 6.3 keV)
nene = fits.open(spfile)[0].header.get('NENE')
simput_file = os.environ.get('XRAYSIM') + '/tests/data/file_created_for_test.simput'

# Isothermal + no velocities
spec_cube = make_speccube(infile, spfile, size=size, npix=npix, redshift=redshift, center=center, proj=proj,
                          nsample=nsample, isothermal=t_iso, novel=True)


def test_file_created(inp=spec_cube, out=simput_file):
    """
    The output SIMPUT file must be correctly created
    :param inp: (dict) spectral cube structure
    :param out: (str) output SIMPUT file
    """
    if os.path.isfile(out):
        os.remove(out)
    cube2simputfile(inp, out)
    assert os.path.isfile(out)
    os.remove(out)


def test_primary_header_keywords(inp=spec_cube, out=simput_file):
    """
    The header of the Primary of the output SIMPUT file must contain a series of keywords whose value depend on the
    input data
    :param inp: (dict) spectral cube structure
    :param out: (str) output SIMPUT file
    """
    if os.path.isfile(out):
        os.remove(out)
    center_map = (12., 34.)
    cube2simputfile(inp, out, pos=center_map)
    header = fits.open(out)[0].header
    os.remove(out)
    assert header.get('SIM_FILE') == inp.get('simulation_file')
    assert header.get('SP_FILE') == inp.get('spectral_table')
    assert header.get('PROJ') == inp.get('proj')
    assert header.get('Z_COS') == inp.get('z_cos')
    assert (header.get('D_C'), header.comments['D_C']) == (inp.get('d_c'), '[Mpc]')
    assert header.get('NPIX') == inp.get('data').shape[0]
    assert header.get('NENE') == inp.get('data').shape[2]
    assert (header.get('ANG_PIX'), header.comments['ANG_PIX']) == (inp.get('pixel_size'), '[arcmin]')
    assert (header.get('ANG_MAP'), header.comments['ANG_MAP']) == (inp.get('size'), '[deg]')
    assert (header.get('RA_C'), header.comments['RA_C']) == (center_map[0], '[deg]')
    assert (header.get('DEC_C'), header.comments['DEC_C']) == (center_map[1], '[deg]')
    assert header.get('SMOOTH') == inp.get('smoothing')
    assert header.get('VPEC') == inp.get('velocities')


def test_isothermal_spectrum(inp=spec_cube, out=simput_file):
    """
    All the spectra contained in the Extension 2 of the SIMPUT file must be isothermal with T equal to the value
    indicated by the spec_cube, with arbitrary normalization.
    :param inp: (dict) spectral cube structure
    :param out: (str) output SIMPUT file
    """

    # Creating SIMPUT file
    if os.path.isfile(out):
        os.remove(out)
    cube2simputfile(inp, out)
    hdulist = fits.open(simput_file)
    os.remove(out)
    header0 = hdulist[0].header

    # Reading the spectral table and the temperature from the header of the SIMPUT file just created
    sptable = read_spectable(header0.get('SP_FILE'))
    temp = header0.get('ISOTHERM') / keV2K  # [keV]

    # Getting energy and (normalized spectrum) from table and from
    z = spec_cube.get('z_cos')
    energy_reference = spec_cube.get('energy')  # [keV]
    spectrum_reference = calc_spec(sptable, z, temp, no_z_interp=True)
    spectrum_reference /= spectrum_reference.mean()  # normalize to mean = 1

    # Checking energy from the SIMPUT file
    for energy in hdulist[2].data['ENERGY']:
        for val, val_reference in zip(energy, energy_reference):
            assert val  == pytest.approx(val_reference, rel=1.e-6)

    # Checking spectra from the SIMPUT file
    for spectrum in hdulist[2].data['FLUXDENSITY']:
        spectrum_norm = spectrum / spectrum.mean()  # normalize to mean = 1
        for val, val_reference in zip(spectrum_norm, spectrum_reference):
            assert val == pytest.approx(val_reference, rel=1.e-6)
