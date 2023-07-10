from src.pkg.specutils.sixte import cube2simputfile
from src.pkg.sphprojection.mapping import make_speccube
from astropy.io import fits
import os

infile = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/snap_128'
spfile = os.environ.get('XRAYSIM') + '/tests/data/test_emission_table.fits'
npix, size, redshift, center, proj, flag_ene, nsample = 10, 1., 0.1, [500e3, 500e3], 'z', False, 10000
nene = fits.open(spfile)[0].header.get('NENE')
simput_file = os.environ.get('XRAYSIM') + '/tests/data/file_created_for_test.simput'

spec_cube = make_speccube(infile, spfile, size=size, npix=npix, redshift=0.1, center=center, proj=proj, nsample=nsample)


def test_file_created(inp=spec_cube, out=simput_file):
    """
    The output SIMPUT file must be correctly created
    :param inp: spectral cube structure
    :param out: (str) output SIMPUT file
    :return: None
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
    :param inp: spectral cube structure
    :param out: (str) output SIMPUT file
    :return: None
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
