import numpy as np
from astropy.io import fits
from src.pkg.specutils import absorption
import copy as cp
from src.pkg.gadgetutils import phys_const
import os
import json

# Initialization of global variables
instruments_config_file = os.path.join(os.path.dirname(__file__), 'sixte_instruments.json')
with open(instruments_config_file) as file:
    json_data = json.load(file)

instruments = {}
for instr in json_data:
    instruments[instr['name'].lower()] = {
        'command': instr.get('command'),
        'xml': os.environ.get('SIXTE_INSTRUMENTS') + '/' + instr['subdir'] + '/' + instr['xml'],
        'adv_xml': os.environ.get('SIXTE_INSTRUMENTS') + '/' + instr['subdir'] + '/' + instr['adv_xml']
    }

del file, json_data, instr

def set_simput_src_cat_header(header: fits.header):
    """
    Adds the SIMPUT keywords to the header of Extension 1
    :param header: (fits.header) Input FITS header
    :return: The original header with added/modified keywords
    """
    header.set('EXTNAME', 'SRC_CAT ', 'name of this binary table extension')
    header.set('HDUCLASS', 'HEASARC/SIMPUT')
    header.set('HDUCLAS1', 'SRC_CAT ')
    header.set('HDUVERS', '1.1.0   ')
    header.set('RADESYS', 'FK5     ')
    header.set('EQUINOX', 2000.)
    return header


def set_simput_spectrum_header(header: fits.header):
    """
    Adds the SIMPUT keywords to the header of Extension 2
    :param header: (fits.header) Input FITS header
    :return: The original header with added/modified keywords
    """
    header.set('EXTNAME', 'SPECTRUM ', 'name of this binary table extension')
    header.set('HDUCLASS', 'HEASARC/SIMPUT')
    header.set('HDUCLAS1', 'SPECTRUM ')
    header.set('HDUVERS', '1.1.0   ')
    header.set('EXTVER', 1)
    return header


def set_simput_headers(hdulist: fits.HDUList):
    """
    Adds the SIMPUT keywords to the headers of the FITS HDU list
    :param hdulist: (fits.HDUList) the HDU list
    :return: the HDU list with added keywords
    """
    set_simput_src_cat_header(hdulist[1].header)
    set_simput_spectrum_header(hdulist[2].header)
    return hdulist


def cube2simputfile(spcube_input, simput_file: str, tag='', pos=(0., 0.), npix=None, fluxsc=1., addto=None,
                    appendto=None, nh=None, preserve_input=True):
    """
    :param spcube_input: spectral cube structure, i.e. output of make_speccube
    :param simput_file: (str) SIMPUT output file
    :param tag: (str) prefix of the source name, default None
    :param pos: (float 2) sky position in RA, DEC [deg]
    :param npix: number of pixels per side of the output spectral map (TODO: if different from input it is rebinned)
    :param fluxsc: (float) flux scaling to be applied to the cube
    :param addto: TODO
    :param appendto: TODO
    :param nh: (float) hydrogen column density [10^22 cm^-2], if included it changes the value of the input object
        converting it to the desired one, default: None (i.e. maintains the original value of nh)
    :param preserve_input: (bool) if set to true the 'data' key is deleted from the spcube_struct, default False
    :return: None
    """

    spcube_struct = cp.deepcopy(spcube_input) if preserve_input else spcube_input

    if nh is not None:
        spcube_struct = absorption.convert_nh(spcube_struct, nh, preserve_input=False)

    spcube = spcube_struct.get('data')  # [counts s^-1 cm^-2 arcmin^-2 keV^-1] or [keV s^-1 cm^-2 arcmin^-2 keV^-1]
    npix0 = spcube.shape[0]
    nene = spcube.shape[2]
    energy = spcube_struct.get('energy')  # [keV]
    d_ene = spcube_struct.get('energy')  # [keV]
    size = spcube_struct.get('size')  # [deg]
    d_area = spcube_struct.get('pixel_size') ** 2  # [arcmin^2]
    spcube *= d_area  # [counts s^-1 cm^-2 keV^-1] or [keV s^-1 cm^-2 keV^-1]

    # Defining RA-DEC position
    try:
        ra0, dec0 = float(pos[0]), float(pos[1])
    except BaseException:
        print("Invalid center: ", pos, "Must be a 2d number vector")
        raise ValueError

    # Correcting energy to counts, if necessary
    if spcube_struct.get('flag_ene'):
        for iene in range(0, nene):
            spcube[:, :, iene] /= energy[iene]  # [counts keV^-1 s^-1 cm^-2]

    # TODO: Implement addto here

    if npix is None:
        npix = npix0
    else:
        npix = npix0

    # Rebinning if necessary TODO: implement something close to the CONGRID function in IDL and remove previous else

    # Creating coordinate arrays
    ra_pix = ra0 + np.linspace(-0.5 * size, 0.5 * size, npix, endpoint=False) + size / (2 * npix)  # [deg]
    dec_pix = dec0 + np.linspace(-0.5 * size, 0.5 * size, npix, endpoint=False) + size / (2 * npix)  # [deg]

    # Creating the SIMPUT file
    hdulist = fits.HDUList()

    # Primary (empty)
    primary = [0]
    hdulist.append(fits.PrimaryHDU(primary))

    # Creating data for Extension 1 (sources) and 2 (spectrum)
    name = []
    ra = []
    dec = []
    flux = []
    spectrum = []
    fluxdensity = []
    energy_out = []

    name_prefix = tag + '-' if tag else ''
    for ipix in range(0, npix):
        for jpix in range(0, npix):
            source_flux = np.sum(spcube[ipix, jpix, :] * energy * d_ene) * phys_const.keV2erg  # [erg s^-1 cm^-2]
            if source_flux > 0.:  # cleaning pixels that have zero flux
                row_name = name_prefix + '(' + str(ipix) + ',' + str(jpix) + ')'
                name.append(row_name)
                ra.append(ra_pix[ipix])  # [deg]
                dec.append(dec_pix[jpix])  # [deg]
                flux.append(source_flux)  # [erg s^-1 cm^-2]
                spectrum.append("[SPECTRUM,1][NAME=='" + row_name + "']")
                energy_out.append(energy)  # [keV]
                fluxdensity.append(spcube[ipix, jpix, :])  # [counts keV^-1 s^-1 cm^-2]

    nsource = len(name)
    src_id = np.arange(1, nsource + 1)
    imgrota = imgscal = np.full(nsource, 0.)
    e_min = np.full(nsource, energy[0] - 0.5 * spcube_struct.get('energy_interval')[0])  # [keV]
    e_max = np.full(nsource, energy[-1] + 0.5 * spcube_struct.get('energy_interval')[-1])  # [keV]
    image = timing = np.full(nsource, 'NULL                            ')

    # Extension 1 (sources)
    src_cat_columns = [fits.Column(name='SRC_ID', format='J', array=src_id),
                       fits.Column(name='SRC_NAME', format='32A', array=name),
                       fits.Column(name='RA', format='E', array=ra, unit='deg'),
                       fits.Column(name='DEC', format='E', array=dec, unit='deg'),
                       fits.Column(name='IMGROTA', format='E', array=imgrota, unit='deg'),
                       fits.Column(name='IMGSCAL', format='E', array=imgscal),
                       fits.Column(name='E_MIN', format='E', array=e_min, unit='keV'),
                       fits.Column(name='E_MAX', format='E', array=e_max, unit='keV'),
                       fits.Column(name='FLUX', format='E', array=flux, unit='erg/s/cm**2'),
                       fits.Column(name='SPECTRUM', format='64A', array=spectrum),
                       fits.Column(name='IMAGE', format='32A', array=image),
                       fits.Column(name='TIMING', format='32A', array=timing)]
    src_cat = fits.BinTableHDU.from_columns(fits.ColDefs(src_cat_columns))
    hdulist.append(src_cat)

    # Extension 2 (spectrum)
    spectrum_columns = [fits.Column(name='NAME', format='32A', array=name),
                        fits.Column(name='ENERGY', format=str(nene) + 'E', array=energy_out, unit='keV'),
                        fits.Column(name='FLUXDENSITY', format=str(nene) + 'E', array=energy_out,
                                    unit='photons/s/cm**2/keV')]
    spectrum_ext = fits.BinTableHDU.from_columns(fits.ColDefs(spectrum_columns))
    hdulist.append(spectrum_ext)

    # Setting headers
    set_simput_headers(hdulist)
    hdulist[0].header.set('SIM_FILE', spcube_struct.get('simulation_file'))
    hdulist[0].header.set('SP_FILE', spcube_struct.get('spectral_table'))
    hdulist[0].header.set('PROJ', spcube_struct.get('proj'))
    hdulist[0].header.set('Z_COS', spcube_struct.get('z_cos'))
    hdulist[0].header.set('D_C', spcube_struct.get('d_c'), '[Mpc]')
    hdulist[0].header.set('NPIX', npix)
    hdulist[0].header.set('NENE', nene)
    hdulist[0].header.set('ANG_PIX', spcube_struct.get('pixel_size'), '[arcmin]')
    hdulist[0].header.set('ANG_MAP', spcube_struct.get('size'), '[deg]')
    hdulist[0].header.set('RA_C', ra0, '[deg]')
    hdulist[0].header.set('DEC_C', dec0, '[deg]')
    if fluxsc != 1.:
        hdulist[0].header.set('FLUXSC', fluxsc)
    if spcube_struct.get('tcut'):
        hdulist[0].header.set('T_CUT', spcube_struct.get('tcut'))
    if spcube_struct.get('isothermal'):
        hdulist[0].header.set('ISOTHERM', spcube_struct.get('isothermal'))
    hdulist[0].header.set('SMOOTH', spcube_struct.get('smoothing'))
    hdulist[0].header.set('VPEC', spcube_struct.get('velocities'))
    if spcube_struct.get('zrange'):
        hdulist[0].header.set('Z_RANGE', spcube_struct.get('zrange'))
    if spcube_struct.get('nsample'):
        hdulist[0].header.set('NSAMPLE', spcube_struct.get('nsample'))
    if nh is not None:
        hdulist[0].header.set('NH', nh, '[10^22 cm^-2]')
    else:
        if 'nh' in spcube_struct:
            hdulist[0].header.set('NH', spcube_struct.get('nh'), '[10^22 cm^-2]')

    hdulist.writeto(simput_file, overwrite=True)
    return None


def create_eventlist(simputfile: str, instrument: str, exposure: float, evtfile: str, pointing=None, xmlfile=None,
                     advxml=None, background=True, overwrite=True, verbosity=None, logfile=None, no_exec=False):
    """
    Creates a simulated X-ray event-list by running the SIXTE simulator (see the manuale from the SIXTE webpage
    http://www.sternwarte.uni-erlangen.de/~sixte/data/simulator_manual.pdf).
    :param simputfile: (str) Simput file
    :param instrument: (str) Instrument
    :param exposure: (float) Exposure [s]
    :param evtfile: (str) Output FITS file containing the simulation results
    :param pointing: (float 2) RA, DEC coordinates of the telescope pointing [deg], default None (uses the RA, DEC
        keywords of the simputfile header)
    :param xmlfile: (str) XML file for the telescope configuration
    :param advxml: (str) Advanced XML configuration file
    :param background: (bool) If set to True includes the instrumental background, default True
    :param overwrite: (bool) If set overwrites previous output file (evtfile) if exists, default True
    :param verbosity: (int) Verbosity level, with 0 being the lowest (see SIXTE manual 'chatter') and 7 highest,
    default None, i.e. SIXTE default (4)
    :param logfile: (str) if set the output is not written on screen but saved in the file
    :param no_exec: (bool) If set to True no simulation is run but the SIXTE command is printed out instead
    :return: None
    """

    if instrument.lower() in instruments:
        sixte_command = instruments[instrument]['command']
        xmlfile_ = xmlfile if xmlfile else instruments[instrument]['xml']
        advxml_ = advxml if advxml else instruments[instrument]['adv_xml']
    else:
        print("ERROR in create_eventlist. Invalid instrument", instrument,
              ": must be one of " + str(list(instruments.keys())) + ". To configure other instruments modify the "
              "instruments configuration file: " + instruments_config_file)
        raise ValueError

    if pointing is None:
        ra = fits.open(simputfile)[0].header.get('RA_C')
        dec = fits.open(simputfile)[0].header.get('DEC_C')
    else:
        try:
            ra, dec = float(pointing[0]), float(pointing[1])
        except BaseException:
            print("ERROR in create_eventlist. Invalid pointing: ", pointing, "Must be a 2d number vector")
            raise ValueError

    background_ = 'yes' if background else 'no'
    clobber_ = 'yes' if overwrite else 'no'

    command = sixte_command + ' XMLFile=' + xmlfile_ + ' AdvXml=' + advxml_ + ' Simput=' + simputfile + ' Exposure=' + \
              str(exposure) + ' RA=' + str(ra) + ' Dec=' + str(dec) + ' background=' + background_ + ' evtfile=' + \
              evtfile + ' clobber=' + clobber_

    if type(verbosity) is int:
        if verbosity < 0:
            command += ' chatter=0'
        else:
            command += ' chatter=' + str(verbosity)

    if type(logfile) is str and logfile != '':
        command += ' > ' + logfile

    print(command) if no_exec else os.system(command)
    return None
