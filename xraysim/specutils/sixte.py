import copy as cp
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from xraysim.gadgetutils import phys_const
from xraysim.specutils import absorption

# Initialization of global variables
instruments_config_file = os.path.join(os.path.dirname(__file__), 'sixte_instruments.json')
with open(instruments_config_file) as file:
    json_data = json.load(file)

sixte_instruments_dir = os.environ.get('SIXTE') + '/share/sixte/instruments'

instruments = {}
for instr in json_data:
    instruments[instr['name'].lower()] = {
        'command': instr.get('command'),
        'xml': sixte_instruments_dir + '/' + instr['subdir'] + '/' + instr['xml'],
        'adv_xml': sixte_instruments_dir + '/' + instr['subdir'] + '/' + instr['adv_xml']
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


def cube2simputfile(spcube: dict, simput_file: str, tag='', pos=(0., 0.), npix=None, fluxsc=1., addto=None,
                    appendto=None, nh=None, preserve_input=True, overwrite=True):
    """
    :param spcube: (dict) spectral cube structure, i.e. output of make_speccube
    :param simput_file: (str) SIMPUT output file
    :param tag: (str) prefix of the source name, default None
    :param pos: (float 2) sky position in RA, DEC [deg]
    :param npix: number of pixels per side of the output spectral map (TODO: if different from input it is rebinned)
    :param fluxsc: (float) flux scaling to be applied to the cube
    :param addto: TODO
    :param appendto: TODO
    :param nh: (float) hydrogen column density [10^22 cm^-2], if included it changes the value of the input object
        converting it to the desired one, default: None (i.e. maintains the original value of nh)
    :param preserve_input: (bool) If set to true the 'data' key in spcube_struct is left untouched and duplicated in
        memory. If there is no need to preserve it, setting to False will save memory. Default: True.
    :param overwrite: (bool) If set to true the file is overwritten. Default: True.
    :return: System output of the writing operation (usually None)
    """

    spcube_struct = cp.deepcopy(spcube) if preserve_input else spcube

    if nh is not None:
        spcube_struct = absorption.convert_nh(spcube_struct, nh, preserve_input=False)

    spcube = spcube_struct.get('data')  # [photons s^-1 cm^-2 arcmin^-2 keV^-1] or [keV s^-1 cm^-2 arcmin^-2 keV^-1]
    npix0 = spcube.shape[0]
    nene = spcube.shape[2]
    energy = spcube_struct.get('energy')  # [keV]
    d_ene = spcube_struct.get('energy_interval')  # [keV]
    size = spcube_struct.get('size')  # [deg]
    d_area = spcube_struct.get('pixel_size') ** 2  # [arcmin^2]
    spcube *= d_area  # [photons s^-1 cm^-2 keV^-1] or [keV s^-1 cm^-2 keV^-1]

    # Defining RA-DEC position
    try:
        ra0, dec0 = float(pos[0]), float(pos[1])
    except BaseException:
        raise ValueError("Invalid center: ", pos, "Must be a 2d number vector")

    # Correcting energy to photons, if necessary
    if spcube_struct.get('flag_ene'):
        for iene in range(0, nene):
            spcube[:, :, iene] /= energy[iene]  # [photons keV^-1 s^-1 cm^-2]

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
                fluxdensity.append(spcube[ipix, jpix, :])  # [photons keV^-1 s^-1 cm^-2]

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
                        fits.Column(name='FLUXDENSITY', format=str(nene) + 'E', array=fluxdensity,
                                    unit='photons/s/cm**2/keV')]
    spectrum_ext = fits.BinTableHDU.from_columns(fits.ColDefs(spectrum_columns))
    hdulist.append(spectrum_ext)

    # Setting headers
    set_simput_headers(hdulist)
    hdulist[0].header.set('INFO', 'Created with Python xraysim and astropy')
    if 'simulation_type' in spcube_struct:
        hdulist[0].header.set('SIM_TYPE', spcube_struct.get('simulation_type'))
    hdulist[0].header.set('SIM_FILE', spcube_struct.get('simulation_file'))
    hdulist[0].header.set('SP_FILE', spcube_struct.get('spectral_table'))
    hdulist[0].header.set('PROJ', spcube_struct.get('proj'))
    hdulist[0].header.set('X_MIN', spcube_struct.get('xrange')[0])
    hdulist[0].header.set('X_MAX', spcube_struct.get('xrange')[1])
    hdulist[0].header.set('Y_MIN', spcube_struct.get('yrange')[0])
    hdulist[0].header.set('Y_MAX', spcube_struct.get('yrange')[1])
    if spcube_struct.get('zrange'):
        hdulist[0].header.set('Z_MIN', spcube_struct.get('zrange')[0])
        hdulist[0].header.set('Z_MAX', spcube_struct.get('zrange')[1])
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
        hdulist[0].header.set('ISO_T', spcube_struct.get('isothermal'))
    hdulist[0].header.set('SMOOTH', spcube_struct.get('smoothing'))
    hdulist[0].header.set('VPEC', spcube_struct.get('velocities'))
    if spcube_struct.get('nsample'):
        hdulist[0].header.set('NSAMPLE', spcube_struct.get('nsample'))
    if nh is not None:
        hdulist[0].header.set('NH', nh, '[10^22 cm^-2]')
    else:
        if 'nh' in spcube_struct:
            hdulist[0].header.set('NH', spcube_struct.get('nh'), '[10^22 cm^-2]')

    # Writing FITS file (returns None)
    return hdulist.writeto(simput_file, overwrite=overwrite)


def inherit_keywords(input_file: str, output_file: str, file_type=None) -> int:
    """
    Writes a list of keywords (if present) from the Primary header of the input file into the Primary header of the
    output file.
    :param input_file: (str) Input FITS file.
    :param output_file: (str) Output FITS file that will be modified.
    :param file_type: (str) Output file type: can be either "evt", "evtlist" or "pha". Default None, i.e. derived
    from file extension.
    :return: (int) System output of the writing operation
    """
    keyword_list = ['INFO', 'SIM_TYPE', 'SIM_FILE', 'SP_FILE', 'SIMPUT_F', 'PARENT_F', 'PROJ', 'X_MIN', 'X_MAX',
                    'Y_MIN', 'Y_MAX', 'Z_MIN', 'Z_MAX', 'Z_COS', 'D_C', 'NPIX', 'NENE', 'ANG_PIX', 'ANG_MAP', 'ISO_T',
                    'SMOOTH', 'VPEC', 'NSAMPLE', 'NH', 'RA_C', 'DEC_C', 'FLUXSC', 'T_CUT']
    header_inp = fits.getheader(input_file, 0)
    hdulist = fits.open(output_file)

    if type(file_type) == str:
        dummy = file_type.lower().strip()
        file_type_ = dummy if dummy in ["evt", "evtlist", "pha"] else output_file.split(".")[-1]
    else:
        file_type_ = output_file.split(".")[-1]

    if file_type_ in ["evt", "evtlist"]:
        hdulist[0].header.set("SIMPUT_F", input_file)
    elif file_type_ == "pha":
        hdulist[0].header.set("EVT_FILE", input_file)
    else:
        hdulist[0].header.set("PARENT_F", input_file, "UNKNOWN_TYPE")

    for key in keyword_list:
        if key in header_inp:
            hdulist[0].header.set(key, header_inp.get(key), header_inp.comments[key])

    return hdulist.writeto(output_file, overwrite=True)


def create_eventlist(simputfile: str, instrument: str, exposure: float, evtfile: str, pointing=None, xmlfile=None,
                     advxml=None, background=True, seed=None, overwrite=True, verbosity=None, logfile=None,
                     no_exec=False):
    """
    Creates a simulated X-ray event-list by running the SIXTE simulator (see the manuale from the SIXTE webpage
    http://www.sternwarte.uni-erlangen.de/~sixte/data/simulator_manual.pdf).
    :param simputfile: (str) Simput file
    :param instrument: (str) Instrument
    :param exposure: (float) Exposure [s]
    :param evtfile: (str) Output FITS file containing the simulation results
    :param pointing: (float 2) RA, DEC coordinates of the telescope pointing [deg]. Default: None. i.e. uses the RA, DEC
        keywords of the simputfile header
    :param xmlfile: (str) XML file for the telescope configuration
    :param advxml: (str) Advanced XML configuration file
    :param background: (bool) If set to True includes the instrumental background, default True
    :param seed: (int) Random seed, default None
    :param overwrite: (bool) If set overwrites previous output file (evtfile) if exists, default True
    :param verbosity: (int) Verbosity level, with 0 being the lowest (see SIXTE manual 'chatter') and 7 highest.
    Default: None, i.e. SIXTE default (4).
    :param logfile: (str) if set the output is not written on screen but saved in the file
    :param no_exec: (bool) If set to True no simulation is run but the SIXTE command is printed out instead. Default:
        False.
    :return: System output of SIXTE command (or string containing the command if no_exec is set to True)
    """

    if instrument.lower() in instruments:
        sixte_command = instruments[instrument]['command']
        xmlfile_ = xmlfile if xmlfile else instruments[instrument]['xml']
        advxml_ = advxml if advxml else instruments[instrument]['adv_xml']
    else:
        raise ValueError("ERROR in create_eventlist. Invalid instrument", instrument,
                         ": must be one of " + str(list(instruments.keys())) + ". To configure other instruments modify the " +
                         "instruments configuration file: " + instruments_config_file)

    if pointing is None:
        ra = fits.open(simputfile)[0].header.get('RA_C')
        dec = fits.open(simputfile)[0].header.get('DEC_C')
    else:
        try:
            ra, dec = float(pointing[0]), float(pointing[1])
        except BaseException:
            raise ValueError("ERROR in create_eventlist. Invalid pointing: ", pointing, "Must be a 2d number vector")

    background_ = 'yes' if background else 'no'
    clobber_ = 'yes' if overwrite else 'no'

    command = sixte_command + ' XMLFile=' + xmlfile_ + ' AdvXml=' + advxml_ + ' Simput=' + simputfile + ' Exposure=' + \
              str(exposure) + ' RA=' + str(ra) + ' Dec=' + str(dec) + ' background=' + background_ + ' evtfile=' + \
              evtfile + ' clobber=' + clobber_

    if type(seed) is int:
        command += ' seed=' + str(seed)

    if type(verbosity) is int:
        if verbosity < 0:
            command += ' chatter=0'
        else:
            command += ' chatter=' + str(verbosity)

    if type(logfile) is str and logfile != '':
        command += ' > ' + logfile + ' 2>&1'

    if no_exec:
        return command
    else:
        sys_out = os.system(command)
        if sys_out == 0:
            inherit_keywords(simputfile, evtfile, file_type="evt")
        return sys_out


def get_fluxmap(simputfile: str):
    hdul = fits.open(simputfile)
    npix = hdul[0].header.get('NPIX')
    ra = np.linspace(hdul[1].data['RA'].min(), hdul[1].data['RA'].max(), npix)  # [deg]
    dec = np.linspace(hdul[1].data['DEC'].min(), hdul[1].data['DEC'].max(), npix)  # [deg]
    ang_pix = hdul[0].header.get('ANG_PIX') / 60.  # [deg]
    if ('X_MIN' in hdul[0].header and 'X_MAX' in hdul[0].header):
        x_min, x_max = hdul[0].header['X_MIN'], hdul[0].header['X_MAX']  # [h^-1 kpc]
        l_pix = (x_max - x_min) / npix  # [h^-1 kpc]
        x = np.linspace(x_min + 0.5 * l_pix, x_max - 0.5 * l_pix, npix)
        y = np.linspace(hdul[0].header['Y_MIN'] + 0.5 * l_pix, hdul[0].header['Y_MAX'] - 0.5 * l_pix, npix)
    else:
        x = np.arange(npix)
        y = np.arange(npix)
        l_pix = None

    flux_map = np.zeros([npix, npix], dtype=np.float32)
    for row in hdul[1].data:
        src_name = row['SRC_NAME']
        istr, jstr = src_name[src_name.find('(') + 1: src_name.find(')')].split(',')
        flux_map[int(istr), int(jstr)] = row['FLUX']

    return {'data': flux_map, 'ra': ra, 'dec': dec, 'ang_pix': ang_pix, 'l_pix': l_pix, 'x': x, 'y': y}


def show_fluxmap(inp, gadget_units=False):
    if type(inp) is str:
        flux_map = get_fluxmap(inp)
    elif type(inp) is dict:
        flux_map = inp
    else:
        raise ValueError("ERROR in show_fluxmap. Invalid input type, must be either str or dict")

    if gadget_units and flux_map['l_pix'] is not None:
        extent = [
            flux_map['x'][0] - 0.5 * flux_map['l_pix'],
            flux_map['x'][-1] + 0.5 * flux_map['l_pix'],
            flux_map['y'][0] - 0.5 * flux_map['l_pix'],
            flux_map['y'][-1] + 0.5 * flux_map['l_pix']
        ]
    else:
        extent = [
            flux_map['ra'][0] - 0.5 * flux_map['ang_pix'],
            flux_map['ra'][-1] + 0.5 * flux_map['ang_pix'],
            flux_map['dec'][0] - 0.5 * flux_map['ang_pix'],
            flux_map['dec'][-1] + 0.5 * flux_map['ang_pix']
        ]
    plt.imshow(flux_map.get('data').transpose(), origin='lower', extent=extent)
    plt.show()
    return None


def get_rsppath(evtfile: str):
    history = fits.open(evtfile)[0].header['HISTORY']
    found = False
    iline = -1
    while not found and iline < len(history) - 1:
        iline += 1
        found = 'XMLFile = ' in history[iline]

    if not found:
        rsppath = None
    else:
        line_split = history[iline].split(' ')
        if len(line_split) == 4:
            p, filename = line_split[0], line_split[3]
        else:
            return None

        addon = True
        while addon and iline < len(history) - 1:
            iline += 1
            line_split = history[iline].split(' ')
            addon = (line_split[0] == p)
            if addon and len(line_split) >= 2:
                filename += line_split[1]
        rsppath = filename[0:filename.rindex('/')]

    return rsppath


def make_pha(evtfile: str, phafile: str, rsppath=None, pixid=None, grading=1, logfile=None, overwrite=True,
             no_exec=False):
    """ Creates a .pha file containing the spectrum extracted from an event file using the SIXTE makespec command
    :param evtfile: (str) Event file
    :param phafile: (str) Output file
    :param rsppath: (str) Path to the .rmf and .arf files
    :param pixid: (int or int list) Pixel id of photons to be included in the spectrum (default, None, i.e. all pixels)
    :param grading: (int or int list) Grading of photons to be included in the spectrum (default, 1)
    :param logfile: (str) If set the output is not written on screen but saved in the file
    :param overwrite: (bool) If set overwrites previous output file (phafile) if exists, default True
    :param no_exec: (bool) If set to True no simulation is run but the SIXTE command is printed out instead
    :return: System output of SIXTE makespec command (or string containing the command if no_exec is set to True)
    """

    # Defining filter list to be used (if not empty) with the EventFilter keyword of makespec
    filter_list = []

    # Grading
    error_msg_grading = "ERROR in make_pha. Grading values must be integer, iterable of integers or None."
    if isinstance(grading, type(None)):
        pass
    elif isinstance(grading, int):
        filter_list.append("GRADING==" + str(grading))
    elif isinstance(grading, tuple) or isinstance(grading, list):
        if all(type(item) is int for item in grading):
            tag_grading = " '(GRADING==" + str(grading[0])
            for item in grading[1:]:
                tag_grading += " || GRADING==" + str(item)
            tag_grading += ")'"
            filter_list.append(tag_grading)
        else:
            raise ValueError(error_msg_grading)
    else:
        raise ValueError(error_msg_grading)

    # Pixel Id
    error_msg_pixid = "ERROR in make_pha. Pixid values must be integer, iterable of integers or None."
    if isinstance(pixid, type(None)):
        pass
    elif isinstance(pixid, int):
        filter_list.append("PIXID==" + str(pixid))
    elif isinstance(pixid, tuple) or isinstance(pixid, list):
        if all(type(item) is int for item in pixid):
            tag_pixid = " '(PIXID==" + str(pixid[0])
            for item in pixid[1:]:
                tag_pixid += " || PIXID==" + str(item)
            tag_pixid += ")'"
            filter_list.append(tag_pixid)
        else:
            raise ValueError(error_msg_pixid)
    else:
        raise ValueError(error_msg_pixid)

    # If rsppath is not provided I try to recover it from the evtfile
    rsppath_ = get_rsppath(evtfile) if rsppath is None else rsppath
    tag_rsppath = "" if rsppath_ is None else " RSPPath=" + rsppath_

    clobber_ = 'yes' if overwrite else 'no'

    command = "makespec EvtFile=" + evtfile + " Spectrum=" + phafile + tag_rsppath + ' clobber=' + clobber_

    # Defining a tag to be used (if not empty) with the EventFilter keyword of makespec
    tag_filter = ' && '.join(filter_list)
    if tag_filter != '':
        command += ' EventFilter="' + tag_filter + '"'

    if type(logfile) is str and logfile != '':
        command += ' > ' + logfile + ' 2>&1'

    if no_exec:
        return command
    else:
        sys_out = os.system(command)
        if sys_out == 0:
            inherit_keywords(evtfile, phafile, file_type="pha")
        return sys_out
