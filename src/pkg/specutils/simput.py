from astropy.io import fits

def set_simput_src_cat_header(header: fits.header):
    """
    Adds the SIMPUT keywords to the header of Extension 1
    :param header: (fits.header) the FITS header
    :return: the header with added keywords
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
    :param header: (fits.header) the FITS header
    :return: the header with added keywords
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
