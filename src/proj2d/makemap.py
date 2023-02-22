import math as mt

import numpy as np
import pygadgetreader as pygr
from tqdm import tqdm

from .intkernel import intkernel
from .linkedlist import linkedlist2d


def makemap(filename: str, qty, npix=256, center=None, size=None, proj='z', tcut=0., sample=1, struct=False):
    """

    :param filename: (str) input file
    :param qty: (str) quantity to integrate, i.e. Int(qty*dl)
    :param npix: (int) number of map pixels per side
    :param center: (float 2) comoving coord. of the map center [h^-1 kpc], default: median point of gas particles
    :param size: (float) map comoving size [h^-1 kpc], default: encloses all gas particles
    :param proj: (str/int) direction of projection ('x', 'y', 'z' or 0, 1, 2)
    :param tcut: (float) if set defines a temperature cut below which particles are removed [K], default: 0.
    :param sample: (int), if set defines a sampling for the particles (useful to speed up), default: 1 (no sampling)
    :param struct: (bool) if set outputs a structure (dictionary) containing several info, default: False TODO: info
    :return:
    """

    intkernel_vec = np.vectorize(intkernel)

    # Reading header variables
    redshift = pygr.readhead(filename, 'redshift')
    h = pygr.readhead(filename, 'h')
    ngas = pygr.readhead(filename, 'gascount')
    f_cooling = pygr.readhead(filename, 'f_cooling')

    # Reading positions of particles
    pos = pygr.readsnap(filename, 'pos', 'gas', units=0)  # [h^-1 kpc] comoving
    if proj == 'x' or proj == 0:
        projInd = 0
        x = pos[:, 1]
        y = pos[:, 2]
    elif proj == 'y' or proj == 1:
        projInd = 1
        x = pos[:, 2]
        y = pos[:, 0]
    elif proj == 'z' or proj == 2:
        projInd = 2
        x = pos[:, 0]
        y = pos[:, 1]
    else:
        print("Invalid projection axis: ", proj, "Choose between 'x' (or 0), 'y' (1), 'z' (2)")
        raise ValueError
    del pos

    # Defining center and map size
    hsml = pygr.readsnap(filename, 'hsml', 'gas', units=0)  # [h^-1 kpc] comoving
    if center is None:
        xmin, xmax = min(x - hsml), max(x + hsml)
        ymin, ymax = min(y - hsml), max(y + hsml)
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)
        if size is None:
            deltaX, deltaY = xmax - xmin, ymax - ymin
            if deltaX >= deltaY:
                size = deltaX
                xmap0, ymap0 = xmin, ymin - 0.5 * (deltaX - deltaY)
            else:
                size = deltaY
                xmap0, ymap0 = xmin - 0.5 * (deltaY - deltaX), ymin
        else:
            xmap0, ymap0 = xc - 0.5 * size, yc - 0.5 * size
    else:
        try:
            xc, yc = float(center[0]), float(center[1])
        except:
            print("Invalid center: ", center, "Must be a 2d number vector")
            raise ValueError

        if size is None:
            xmin, xmax = min(x - hsml), max(x + hsml)
            ymin, ymax = min(y - hsml), max(y + hsml)
            if not (xmin <= xc <= xmax and ymin <= yc <= ymax):
                print("WARNING: Map center is outside the simulation box")
            size = 2. * max(abs(xc - xmin), abs(xc - xmax), abs(yc - ymin), abs(yc - ymax))

        xmap0, ymap0 = xc - 0.5 * size, yc - 0.5 * size

    # Normalizing coordinates in pixel units (0 = left/bottom border, npix = right/top border)
    x = (x - xmap0) / size * npix
    y = (y - ymap0) / size * npix
    hsml = hsml / size * npix
    pixsize = size / npix  # comoving [h^-1 kpc]

    # Create linked list and cutting out particles
    if tcut > 0.:
        temp = pygr.readsnap(filename, 'u', 'gas', units=1)  # [K]
        valid = np.where((x + hsml > 0) & (x - hsml < npix) & (y + hsml > 0) & (y - hsml < npix) & (temp > tcut))[0]
        if qty not in ['Tmw', 'Tew', 'Tsl']:
            del temp
    else:
        valid = np.where((x + hsml > 0) & (x - hsml < npix) & (y + hsml > 0) & (y - hsml < npix))[0]
    particle_list = valid[linkedlist2d(x[valid], y[valid], npix, npix)]
    del valid

    # Calculating quantity (q) to integrate and weight (w)
    mass = pygr.readsnap(filename, 'mass', 'gas', units=0)  # [10^10 h^-1 M_Sun]
    if qty == 'rho':  # Int(rho*dl)
        q = mass / pixsize ** 2  # comoving [10^10 h M_Sun kpc^-2]
        w = np.full(ngas, 0.)  # [---]
    elif qty == 'rho2':  # Int(rho2*dl)
        q = mass * pygr.readsnap(filename, 'rho', 'gas', units=0) / pixsize ** 2  # comoving [10^20 h^3 M_Sun^2 kpc^-1]
        w = np.full(ngas, 0.)  # [---]
    elif qty in ['Tmw', 'Tew', 'Tsl']:
        if not 'temp' in locals():
            temp = pygr.readsnap(filename, 'u', 'gas', units=1)  # internal energy per unit mass [km^2 s^-2]

        if qty == 'Tmw':
            q = mass * temp  # [10^10 h^-1 M_Sun K]
            w = mass / pixsize ** 2  # comoving [10^10 h M_Sun kpc^-2]
        elif qty == 'Tew':
            rho = pygr.readsnap(filename, 'rho', 'gas', units=0)  # [10^10 h^2 M_Sun kpc^-3]
            q = mass * rho * temp  # [10^20 h M_Sun^2 kpc^-3 K]
            w = mass * rho / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5]
            del rho
        elif qty == 'Tsl':
            rho = pygr.readsnap(filename, 'rho', 'gas', units=0)  # [10^10 h^2 M_Sun kpc^-3]
            q = mass * rho * temp ** 0.25  # [10^20 h M_Sun^2 kpc^-3 K^0.25]
            w = mass * rho * temp ** (-0.75) / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 K^-0.75]
            del rho
        del mass, temp

    elif qty in ['vmw', 'vew']:
        vel = pygr.readsnap(filename, 'vel', 'gas', units=0)[:, projInd] / (1 + redshift)  # [km s^-1]
        if qty == 'vmw':
            q = mass * vel / pixsize ** 2  # [10^10 h M_Sun kpc^-2 km s^-1]
            w = mass / pixsize ** 2  # [10^10 h M_Sun kpc^-2]
        elif qty == 'vew':
            rho = pygr.readsnap(filename, 'rho', 'gas', units=0)  # [10^10 h^2 M_Sun kpc^-3]
            q = mass * rho * vel / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 km s^-1]
            w = mass * rho / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5]
            del rho
        del mass, vel
    else:
        print("Invalid mapping quantity: ", qty, "Must be one of 'rho', 'rho2', 'Tmw', 'Tew', 'Tsl', 'vmw', 'vew'")
        raise ValueError

    # Mapping
    qmap = np.full((npix, npix), 0.)
    wmap = np.full((npix, npix), 0.)

    for ipart in tqdm(particle_list[::sample]):

        i_beg = max(mt.floor(x[ipart] - hsml[ipart]), 0)
        i_end = min(mt.floor(x[ipart] + hsml[ipart]), npix - 1)
        j_beg = max(mt.floor(y[ipart] - hsml[ipart]), 0)
        j_end = min(mt.floor(y[ipart] + hsml[ipart]), npix - 1)

        if (i_end >= i_beg) & (j_end >= j_beg):
            step = 1. / hsml[ipart]  # 1 pixel-shift in units of hsml
            nx, ny = i_end - i_beg + 1, j_end - j_beg + 1
            wx, wy = np.empty(nx), np.empty(ny)

            xpix = (np.arange(i_beg, i_end + 2) - x[ipart]) * step
            intwx = intkernel_vec(xpix)
            for i in range(nx):
                wx[i] = intwx[i + 1] - intwx[i]

            ypix = (np.arange(j_beg, j_end + 2) - y[ipart]) * step
            intwy = intkernel_vec(ypix)
            for j in range(ny):
                wy[j] = intwy[j + 1] - intwy[j]

            for i in range(nx):
                imap = i_beg + i
                for j in range(ny):
                    ww = wx[i] * wy[j]
                    jmap = j_beg + j
                    qmap[imap, jmap] += q[ipart] * ww
                    wmap[imap, jmap] += w[ipart] * ww

    qmap[np.where(wmap != 0.)] /= wmap[np.where(wmap != 0.)]

    # Output
    if struct:

        units = {
            'rho': {'map': '10^10 h M_Sun kpc^-2', 'norm': '---'},
            'rho2': {'map': '10^20 h^3 M_Sun^2 kpc^-1', 'norm': '---'},
            'Tmw': {'map': 'K', 'norm': '10^10 h M_Sun kpc^-2'},
            'Tew': {'map': 'K', 'norm': '10^20 h^3 M_Sun^2 kpc^-5'},
            'Tsl': {'map': 'K', 'norm': '10^20 h^3 M_Sun^2 kpc^-5 K^-0.75'},
            'vmw': {'map': 'km s^-1', 'norm': '10^10 h M_Sun kpc^-2'},
            'vew': {'map': 'km s^-1', 'norm': '10^20 h^3 M_Sun^2 kpc^-5'}
        }

        result = {
            'map': qmap,
            'norm': wmap,
            'xrange': (xmap0, xmap0 + size),  # [h^-1 kpc] comoving
            'yrange': (ymap0, ymap0 + size),  # [h^-1 kpc] comoving
            'pixel_size': pixsize,  # [h^-1 kpc] comoving
            'units': units[qty]['map'],
            'norm_units': units[qty]['norm'],
            'coord_units': 'h^-1 kpc'
        }

        return result

    else:

        return qmap
