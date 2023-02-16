import numpy as np
import pygadgetreader as pygr

from .phys_const import k_B, m_p, Xp, Yp, mu0
from .linkedlist import linkedlist2d
from .intkernel import intkernel

def makemap(filename: str, qty, center=None, size=None, proj='z', npix=256):
    """

    :param filename: (str) input file
    :param qty: (str) quantity to integrate, i.e. Int(qty*dl)
    :param center: (float 2) comoving coord. of the map center [h^-1 kpc], default: median point of gas particles
    :param size: (float) map comoving size [h^-1 kpc], default: encloses all gas particles
    :param proj: (str/int) direction of projection ('x', 'y', 'z' or 0, 1, 2)
    :param npix: (int) number of map pixels per side
    :return:
    """

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
        xMin, xMax = min(x - hsml), max(x + hsml)
        yMin, yMax = min(y - hsml), max(y + hsml)
        xc = 0.5 * (xMin + xMax)
        yc = 0.5 * (yMin + yMax)
        if size is None:
            deltaX, deltaY = xMax - xMin, yMax - yMin
            if deltaX >= deltaY:
                size = deltaX
                x0, y0 = xMin, yMin - 0.5 * (deltaX - deltaY)
            else:
                size = deltaY
                x0, y0 = xMin - 0.5 * (deltaY - deltaX), yMin
        else:
            x0, y0 = xc - 0.5 * size, yc - 0.5 * size
    else:
        try:
            xc, yc = float(center[0]), float(center[1])
        except:
            print("Invalid center: ", center, "Must be a 2d number vector")
            raise ValueError

        if size is None:
            xMin, xMax = min(x - hsml), max(x + hsml)
            yMin, yMax = min(y - hsml), max(y + hsml)
            if not (xMin <= xc <= xMax and yMin <= yc <= yMax):
                print("WARNING: Map center is outside the simulation box")
                size = max(abs(xc - xMin), abs(xc - xMax), abs(yc - yMin), abs(yc - yMax))

        x0, y0 = xc - 0.5 * size, yc - 0.5 * size

    # Normalizing coordinates in pixel units (between 0 and nPix)
    x = (x - x0) / size * npix
    y = (y - y0) / size * npix
    hsml = hsml / size * npix
    pixSize = size / npix  # comoving [h^-1 kpc]
    pixSize_physical = pixSize / (1 + redshift)  # physical [h^-1 kpc]

    # Create linked lists
    lkdlist_first, lkdlist_next = linkedlist2d(x, y, npix, npix)

    # Calculating quantity (q) to integrate and weight (w)
    mass = pygr.readsnap(filename, 'mass', 'gas', units=0)  # [10^10 h^-1 M_Sun]
    if qty == 'rho':  # Int(rho*dl)
        q = mass  # [10^10 h^-1 M_Sun]
        w = np.full(ngas, 1.)
    elif qty == 'rho2':  # Int(rho2*dl)
        q = mass * pygr.readsnap(filename, 'rho', 'gas', units=0)  # [10^20 h M_Sun^2 kpc^-3]
        w = np.full(ngas, 1.)
    elif qty in ['Tmw', 'Tew', 'Tsl']:
        u = pygr.readsnap(filename, 'u', 'gas', units=0)  # internal energy per unit mass [km^2 s^-2]
        if f_cooling == 0:
            temp = 2. / 3. * u * (1.e5 ** 2) * mu0 * m_p / k_B  # [K] (full ionization)
        else:
            ne = pygr.readsnap(filename, 'ne', 'gas', units=0)
            temp = 2. / 3. * u * (1.e5 ** 2) / ((1. + ne) * Xp + 0.25 * Yp) * m_p / k_B  # [K]

        if qty == 'Tmw':
            q = mass * temp  # [10^10 h^-1 M_Sun K]
            w = mass  # [10^10 h^-1 M_Sun]
        elif qty == 'Tew':
            rho = pygr.readsnap(filename, 'rho', 'gas', units=0)  # [10^10 h^2 M_Sun kpc^-3]
            q = mass * rho * temp  # [10^20 h M_Sun^2 kpc^-3 K]
            w = mass * rho  # [10^20 h M_Sun^2 kpc^-3]
            del rho
        elif qty == 'Tsl':
            rho = pygr.readsnap(filename, 'rho', 'gas', units=0)  # [10^10 h^2 M_Sun kpc^-3]
            q = mass * rho * temp ** 0.25  # [10^20 h M_Sun^2 kpc^-3 K^0.25]
            w = mass * rho * temp ** (-0.75)  # [10^20 h M_Sun^2 kpc^-3 K^-0.75]
            del rho
        del mass, temp

    elif qty in ['vmw', 'vew']:
        vel = pygr.readsnap(filename, 'vel', 'gas', units=0)[:, projInd] / (1 + redshift)  # [km s^-1]
        if qty == 'vmw':
            q = mass * vel  # [10^10 h^-1 M_Sun km s^-1]
            w = mass  # [10^10 h^-1 M_Sun]
        elif qty == 'vew':
            rho = pygr.readsnap(filename, 'rho', 'gas', units=0)  # [10^10 h^2 M_Sun kpc^-3]
            q = mass * rho * vel  # [10^20 h M_Sun^2 kpc^-3 km s^-1]
            w = mass * rho  # [10^20 h M_Sun^2 kpc^-3]
            del rho
        del mass, vel
    else:
        print("Invalid mapping quantity: ", qty, "Must be one of 'rho', 'rho2', 'Tmw', 'Tew', 'Tsl', 'vmw', 'vew'")
        raise ValueError

    # Mapping
    qmap = np.full((npix, npix), 0.)
    wmap = np.full((npix, npix), 0.)

    ipart = lkdlist_first
    while ipart != -1:

        step = 1. / hsml[ipart]  # 1 pixel-shift in units of hsml

        i_beg = max(int(np.floor(x[ipart] - hsml[ipart])), 0)
        i_end = min(int(np.ceil(x[ipart] + hsml[ipart])), npix - 1)
        j_beg = max(int(np.floor(y[ipart] - hsml[ipart])), 0)
        j_end = min(int(np.ceil(y[ipart] + hsml[ipart])), npix - 1)
        x0 = (i_beg - x[ipart]) * step
        wkx0 = intkernel(x0)
        for imap in range(i_beg, i_end):
            x1 = x0 + step
            wkx1 = intkernel(x1)
            wkx = wkx1-wkx0
            y0 = (j_beg - y[ipart]) * step
            wky0 = intkernel(y0)
            for jmap in range(j_beg, j_end):
                y1 = y0 + step
                wky1 = intkernel(y1)
                wk = (wky1-wky0) * wkx
                qmap[imap, jmap] += q[ipart] * wk
                wmap[imap, jmap] += w[ipart] * wk
                y0, wky0 = y1, wky1
            x0, wkx0 = x1, wkx1

        ipart = lkdlist_next[ipart]

    qmap[np.where(wmap != 0.)] /= wmap[np.where(wmap != 0.)]
    return qmap, wmap
