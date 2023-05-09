import math as mt

import astropy as apy
import numpy as np
import pygadgetreader as pygr
from src.pkg.gadgetutils.readspecial import readtemperature
from src.pkg.gadgetutils import convert, phys_const
from sphprojection.kernel import intkernel, kernel_weight_2d
from sphprojection.linkedlist import linkedlist2d
from tqdm import tqdm
from src.pkg.specutils import tables

intkernel_vec = np.vectorize(intkernel)


def get_proj_index(proj):
    if proj == 'x' or proj == 0:
        return 0
    elif proj == 'y' or proj == 1:
        return 1
    elif proj == 'z' or proj == 2:
        return 2
    else:
        print("Invalid projection axis: ", proj, "Choose between 'x' (or 0), 'y' (1), 'z' (2)")
        raise ValueError


def get_map_coord(simfile, proj_index, z=False):
    if proj_index == 0:
        index_list = [1, 2, 0]
    elif proj_index == 1:
        index_list = [2, 0, 1]
    else:
        index_list = [0, 1, 2]

    pos = pygr.readsnap(simfile, 'pos', 'gas', units=0, suppress=1)  # [h^-1 kpc] comoving
    x = pos[:, index_list[0]]
    y = pos[:, index_list[1]]

    if z:
        return x, y, pos[:, index_list[1]]
    else:
        return x, y


def make_map(simfile: str, quantity, npix=256, center=None, size=None, proj='z', zrange=None, tcut=0., nsample=None,
             struct=False, nosmooth=False, progress=False):
    """
    :param simfile: (str) simulation file (Gadget)
    :param quantity: (str) physical quantity to map (one of rho, rho2, Tmw, Tew, Tsl, vmw, vew, wmw, wew)
    :param npix: (int) number of map pixels per side
    :param center: (float 2) comoving coord. of the map center [h^-1 kpc], default: median point of gas particles
    :param size: (float) map comoving size [h^-1 kpc], default: encloses all gas particles
    :param proj: (str/int) direction of projection ('x', 'y', 'z' or 0, 1, 2)
    :param zrange: (float 2) range in the l.o.s. axis
    :param tcut: (float) if set defines a temperature cut below which particles are removed [K], default: 0.
    :param nsample: (int), if set defines a sampling for the particles (useful to speed up), default: 1 (no sampling)
    :param struct: (bool) if set outputs a structure (dictionary) containing several info, default: False
                    - norm: normalization map
                    - x(y)range: map range in the x(y) direction
                    - pixel_size: pixel size
                    - units: units of the map
                    - norm_units: units of the normalization map
                    - coord_units: units of x(y) range, i.e. h^-1 kpc comoving
                    - other info present for some specific options
    :param nosmooth: (bool) if set the SPH smoothing is turned off, and particles ar treated as points, default: False
    :param progress: (bool) if set the progress bar is shown in output, default: False
    :return: map or structure if struct keyword is set to True
    """

    # Initialization
    if nsample is None:
        nsample = 1

    # Reading header variables
    redshift = pygr.readhead(simfile, 'redshift')
    ngas = pygr.readhead(simfile, 'gascount')
    f_cooling = pygr.readhead(simfile, 'f_cooling')

    # Reading positions of particles
    proj_index = get_proj_index(proj)
    if zrange:
        x, y, z = get_map_coord(simfile, proj_index, True)
    else:
        x, y = get_map_coord(simfile, proj_index)
        z = None

    # Reading smoothing length or assigning it to zero if smoothing is turned off
    hsml = np.full(ngas, 1.e-300) if nosmooth else pygr.readsnap(simfile, 'hsml', 'gas',
                                                                 units=0, suppress=1)  # [h^-1 kpc] comoving

    # Defining center and map size
    if center is None:
        xmin, xmax = min(x - hsml), max(x + hsml)
        ymin, ymax = min(y - hsml), max(y + hsml)
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)
        if size is None:
            delta_x, delta_y = xmax - xmin, ymax - ymin
            if delta_x >= delta_y:
                size = delta_x
                xmap0, ymap0 = xmin, ymin - 0.5 * (delta_x - delta_y)
            else:
                size = delta_y
                xmap0, ymap0 = xmin - 0.5 * (delta_y - delta_x), ymin
        else:
            xmap0, ymap0 = xc - 0.5 * size, yc - 0.5 * size
    else:
        try:
            xc, yc = float(center[0]), float(center[1])
        except BaseException:
            print("Invalid center: ", center, "Must be a 2d number vector")
            raise ValueError

        if size is None:
            xmin, xmax = min(x - hsml), max(x + hsml)
            ymin, ymax = min(y - hsml), max(y + hsml)
            if not (xmin <= xc <= xmax and ymin <= yc <= ymax):
                print("WARNING: Map center is outside the simulation box")
            size = 2. * max(abs(xc - xmin), abs(xc - xmax), abs(yc - ymin), abs(yc - ymax))

        xmap0, ymap0 = xc - 0.5 * size, yc - 0.5 * size

    pixsize = size / npix  # [h^-1 kpc] comoving

    # Normalizing coordinates in pixel units (0 = left/bottom border, npix = right/top border)
    x = (x - xmap0) / size * npix  # [pixel units]
    y = (y - ymap0) / size * npix  # [pixel units]
    hsml_z = hsml if zrange else None  # saving hsml in comoving coordinates [h^-1 kpc]
    hsml = hsml / size * npix  # [pixel units]

    # Cutting out particles outside the f.o.v. and for other conditions
    valid_mask = (x + hsml > 0) & (x - hsml < npix) & (y + hsml > 0) & (y - hsml < npix)
    if tcut > 0.:
        temp = readtemperature(simfile, f_cooling=f_cooling, suppress=1)  # [K]
        valid_mask = valid_mask & (temp > tcut)
        if quantity not in ['Tmw', 'Tew', 'Tsl']:
            del temp

    if zrange:
        valid_mask = valid_mask & (z + hsml_z > zrange[0]) & (z - hsml_z < zrange[1])

    valid = np.where(valid_mask)[0]
    del valid_mask

    # Creating linked list
    particle_list = valid[linkedlist2d(x[valid], y[valid], npix, npix)]
    del valid

    # Calculating quantity (q) to integrate and weight (w)
    mass = pygr.readsnap(simfile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    if zrange:
        # If a l.o.s. range is defined I modify the particle mass according to the smoothing kernel
        for ipart in particle_list[::nsample]:
            mass[ipart] *= intkernel_vec((zrange[1] - z[ipart]) / hsml_z[ipart]) - intkernel_vec((zrange[0] - z[ipart]))
        del z, hsml_z

    if quantity == 'rho':  # Int(rho*dl)
        qty = mass / pixsize ** 2  # comoving [10^10 h M_Sun kpc^-2]
        nrm = np.full(ngas, 0.)  # [---]
    elif quantity == 'rho2':  # Int(rho2*dl)
        qty = mass * pygr.readsnap(simfile, 'rho', 'gas',
                                   units=0, suppress=1) / pixsize ** 2  # comoving [10^20 h^3 M_Sun^2 kpc^-1]
        nrm = np.full(ngas, 0.)  # [---]
    elif quantity in ['Tmw', 'Tew', 'Tsl']:
        if 'temp' not in locals():
            temp = readtemperature(simfile, f_cooling=f_cooling, suppress=1)  # [K]

        if quantity == 'Tmw':
            qty = mass * temp / pixsize ** 2  # [10^10 h M_Sun kpc^-2 K]
            nrm = mass / pixsize ** 2  # comoving [10^10 h M_Sun kpc^-2]
        elif quantity == 'Tew':
            rho = pygr.readsnap(simfile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
            qty = mass * rho * temp / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 K]
            nrm = mass * rho / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5]
            del rho
        elif quantity == 'Tsl':
            rho = pygr.readsnap(simfile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
            qty = mass * rho * temp ** 0.25 / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 K^0.25]
            nrm = mass * rho * temp ** (-0.75) / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 K^-0.75]
            del rho
        del mass, temp
    elif quantity in ['vmw', 'vew', 'wmw', 'wew']:
        vel = pygr.readsnap(simfile, 'vel', 'gas', units=0, suppress=1)[:, proj_index] / (1 + redshift)  # [km s^-1]
        if quantity == 'vmw':
            qty = mass * vel / pixsize ** 2  # [10^10 h M_Sun kpc^-2 km s^-1]
            nrm = mass / pixsize ** 2  # [10^10 h M_Sun kpc^-2]
        elif quantity == 'vew':
            rho = pygr.readsnap(simfile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
            qty = mass * rho * vel / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 km s^-1]
            nrm = mass * rho / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5]
            del rho
        elif quantity == 'wmw':
            qty = mass * vel ** 2 / pixsize ** 2  # [10^10 h M_Sun kpc^-2 km^2 s^-2]
            nrm = mass / pixsize ** 2  # [10^10 h M_Sun kpc^-2]
            qty2 = mass * vel / pixsize ** 2  # [10^10 h M_Sun kpc^-2 km s^-1]
        elif quantity == 'wew':
            rho = pygr.readsnap(simfile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
            qty = mass * rho * vel ** 2 / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 km^2 s^-2]
            nrm = mass * rho / pixsize ** 2  # [10^10 h M_Sun kpc^-2]
            qty2 = mass * rho * vel / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 km s^-1]
            del rho
        del mass, vel
    else:
        print("Invalid mapping quantity: ", quantity,
              "Must be one of 'rho', 'rho2', 'Tmw', 'Tew', 'Tsl', 'vmw', 'vew', 'wmw', 'wew'")
        raise ValueError

    # Mapping
    qty_map = np.full((npix, npix), 0.)
    nrm_map = np.full((npix, npix), 0.)
    qty2_map = np.full((npix, npix), 0.) if quantity in ['wmw', 'wew'] else None

    iter_ = tqdm(particle_list[::nsample]) if progress else particle_list[::nsample]
    for ipart in iter_:
        # Indexes of first and last pixel to map in both axes
        i_beg = max(mt.floor(x[ipart] - hsml[ipart]), 0)
        i_end = min(mt.floor(x[ipart] + hsml[ipart]), npix - 1)
        j_beg = max(mt.floor(y[ipart] - hsml[ipart]), 0)
        j_end = min(mt.floor(y[ipart] + hsml[ipart]), npix - 1)

        # Defining weight vectors for x and y-axis
        xpix = (np.arange(i_beg, i_end + 2) - x[ipart]) / hsml[ipart]
        ypix = (np.arange(j_beg, j_end + 2) - y[ipart]) / hsml[ipart]

        # Using weight vectors to construct weight matrix
        wk_matrix = kernel_weight_2d(xpix, ypix)

        # Adding to maps
        qty_map[i_beg:i_end + 1, j_beg:j_end + 1] += wk_matrix * qty[ipart]
        nrm_map[i_beg:i_end + 1, j_beg:j_end + 1] += wk_matrix * nrm[ipart]
        if quantity in ['wmw', 'wew']:
            qty2_map[i_beg:i_end + 1, j_beg:j_end + 1] += wk_matrix * qty2[ipart]

    qty_map[np.where(nrm_map != 0.)] /= nrm_map[np.where(nrm_map != 0.)]
    if quantity in ['wmw', 'wew']:
        qty2_map[np.where(nrm_map != 0.)] /= nrm_map[np.where(nrm_map != 0.)]
        # The numerical noise may cause some pixels of qty_map to have smaller values than the corresponding ones,
        # squared, in qty2_map: this would cause the presence of nan in the result. The loop below puts 0 in those
        # pixels.
        for ipix in range(npix):
            for jpix in range(npix):
                qty_map[ipix, jpix] = np.sqrt(max(qty_map[ipix, jpix] - qty2_map[ipix, jpix] ** 2, 0))


    # Conversion to float32 for output
    qty_map = np.float32(qty_map)
    nrm_map = np.float32(nrm_map)
    if quantity in ['wmw', 'wew']:
        qty2_map = np.float32(qty2_map)

    # Output
    if struct:

        units = {
            'rho': {'map': '10^10 h M_Sun kpc^-2', 'norm': '---'},
            'rho2': {'map': '10^20 h^3 M_Sun^2 kpc^-1', 'norm': '---'},
            'Tmw': {'map': 'K', 'norm': '10^10 h M_Sun kpc^-2'},
            'Tew': {'map': 'K', 'norm': '10^20 h^3 M_Sun^2 kpc^-5'},
            'Tsl': {'map': 'K', 'norm': '10^20 h^3 M_Sun^2 kpc^-5 K^-0.75'},
            'vmw': {'map': 'km s^-1', 'norm': '10^10 h M_Sun kpc^-2'},
            'vew': {'map': 'km s^-1', 'norm': '10^20 h^3 M_Sun^2 kpc^-5'},
            'wmw': {'map': 'km s^-1', 'map2': 'km s^-1', 'norm': '10^10 h M_Sun kpc^-2'},
            'wew': {'map': 'km s^-1', 'map2': 'km s^-1', 'norm': '10^20 h^3 M_Sun^2 kpc^-5'}
        }

        result = {
            'map': qty_map,
            'norm': nrm_map,
            'xrange': (xmap0, xmap0 + size),  # [h^-1 kpc] comoving
            'yrange': (ymap0, ymap0 + size),  # [h^-1 kpc] comoving
            'pixel_size': pixsize,  # [h^-1 kpc] comoving
            'units': units[quantity]['map'],
            'norm_units': units[quantity]['norm'],
            'coord_units': 'h^-1 kpc'
        }
        if nosmooth:
            result['smoothing'] = 'OFF'
        if quantity in ['wmw', 'wew']:
            result['map2'] = qty2_map
            result['map2_units'] = units[quantity]['map2']
        if zrange:
            result['zrange'] = zrange  # [h^-1 kpc] comoving

        return result

    else:

        return qty_map


from astropy import cosmology


def make_speccube(simfile: str, spfile: str, size: float, npix=256, redshift=None, center=None, proj='z', zrange=None,
                  energy_cut=None, tcut=0., nsample=None, struct=False, isothermal=None, novel=None, nosmooth=False,
                  progress=False):
    """
    :param simfile: (str) simulation file (Gadget)
    :param spfile: (str) spectrum file (FITS)
    :param size: (float) angular size of the map [deg]
    :param npix: (int) number of pixels per map side (default=256)
    :param redshift: (float) redshift there to place the simulation (default: the redshift of the Gadget snapshot file)
    :param center: (float 2) comoving coord. of the map center [h^-1 kpc], default: median point of gas particles
    :param proj: (str/int) direction of projection ('x', 'y', 'z' or 0, 1, 2)
    :param zrange: (float 2) range in the l.o.s. axis
    :param energy_cut: (float 2) energy interval to cpmpute (default: assumes the one from the spfile)
    :param tcut: (float) if set defines a temperature cut below which particles are removed [K], default: 0.
    :param nsample: (int), if set defines a sampling for the particles (useful to speed up), default: 1 (no sampling)
    :param struct: (bool) if set outputs a structure (dictionary) containing several info, default: False
                    - data: spectral cube
                    - x(y)range: map range in the x(y) direction in Gadget units [h^-1 kpc]
                    - size: map size [deg]
                    - pixel_size: pixel size [arcmin]
                    - energy: energy of the spectrum [keV]
                    - energy_interval: energy bin size of the spectrum [keV]
                    - units: units of the spectral cube contained in 'data'
                    - coord_units: units of the coordinates of the map, i.e. x(y)range
                    - energy_units: units of energy and energy_interval
    :param isothermal: (float) if set to a value it assumes an isothermal gas with temperature fixed to the input value
        [K], default: the temperature is read from the Gadget file
    :param novel: (bool) if set to True peculiar velocities are turned off, default: False
    :param nosmooth: (bool) if set the SPH smoothing is turned off, and particles ar treated as points, default: False
    :param progress: (bool) if set the progress bar is shown in output, default: False
    :return: spectral cube or structure if struct keyword is set to True
    """

    # Initialization
    pixsize = size / npix * 60.  # [arcmin]

    if nsample is None:
        nsample = 1

    # Reading header variables
    if redshift is None:
        redshift = pygr.readhead(simfile, 'redshift')
    h_hubble = pygr.readhead(simfile, 'hubble')
    ngas = pygr.readhead(simfile, 'gascount')
    f_cooling = pygr.readhead(simfile, 'f_cooling')

    # Reading positions of particles
    proj_index = get_proj_index(proj)
    if zrange:
        x, y, z = get_map_coord(simfile, proj_index, True)  # [h^-1 kpc] comoving
    else:
        x, y = get_map_coord(simfile, proj_index)  # [h^-1 kpc] comoving
        z = None

    # Reading smoothing length or assigning it to zero if smoothing is turned off
    hsml = np.full(ngas, 1.e-300) if nosmooth else pygr.readsnap(simfile, 'hsml', 'gas',
                                                                 units=0, suppress=1)  # [h^-1 kpc] comoving

    # Geometry conversion
    cosmo = apy.cosmology.FlatLambdaCDM(H0=100., Om0=0.3)
    gadget2deg = cosmo.arcsec_per_kpc_comoving(redshift).to_value() / 3600.  # 1 deg / 1 h^-1 kpc (comoving)
    size_gadget = size / gadget2deg  # [h^-1 kpc]

    # Defining center
    if center is None:
        xmin, xmax = min(x - hsml), max(x + hsml)  # [h^-1 kpc]
        ymin, ymax = min(y - hsml), max(y + hsml)  # [h^-1 kpc]
        xc = 0.5 * (xmin + xmax)  # [h^-1 kpc]
        yc = 0.5 * (ymin + ymax)  # [h^-1 kpc]
    else:
        try:
            xc, yc = float(center[0]), float(center[1])  # [h^-1 kpc]
        except BaseException:
            print("Invalid center: ", center, "Must be a 2d number vector")
            raise ValueError

    xmap0, ymap0 = xc - 0.5 * size_gadget, yc - 0.5 * size_gadget  # [h^-1 kpc]

    # Normalizing coordinates in pixel units (0 = left/bottom border, npix = right/top border)
    x = (x - xmap0) / size_gadget * npix  # [pixel units]
    y = (y - ymap0) / size_gadget * npix  # [pixel units]
    hsml_z = hsml if zrange else None  # saving hsml in comoving coordinates [h^-1 kpc]
    hsml = hsml / size_gadget * npix  # [pixel units]

    # Cutting out particles outside the f.o.v. and for other conditions
    if isothermal:
        temp = np.full(ngas, isothermal)
    else:
        temp = readtemperature(simfile, f_cooling=f_cooling, suppress=1)  # [K]
    valid_mask = (x + hsml > 0) & (x - hsml < npix) & (y + hsml > 0) & (y - hsml < npix)

    if tcut > 0.:
        valid_mask = valid_mask & (temp > tcut)
    temp_keV = temp / phys_const.keV2K  # [keV]
    del temp

    if zrange:
        valid_mask = valid_mask & (z + hsml_z > zrange[0]) & (z - hsml_z < zrange[1])

    valid = np.where(valid_mask)[0]
    del valid_mask

    # Creating linked list
    particle_list = valid[linkedlist2d(x[valid], y[valid], npix, npix)]
    del valid

    # Calculating quantity (q) to integrate and weight (w)
    mass = pygr.readsnap(simfile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    if zrange:
        # If a l.o.s. range is defined I modify the particle mass according to the smoothing kernel
        for ipart in particle_list[::nsample]:
            mass[ipart] *= intkernel_vec((zrange[1] - z[ipart]) / hsml_z[ipart]) - intkernel_vec((zrange[0] - z[ipart]))
        del z, hsml_z

    # Calculating effective redshift (Hubble + peculiar veolcity) of the particles
    if novel:
        # If peculiar velocities are switched off
        z_eff = np.full(ngas, redshift)
    else:
        vel = pygr.readsnap(simfile, 'vel', 'gas', units=0, suppress=1)[:, proj_index] / (1 + redshift)  # [km s^-1]
        z_eff = convert.vpec2zobs(vel, redshift, units='km/s')

    # Reading density
    rho = pygr.readsnap(simfile, 'rho', 'gas', units=0, suppress=1) / (
                1 + redshift) ** 3  # physical [10^10 h^2 M_Sun kpc^-3]

    ne = pygr.readsnap(simfile, 'ne', 'gas', units=0, suppress=1) if f_cooling else None

    norm = convert.gadgget2xspecnorm(mass, rho, 1.e3 * cosmo.comoving_distance(z_eff).to_value(), h_hubble, ne)  # [10^14 cm^-5]
    del mass, rho, ne

    # Reading emission table
    spectable = tables.read_spectable(spfile, z_cut=(np.min(z_eff), np.max(z_eff)),
                                      temperature_cut=(np.min(temp_keV), np.max(temp_keV)),
                                      energy_cut=energy_cut)  # [10^-14 counts s^-1 cm^3]

    energy = spectable.get('energy')  # [keV]
    nene = len(energy)
    d_ene = (energy[-1] - energy[0]) / (
            nene - 1)  # [keV] Assuming uniform energy interval. TODO: include d_ene while generating the table

    # Mapping
    spcube = np.full((npix, npix, nene), 0.)

    iter_ = tqdm(particle_list[::nsample]) if progress else particle_list[::nsample]
    for ipart in iter_:
        # Indexes of first and last pixel to map in both axes
        i_beg = max(mt.floor(x[ipart] - hsml[ipart]), 0)
        i_end = min(mt.floor(x[ipart] + hsml[ipart]), npix - 1)
        j_beg = max(mt.floor(y[ipart] - hsml[ipart]), 0)
        j_end = min(mt.floor(y[ipart] + hsml[ipart]), npix - 1)

        # Defining weight vectors for x and y-axis
        xpix = (np.arange(i_beg, i_end + 2) - x[ipart]) / hsml[ipart]
        ypix = (np.arange(j_beg, j_end + 2) - y[ipart]) / hsml[ipart]

        nx = i_end - i_beg + 1
        ny = j_end - j_beg + 1

        # Using weight vectors to construct weight matrix
        wk_matrix = kernel_weight_2d(xpix, ypix)

        # Calculating spectrum of the particle [10^-14 counts s^-1 cm^3]
        spectrum = tables.calc_spec(spectable, z_eff[ipart], temp_keV[ipart], no_z_interp=True, flag_ene=False)

        # TODO: fix this attempt of vectorialization to speed up code
        spectrum_wk = np.repeat(wk_matrix[:, :, np.newaxis], nene, axis=2) * np.tile(spectrum, (nx, ny, 1))

        # Adding to the spec cube: units [counts s^-1 cm^-2]
        spcube[i_beg:i_end + 1, j_beg:j_end + 1, :] += norm[ipart] * spectrum_wk

    # Renormalizing result
    spcube /= d_ene * pixsize ** 2  # [counts keV^-1 s^-1 cm^-2 arcmin^-2]

    # Conversion to float32 for output
    spcube = np.float32(spcube)
    energy = np.float32(energy)

    # Output
    if struct:

        result = {
            'data': spcube,
            'xrange': (xmap0, xmap0 + size_gadget),  # [h^-1 kpc]
            'yrange': (ymap0, ymap0 + size_gadget),  # [h^-1 kpc]
            'size': size,  # [deg]
            'pixel_size': pixsize,  # [arcmin]
            'energy': energy,
            'energy_interval': np.float32(np.full(nene, d_ene)),
            'units': 'counts keV^-1 s^-1 cm^-2 arcmin^-2',
            'coord_units': 'h^-1 kpc',
            'energy_units': 'keV'
        }
        if nosmooth:
            result['smoothing'] = 'OFF'
        if zrange:
            result['zrange'] = zrange  # [h^-1 kpc] comoving

        return result

    else:

        return spcube

# [counts keV^-1 s^-1 cm^-2 arcmin^-2]
# TODO: The first 7 code lines in the mapping loop can be wrapped into a method, with inputs (x, y, hsml, nx, ny) and
# output (i_beg, i_end, j_beg, j_end, wk_matrix)
