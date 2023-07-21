import math as mt

import astropy as apy
import numpy as np
import pygadgetreader as pygr
from src.pkg.gadgetutils.readspecial import readtemperature, readvelocity
from src.pkg.gadgetutils import convert, phys_const
from src.pkg.sphprojection.kernel import intkernel, kernel_weight_2d
from src.pkg.sphprojection.linkedlist import linkedlist2d
from src.pkg.sphprojection.matrix_operations import multiply_2d_1d
from tqdm import tqdm
from src.pkg.specutils import tables, absorption
from astropy.io import fits

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


def kernel_2d(x: float, y: float, h: float, nx: int, ny: int):
    """
    Calculates a 2d-kernel map based on the (x, y) coordinates and the smoothing length. Both coordinates and the 
    smoothing length must be normalized to map units, i.e. in pixel units with value 0 corresponding to the map 
    starting borders (i.e. left and bottom borders). 
    :param x: (float) The normalized x-coordinate
    :param y: (float) The normalized y-coordinate
    :param h: (float) The normalized smoothing length
    :param nx: (int) Number of map pixels in the x direction
    :param ny: (int) Number of map pixels in the y direction
    :return: A 3-elements tuple with these elements:
        1: The 2d-kernel map
        2: A 2-elements tuple with the true map pixel limits in the x-direction
        3: A 2-elements tuple with the true map pixel limits in the y-direction
    """
    
    # Indexes of first and last pixel to map in both axes
    i0 = max(mt.floor(x - h), 0)
    i1 = min(mt.floor(x + h), nx - 1)
    j0 = max(mt.floor(y - h), 0)
    j1 = min(mt.floor(y + h), ny - 1)

    # Defining weight vectors for x and y-axis
    xpix = (np.arange(i0, i1 + 2) - x) / h
    ypix = (np.arange(j0, j1 + 2) - y) / h

    # Using weight vectors to construct weight matrix
    wk_matrix = kernel_weight_2d(xpix, ypix)
    
    return wk_matrix, (i0, i1), (j0, j1)


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
        vel = readvelocity(simfile, units='km/s', suppress=1)[:, proj_index]  # [km s^-1]
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
        # Getting the 2d kernel map and pixel ranges
        wk_matrix, i_range, j_range = kernel_2d(x[ipart], y[ipart], hsml[ipart], npix, npix)

        # Adding to maps
        qty_map[i_range[0]:i_range[1] + 1, j_range[0]:j_range[1] + 1] += wk_matrix * qty[ipart]
        nrm_map[i_range[0]:i_range[1] + 1, j_range[0]:j_range[1] + 1] += wk_matrix * nrm[ipart]
        if quantity in ['wmw', 'wew']:
            qty2_map[i_range[0]:i_range[1] + 1, j_range[0]:j_range[1] + 1] += wk_matrix * qty2[ipart]

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
                  energy_cut=None, tcut=0., flag_ene=False, nsample=None, isothermal=None, novel=None, gaussvel=None,
                  seed=0, nosmooth=False, nh=None, progress=False):
    """
    :param simfile: (str) Simulation file (Gadget)
    :param spfile: (str) Spectrum file (FITS)
    :param size: (float) Angular size of the map [deg]
    :param npix: (int) Number of pixels per map side (default=256)
    :param redshift: (float) Redshift where to place the simulation (default: the redshift of the Gadget snapshot file)
    :param center: (float 2) Comoving coord. of the map center [h^-1 kpc], default: median point of gas particles
    :param proj: (str/int) Direction of projection ('x', 'y', 'z' or 0, 1, 2)
    :param zrange: (float 2) Range in the l.o.s. axis
    :param energy_cut: (float 2) Energy interval to compute (default: assumes the one from the spfile)
    :param tcut: (float) If set defines a temperature cut below which particles are removed [K]. Default: 0.
    :param flag_ene: (bool) if set to True forces the computation to be in energy units, i.e. [keV keV^-1 s^-1 cm^-2
        arcmin^-2], with False in count units, i.e. [photons keV^-1 s^-1 cm^-2 arcmin^-2], default: False
    :param nsample: (int), if set defines a sampling for the particles (useful to speed up), default: 1 (no sampling)
    :param isothermal: (float) if set to a value it assumes an isothermal gas with temperature fixed to the input value
        [K], default: the temperature is read from the Gadget file
    :param novel: (bool) If set to True peculiar velocities are turned off. Default: False
    :param gaussvel: (float 2) If set velocites are set randomly with a gaussian pdf, with mean and standard deviation
        given by the first and second argument in [km/s]. If the second argument is not present it is considered to be
        0. Applies only if novel=False. Default: None.
    :param seed: (int) Seed for the random generator (used for gaussvel). Default: 0.
    :param nosmooth: (bool) if set the SPH smoothing is turned off, and particles ar treated as points, default: False
    :param nh: (float) hydrogen column density [cm^-2], overrides the value from the spectral table
    :param progress: (bool) if set the progress bar is shown in output, default: False
    :return: a structure (dictionary) containing several info, including:
                    - data: spectral cube [photons keV^-1 s^-1 cm^-2 arcmin^-2] (or [keV keV^-1 s^-1 cm^-2 arcmin^-2]
                        if flag_ene is True)
                    - x(y)range: map range in the x(y) direction in Gadget units [h^-1 kpc]
                    - size: map size [deg]
                    - pixel_size: pixel size [arcmin]
                    - energy: energy of the spectrum [keV]
                    - energy_interval: energy bin size of the spectrum [keV]
                    - units: units of the spectral cube contained in 'data'
                    - coord_units: units of the coordinates of the map, i.e. x(y)range
                    - energy_units: units of energy and energy_interval
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

    # Reading temperature or assigning it to a single value if isothermal is set
    if isothermal:
        temp = np.full(ngas, isothermal)
    else:
        temp = readtemperature(simfile, f_cooling=f_cooling, suppress=1)  # [K]

    # Cutting out particles outside the f.o.v.
    valid_mask = (x + hsml > 0) & (x - hsml < npix) & (y + hsml > 0) & (y - hsml < npix)

    # If tcut is set, cutting out particles with temperature lower than the limit
    if tcut > 0.:
        valid_mask = valid_mask & (temp > tcut)
    temp_keV = temp / phys_const.keV2K  # [keV]
    del temp

    # If zrange is set, cutting out particles outside the l.o.s. range
    if zrange:
        valid_mask = valid_mask & (z + hsml_z > zrange[0]) & (z - hsml_z < zrange[1])

    valid = np.where(valid_mask)[0]
    del valid_mask

    # Creating linked list with valid particles only
    particle_list = valid[linkedlist2d(x[valid], y[valid], npix, npix)]
    del valid

    # Calculating quantity (q) to integrate and weight (w)
    mass = pygr.readsnap(simfile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    if zrange:
        # If a l.o.s. range is defined I modify the particle mass according to the smoothing kernel
        for ipart in particle_list[::nsample]:
            mass[ipart] *= intkernel_vec((zrange[1] - z[ipart]) / hsml_z[ipart]) - intkernel_vec((zrange[0] - z[ipart]))
        del z, hsml_z

    # Calculating effective redshift (Hubble + peculiar velocity) of the particles
    if novel:
        # If peculiar velocities are switched off all particles effective redshift are set to the cosmological one
        z_eff = np.full(ngas, redshift)
    elif gaussvel:
        # Random gaussian velocity distribution
        try:
            v_mean = float(gaussvel[0])  # [km/s]
            v_std = float(gaussvel[1]) if len(gaussvel) > 1 else 0.  # [km/s]
            rng = np.random.default_rng(np.abs(seed))
            z_eff = convert.vpec2zobs(rng.normal(loc=v_mean, scale=v_std, size=ngas),  # [km/s]
                                      redshift,
                                      units='km/s')
        except BaseException:
            print("Invalid value for gaussvel: ", gaussvel, "Must be a 2d number vector")
            raise ValueError

    else:
        z_eff = convert.vpec2zobs(readvelocity(simfile, units='km/s', suppress=1)[:, proj_index],  # [km/s]
                                  redshift,
                                  units='km/s')

    # Reading density (physical [10^10 h^2 M_Sun kpc^-3])
    rho = pygr.readsnap(simfile, 'rho', 'gas', units=0, suppress=1) / (1 + redshift) ** 3

    # Reading ionization fraction if f_cooling is on
    ne = pygr.readsnap(simfile, 'ne', 'gas', units=0, suppress=1) if f_cooling else None

    # Calculating Xspec normalization [10^14 cm^-5]
    norm = convert.gadgget2xspecnorm(mass, rho, 1.e3 * cosmo.comoving_distance(z_eff).to_value(), h_hubble, ne)
    del mass, rho, ne

    # Reading emission table [10^-14 photons s^-1 cm^3]
    spectable = tables.read_spectable(spfile, z_cut=(np.min(z_eff), np.max(z_eff)),
                                      temperature_cut=(np.min(temp_keV), np.max(temp_keV)),
                                      energy_cut=energy_cut)

    # In nh is provided the spectral table is converted
    if nh is not None:
        spectable = absorption.convert_nh(spectable, nh)

    energy = spectable.get('energy')  # [keV]
    nene = len(energy)
    # TODO: include d_ene while generating the table
    d_ene = (energy[-1] - energy[0]) / (nene - 1)  # [keV] Assuming uniform energy interval.

    # Converting photons to energy o viceversa, if necessary
    if flag_ene != spectable.get('flag_ene'):
        if flag_ene:
            for iene in range(0, nene):
                spectable['data'][:, :, iene] *= energy[iene]  # [10^-14 keV s^-1 cm^3]
        else:
            for iene in range(0, nene):
                spectable['data'][:, :, iene] /= energy[iene]  # [10^-14 photons s^-1 cm^3]

    # Initializing 3d spec-cube
    spcube = np.full((npix, npix, nene), 0.)

    # Defining iterable to iterate through particles
    iter_ = tqdm(particle_list[::nsample]) if progress else particle_list[::nsample]
    for ipart in iter_:
        # Getting the 2d kernel map and pixel ranges
        wk_matrix, i_range, j_range = kernel_2d(x[ipart], y[ipart], hsml[ipart], npix, npix)

        # Calculating spectrum of the particle [photons s^-1 cm^-2]
        spectrum = norm[ipart] * tables.calc_spec(spectable, z_eff[ipart], temp_keV[ipart], no_z_interp=True,
                                                  flag_ene=False)

        # Calculating 3d spec-cube of the single SPH particle
        spectrum_wk = multiply_2d_1d(wk_matrix, spectrum)  # [photons s^-1 cm^-2]

        # Adding to the spec-cube: units [photons s^-1 cm^-2]
        spcube[i_range[0]:i_range[1] + 1, j_range[0]:j_range[1] + 1, :] += spectrum_wk

    # Renormalizing result
    spcube /= d_ene * pixsize ** 2  # [photons s^-1 cm^-2 arcmin^-2 keV^-1]

    # Output
    result = {
        'data': np.float32(spcube),
        'xrange': (xmap0, xmap0 + size_gadget),  # [h^-1 kpc]
        'yrange': (ymap0, ymap0 + size_gadget),  # [h^-1 kpc]
        'size': np.float32(size),  # [deg]
        'size_units': 'deg',
        'pixel_size': np.float32(pixsize),  # [arcmin]
        'pixel_size_units': 'arcmin',
        'energy': np.float32(energy),
        'energy_interval': np.float32(np.full(nene, d_ene)),
        'units': 'keV keV^-1 s^-1 cm^-2 arcmin^-2' if flag_ene else 'photons keV^-1 s^-1 cm^-2 arcmin^-2',
        'coord_units': 'h^-1 kpc',
        'energy_units': 'keV',
        'simulation_file': simfile,
        'spectral_table': spfile,
        'proj': proj,
        'z_cos': redshift,
        'd_c': cosmo.comoving_distance(redshift).to_value(),  # h^-1 Mpc
        'flag_ene': flag_ene
    }
    if tcut:
        result['tcut'] = tcut
    if isothermal:
        result['isothermal'] = isothermal
    result['smoothing'] = 'OFF' if nosmooth else 'ON'
    result['velocities'] = 'OFF' if novel else 'ON'
    if zrange:
        result['zrange'] = zrange  # [h^-1 kpc] comoving
    if nsample and nsample != 1:
        result['nsample'] = nsample
    if nh is not None:
        result['nh'] = nh  # [10^22 cm^-2]
        result['nh_units'] = '10^22 cm^-2'
    else:
        if 'nh' in spectable:
            result['nh'] = spectable.get('nh')  # [10^22 cm^-2]
            result['nh_units'] = '10^22 cm^-2'

    return result


def write_speccube(spec_cube: dict, outfile: str, overwrite=True):
    """
    Writes a spectral-cube into a FITS file.
    :param spec_cube: (dict) Spectral-cube, i.e. output of make_speccube
    :param outfile: (str) FITS file
    :param overwrite: (bool) If set to true the file is overwrittend, default True
    :return: Output of the write FITS procedure
    """
    hduList = fits.HDUList()
    data = spec_cube.get('data')
    xrange = spec_cube.get('xrange')
    yrange = spec_cube.get('yrange')
    zrange = spec_cube.get('zrange')
    nh = spec_cube.get('nh')
    nsample = spec_cube.get('nsample')

    # Primary
    hduList.append(fits.PrimaryHDU(data.transpose()))
    hduList[-1].header.set('SIM_FILE', spec_cube.get('simulation_file'))
    hduList[-1].header.set('SP_FILE', spec_cube.get('spectral_table'))
    hduList[-1].header.set('PROJ', spec_cube.get('proj'))
    hduList[-1].header.set('Z_COS', spec_cube.get('z_cos'))
    hduList[-1].header.set('D_C', spec_cube.get('d_c'))
    if nsample:
        hduList[-1].header.set('NSAMPLE', nsample)
    hduList[-1].header.set('NPIX', data.shape[0])
    hduList[-1].header.set('NENE', data.shape[2])
    hduList[-1].header.set('ANG_PIX', spec_cube.get('pixel_size'), '[' + spec_cube.get('pixel_size_units') + ']')
    hduList[-1].header.set('ANG_MAP', spec_cube.get('size'), '[' + spec_cube.get('size_units') + ']')
    hduList[-1].header.set('E_MIN', spec_cube.get('energy')[0] - 0.5 * spec_cube.get('energy_interval')[0])
    hduList[-1].header.set('E_MAX', spec_cube.get('energy')[-1] + 0.5 * spec_cube.get('energy_interval')[-1])
    hduList[-1].header.set('FLAG_ENE', 1 if spec_cube.get('flag_ene') else 0)
    hduList[-1].header.set('UNITS', '[' + spec_cube.get('units') + ']')
    hduList[-1].header.set('X_MIN', xrange[0])
    hduList[-1].header.set('X_MAX', xrange[1])
    hduList[-1].header.set('Y_MIN', yrange[0])
    hduList[-1].header.set('Y_MAX', yrange[1])
    if zrange:
        hduList[-1].header.set('Z_MIN', zrange[0])
        hduList[-1].header.set('Z_MAX', zrange[1])
    hduList[-1].header.set('C_UNITS', '[' + spec_cube.get('coord_units') + ']', 'X-Y(-Z) coordinate units')
    if spec_cube.get('tcut'):
        hduList[-1].header.set('T_CUT', spec_cube.get('tcut'), '[K]')
    if spec_cube.get('isothermal'):
        hduList[-1].header.set('ISO_T', spec_cube.get('isothermal'), '[K]')
    hduList[-1].header.set('SMOOTH', spec_cube.get('smoothing'))
    hduList[-1].header.set('VEL', spec_cube.get('velocities'))
    if nh:
        hduList[-1].header.set('N_H', spec_cube.get('nh'), '[' + spec_cube.get('nh_units') + ']')

    # Extension 1
    hduList.append(fits.ImageHDU(spec_cube.get('energy')))
    hduList[-1].header.set('NENE', data.shape[2])
    hduList[-1].header.set('UNITS', '[' + spec_cube.get('energy_units') + ']')

    # Extension 2
    hduList.append(fits.ImageHDU(spec_cube.get('energy_interval')))
    hduList[-1].header.set('NENE', data.shape[2])
    hduList[-1].header.set('UNITS', '[' + spec_cube.get('energy_units') + ']')

    # Writing FITS file
    return hduList.writeto(outfile, overwrite=overwrite)


def read_speccube(infile: str):
    hdulist = fits.open(infile)
    header0 = hdulist[0].header
    header1 = hdulist[1].header
    result = {
        'data': hdulist[0].data,
        'xrange': (np.float32(header0.get('X_MIN')), np.float32(header0.get('X_MAX'))),  # [h^-1 kpc]
        'yrange': (np.float32(header0.get('Y_MIN')), np.float32(header0.get('Y_MAX'))),  # [h^-1 kpc]
        'size': np.float32(header0.get('ANG_MAP')),  # [deg]
        'size_units': header0.comment.get('ANG_MAP').replace('[', '').replace(']', ''),
        'pixel_size': np.float32(header0.get('ANG_PIX')),  # [arcmin]
        'pixel_size_units': header0.comment.get('ANG_PIX').replace('[', '').replace(']', ''),
        'energy': hdulist[1].data,
        'energy_interval': hdulist[2].data,
        'units': header0.get('ANG_MAP').replace('[', '').replace(']', ''),
        'coord_units': header0.get('C_UNITS').replace('[', '').replace(']', ''),
        'energy_units': header1.get('UNITS').replace('[', '').replace(']', ''),
        'simulation_file': header0.get('SIM_FILE'),
        'spectral_table': header0.get('SP_FILE'),
        'proj': header0.get('PROJ'),
        'z_cos': header0.get('Z_COS'),
        'd_c': header0.get('D_C'),  # h^-1 Mpc
        'flag_ene': header0.get('FLAG_ENE') == 1
    }
    if 'T_CUT' in header0:
        result['tcut'] = header0.get('T_CUT')  # [K]
    if 'ISO_T' in header0:
        result['isothermal'] = header0.get('ISO_T')  # [K]
    result['smoothing'] = header0.get('SMOOTH')
    result['velocities'] = header0.get('VEL')
    if 'Z_MIN' in header0:
        result['zrange'] = (header0.get('Z_MIN'), header0.get('Z_MAX'))
    if 'N_H' in header0:
        result['nh'] = header0.get('N_H')
        result['nh_units'] = header0.comment.get('N_H').replace('[', '').replace(']', '')

    return result
