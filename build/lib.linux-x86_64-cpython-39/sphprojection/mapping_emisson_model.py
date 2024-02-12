import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from astropy.io import fits
from multiprocessing import Pool
from astropy import cosmology

import pygadgetreader as pygr
from xraysim.sphprojection.mapping import get_map_coord, get_proj_index
from xraysim.gadgetutils import convert
from xraysim.gadgetutils.readspecial import readtemperature, readvelocity

from xraysim.sphprojection.kernel import intkernel
from xraysim.sphprojection.linkedlist import linkedlist2d
from xraysim.sphprojection.kernel import kernel_mapping_p
from xraysim.specutils.emisson_models import EmissionModels
from xraysim.specutils.sixte import cube2simputfile
from xraysim.sphprojection.mapping import write_speccube

intkernel_vec = np.vectorize(intkernel)


def make_spectrum_cube(iterator, nx, ny, nz, iter_, x, y, hsml, norm, z_eff, temp_kev, metallicity, energy, method,
                       flag_ene, n_jobs, chunk_size):
    print(len(iter_))
    chunks = np.array_split(iter_, chunk_size)
    [print(len(i), i) for i in chunks]
    print(metallicity.shape)
    with Pool(processes=n_jobs) as pool:
        results = pool.starmap(spectrum_and_add_to_cube,
                               [(
                                   z_eff[i], temp_kev[i], metallicity[i], norm[i], x[i], y[i], hsml[i], nx, ny, nz,
                                   energy,
                                   method, flag_ene, f"Chunk {idx + 1} Progress :")
                                   for idx, i in enumerate(chunks)])

    final_cube = np.sum(np.stack(results), axis=0)
    print(final_cube.shape)
    return final_cube


def spectrum_and_add_to_cube(z_eff, temp_kev, metals, norm, x, y, hsml, nx, ny, nz, energy, method,
                             flag_ene, chunk_name):
    spcube = np.full((nx, ny, nz), 0., dtype=np.float64)  # mind it nz is energy bins

    progress_bar = tqdm(total=len(z_eff), position=0, leave=True, desc=chunk_name)

    with (EmissionModels(method, energy) as em):

        for (redshift, temperature, Z, normalization, xi, yi, hsml_i) in zip(z_eff, temp_kev, metals, norm, x, y, hsml):
            spectrum = em.compute_spectrum(redshift, temperature, Z, normalization, flag_ene)
            wx, i0 = kernel_mapping_p(xi, hsml_i, nx)
            wy, j0 = kernel_mapping_p(yi, hsml_i, ny)

            wx = np.asarray(wx, dtype=np.float32)
            wy = np.asarray(wy, dtype=np.float32)

            if not (i0 >= 0 and i0 + len(wx) <= nx):
                print(f"ERROR in {chunk_name}. Out of bounds in 1st index.")
                raise ValueError
            if not (j0 >= 0 and j0 + len(wy) <= ny):
                print(f"ERROR in {chunk_name}. Out of bounds in 2nd index.")
                raise ValueError
            if nz > spcube.shape[2]:
                print(f"ERROR in {chunk_name}. Out of bounds in 3rd index.")
                raise ValueError

            # Computing the weighted spectrum and assigning them to the grids using numpy now
            matrix_wx, matrix_wy = np.meshgrid(wx, wy)
            weight_matrix = (matrix_wx * matrix_wy).T
            spcube[i0:i0 + weight_matrix.shape[0], j0:j0 + weight_matrix.shape[1], :] += weight_matrix[:, :,
                                                                                         np.newaxis] * spectrum
            # (second option to assign spectrum !)
            # index_iterable = np.ndindex(*weight_matrix.shape)
            # for ix in index_iterable:
            #    spcube[i0 + ix[0], j0 + ix[1], :] = spcube[i0 + ix[0], j0 + ix[1], :] + weight_matrix[
            #        ix[0], ix[1]] * spectrum

            progress_bar.update(1)
        progress_bar.close()
        print('here-end')
    return spcube


def make_simput_emission_model(simfile: str, size: float, emin: float, emax: float, bins: int, method: str, npix=256,
                               redshift=None, center=None, proj='z', zrange=None, tcut=0., flag_ene=False, novel=None,
                               gaussvel=None, seed=0, n_jobs=2, chunk_size=2):
    pixsize = size / npix * 60.  # [arcmin]

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
    hsml = pygr.readsnap(simfile, 'hsml', 'gas', units=0, suppress=1)  # [h^-1 kpc] comoving

    # Geometry conversion
    cosmo = cosmology.FlatLambdaCDM(H0=100., Om0=0.3)
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
    temp_kev = readtemperature(simfile, f_cooling=f_cooling, units='keV', suppress=1)  # [keV]

    # Cutting out particles outside the f.o.v.
    valid_mask = (x + hsml > 0) & (x - hsml < npix) & (y + hsml > 0) & (y - hsml < npix)

    # If tcut is set, cutting out particles with temperature lower than the limit. IMPORTANT: the cut on the simulation
    # temperature even in the isothermal case, to eliminate particles with extremely high densities that would dominate
    # the emission.
    if tcut > 0.:  # [K]
        valid_mask = valid_mask & (readtemperature(simfile, f_cooling=f_cooling, suppress=1) > tcut)

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
        for ipart in particle_list[::1]:
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
            vel = rng.normal(loc=v_mean, scale=v_std, size=ngas)
            z_eff = convert.vpec2zobs(vel,  # [km/s]
                                      redshift,
                                      units='km/s')
            del vel
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

    try:
        # try first opening for Gadget format, if unsuccessful try with Gizmo-Simba setting, if not
        # wrong file
        metallicity = pygr.readsnap(simfile, 'Z   ', 'gas', units=0, suppress=1)
        if len(metallicity) == 0:
            raise ValueError("No metallicity Data, file is Gizmo-Simba")
    except ValueError:
        try:
            metallicity = pygr.readsnap(simfile, 'Metallicity', 'gas', units=0, suppress=1)
        except KeyError:
            print('Wrong keyword or unknown simulation file')

    if len(metallicity.shape) == 1:
        metallicity = np.reshape(metallicity, newshape=(-1, 1))
    elif len(metallicity.shape) == 2:
        metallicity = metallicity[:, 2:]

    print(metallicity)
    # Calculating Xspec normalization [10^14 cm^-5]
    norm = convert.gadget2xspecnorm(mass, rho, 1.e3 * cosmo.comoving_distance(z_eff).to_value(), h_hubble, ne)

    del mass, rho, ne

    # removing particles for which norm is negative
    for i in np.argwhere(norm < 0):
        idx = np.argwhere(particle_list == i)
        particle_list = np.delete(particle_list, idx)

    # Defining iterable to iterate through particles
    iter_ = particle_list[::1]
    print('Negative norm', np.sum(norm[iter_] < 0))

    energy = np.linspace(emin, emax, bins + 1)  # Energy edges in KeV
    nene = len(energy) - 1
    d_ene = (energy[-1] - energy[0]) / (nene - 1)  # [keV] Assuming uniform energy interval.

    print(nene)
    spectrum_cube = make_spectrum_cube(iter_, npix, npix, nene, iter_, x, y, hsml, norm, z_eff, temp_kev, metallicity,
                                       energy, method, flag_ene, n_jobs, chunk_size)

    spectrum_cube = spectrum_cube / (d_ene * pixsize ** 2)  # [photons s^-1 cm^-2 arcmin^-2 keV^-1]

    result = {
        'data': np.float32(spectrum_cube),
        'xrange': (xmap0, xmap0 + size_gadget),  # [h^-1 kpc]
        'yrange': (ymap0, ymap0 + size_gadget),  # [h^-1 kpc]
        'size': np.float32(size),  # [deg]
        'size_units': 'deg',
        'pixel_size': np.float32(pixsize),  # [arcmin]
        'pixel_size_units': 'arcmin',
        'energy': np.float32(0.5 * (energy[1:] + energy[:-1])),
        'energy_interval': np.float32(np.full(nene, d_ene)),
        'units': 'keV keV^-1 s^-1 cm^-2 arcmin^-2' if flag_ene else 'photons keV^-1 s^-1 cm^-2 arcmin^-2',
        'coord_units': 'h^-1 kpc',
        'energy_units': 'keV',
        'simulation_file': simfile,
        'proj': proj,
        'z_cos': redshift,
        'd_c': cosmo.comoving_distance(redshift).to_value(),  # [h^-1 Mpc]
        'flag_ene': flag_ene
    }
    if tcut:
        result['tcut'] = tcut  # [K]
    result['velocities'] = 'OFF' if novel else 'ON'
    if zrange:
        result['zrange'] = zrange  # [h^-1 kpc] comoving

    write_speccube(result, 'check.spcube')
    cube2simputfile(result, 'check-xspec.simput')

    return result
