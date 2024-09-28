import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from astropy import cosmology

import pygadgetreader as pygr
from xraysim.gadgetutils import convert
from xraysim.gadgetutils.readspecial import readtemperature, readvelocity

from xraysim.sphprojection.kernel import intkernel
from xraysim.sphprojection.linkedlist import linkedlist2d

from xraysim.specutils.emisson_models import EmissionModels
from xraysim.specutils.sixte import cube2simputfile
from xraysim.sphprojection.mapping import write_speccube
from numba import njit

# intkernel_vec = np.vectorize(intkernel)


# # Wendland C4 1D
def sph_kernel_wendland4_1D(x):
    kernel = np.zeros(len(x))

    idx = (np.abs(x) > 0) & (np.abs(x) < 2)

    kernel[idx] = (3 / 4) * (1 - x[idx] / 2) ** 5 * (2 * x[idx] ** 2 + 2.5 * x[idx] + 1)

    idx = x > 2

    kernel[idx] = 1

    # for the rest i.e. x<-2 it is zero
    return kernel


def get_projection_keys(proj):
    axis_map = {
        'z': (0, 1, 2), '2': (0, 1, 2), 'Z': (0, 1, 2),
        'x': (1, 2, 0), '0': (1, 2, 0), 'X': (1, 2, 0),
        'y': (2, 0, 1), '1': (2, 0, 1), 'Y': (2, 0, 1)
    }

    axis = {
        True: (1, 2, 0),  # for  x
        False: (2, 0, 1),  # for  y
        None: (0, 1, 2)  # for the default z
    }

    x_projections = {'x', 'X', '0'}
    y_projections = {'y', 'Y', '1'}

    key = True if proj in x_projections else False if proj in y_projections else None

    return axis[key]


def make_spectrum_cube(nx, ny, nz, iter_, x, y, hsml, norm, z_eff, temp_kev, metallicity, energy, method,
                       flag_ene, n_jobs):
    print(len(iter_))
    chunks = np.array_split(iter_, n_jobs)

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


@njit()
def make_pixels(x, y):
    nx = x.shape[0]
    ny = y.shape[0]

    pixels = np.empty((nx, ny, 2), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            pixels[i, j, 0] = x[i]
            pixels[i, j, 1] = y[j]

    return pixels


@njit()
def wendland_c4_2d(dist, hsml):
    q = dist / hsml
    alpha_2d = 9 / (4 * np.pi * hsml ** 2)
    W = np.zeros(len(q), dtype=np.float32)
    mask = (q <= 2)
    W[mask] = alpha_2d * (1 - q[mask] / 2) ** 6 * (35 * (q[mask] ** 2) / 12 + 3 * q[mask] + 1)
    return W


@njit()
def compute_weights_and_mapping(nx, ny, xi, yi, hsmli):
    # Create a meshgrid of coordinates

    pixels = make_pixels(np.arange(nx), np.arange(nx))

    center_point = np.array([xi, yi])

    distances = np.sqrt((pixels[:, :, 0] - center_point[0]) ** 2 + (pixels[:, :, 1] - center_point[1]) ** 2)

    mask = (distances <= hsmli) & (distances != 0)  # Exclude the point itself

    selected_distances = []

    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if mask[i, j]:
                selected_distances.append(distances[i, j])
    selected_distances = np.array(selected_distances)
    weight = wendland_c4_2d(selected_distances,hsmli)
    neighbors = np.column_stack(np.where(mask))  # Get indices of neighboring pixels

    return  weight, neighbors

@njit()
def assign_spectrum(spec_cube, spectrum, neighbors, weight):
    for w,pixel in zip(weight, neighbors):
        spec_cube[pixel[0],pixel[1],:] = spec_cube[pixel[0],pixel[1],:] + w*spectrum
    return spec_cube


def spectrum_and_add_to_cube(z_eff, temp_kev, metals, norm, x, y, hsml, nx, ny, nz, energy, method,
                             flag_ene, chunk_name):
    spcube = np.full((nx, ny, nz), 0., dtype=np.float64)  # mind it nz is energy bins
    progress_bar = tqdm(total=len(z_eff), position=0, leave=True, desc=chunk_name)

    with (EmissionModels(method, energy) as em):
        for (redshift, temperature, Z, normalization, xi, yi, hsml_i) in zip(z_eff, temp_kev, metals, norm, x, y, hsml):
            # print(redshift, temperature, Z, normalization)

            spectrum = em.compute_spectrum(redshift, temperature, Z, normalization, flag_ene)

            # print(xi, yi, hsml_i)
            Weight, Neighbors= compute_weights_and_mapping(nx, ny, xi, yi, hsml_i)
            spcube = assign_spectrum(spcube,spectrum,Neighbors,Weight)
            progress_bar.update(1)
        # progress_bar.close()
    return spcube


def make_simput_emission_model(simfile: str, angular_size: float, energy_min: float, energy_max: float, bins: int,
                               method: str, Npixel=256,
                               redshift=None, halo_center=None, halo_radius=None, proj='z', zrange=None, tcut=0.,
                               flag_ene=False,
                               novel=None,
                               gaussvel=None, seed=0, n_jobs=2, chunk_size=2):
    # pixel size across both axis ; it will be same across x and y
    pixel_size = (angular_size / Npixel) * 60  # [arcmins]

    if redshift is None:
        redshift = pygr.readhead(simfile, 'redshift')
    print('redshift', redshift)

    h_hubble = pygr.readhead(simfile, 'hubble')
    ngas = pygr.readhead(simfile, 'gascount')
    f_cooling = pygr.readhead(simfile, 'f_cooling')

    print('redshift, hubble, ngas, fcooling', redshift, h_hubble, ngas, f_cooling)

    pos = pygr.readsnap(simfile, 'pos', 'gas', units=0).T  # [h^-1 kpc] comoving
    proj_idx = get_projection_keys(proj)

    x_coord, y_coord, z_coord = pos[proj_idx[0]], pos[proj_idx[1]], pos[proj_idx[2]]

    # Reading smoothing length or assigning it to zero if smoothing is turned off
    hsml = pygr.readsnap(simfile, 'hsml', 'gas', units=0)  # [h^-1 kpc] comoving

    # Geometry conversion
    cosmo = cosmology.FlatLambdaCDM(H0=100., Om0=0.3)
    gadget2deg = cosmo.arcsec_per_kpc_comoving(redshift).to_value() / 3600.  # 1 deg / 1 h^-1 kpc (comoving)
    # the amount of size of the simulation in [h^-1 kpc] we are covering from the central position
    size_gadget = angular_size / gadget2deg  # [h^-1 kpc]
    print(size_gadget)

    # Defining center, units [h^-1 kpc]
    if halo_center is None:
        xc, yc = (np.min(x_coord - hsml) + np.max(x_coord + hsml)) * 0.5, (
                np.min(y_coord - hsml) + np.max(y_coord + hsml)) * 0.5
    else:
        try:
            xc, yc = float(halo_center[0]), float(halo_center[1])  # [h^-1 kpc]
        except BaseException:
            print("Invalid center: ", halo_center, "Must be a 2d number vector")
            raise ValueError

    # plt.scatter(x_coord[::100], y_coord[::100], s=0.01)
    # plt.scatter(xc, yc, s=100)
    # plt.vlines(xc - 0.5 * size_gadget, np.min(y_coord), np.max(y_coord), colors='k')
    # plt.vlines(xc + 0.5 * size_gadget, np.min(y_coord), np.max(y_coord), colors='k')
    # plt.hlines(yc - 0.5 * size_gadget, np.min(x_coord), np.max(x_coord), colors='k')
    # plt.hlines(yc + 0.5 * size_gadget, np.min(x_coord), np.max(x_coord), colors='k')
    # plt.show()

    x_map_origin, y_map_origin = xc - 0.5 * size_gadget, yc - 0.5 * size_gadget  # [h^-1 kpc]

    # Normalizing coordinates in pixel units (0 = left/bottom border, npix = right/top border)
    x_coord = (x_coord - x_map_origin) / size_gadget * Npixel  # [pixel units]
    y_coord = (y_coord - y_map_origin) / size_gadget * Npixel  # [pixel units]

    # plt.scatter(x_coord[::5], y_coord[::5], s=.1, color='r')
    # plt.xlim(0, Npixel)
    # plt.ylim(0, Npixel)
    # plt.show()

    hsml_z = hsml if zrange else None  # saving hsml in comoving coordinates [h^-1 kpc]
    hsml = hsml / size_gadget * Npixel  # [pixel units]

    # Cutting out particles outside the f.o.v.
    valid_mask = (x_coord + hsml > 0) & (x_coord - hsml < Npixel) & (y_coord + hsml > 0) & (y_coord - hsml < Npixel)

    # gas particles are selected in 2D Projected plane for selected halo
    if halo_radius is not None:
        halo_radius = halo_radius / size_gadget * Npixel
        print(halo_radius)
        halo_center = [(xc - x_map_origin) / size_gadget * Npixel, (yc - y_map_origin) / size_gadget * Npixel]
        valid_mask = valid_mask & (
                np.sqrt((x_coord - halo_center[0]) ** 2 + (y_coord - halo_center[1]) ** 2) < halo_radius)

    # plt.scatter(x_coord[valid_mask], y_coord[valid_mask], s=.01, color='k')
    # plt.xlim(0, Npixel)
    # plt.ylim(0, Npixel)
    # plt.show()

    # Reading temperature or assigning it to a single value if isothermal is set
    temp_kev = readtemperature(simfile, f_cooling=f_cooling, units='keV', suppress=1)  # [keV]

    if tcut > 0.:  # [K]
        valid_mask = valid_mask & (readtemperature(simfile, f_cooling=f_cooling, suppress=1) > tcut)

    # If zrange is set, cutting out particles outside the l.o.s. range
    if zrange:
        valid_mask = valid_mask & (z_coord + hsml_z > zrange[0]) & (z_coord - hsml_z < zrange[1])

    valid = np.where(valid_mask)[0]
    del valid_mask

    # Creating linked list with valid particles only
    particle_list = valid[linkedlist2d(x_coord[valid], y_coord[valid], Npixel, Npixel)]
    print(len(valid), len(particle_list))

    # Calculating quantity (q) to integrate and weight (w)
    mass = pygr.readsnap(simfile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]
    print(mass[particle_list])
    if zrange:
        # If a l.o.s. range is defined I modify the particle mass according to the smoothing kernel
        mass[particle_list] *= sph_kernel_wendland4_1D((zrange[1] - z_coord[particle_list]) / hsml_z[particle_list]) \
                               - sph_kernel_wendland4_1D((zrange[0] - z_coord[particle_list]))
        del z_coord, hsml_z
    print(mass[particle_list])

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
        z_eff = convert.vpec2zobs(readvelocity(simfile, units='km/s', suppress=1)[:, proj_idx[-1]],  # [km/s]
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
    iter_ = particle_list[::100]
    print('Num_particle',len(iter_))
    print('Negative norm', np.sum(norm[iter_] < 0))

    energy = np.linspace(energy_min, energy_max, bins + 1)  # Energy edges in KeV
    nene = len(energy) - 1
    d_ene = (energy[-1] - energy[0]) / (nene - 1)  # [keV] Assuming uniform energy interval.

    print(x_coord[particle_list], y_coord[particle_list], hsml[particle_list])

    print(nene)
    spectrum_cube = make_spectrum_cube(Npixel, Npixel, nene, iter_, x_coord, y_coord, hsml, norm, z_eff,
                                       temp_kev, metallicity,
                                       energy, method, flag_ene, n_jobs)
    spectrum_cube = spectrum_cube / (d_ene * pixel_size ** 2)  # [photons s^-1 cm^-2 arcmin^-2 keV^-1]

    result = {
        'data': np.float32(spectrum_cube),
        'xrange': (x_map_origin, x_map_origin + size_gadget),  # [h^-1 kpc]
        'yrange': (y_map_origin, y_map_origin+ size_gadget),  # [h^-1 kpc]
        'size': np.float32(angular_size),  # [deg]
        'size_units': 'deg',
        'pixel_size': np.float32(pixel_size),  # [arcmin]
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

    write_speccube(result, f'{simfile}.speccube')
    #cube2simputfile(result, f'{simfile}.simput')
    return spectrum_cube


import os

inputDir = os.environ.get('XRAYSIM') + '/Gadget_simulations/'
snapshotFile = inputDir + 'snap_119'
print(snapshotFile)
spectrumcube = make_simput_emission_model(snapshotFile, angular_size=2 * (0.1974122751756834), energy_min=0.1, energy_max=1, bins=1001,
                           method='TheThreeHundred-2',
                           proj='z', Npixel=128, halo_center=[500739.49134711, 498543.13815483],
                            n_jobs=1,
                           tcut=937645.6204982)

print(spectrumcube)
#, halo_radius=1624.69,