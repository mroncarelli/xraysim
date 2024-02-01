# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

from xraysim.specutils.tables import calc_spec


def intkernel(x):
    """
    Computes the integral of the 1D SPH smoothing kernel w(x): W(x) = Int_{-1}^{x} w(x) dx.
    Here w(x) is centered in x=0 and defined positive between -1 and -1, with w(x)=0 for x<=-1 and x>=1.
    :param x: (float) Argument of the integral.
    :return: (float) Result of the integral.
    :examples: intKernel(-1.) returns 0 (same for any argument <-1)
           intKernel(0.) returns 0.5
           intKernel(0.3) returns 0.8442
           intKernel(0.5) returns 0.958333
           intKernel(1.) returns 1 (same for any argument >1)
    """
    cdef xc = x
    return intkernel_c(xc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float intkernel_c(x: float):
    """
    Computes the integral of the 1D SPH smoothing kernel w(x): W(x) = Int_{-1}^{x} w(x) dx.
    Here w(x) is centered in x=0 and defined positive between -1 and -1, with w(x)=0 for x<=-1 and x>=1.
    :param x: (float) Argument of the integral.
    :return: (float) Result of the integral.
    :examples: intKernel(-1.) returns 0 (same for any argument <-1)
               intKernel(0.) returns 0.5
               intKernel(0.3) returns 0.8442
               intKernel(0.5) returns 0.958333
               intKernel(1.) returns 1 (same for any argument >1)
    """
    if x < 0:
        return 1. - intkernel_c_ge0(-x)
    elif x < 0.5:
        return 0.5 + 4. / 3. * x - 8. / 3. * x ** 3 + 2. * x ** 4
    elif x < 1:
        return 0.5 - 1. / 6. + 8. / 3. * x - 4. * x ** 2 + 8. / 3. * x ** 3 - 2. / 3. * x ** 4
    else:
        return 1.


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float intkernel_c_ge0(x: float):
    """
    Computes the integral of the 1D SPH smoothing kernel w(x): W(x) = Int_{-1}^{x} w(x) dx, but works only for x>=0.
    Useful to speed up computation. It is supposed to be called by intkernel only, and should not be used standalone.
    Here w(x) is centered in x=0 and defined positive between -1 and -1, with w(x)=0 for x<=-1 and x>=1.
    :param x: (float) Argument of the integral.
    :return: (float) Result of the integral.
    :examples: intKernel(0.) returns 0.5
               intKernel(0.3) returns 0.8442
               intKernel(0.5) returns 0.958333
               intKernel(1.) returns 1 (same for any argument >1)
    """
    if  x < 0.5:
        return 0.5 + 4. / 3. * x - 8. / 3. * x ** 3 + 2. * x ** 4
    elif x < 1:
        return 0.5 - 1. / 6. + 8. / 3. * x - 4. * x ** 2 + 8. / 3. * x ** 3 - 2. / 3. * x ** 4
    else:
        return 1.


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float[:] kernel_weight(float[:] x):
    """
    Computes the definite integral of the 1D SPH smoothing kernel w(x) with limits given by the input vector, i.e.:
    W_i = Int_{x_i}^{x_{i+1}} w(x) dx.
    :param x: (float vector) Coordinates to be used as integral limits, must be at least of size 2.
    :return: (float vector) Results of the definite integral, with size equal to the len(x)-1.
    """
    cdef Py_ssize_t i, n = x.shape[0] - 1
    cdef float[:] result = np.empty(n, dtype=np.float32)
    cdef float w0, w1
    w0  = intkernel_c(x[0])
    for i in range(n):
        w1 = intkernel_c(x[i + 1])
        result[i] = w1 - w0
        w0 = w1
    return result

# TODO: Move the stuff after this comment in another file called mapping_loops.pyx. I tried to do this but got stuck
# with import issues

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef kernel_mapping(float x, float h, int n):
    """
    Calculates a 1d-kernel weight based on the coordinates and the smoothing length. Thee coordinates and the 
    smoothing length must be normalized to map units, i.e. in pixel units with value 0 corresponding to the starting 
    border (e.g. left/bottom border) and 1 to the end border (e.g. right/top border) of the map.
    :param x: (float) The normalized coordinate, must be in increasing order.
    :param h: (float) The normalized smoothing length.
    :param n: (int) Number of map pixels in the coordinate direction.
    :return: A 2-elements tuple with these elements:
        0: (float) the 1d-kernel weight,
        1: (int) the index of the starting pixel in the coordinate direction.
    """
    # Indexes of first and last pixel to map in both axes
    cdef Py_ssize_t i0 = max(int(np.floor(x - h)), 0)
    cdef Py_ssize_t i1 = min(int(np.floor(x + h)), n - 1)

    # Defining weight vectors for x and y-axis
    cdef float[:] xpix = (np.arange(i0, i1 + 2, dtype=np.float32) - x) / h

    return kernel_weight(xpix), i0


cdef bint isomorphic(double[:, ::1] var1, double[:, ::1] var2):
    """
    Checks that two 2d-arrays have the same shape.
    :param var1: (double-2d) First array.
    :param var2: (double-2d) Second array.
    :return: (bint) True if the arrays have the same shape, False otherwise.
    """
    cdef bint result = True
    cdef Py_ssize_t i, dim
    for i, dim in enumerate(var1.shape):
        result = result and dim == var2.shape[i]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef add_to_map(double[:, ::1] qty_map, double[:, ::1] nrm_map, float qty, float nrm, float[:] wx, float[:] wy,
                Py_ssize_t i0, Py_ssize_t j0):
    """
    Adds particle quantities to qty_map and nrm_map using kernel weights. Full Cython to maximize speed. Assumes that
    qty_map and nrm_map have the same shape, this must be checked before calling this method.
    """
    cdef Py_ssize_t len_wx = wx.shape[0]
    cdef Py_ssize_t len_wy = wy.shape[0]
    cdef Py_ssize_t i, j, ipix, jpix
    cdef float qty_wx, nrm_wx

    # Checks to avoid bound errors
    # (checks that qty_map and nrm_map have the same shape must be done before calling this method)
    if not (i0 >= 0 and i0 + len_wx <= qty_map.shape[0]):
        print("ERROR in add_to_map. Out of bounds in 1st index.")
        raise ValueError
    if not (j0 >= 0 and j0 + len_wy <= qty_map.shape[1]):
        print("ERROR in add_to_map. Out of bounds in 2nd index.")
        raise ValueError

    for i in range(len_wx):
        qty_wx = qty * wx[i]
        nrm_wx = nrm * wx[i]
        for j in range(len_wy):
            ipix = i0 + i
            jpix = j0 + j
            qty_map[ipix, jpix] += qty_wx * wy[j]
            nrm_map[ipix, jpix] += nrm_wx * wy[j]

    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef add_to_map2(double[:, ::1] qty_map, double[:, ::1] qty2_map, double[:, ::1] nrm_map, float qty, float qty2,
                 float nrm, float[:] wx, float[:] wy, Py_ssize_t i0, Py_ssize_t j0):
    """
    Adds particle quantities to qty_map, qty2_map and nrm_map using kernel weights. Full Cython to maximize speed. 
    Assumes that qty_map, qty2_map and nrm_map have the same shape, this must be checked before calling this method.
    """
    cdef Py_ssize_t len_wx = wx.shape[0]
    cdef Py_ssize_t len_wy = wy.shape[0]
    cdef Py_ssize_t i, j, ipix, jpix
    cdef float qty_wx, qty2_wx, nrm_wx

    # Checks to avoid bound errors
    # (checks that qty_map, qty2_map and nrm_map have the same shape must be done before calling this method)
    if not (i0 >= 0 and i0 + len_wx <= qty_map.shape[0]):
        print("ERROR in add_to_map2. Out of bounds in 1st index.")
        raise ValueError
    if not (j0 >= 0 and j0 + len_wy <= qty_map.shape[1]):
        print("ERROR in add_to_map2. Out of bounds in 2nd index.")
        raise ValueError

    for i in range(len_wx):
        qty_wx = qty * wx[i]
        qty2_wx = qty2 * wx[i]
        nrm_wx = nrm * wx[i]
        for j in range(len_wy):
            ipix = i0 + i
            jpix = j0 + j
            qty_map[ipix, jpix] += qty_wx * wy[j]
            qty2_map[ipix, jpix] += qty2_wx * wy[j]
            nrm_map[ipix, jpix] += nrm_wx * wy[j]

    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef add_to_spcube(double[:, :, ::1] spcube, float[:] spectrum, float[:] wx, float[:] wy,
                   Py_ssize_t i0, Py_ssize_t j0, Py_ssize_t nz):
    """
    Adds spectrum to spcube using kernel weights. Full Cython to maximize speed.
    """
    cdef Py_ssize_t len_wx = wx.shape[0]
    cdef Py_ssize_t len_wy = wy.shape[0]
    cdef Py_ssize_t i, j, k
    cdef float w, ww

    # Checks to avoid bound errors
    if not (i0 >= 0 and i0 + len_wx <= spcube.shape[0]):
        print("ERROR in add_to_spcube. Out of bounds in 1st index.")
        raise ValueError
    if not (j0 >= 0 and j0 + len_wy <= spcube.shape[1]):
        print("ERROR in add_to_spcube. Out of bounds in 2nd index.")
        raise ValueError
    if nz > spcube.shape[2]:
        print("ERROR in add_to_spcube. Out of bounds in 3rd index.")
        raise ValueError

    for i in range(len_wx):
        w = wx[i]
        for j in range(len_wy):
            ww = w * wy[j]
            for k in range(nz):
                spcube[i0 + i, j0 + j, k] += ww * spectrum[k]

    return None


def make_map_loop(double[:, ::1] qty_map, double[:, ::1] nrm_map, iter_, float[:] x, float[:] y, float[:] hsml,
                  float[:] qty, float[:] nrm):
    """
    Cython version of make_map loop in mapping.py.
    """
    if not isomorphic(qty_map, nrm_map):
        print("ERROR in make_map_loop. Arguments qty_map and nrm_map must have the same shape")
        raise ValueError

    cdef float[:] wx, wy
    cdef int nx = qty_map.shape[0]
    cdef int ny = qty_map.shape[1]
    cdef Py_ssize_t i0, j0

    for ipart in iter_:
        # Getting the kernel weights in the two directions
        wx, i0 = kernel_mapping(x[ipart], hsml[ipart], nx)
        wy, j0 = kernel_mapping(y[ipart], hsml[ipart], ny)
        # Adding spectrum to the speccube using weights
        add_to_map(qty_map, nrm_map, qty[ipart], nrm[ipart], wx, wy, i0, j0)

    return None


def make_map_loop2(double[:, ::1] qty_map, double[:, ::1] qty2_map, double[:, ::1] nrm_map, iter_, float[:] x,
                   float[:] y, float[:] hsml, float[:] qty, float[:] qty2, float[:] nrm):
    """
    Cython version of second make_map loop in mapping.py.
    """
    if not (isomorphic(qty_map, nrm_map) and isomorphic(qty_map, qty2_map)):
        print("ERROR in make_map_loop2. Arguments qty_map, nrm_map and qty2_map must have the same shape")
        raise ValueError

    cdef float[:] wx, wy
    cdef int nx = qty_map.shape[0]
    cdef int ny = qty_map.shape[1]
    cdef Py_ssize_t i0, j0

    for ipart in iter_:
        # Getting the kernel weights in the two directions
        wx, i0 = kernel_mapping(x[ipart], hsml[ipart], nx)
        wy, j0 = kernel_mapping(y[ipart], hsml[ipart], ny)
        # Adding particles quantities to the maps using weights
        add_to_map2(qty_map, qty2_map, nrm_map, qty[ipart], qty2[ipart], nrm[ipart], wx, wy, i0, j0)

    return None

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def make_speccube_loop(double[:, :, ::1] spcube, iter_, float[:] x, float[:] y, float[:] hsml, spectable, norm, z_eff,
                       temp_kev):
    """
    Cython version of make_speccube loop in mapping.py.
    """
    cdef float[:] spectrum, wx, wy
    cdef int nx = spcube.shape[0]
    cdef int ny = spcube.sha
# this is set up by me in order to calculate spectra for the particles in a precise way
pe[1]
    cdef Py_ssize_t nz = spcube.shape[2]
    cdef Py_ssize_t i0, i1

    for ipart in iter_:
        # Calculating spectrum of the particle [photons s^-1 cm^-2]
        spectrum = norm[ipart] * calc_spec(spectable, z_eff[ipart], temp_kev[ipart], no_z_interp=True, flag_ene=False)
        # Getting the kernel weights in the two directions
        wx, i0 = kernel_mapping(x[ipart], hsml[ipart], nx)
        wy, j0 = kernel_mapping(y[ipart], hsml[ipart], ny)
        # Adding spectrum to the speccube using weights
        add_to_spcube(spcube, spectrum, wx, wy, i0, j0, nz)

    return None
