# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

from src.pkg.specutils.tables import calc_spec

DATA_TYPE = np.float32

def intkernel_p(x):
    return intkernel(x)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float intkernel(x: float):
    """
    Computes the integral of the 1D SPH smoothing kernel w(x): W(x) = Int_{-1}^{x} w(x) dx.
    Here w(x) is centered in x=0 and defined positive between -1 and -1, with w(x)=0 for x<=-1 and x>=1.
    :param x: the argument of the integral
    :return: the result of the integral
    :examples: intKernel(-1.) returns 0 (same for any argument <-1)
               intKernel(0.) returns 0.5
               intKernel(0.3) returns 0.8442
               intKernel(0.5) returns 0.958333
               intKernel(1.) returns 1 (same for any argument >1)
    """
    if x < 0:
        return 1. - intkernel_ge0(-x)
    elif x < 0.5:
        return 0.5 + 4. / 3. * x - 8. / 3. * x ** 3 + 2. * x ** 4
    elif x < 1:
        return 0.5 - 1. / 6. + 8. / 3. * x - 4. * x ** 2 + 8. / 3. * x ** 3 - 2. / 3. * x ** 4
    else:
        return 1.


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float intkernel_ge0(x: float):
    """
    Computes the integral of the 1D SPH smoothing kernel w(x): W(x) = Int_{-1}^{x} w(x) dx, but works only for x>=0.
    Useful to speed up computation. It is supposed to be called by intkernel only, and should not be used standalone.e
    Here w(x) is centered in x=0 and defined positive between -1 and -1, with w(x)=0 for x<=-1 and x>=1.
    :param x: the argument of the integral
    :return: the result of the integral
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
    cdef Py_ssize_t i, n = x.shape[0] - 1
    cdef float[:] result = np.empty(n, dtype=DATA_TYPE)
    cdef float w0, w1
    w0  = intkernel(x[0])
    for i in range(n):
        w1 = intkernel(x[i+1])
        result[i] = w1 - w0
        w0 = w1
    return result


intkernel_vec = np.vectorize(intkernel)


def kernel_weight_2d(x, y):
    """
    Computes the SPH kernel weight of a 2d map, whose pixel borders are specified by the input.
    :param x: the normalised coordinates of the pixels in the x-axis, must be in increasing order
    :param y: the normalised coordinates of the pixels in the y-axis, must be in increasing order
    :return: a 2d array of size len(x)-1, len(y)-1 with the weights
    :examples: intKernel([-0.3, -0.04, 0., 0.2, 0.8, 1.3], [-2., -0.9, 0., 0.75]) returns:
     array([[1.94053680e-05, 1.45496708e-01, 1.44758217e-01],
       [3.54510721e-06, 2.65803476e-02, 2.64454350e-02],
       [1.65716385e-05, 1.24250097e-01, 1.23619446e-01],
       [1.66960967e-05, 1.25183254e-01, 1.24547867e-01],
       [7.11294143e-08, 5.33310968e-04, 5.30604066e-04]])
    """
    nx = len(x) - 1
    int_wk_x = intkernel_vec(x)
    wk_x = [int_wk_x[i + 1] - int_wk_x[i] for i in range(nx)]
    ny = len(y) - 1
    int_wk_y = intkernel_vec(y)
    wk_y = [int_wk_y[j + 1] - int_wk_y[j] for j in range(ny)]
    return np.full([ny, nx], wk_x).astype(DATA_TYPE).transpose() * np.full([nx, ny], wk_y).astype(DATA_TYPE)


def add_2dweight_vector(double[:, :, ::1] array3, int is0, int is1, double[:, ::1] w, double[:] v):
    cdef double[:, :, :] view_array3 = array3
    cdef double[:, :] view_w = w
    cdef double[:] view_v = v
    cdef Py_ssize_t nx = w.shape[0]
    cdef Py_ssize_t ny = w.shape[1]
    cdef Py_ssize_t nz = v.shape[0]
    for i0 in range(nx):
        for i1 in range(ny):
            for i2 in range(nz):
                 view_array3[is0 + i0, is1 + i1, i2] += view_w[i0, i1] * view_v[i2]
    return None


# TODO: Move the stuff after this comment in another file called mapping_loops.pyx. I tried to do it but got stuck
# with import issues

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef kernel_mapping(float x, float h, int n):
    """
    Calculates a 1d-kernel weight based on the 1d coordinates and the smoothing length. Thee coordinates and the 
    smoothing length must be normalized to map units, i.e. in pixel units with value 0 corresponding to the map 
    starting borders (i.e. left and bottom borders). 
    :param x: (float) The normalized coordinate, must be in increasing order
    :param h: (float) The normalized smoothing length
    :param n: (int) Number of map pixels in the coordinate direction
    :return: A 2-elements tuple with these elements:
        1: The 1d-kernel weight
        2: A 2-elements tuple with the true map pixel limits in the coordinate direction
    """

    # Indexes of first and last pixel to map in both axes
    cdef Py_ssize_t i0 = max(int(np.floor(x - h)), 0)
    cdef Py_ssize_t i1 = min(int(np.floor(x + h)), n - 1)

    # Defining weight vectors for x and y-axis
    cdef float[:] xpix = (np.arange(i0, i1 + 2, dtype=DATA_TYPE) - x) / h

    return kernel_weight(xpix), (i0, i1)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef add_to_spcube(double[:, :, ::1] spcube, float[:] spectrum, float[:] wx, float[:] wy,
                   Py_ssize_t i0, Py_ssize_t j0, Py_ssize_t nz):

    cdef Py_ssize_t len_wx = wx.shape[0]
    cdef Py_ssize_t len_wy = wy.shape[0]
    cdef Py_ssize_t i, j, k
    cdef float w, ww
    for i in range(len_wx):
        w = wx[i]
        for j in range(len_wy):
            ww = w * wy[j]
            for k in range(nz):
                spcube[i0 + i, j0 + j, k] += ww * spectrum[k]

    return None

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def make_speccube_loop(double[:, :, ::1] spcube, spectable, iter_, float[:] x, float[:] y, float[:] hsml, norm, z_eff, temp_kev):

    cdef float[:] spectrum, wx, wy
    cdef int nx = spcube.shape[0]
    cdef int ny = spcube.shape[1]
    cdef Py_ssize_t nz = spcube.shape[2]
    cdef Py_ssize_t i0, i1, j0, j1

    for ipart in iter_:

        # Calculating spectrum of the particle [photons s^-1 cm^-2]
        spectrum = norm[ipart] * calc_spec(spectable, z_eff[ipart], temp_kev[ipart], no_z_interp=True, flag_ene=False)

        # Getting the kernel weights in the two directions
        wx, (i0, i1) = kernel_mapping(x[ipart], hsml[ipart], nx)
        wy, (j0, j1) = kernel_mapping(y[ipart], hsml[ipart], ny)

        # Checks
        assert len(wx) == i1 - i0 + 1
        assert len(wy) == j1 - j0 + 1
        assert len(spectrum) == nz

        # Adding spectrum to the speccube using weights
        add_to_spcube(spcube, spectrum, wx, wy, i0, j0, nz)

    return None
