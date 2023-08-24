# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

DATA_TYPE = np.float32

def intkernel(x: float) -> float:
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
    cdef float xc
    xc = x
    if xc < 0:
        return 1. - intkernel_ge0(-xc)
    elif xc < 0.5:
        return 0.5 + 4. / 3. * xc - 8. / 3. * xc ** 3 + 2. * xc ** 4
    elif xc < 1:
        return 0.5 - 1. / 6. + 8. / 3. * xc - 4. * xc ** 2 + 8. / 3. * xc ** 3 - 2. / 3. * xc ** 4
    else:
        return 1.

def intkernel_ge0(x: float) -> float:
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
    cdef float xc
    xc = x
    if  xc < 0.5:
        return 0.5 + 4. / 3. * xc - 8. / 3. * xc ** 3 + 2. * xc ** 4
    elif xc < 1:
        return 0.5 - 1. / 6. + 8. / 3. * xc - 4. * xc ** 2 + 8. / 3. * xc ** 3 - 2. / 3. * xc ** 4
    else:
        return 1.


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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def add_2dweight_vector(double[:, :, ::1] array3, int is0, int is1, float[:, ::1] w, float[:] v):
    cdef double[:, :, :] view_array3 = array3
    cdef float[:, :] view_w = w
    cdef float[:] view_v = v
    cdef Py_ssize_t nx = w.shape[0]
    cdef Py_ssize_t ny = w.shape[1]
    cdef Py_ssize_t nz = v.shape[0]
    for i0 in range(nx):
        for i1 in range(ny):
            for i2 in range(nz):
                 view_array3[is0 + i0, is1 + i1, i2] += view_w[i0, i1] * view_v[i2]
    return None
