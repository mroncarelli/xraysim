# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython


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
