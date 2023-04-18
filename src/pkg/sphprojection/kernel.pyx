# cython: language_level=3


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
    Useful to speed up computation.
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
