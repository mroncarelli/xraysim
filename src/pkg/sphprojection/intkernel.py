from math import floor


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
    x = max(min(x, 0.999999999999999), -1.)
    icase = floor(2. * (x + 1.))
    if icase == 0:
        return 0.5 - (-1. / 6. - 8. / 3. * x - 4. * x ** 2 - 8. / 3. * x ** 3 - 2. / 3. * x ** 4)
    elif icase == 1:
        return 0.5 - (-4. / 3. * x + 8. / 3. * x ** 3 + 2. * x ** 4)
    elif icase == 2:
        return 0.5 + 4. / 3. * x - 8. / 3. * x ** 3 + 2. * x ** 4
    else:
        return 0.5 - 1. / 6. + 8. / 3. * x - 4. * x ** 2 + 8. / 3. * x ** 3 - 2. / 3. * x ** 4
