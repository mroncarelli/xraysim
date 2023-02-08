import numpy as np

def intKernel(x: float) -> float:

    case = int(np.floor(2. * (x + 1.)))
    if case <= -1:
        return 0
    elif case == 0:
        return 0.5 - (-1. / 6. - 8. / 3. * x - 4. * x**2 - 8. / 3. * x**3 - 2. / 3. * x**4)
    elif case == 1:
        return 0.5 - (-4. / 3. * x + 8. / 3. * x**3 + 2. * x**4) 
    elif case == 2:
        return 0.5 + 4. / 3. * x - 8. / 3. * x**3 + 2. * x**4
    elif case == 3:
        return 0.5 - 1. / 6. + 8. / 3. * x - 4. * x**2 + 8. / 3. * x**3 - 2. / 3. * x**4
    else:
        return 1.
