# cython: language_level=3
import numpy as np

def multiply_2d_1d(x, y) :
    """
    Multiplies a 2-d map by a 1-d vector
    :param x: The 2-d map
    :param y: The 1-d vector
    :return: A 3-d datacube
    """
    n0, n1 = x.shape[0], x.shape[1]
    result = np.ndarray([n0, n1, len(y)])
    for i in range(n0):
        for j in range(n1):
            result[i, j, :] = x[i, j] * y

    return result
