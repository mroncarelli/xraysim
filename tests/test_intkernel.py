import numpy as np
import pytest
from src.pkg.sphprojection.kernel import intkernel


# This is a series of tests on the intkernel function

def test_left_extreme():
    # Value for argument < -1 must be = 0
    assert intkernel(-10.) == pytest.approx(0.)


def test_left_limit():
    # Value for argument = -1 must be = 0
    assert intkernel(-1.) == pytest.approx(0.)


def test_right_extreme():
    # Value for argument > 1 must be = 1
    assert intkernel(10.) == pytest.approx(1.)


def test_right_limit():
    # Value for argument = 1 must be = 1
    assert intkernel(1.) == pytest.approx(1.)


def test_half():
    # Value for argument = 0 must be = 0.5
    assert intkernel(0.) == pytest.approx(0.5)


def test_val1():
    # Value for argument = 0.3 must be = 0.844200
    assert intkernel(0.3) == pytest.approx(0.844200)


def test_val2():
    # Value for argument = 0.5 must be = 0.958333
    assert intkernel(0.5) == pytest.approx(0.958333)


def test_negative_vs_positive():
    # For any argument x, f(x) = 1-f(-x)
    intkernel_vec = np.vectorize(intkernel)
    x = np.arange(0., 1., 0.01)
    assert all(ypos == pytest.approx(1 - yneg) for ypos, yneg in zip(intkernel_vec(x), intkernel_vec(-x)))


def test_norm_eq_1():
    # A 2D-map of weighted pixels with ranges enclosing [-1, 1] on both sides must sum to 1
    intkernel_vec = np.vectorize(intkernel)
    wx = intkernel_vec(np.arange(-2., 2., 0.01))
    nx = len(wx) - 1
    wy = intkernel_vec(np.arange(-1.9, 2.1, 0.02))
    ny = len(wy) - 1

    map2d = np.empty((nx, ny))
    for i in range(nx):
        for j in range(ny):
            map2d[i, j] = (wx[i + 1] - wx[i]) * (wy[j + 1] - wy[j])

    assert np.sum(map2d) == pytest.approx(1.)
