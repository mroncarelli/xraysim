import numpy as np
import pytest
from src.pkg.sphprojection.kernel import kernel_weight_2d

def test_norm_eq_1():
    # A 2D-map of weighted pixels with ranges enclosing [-1, 1] on both sides must sum to 1
    x = np.arange(-2., 2., 0.01, dtype='float32')
    y = np.arange(-1.9, 2.1, 0.02, dtype='float32')
    map2d = kernel_weight_2d(x, y)
    assert np.sum(map2d) == pytest.approx(1.)
