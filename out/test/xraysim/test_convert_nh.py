import os
import pytest
from xraysim.specutils import absorption as spabs
from xraysim.specutils import tables
import numpy as np

spfile_path = os.path.join(os.path.dirname(__file__), "inp/test_emission_table.fits")
spectable = tables.read_spectable(spfile_path)


def test_table_values_decrease_exponentially_with_nh(delta_nh=0.02):
    if 'nh' in spectable:
        nh_old = spectable.get('nh')  # [10^22 cm^-2]
    else:
        nh_old = 0.

    nh_new = nh_old + delta_nh
    spectable_new = spabs.convert_nh(spectable, nh_new)
    energy = spectable.get('energy')
    expected_ratio = np.exp(-delta_nh * spabs.sigma_abs_galactic(energy))
    nz, nt = spectable.get('data').shape[0:2]
    for iz in range(nz):
        for it in range(nt):
            assert spectable.get('data')[iz, it, :] * expected_ratio == pytest.approx(
                spectable_new.get('data')[iz, it, :])
