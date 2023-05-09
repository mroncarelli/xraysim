import os
import pytest
from specutils.tables import read_spectable, calc_spec

spectable_file = os.environ.get('XRAYSIM') + '/tests/data/emission_table.fits'


def test_table_values():
    # A spectrum computed at a z and temperature that match table values must return the values of the table
    spec_table = read_spectable(spectable_file)
    iz = 10
    it = 20
    spec = calc_spec(spec_table, spec_table.get('z')[iz], spec_table.get('temperature')[it])
    assert all(v / v_table == pytest.approx(1) for v, v_table in zip(spec, spec_table.get('data')[iz, it, :]))
