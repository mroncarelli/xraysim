import pytest
import numpy as np

from xraysim.specutils.emisson_models import XspecModel
@pytest.fixture
def xspec_model():
    energy = np.linspace(1.0, 10.0, 100)
    model = XspecModel("apec", energy)
    return model

def test_xspec_model_initialization(xspec_model):
    assert isinstance(xspec_model, XspecModel)
    assert xspec_model.xspec_model.name == "apec"
    # Add more assertions as needed