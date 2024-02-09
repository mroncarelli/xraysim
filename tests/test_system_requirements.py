from importlib.util import find_spec
import os
from distutils.spawn import find_executable

commandList = ['xifupipeline', 'makespec']
envVariableList = ['XRAYSIM', 'SIXTE_INSTRUMENTS']
packageList = ['xspec']


def test_programs_are_installed():
    """
    Tests that the necessary programs are installed by checking that the corresponding programs are callable by the
    prompt.
    """
    for command in commandList:
        assert find_executable(command) is not None


def test_environment_variables_are_set():
    """
    Tests that the necessary environment variables are set.
    """
    for env_var in envVariableList:
        assert os.environ.get(env_var) is not None


def test_packages_are_present():
    """
    Tests that the necessary packages are present.
    """
    for package in packageList:
        dummy = find_spec(package)
        assert dummy is not None
