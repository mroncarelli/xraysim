def pytest_addoption(parser):
    parser.addoption("--full_run", action="store", default="standard")
