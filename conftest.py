def pytest_addoption(parser):
    parser.addoption("--eventlist", action="store", default="standard")
