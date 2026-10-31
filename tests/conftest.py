def pytest_configure(config):
    config.addinivalue_line("markers", "slow: downloads HF weights / needs network (deselect with -m 'not slow')")
