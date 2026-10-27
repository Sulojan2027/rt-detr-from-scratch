def pytest_configure(config):
    config.addinivalue_line("markers", "slow: downloads HF weights / needs network (skipped in CI)")
