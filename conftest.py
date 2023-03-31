"""Pytest setup"""


def pytest_configure(config):
    # NB this causes `psline_psd/__init__.py` to run
    import psline_psd  # noqa
