"""Pytest setup"""


def pytest_configure(config):
    # NB this causes `pspline_psd/__init__.py` to run
    import pspline_psd  # noqa
