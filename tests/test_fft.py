from pspline_psd.utils import get_fz
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


def test_fft(helpers):
    """
    Test that the FFT function works
    """
    data_obj = helpers.load_data_0()
    ar4_data = data_obj['data']
    expected_fz = data_obj['anSpecif']['FZ']
    py_fz = get_fz(ar4_data)
    fig = helpers.plot_comparison(expected_fz, py_fz, "FZ")
    # plt.show()
    assert np.allclose(expected_fz, py_fz, atol=1e-5)
