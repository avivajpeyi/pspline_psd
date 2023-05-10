from pspline_psd.utils import periodogram
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


def test_periodogram(helpers):
    """
    Test that the FFT function works
    """
    data_obj = helpers.load_data_0()
    ar4_data = data_obj['data']
    expected_pdgm = data_obj['pdgrm'] ## THIS IS PDGM SCALED FOR THE PSD -- NOT THE PDGM
    py_pdgm = periodogram(ar4_data)
    fig = helpers.plot_comparison(expected_pdgm, py_pdgm, "pdgrm")
    fig.show()
    assert np.allclose(expected_pdgm, py_pdgm, atol=1e-5)


