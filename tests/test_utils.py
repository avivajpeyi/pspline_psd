from pspline_psd.utils import get_periodogram, get_fz
from pspline_psd.sample.gibbs_pspline_simple import _format_data
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
    fz = get_fz(ar4_data)
    py_pdgm = get_periodogram(fz)
    # only keep every 2nd value
    py_pdgm = py_pdgm[::2]
    # add a zero to the end
    py_pdgm = np.append(py_pdgm, 0)

    fig = helpers.plot_comparison(expected_pdgm/np.sum(expected_pdgm), py_pdgm/np.sum(py_pdgm), "pdgrm")
    fig.show()
    assert np.allclose(expected_pdgm, py_pdgm, atol=1e-5)


