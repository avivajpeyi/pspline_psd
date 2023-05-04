from pspline_psd.utils import periodogram
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


def test_fft(ar4_data):
    """
    Test that the FFT function works
    """
    expected_pdgm = [3.923472e-33, 0.08510271, 0.04090759, 0.05818444, 0.01196102, 0.008774831, 0.04602032, 0.0788349, 0.2684149, 0.03964334, 0.4944483, 0.2547649, 0.446567, 2.26811, 0.4006971, 2.045688, 0.1355134, 0.1357872, 0.1931103, 0.001156087, 0.08617651, 0.1005989, 0.02551009, 0.00873342, 0.01114421, 0.01230256, 0.0466447, 0.0008572933, 0.001529329, 0.01068104, 0.02889412, 0.06800997, 0.03877467, 0.05758103, 0.04332073, 0.02385658, 0.1156185, 0.1746326, 0.1132719, 1.357271, 0.4737421, 0.09911091, 0.0478397, 0.05335764, 0.01433003, 0.004120592, 0.01220148, 0.009405838, 0.01092784, 0.006972285, 0.0008098067, 0.008745403, 0.0002858206, 0.004740306, 0.008888544, 0.0003643653, 0.000440252, 0.001757512, 0.0003703224, 0.003596092, 0.001239955, 0.0001372894, 0.00108088, 0.001518945, 0.002519573]
    py_pdgm = periodogram(ar4_data)
    freq = np.arange(0, np.pi, len(expected_pdgm))
    # skip 1st element because it's 0
    plt.plot(freq[1:], np.log(expected_pdgm[1:]), label="Expected")
    plt.plot(freq[1:], np.log(py_pdgm[1:]), label="Python")
    plt.ylim(-20, 0)
    plt.show()

    assert np.allclose(expected_pdgm, py_pdgm, atol=1e-5)


