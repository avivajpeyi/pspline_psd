from tqdm.auto import trange
from pspline_psd.bayesian_utilities.bayesian_functions import psd_model
from scipy.stats import median_abs_deviation
import numpy as np


def generate_psd_posterior(freq, db_list, tau_samples, v_samples, ):
    n = len(tau_samples)
    psd = np.zeros((n, len(freq)))
    kwargs = dict(db_list=db_list, n=len(freq))
    for i in trange(n, desc='Generating PSD posterior'):
        psd[i, :] = psd_model(v=v_samples[i, :], **kwargs) * tau_samples[i]
    return psd


def generate_psd_quantiles(freq, db_list, tau_samples, v_samples, uniform_bands=True):
    psds = generate_psd_posterior(freq, db_list, tau_samples, v_samples)
    psd_median = np.quantile(psds, 0.5, axis=0)
    psd_quants = np.quantile(psds, [0.05, 0.95], axis=0)

    lnpsds = logfuller(psds)
    lnpsd_median = np.median(lnpsds, axis=0)
    lnpsd_mad = median_abs_deviation(lnpsds, axis=0)
    lnpsd_uniform_max = uniformmax(lnpsds)
    lnpsd_c_value = np.quantile(lnpsd_uniform_max, 0.9) * lnpsd_mad

    uniform_psd_quants = np.array(
        [
            np.exp(lnpsd_median - lnpsd_c_value),
            np.exp(lnpsd_median + lnpsd_c_value),
        ]
    )

    if uniform_bands:
        psd_with_unc = np.vstack([psd_median, uniform_psd_quants])
    else:
        psd_with_unc = np.vstack([psd_median, psd_quants])

    assert psd_with_unc.shape == (3, len(freq))
    assert np.all(psd_with_unc > 0)
    return psd_with_unc


def uniformmax(sample):
    mad = median_abs_deviation(sample, nan_policy='omit', axis=0)
    # replace 0 with very small number
    mad[mad == 0] = 1e-10
    return np.max(np.abs(sample - np.median(sample, axis=0)) / mad, axis=0)


def logfuller(x, xi=0.001):
    return np.log(x + xi) - xi / (x + xi)
