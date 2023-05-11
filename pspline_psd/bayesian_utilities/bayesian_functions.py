import numpy as np
import scipy.stats as stats

import numpy as np
from scipy.stats import gaussian_kde
from .whittle_utilities import unrollPsd, densityMixture

def qpsd(omega, k, v, degree, db_list):
    # TODO: knots and degree are not used here -- is this an artefact from the 'B-Spline' PSD?
    v = np.array(v)
    expV = np.exp(v)

    if np.any(np.isinf(expV)):
        ls = np.logaddexp(0, v)
        weight = np.exp(v - ls)
    else:
        ls = 1 + np.sum(expV)
        weight = expV / ls

    s = 1 - np.sum(weight)
    weight = np.append(weight, 0 if s < 0 else s)
    psd = gaussian_kde(db_list, weights=weight)(omega)
    epsilon = 1e-20
    psd = np.maximum(psd, epsilon)

    return psd

def lprior(k, v, tau, tau_alpha, tau_beta, phi, phi_alpha, phi_beta, delta, delta_alpha, delta_beta, P):
    # TODO: this is a mess... whats going on lol
    # TODO: Move to using bilby priors
    logprior = (k - 1) * np.log(phi) / 2 - phi * np.dot(np.dot(np.transpose(v), P), v) / 2 + phi_alpha * np.log(
        delta) + (phi_alpha - 1) * np.log(phi) - phi_beta * delta * phi + (delta_alpha - 1) * np.log(
        delta) - delta_beta * delta - (tau_alpha + 1) * np.log(tau) - tau_beta / tau
    return logprior


def llike(omega, FZ, k, v, tau, pdgrm, degree, db_list):
    # TODO: Move to using bilby likelihood
    n = len(FZ)

    if n % 2:
        bFreq = 1
    else:
        bFreq = [1, n]

    qq_psd = qpsd(omega, k, v, degree, db_list)
    q = unrollPsd(qq_psd, n)

    f = tau * q

    # whittle log likelihood
    llike = -np.sum(np.log(f[1:-1:2]) + pdgrm[1:-1:2] / (f[1:-1:2] * 2 * np.pi)) / 2
    return llike


def lpost(omega, FZ, k, v, tau, tau_alpha, tau_beta, phi, phi_alpha, phi_beta, delta, delta_alpha, delta_beta, pdgrm,
          degree, db_list, P):
    logprior = lprior(k, v, tau, tau_alpha, tau_beta, phi, phi_alpha, phi_beta, delta, delta_alpha, delta_beta, P)
    loglike = llike(omega, FZ, k, v, tau, pdgrm, degree, db_list)
    logpost = logprior + loglike
    return logpost
