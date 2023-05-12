import numpy as np
from .whittle_utilities import psd_model


def lprior(k, v, tau, tau_alpha, tau_beta, phi, phi_alpha, phi_beta, delta, delta_alpha, delta_beta, P):
    # TODO: this is a mess... whats going on lol
    # TODO: Move to using bilby priors
    logprior = (k - 1) * np.log(phi) / 2 - phi * np.dot(np.dot(np.transpose(v), P), v) / 2 + phi_alpha * np.log(
        delta) + (phi_alpha - 1) * np.log(phi) - phi_beta * delta * phi + (delta_alpha - 1) * np.log(
        delta) - delta_beta * delta - (tau_alpha + 1) * np.log(tau) - tau_beta / tau
    return logprior


def llike(v, tau, pdgrm, db_list):
    """Whittle log likelihood"""
    # TODO: Move to using bilby likelihood
    # TODO: the parameters to this function should be the sampling parameters, not the matrix itself!

    # todo: V should be computed in here
    psd = psd_model(v, db_list, n=len(pdgrm))
    f = tau * psd
    integrand = np.log(f[1:-1:2]) + pdgrm[1:-1:2] / (f[1:-1:2] * 2 * np.pi)
    return -np.sum(integrand) / 2


def lpost(k, v, tau, tau_alpha, tau_beta, phi, phi_alpha, phi_beta, delta, delta_alpha, delta_beta, pdgrm,
          db_list, P):
    logprior = lprior(k, v, tau, tau_alpha, tau_beta, phi, phi_alpha, phi_beta, delta, delta_alpha, delta_beta, P)
    loglike = llike(v, tau, pdgrm, db_list)
    logpost = logprior + loglike
    return logpost
