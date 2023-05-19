import numpy as np
from bilby.core.prior import Gamma, PriorDict
from numpy import dot

from .whittle_utilities import psd_model


def _vPv(v, P):
    return dot(dot(v.T, P), v)


def lprior(k, v, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, P):
    # TODO: Move to using bilby priors

    vTPv = _vPv(v, P)
    logφ = np.log(φ)
    logδ = np.log(δ)
    logτ = np.log(τ)

    lnpri_weights = (k - 1) * logφ * 0.5 - φ * vTPv * 0.5
    lnpri_φ = φα * logδ + (φα - 1) * logφ - φβ * δ * φ
    lnpri_δ = (δα - 1) * logδ - δβ * δ
    lnpri_τ = -(τα + 1) * logτ - τβ / τ
    log_prior = lnpri_weights + lnpri_φ + lnpri_δ + lnpri_τ
    return log_prior


def φ_prior(k, v, P, φα, φβ, δ):
    vTPv = dot(dot(v.T, P), v)
    shape = (k - 1) / 2 + φα
    rate = φβ * δ + vTPv / 2
    return Gamma(k=shape, theta=1 / rate)


def δ_prior(φ, φα, φβ, δα, δβ):
    """Gamma prior for pi(δ|φ)"""
    shape = φα + δα
    rate = φβ * φ + δβ
    return Gamma(k=shape, theta=1 / rate)


def inv_τ_prior(v, periodogram, db_list, τα, τβ):
    """Inverse(?) prior for tau -- tau = 1/inv_tau_sample"""

    # TODO: ask about the even/odd difference, and what 'bFreq' is

    n = len(periodogram)
    psd = psd_model(v, db_list, n=n)
    is_even = n % 2 == 0
    if is_even:
        whtn_pdgm = periodogram[1:-1] / psd[1:-1]
    else:
        whtn_pdgm = periodogram[1:] / psd[1:]

    n = len(whtn_pdgm)

    shape = τα + n / 2
    rate = τβ + np.sum(whtn_pdgm) / (2 * np.pi) / 2
    return Gamma(k=shape, theta=rate)


def sample_φδτ(k, v, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, periodogram, db_list, P):
    φ = φ_prior(k, v, P, φα, φβ, δ).sample().flat[0]
    δ = δ_prior(φ, φα, φβ, δα, δβ).sample().flat[0]
    τ = 1 / inv_τ_prior(v, periodogram, db_list, τα, τβ).sample()
    return φ, δ, τ


# devtools::install_github("pmat747/psplinePsd")


def llike(v, τ, pdgrm, db_list):
    """Whittle log likelihood"""
    # TODO: Move to using bilby likelihood
    # TODO: the parameters to this function should be the sampling parameters, not the matrix itself!
    # todo: V should be computed in here

    n = len(pdgrm)
    psd = psd_model(v, db_list, n=n)
    f = τ * psd

    is_even = n % 2 == 0
    if is_even:
        f = f[1:]
        pdgrm = pdgrm[1:]
    else:
        f = f[1:-1]
        pdgrm = pdgrm[1:-1]

    integrand = np.log(f) + pdgrm / (f * 2 * np.pi)
    lnlike = -np.sum(integrand) / 2
    assert np.isfinite(lnlike), f"lnlike is not finite: {lnlike}"
    return lnlike


def lpost(k, v, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, pdgrm, db_list, P):
    logprior = lprior(k, v, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, P)
    loglike = llike(v, τ, pdgrm, db_list)
    logpost = logprior + loglike
    assert (np.isfinite(logpost), f"logpost is not finite: lnpri{logprior}, lnlike{loglike}, lnpost{logpost}")
    return logpost
