import random
import time
from arviz import InferenceData
import xarray as xr
import arviz as az

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.fft import fft
from tqdm.auto import trange

from ..bayesian_utilities import lpost, sample_φδτ
from ..bayesian_utilities.whittle_utilities import psd_model
from ..logger import logger
from ..splines import initialise_splines
from ..utils import get_fz, get_periodogram


def gibbs_pspline_simple(
    data: np.ndarray,
    Ntotal: int,
    burnin: int,
    thin: int = 1,
    τα: float = 0.001,
    τβ: float = 0.001,
    φα: float = 1,
    φβ: float = 1,
    δα: float = 1e-04,
    δβ: float = 1e-04,
    k: int = None,
    eqSpacedKnots: bool = False,
    degree: int = 3,
    diffMatrixOrder: int = 2,
    compute_psds: bool = False,
    metadata_plotfn: str = "",
):
    """
    Gibbs sampler for the Whittle likelihood with a P-spline prior on the log spectrum.

    Returns
    -------
    samples : np.ndarray
        Array of shape (Ntotal, 3) containing the samples for φ, δ, τ
    samples_V : np.ndarray
        Array of shape (Ntotal, k, len(data)) containing the samples for the P-spline coefficients

    """
    kwargs = locals()
    raw_data = data.copy()
    data_scale = np.std(data)
    data, k = _argument_preconditions(**kwargs)
    kwargs.update({'data': data, 'k': k})
    τ0, δ0, φ0, fz, periodogram, omega = _get_initial_values(**kwargs)
    V0, knots, db_list, P = initialise_splines(data, k, degree, omega, diffMatrixOrder, eqSpacedKnots)

    # Empty lists for the MCMC samples
    n_samples = round(Ntotal / thin)
    samples = np.zeros((n_samples, 3))
    samples_V = np.zeros((n_samples, *V0.shape))
    samples[0, :] = np.array([φ0, δ0, τ0])
    samples_V[0, :] = V0

    # initial values for proposal # log likelihood trace
    lpost_trace = np.zeros(Ntotal)
    accep_frac_list = np.zeros(Ntotal)  # accept_frac of accepted proposals
    sigma = 1  # proposal distribution variance for weights
    accept_frac = 0.4  # starting value for accept_frac of accepted proposals
    Ntot_1 = Ntotal - 1
    φ, τ, δ, V = φ0, τ0, δ0, V0

    ptime = time.process_time()
    for j in trange(n_samples, desc='MCMC sampling'):

        adj = j * thin
        V_star = V.copy()
        aux = np.arange(0, k - 1)
        np.random.shuffle(aux)

        for i in range(thin):
            itr = i + adj
            args = [k, V, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, periodogram, db_list, P]
            f_store = lpost(*args)
            # 1. explore the parameter space for new V
            V, V_star, accept_frac, sigma = _tune_proposal_distribution(
                aux, accept_frac, sigma, V, V_star, f_store, args
            )
            args[1] = V
            accep_frac_list[itr] = accept_frac  # Acceptance probability
            lpost_trace[itr] = f_store  # log posterior trace
            # 2. sample new values for φ, δ, τ
            φ, δ, τ = sample_φδτ(*args)

        samples[j, :] = np.array([φ, δ, τ])
        samples_V[j, :] = V

    # remove burnin
    burn = round(burnin / thin)
    samples = samples[burn:, :]
    samples_V = samples_V[burn:, :]
    accep_frac_list = accep_frac_list[burn:]

    psd_quants = generate_psd_posterior(omega, db_list, samples[:, 2], samples_V)
    psd_quants = psd_quants * np.power(data_scale, 2)

    idata = _create_inference_object(samples, samples_V, lpost_trace, accep_frac_list, db_list, knots, periodogram, omega, raw_data)





    if metadata_plotfn:
        make_summary_plot(idata, metadata_plotfn)

    return dict(idata=idata, psd_quants=psd_quants)



def make_summary_plot(idata, metadata_plotfn):
    raw_data = idata.observed_data.raw_data.values
    omega = idata.observed_data.periodogram.frequency.values
    data_scale = np.std(raw_data)
    raw_data = raw_data / data_scale
    db_list = idata.constant_data.db_list.values
    tau_samples = idata.posterior.tau.values.flatten()
    v_samples = idata.posterior.v.values
    accept_frac = idata.sample_stats.acceptance_rate.values
    samples = np.array([
        idata.posterior.phi.values.flatten(),
        idata.posterior.delta.values.flatten(),
        idata.posterior.tau.values.flatten(),
    ]).T
    psd_quants = generate_psd_posterior(
        omega, db_list, tau_samples, v_samples
    )
    psd_quants = psd_quants * np.power(data_scale, 2)
    n, newn = len(raw_data), len(omega)
    periodogram = np.abs(np.power(fft(raw_data), 2) / (2 * np.pi * n))[0:newn]
    periodogram = periodogram * np.power(data_scale, 2)
    _plot_metadata(samples, accept_frac, psd_quants, periodogram, db_list, metadata_plotfn)


def _create_inference_object(
    posterior_samples, v_samples, lpost_trace, frac_accept, db_list, knots,
    periodogram, omega, raw_data) -> InferenceData:
    nsamp, k, chain = v_samples.shape

    ndraws = np.arange(nsamp)
    nknots = np.arange(k)
    posterior = az.dict_to_dataset(
        dict(
            phi=posterior_samples[:, 0],
            delta=posterior_samples[:, 1],
            tau=posterior_samples[:, 2],
            v=v_samples[...,0]
        ),
        coords=dict(knots=nknots, draws=ndraws),
        dims=dict(
            phi=["draws"],
            delta=["draws",],
            tau=["draws"],
            v=["draws", "knots"]
        ),
        default_dims=[],
        attrs={},
    )
    sample_stats = az.dict_to_dataset(dict(
        acceptance_rate=frac_accept,
        lp=lpost_trace
    ))
    observed_data = az.dict_to_dataset(
        dict(periodogram=periodogram[0:len(omega)], raw_data=raw_data),
        library=None,
        coords={"frequency":omega, "idx": np.arange(len(raw_data))},
        dims={'periodogram': ['frequency'], 'raw_data': ['idx']},
        default_dims=[],
        attrs={},
        index_origin=None
    )

    spline_data = az.dict_to_dataset(
        dict(knots=knots, db_list=db_list),
        library=None,
        coords={},
        dims={'knots': ['location'], 'db_list': ['PSD', 'basis']},
        default_dims=[],
        attrs={},
        index_origin=None
    )

    return InferenceData(
        posterior=posterior,
        sample_stats=sample_stats,
        observed_data=observed_data,
        constant_data=spline_data
    )


def generate_psd_posterior(freq, db_list, tau_samples, v_samples):
    nsamp = len(tau_samples)
    psds = np.zeros((nsamp, len(freq)))
    kwgs = dict(db_list=db_list, n=len(freq))
    assert v_samples.shape[0] == nsamp
    assert v_samples.shape[1] == db_list.shape[1]-1
    for i in trange(nsamp, desc='Generating PSD posterior'):
        psds[i, :] = psd_model(v=v_samples[i, :], **kwgs) * tau_samples[i]

    psd_quants = np.quantile(psds, [0.05, 0.5, 0.95], axis=0)
    assert psd_quants.shape == (3, len(freq))
    return psd_quants


def _tune_proposal_distribution(aux, accept_frac, sigma, V, V_star, f_store, args):
    k_1 = args[0] - 1

    # tunning proposal distribution
    if accept_frac < 0.30:  # increasing acceptance pbb
        sigma = sigma * 0.90  # decreasing proposal moves
    elif accept_frac > 0.50:  # decreasing acceptance pbb
        sigma = sigma * 1.1  # increasing proposal moves

    accept_count = 0  # ACCEPTANCE PROBABILITY

    # Update "V_store" (weights)
    for g in range(k_1):

        Z = np.random.normal()
        U = np.log(np.random.uniform())

        pos = aux[g]
        V_star[pos] = V[pos] + sigma * Z
        args[1] = V_star  # update V_star
        f_star = lpost(*args)

        # is the proposed V_star better than the current V_store?
        alpha1 = np.min([0, (f_star - f_store).ravel()[0]])  # log acceptance ratio
        if U < alpha1:
            V[pos] = V_star[pos]  # Accept W.star
            f_store = f_star
            accept_count += 1  # acceptance probability
        else:
            V_star[pos] = V[pos]  # reset proposal value

    accept_frac = accept_count / k_1
    return V, V_star, accept_frac, sigma  # return updated values


def _get_initial_values(data, k, φα: float = 1, φβ: float = 1, δα: float = 1e-04, δβ: float = 1e-04, **kwargs):
    τ = np.var(data) / (2 * np.pi)
    δ = δα / δβ
    φ = φα / (φβ * δ)
    fz = get_fz(data)
    periodogram = get_periodogram(fz)
    n = len(data)
    omega = 2 * np.arange(0, n / 2 + 1) / n
    return τ, δ, φ, fz, periodogram, omega


def _argument_preconditions(
    data: np.ndarray,
    Ntotal: int,
    burnin: int,
    thin: int = 1,
    τα: float = 0.001,
    τβ: float = 0.001,
    φα: float = 1,
    φβ: float = 1,
    δα: float = 1e-04,
    δβ: float = 1e-04,
    k: int = None,
    eqSpacedKnots: bool = False,
    degree: int = 3,
    diffMatrixOrder: int = 2,
    metadata_plotfn: str = None,
    **kwargs,
):
    assert data.shape[0] > 2, "data must be a non-empty np.array"
    assert burnin < Ntotal, "burnin must be less than Ntotal"
    pos_ints = np.array([thin, Ntotal, burnin])
    assert np.all(pos_ints >= 0) and np.all(pos_ints % 1 == 0), "thin, Ntotal, burnin must be +ive ints"
    assert Ntotal > 0, "Ntotal must be a positive integer"
    pos_flts = np.array([τα, τβ, φα, φβ, δα, δβ])
    assert np.all(pos_flts > 0), "τ.α, τ.β, φ.α, φ.β, δ.α, δ.β must be +ive"
    assert isinstance(eqSpacedKnots, bool), "eqSpacedKnots must be a boolean"
    assert degree in [0, 1, 2, 3, 4, 5], "degree must be between 0 and 5"
    assert diffMatrixOrder in [0, 1, 2], "diffMatrixOrder must be either 0, 1, or 2"
    assert degree > diffMatrixOrder, "penalty order must be lower than the bspline density degree"
    assert isinstance(metadata_plotfn, str), "metadata_plotdir must be a string"

    n = len(data)
    if k is None:
        k = min(round(n / 4), 40)

    if abs(np.mean(data)) > 1e-4:
        logger.exception("data must be mean-centered before fitting")

    assert k >= degree + 2, "k must be at least degree + 2"
    assert (Ntotal - burnin) / thin > k, f"Must have (Ntotal-burnin)/thin > k, atm:({Ntotal} - {burnin}) / {thin} < {k}"
    assert k - 2 >= diffMatrixOrder, "diffMatrixOrder must be lower than or equal to k-2"

    return data, k


def _plot_metadata(samples, counts, psd_quants, periodogram, db_list, metadata_plotfn):
    fig = plt.figure(figsize=(5, 8), layout="constrained")
    gs = GridSpec(5, 2, figure=fig)
    for i, p in enumerate(['φ', 'δ', 'τ']):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(samples[:, i], color=f'C{i}')
        ax.set_ylabel(p)
        ax.set_xlabel("Iteration")
        ax = fig.add_subplot(gs[i, 1])
        ax.hist(samples[:, i], bins=50, color=f'C{i}')
        ax.set_xlabel(p)
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(counts, color='C3')
    ax.set_ylabel("Frac accepted")
    ax.set_xlabel("Iteration")
    ax = fig.add_subplot(gs[3, 1])
    for i, db in enumerate(db_list.T):
        ax.plot(db, color=f'C{i}', alpha=0.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("Splines")
    ax = fig.add_subplot(gs[4, :])


    psd_up, psd_low = psd_quants[2, :], psd_quants[0, :]
    psd_x = np.arange(len(psd_up))
    ax.plot(psd_quants[1, :], color='C4', label='Posterior Median')
    ax.fill_between(psd_x, psd_low, psd_up, color='C4', alpha=0.2, label='90% CI')
    ylims = ax.get_ylim()
    ax.plot([], [], color='k', label='Periodogram', zorder=-10, alpha=0.5)
    ax.plot(periodogram, color='k', zorder=-10, alpha=0.5)
    ax.set_ylim(ylims)
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylabel("PSD")
    fig.tight_layout()
    fig.savefig(metadata_plotfn)
    plt.close(fig)
