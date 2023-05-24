import time
import numpy as np
from tqdm.auto import trange

from ..bayesian_utilities import lpost, sample_φδτ
from ..splines import _get_initial_spline_data
from .sampler_initialisation import _get_initial_values, _argument_preconditions


from .sampling_result import Result


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
) -> Result:
    """
    Gibbs sampler for the Whittle likelihood with a P-spline prior on the log spectrum.
    """
    kwargs = locals()
    data_scale = np.std(data)
    raw_data = data.copy()
    data, k = _argument_preconditions(**kwargs)
    kwargs.update({'data': data, 'k': k})
    τ0, δ0, φ0, fz, periodogram, omega = _get_initial_values(**kwargs)
    V0, db_list, P, knots = _get_initial_spline_data(periodogram, k, degree, omega, diffMatrixOrder, eqSpacedKnots)

    # Empty lists for the MCMC samples
    n_samples = round(Ntotal / thin)
    samples = np.zeros((n_samples, 3))
    samples_V = np.zeros((n_samples, *V0.shape))
    samples[0, :] = np.array([φ0, δ0, τ0])
    samples_V[0, :] = V0

    # initial values for proposal
    lpost_trace = np.zeros(Ntotal)  # log likelihood trace
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
            lpost_trace[itr] = f_store  # log post trace
            # 2. sample new values for φ, δ, τ
            φ, δ, τ = sample_φδτ(*args)

        samples[j, :] = np.array([φ, δ, τ])
        samples_V[j, :] = V

    # remove burnin
    burn = round(burnin / thin)
    samples = samples[burn:, :]
    samples_V = samples_V[burn:, :]
    accep_frac_list = accep_frac_list[burn:]
    lpost_trace = lpost_trace[burn:]

    sampling_result = Result.compile_idata_from_sampling_results(
        posterior_samples=samples, v_samples=samples_V,
        lpost_trace=lpost_trace, frac_accept=accep_frac_list,
        db_list=db_list, knots=knots,
        periodogram=periodogram, omega=omega, raw_data=raw_data
    )
    if metadata_plotfn:
        sampling_result.make_summary_plot(metadata_plotfn)

    return sampling_result


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
