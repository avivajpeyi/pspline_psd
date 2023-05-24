import numpy as np
from pspline_psd.utils import get_fz, get_periodogram
from pspline_psd.logger import logger


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
