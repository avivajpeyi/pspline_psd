import arviz as az
from arviz import InferenceData
import numpy as np
from .plotting import _plot_metadata
from .post_processing import generate_psd_posterior
from scipy.fft import fft


class Result:
    def __init__(self, idata):
        self.idata = idata.to_dict()

    @classmethod
    def compile_idata_from_sampling_results(
        cls,
        posterior_samples, v_samples, lpost_trace, frac_accept, db_list, knots,
        periodogram, omega, raw_data) -> "Result":
        nsamp, k, _ = v_samples.shape

        ndraws = np.arange(nsamp)
        nknots = np.arange(k)
        posterior = az.dict_to_dataset(
            dict(
                phi=posterior_samples[:, 0],
                delta=posterior_samples[:, 1],
                tau=posterior_samples[:, 2],
                v=v_samples
            ),
            coords=dict(knots=nknots, draws=ndraws),
            dims=dict(
                phi=["draws"],
                delta=["draws", ],
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
            coords={"frequency": omega, "idx": np.arange(len(raw_data))},
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

        return cls(InferenceData(
            posterior=posterior,
            sample_stats=sample_stats,
            observed_data=observed_data,
            constant_data=spline_data
        ))

    @property
    def post_samples(self):
        post = self.idata['posterior']
        post_samples = np.array([post["phi"], post["delta"], post["tau"]]).T
        return post_samples

    @property
    def v(self):
        return self.idata['posterior']['v']

    @property
    def omega(self):
        return self.idata['coords']['frequency']

    @property
    def db_list(self):
        return self.idata['constant_data']['db_list']

    @property
    def sample_stats(self):
        return self.idata['sample_stats']

    def make_summary_plot(self, fn: str):
        raw_data = self.idata["observed_data"]["raw_data"]
        data_scale = np.std(raw_data)
        raw_data = raw_data / data_scale

        accept_frac = self.sample_stats["acceptance_rate"].flatten()

        psd_quants = self.psd_quantiles * np.power(data_scale, 2)
        n, newn = len(raw_data), len(self.omega)
        periodogram = np.abs(np.power(fft(raw_data), 2) / (2 * np.pi * n))[0:newn]
        periodogram = periodogram * np.power(data_scale, 2)

        _plot_metadata(self.post_samples, accept_frac, psd_quants, periodogram, self.db_list, fn)

    @property
    def psd_quantiles(self):
        """return quants if present, else compute cache and return """
        # if attribute exists return
        if not hasattr(self, '_psd_quant'):
            self._psd_quant = generate_psd_posterior(
                self.omega, self.db_list, self.post_samples[:, 2], self.v
            )
        return self._psd_quant
