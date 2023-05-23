import os

from pspline_psd.sample.gibbs_pspline_simple import gibbs_pspline_simple


def test_simple_example(helpers):
    data = helpers.load_raw_data()
    data = data - data.mean()


    fn = f"{helpers.OUTDIR}/sample_metadata.png"
    mcmc = gibbs_pspline_simple(
        data=data, Ntotal=1000   , burnin=200, degree=3,
        eqSpacedKnots=True, compute_psds=True, metadata_plotfn=fn
    )
    assert os.path.exists(fn)
