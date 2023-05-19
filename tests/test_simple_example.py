import os

from pspline_psd.sample.gibbs_pspline_simple import gibbs_pspline_simple


def test_simple_example(helpers):
    data = helpers.load_raw_data()

    fn = f"{helpers.OUTDIR}/sample_metadata.png"
    gibbs_pspline_simple(
        data=data, Ntotal=1000, burnin=100, degree=3, k=50, eqSpacedKnots=True, compute_psds=True, metadata_plotfn=fn
    )
    assert os.path.exists(fn)
