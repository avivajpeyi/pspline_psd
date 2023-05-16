import os

from pspline_psd.sample.gibbs_pspline_simple import gibbs_pspline_simple


def test_simple_example(helpers):
    data = helpers.load_data_0()['data']
    fn = f"{helpers.OUTDIR}/sample_metadata.png"
    gibbs_pspline_simple(
        data=data,
        Ntotal=1000,
        burnin=50,
        degree=3,
        eqSpacedKnots=True,
        compute_psds=True,
        metadata_plotfn=fn
    )
    assert os.path.exists(fn)
