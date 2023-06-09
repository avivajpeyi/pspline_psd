"""Pytest setup"""
import glob
import os.path
from collections import namedtuple
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest

DIR = Path(__file__).parent
DATA_DIR = DIR / 'data'
DATA_PATHS = dict(
    data_0=DATA_DIR / 'data_0.Rdata',
    raw_data=DATA_DIR / 'data.txt',
    db_list=DATA_DIR / 'db_list.txt',
    tau=DATA_DIR / 'tau_trace.txt',
    v=DATA_DIR / 'v_trace.txt',
    ll=DATA_DIR / 'll_trace.txt',
)


def pytest_configure(config):
    # NB this causes `pspline_psd/__init__.py` to run
    import pspline_psd  # noqa


def load_rdata(path):
    import rpy2.robjects as robjects

    robjects.r['load'](str(path))
    d = dict(data=np.array(robjects.r['data']))
    d.update(dict(**r_obj_as_dict(robjects.r['mcmc'])))
    return d


def r_obj_as_dict(vector):
    import rpy2.robjects as robjects

    """Convert an RPy2 ListVector to a Python dict"""
    result = {}
    r2np_types = [robjects.FloatVector, robjects.IntVector, robjects.Matrix, robjects.vectors.FloatMatrix]
    for i, name in enumerate(vector.names):
        if isinstance(vector[i], robjects.ListVector):
            result[name] = r_obj_as_dict(vector[i])
        elif len(vector[i]) == 1:
            result[name] = vector[i][0]
        elif type(vector[i]) in r2np_types:
            result[name] = np.array(vector[i])
        else:
            result[name] = vector[i]
    return result


def mkdir(path):
    path = str(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Helpers:
    SAVE_PLOTS = True
    OUTDIR = mkdir(os.path.join(DIR, 'test_output'))

    @staticmethod
    def load_raw_data():
        return np.loadtxt(DATA_PATHS['raw_data'])

    @staticmethod
    def load_data_0():
        return load_rdata(DATA_PATHS['data_0'])

    @staticmethod
    def load_db_list():
        return np.loadtxt(DATA_PATHS['db_list'])

    @staticmethod
    def load_v():
        return np.loadtxt(DATA_PATHS['v'])

    @staticmethod
    def load_ll():
        return np.loadtxt(DATA_PATHS['ll'])

    @staticmethod
    def load_tau():
        return np.loadtxt(DATA_PATHS['tau'])

    @staticmethod
    def plot_comparison(expected, actual, label):
        fig, (ax0, ax1) = plt.subplots(
            2,
            1,
            gridspec_kw={"height_ratios": [3, 1], "wspace": 0, "hspace": 0},
            sharex=True,
        )
        ax0.plot(expected, label='True', color="C0")
        ax0.plot(actual, label='computed', color="C1", ls='--')
        ax0.legend()
        try:
            ax1.errorbar(
                [i for i in range(len(expected))],
                [0] * len(expected),
                yerr=abs(expected - actual),
                fmt=".",
                ms=0.5,
                color='k',
            )
        except Exception as e:
            print(e)
        ax1.set_xlabel('index')
        ax1.set_ylabel(r"$\delta$" + label)
        ax0.set_ylabel(label)
        fig.tight_layout()
        if Helpers.SAVE_PLOTS:
            fig.savefig(os.path.join(Helpers.OUTDIR, f'{label}.png'), dpi=300)
        return fig






@pytest.fixture
def helpers():
    return Helpers
