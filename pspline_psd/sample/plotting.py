import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from .post_processing import generate_psd_quantiles

from arviz import InferenceData





def _plot_metadata(post_samples, counts, psd_quants, periodogram, db_list, knots, v, metadata_plotfn):
    fig = plt.figure(figsize=(5, 8), layout="constrained")
    gs = plt.GridSpec(5, 2, figure=fig)
    for i, p in enumerate(['φ', 'δ', 'τ']):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(post_samples[:, i], color=f'C{i}')
        ax.set_ylabel(p)
        ax.set_xlabel("Iteration")
        ax = fig.add_subplot(gs[i, 1])
        ax.hist(post_samples[:, i], bins=50, color=f'C{i}')
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


    psd_up, psd_low = psd_quants[2, 1:], psd_quants[1, 1:]
    psd_x = np.linspace(0,1, len(psd_up))
    ax.plot(psd_x, psd_quants[0, 1:], color='C4')
    ax.fill_between(psd_x, psd_low, psd_up, color='C4', alpha=0.2, label='Posterior (median, 90% CI)')
    ylims = ax.get_ylim()
    ax.plot([], [], color='k', label='Periodogram', zorder=-10, alpha=0.5)
    ax.plot(np.linspace(0,1, len(periodogram)), periodogram, color='k', zorder=-10, alpha=0.5)
    # plot the knots vs V here as well
    ax.plot(knots, v.flatten()[1:], 'x', color='C3', label='Knots', alpha=0.25)
    ax.set_ylim(ylims)
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylabel("PSD")
    fig.tight_layout()
    fig.savefig(metadata_plotfn)
    plt.close(fig)
