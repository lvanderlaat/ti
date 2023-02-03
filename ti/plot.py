#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import colors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter


# Local files
import model
import constants
import parameters


__author__ = 'Leonardo van der Laat'
__email__  = 'laat@umich.edu'


mm = 1/25.6
UNIT = dict(displacement='m', velocity='m/s')


def ssam(
    t, f, Sxx, lognorm=False, qmin=0.025, qmax=0.995, yscale='log',
    smooth_window='6H', normalize=True
):
    # Sxx = Sxx/Sxx.max(axis=0)
    rsam = Sxx.mean(axis=0)

    label = f'Ground velocity [m/s]'
    cbar_label = label

    if normalize:
        Sxx = Sxx/Sxx.max(axis=0)
        cbar_label = 'Normalized amplitude'

    # Figure
    fig, ax = plt.subplots(figsize=(190*mm, 100*mm))
    ax.set_ylabel('Frequency [Hz]')

    vmin, vmax = np.quantile(Sxx, qmin), np.quantile(Sxx, qmax)

    if lognorm:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(t, f, Sxx, norm=norm)

    ax.set_yscale(yscale)
    ax.set_ylim(1e-1, f.max())

    divider = make_axes_locatable(ax)
    cax = ax.inset_axes([0.0, 1.05, 1, 0.05], transform=ax.transAxes)
    plt.colorbar(im, cax=cax, orientation='horizontal', label=cbar_label)
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')

    # Pseudo-RSAM
    ax1 = ax.twinx()
    ax1.scatter(t, rsam, s=2, c='w')
    _rsam = pd.Series(rsam, index=t)
    _rsam = _rsam.rolling(smooth_window, center=True).median()
    ax1.plot(t, _rsam, lw=2, c='w')
    ax1.set_ylim(_rsam.min()*0.9, _rsam.max()*1.1)
    ax1.set_ylabel(label)
    ax1.set_yscale('log')

    fig.tight_layout()
    return fig


def _ssam(ax, cax, t, f, Sxx, qmin, qmax, lognorm, cmap, label, centered=False):
    vmin, vmax = np.quantile(Sxx, qmin), np.quantile(Sxx, qmax)
    if centered:
        vabs = max(abs(vmin), abs(vmax))
        vmin = -vabs
        vmax = vabs
        linthresh = np.quantile(np.abs(Sxx), qmax)

    if lognorm and not centered:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif lognorm and centered:
        norm = colors.SymLogNorm(linthresh, vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap)
    # ax.contour(t, f, Sxx, norm=norm, levels=[0.5], colors='k')

    plt.colorbar(im, cax=cax, label=label)

    # ax.set_yscale('log')
    # ax.set_ylim(1e-1, f.max())

    ax.set_ylim(f.min(), 7)
    return


def _rsam(ax, t, rsam, c, label, window='6H'):
    ax.scatter(t, rsam, s=2, c=c)
    _rsam = pd.Series(rsam, index=t)
    _rsam = _rsam.rolling('6H', center=True).median()
    ax.plot(t, _rsam, lw=2, c=c, label=label)
    # ax.set_yscale('log')
    return


def obs_vs_synth(
    t, f, Sxx_obs, Sxx_syn, measurement='velocity',
    lognorm=False, qmin=0.005, qmax=0.995, normalize=True
):
    unit = UNIT[measurement]
    label = f'Ground {measurement} [{unit}]'
    cbar_label = label
    cbar_label_diff = label

    rsam_obs = Sxx_obs.sum(axis=0)
    rsam_syn = Sxx_syn.sum(axis=0)
    error = np.abs(rsam_obs-rsam_syn)
    Sxx_err = Sxx_obs - Sxx_syn

    if normalize:
        Sxx_obs = Sxx_obs/Sxx_obs.max(axis=0)
        Sxx_syn = Sxx_syn/Sxx_syn.max(axis=0)
        cbar_label = 'Normalized amplitude'

    fig = plt.figure(figsize=(190*mm, 240*mm))
    gs = GridSpec(
        nrows=4,
        ncols=2,
        left=.12,
        bottom=.04,
        right=.88,
        top=.95,
        wspace=.02,
        hspace=.3,
        width_ratios=[1, 0.05]
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)

    cax2 = fig.add_subplot(gs[1, 1])
    cax3 = fig.add_subplot(gs[2, 1])
    cax4 = fig.add_subplot(gs[3, 1])

    ax1.set_title('SSAM sum')
    ax2.set_title('SSAM Observed')
    ax3.set_title('SSAM Synthetic')
    ax4.set_title('SSAM Observed-Synthetic')


    ax1.set_ylabel(label)
    for ax in [ax2, ax3, ax4]:
        ax.set_ylabel('Frequency [Hz]')

    _rsam(ax1, t, rsam_obs, 'k', 'Observed')
    _rsam(ax1, t, rsam_syn, 'r', 'Synthetic')
    ax1.set_ylim(0, max(rsam_obs.max(), rsam_syn.max())*1.1)

    ax1.plot(t, error, alpha=0.5, c='gray', label='Absolute error')
    ax1.legend()

    _ssam(ax2, cax2, t, f, Sxx_obs, qmin, qmax, False, 'turbo', cbar_label)
    _ssam(ax3, cax3, t, f, Sxx_syn, qmin, qmax, False, 'turbo', cbar_label)
    _ssam(ax4, cax4, t, f, Sxx_err, qmin, qmax, False, 'coolwarm',
          cbar_label_diff, centered=True)
    return fig


def optimized_spectrum(f, Sx_obs, Sx_syn, Sx_obs_s, Sx_syn_s, top_q):
    normalize = False
    if normalize:
        Sx_obs /= Sx_obs.max()
        Sx_syn /= Sx_syn.max()
        Sx_obs_s /= Sx_obs_s.max()
        Sx_syn_s /= Sx_syn_s.max()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    ax.plot(f, Sx_obs, c='k', lw=0.1, label='Observation')
    ax.plot(f, Sx_syn, c='r', lw=0.1, label='Synthetic')

    # Smooth spectra
    ax.plot(f, Sx_obs_s, c='k', lw=1.5, label='Smoothed observation', alpha=0.7)
    ax.plot(f, Sx_syn_s, c='r', lw=1.5, label='Smoothed synthetic', alpha=0.7)

    # ax.axhline(np.quantile(Sx_obs_s, 1-top_q), lw=0.5, ls='--', label='Top-q')

    ax.set_ylabel(f'Ground velocity [m/s]')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_xlim(0.1, 25)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.legend()

    # ax1 = ax.twinx()
    # ax1.plot(f, w, c='gray', alpha=0.5)
    # ax1.set_xscale('log')
    # ax1.grid('off')

    fig.tight_layout()
    return fig


def surfaces(df, estimated):
    import os

    output_dir = '/nfs/turbo/lsa-zspica/work/laat/Halema/porous_gas_flow/OPTIMIZATION/surface/'
    n_params = len(estimated)

    p = df.p
    error = df.error
    df = df[estimated]

    vmin = np.quantile(error, 0.0001)
    vmax = np.quantile(error, 0.9)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    for i in range(n_params-1):
        for j in range(i+1, n_params):
            pi = df.columns[i]
            pj = df.columns[j]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.tricontourf(df[pi], df[pj], error)
            ax.set_xlabel(pi)
            ax.set_ylabel(pj)
            if pi in constants.log_params:
                ax.set_xscale('log')
            if pj in constants.log_params:
                ax.set_yscale('log')
            fig.savefig(os.path.join(output_dir, f'{pi}_{pj}.png'), dpi=250)
            plt.close()


def param_hist(df, estimated, nbins=50):
    p = parameters.as_dict()

    for key in estimated:
        df[key] *= p[key]['conversion']
    optimal = df[df.error == df.error.min()].iloc[0]
    error = df.error
    df = df[estimated]

    n = len(df.columns) + 1
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n/nrows))

    fig, _axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(90*nrows*mm, 90*ncols*mm)
    )
    i = 0
    for _axes_ in _axes:
        for ax in _axes_:
            if i == n - 1:
                bins = np.logspace(
                    np.log10(error.min()),
                    np.log10(error.max()),
                    nbins
                )
                ax.hist(error, bins=bins, lw=0.5, ec='gray')
                ax.set_xlabel('Error')
                ax.set_ylabel('Count')
                ax.set_xscale('log')
                i += 1
                continue
            if i >= n:
                ax.remove()
                continue
            column = estimated[i]

            # TODO function
            _p = p[column]
            unit = _p['unit'].replace('\\', '')
            ylabel = _p['math_notation']
            if unit != ' ':
                ylabel += f' [{unit}]'
            ax.set_xlabel(ylabel)

            ax.set_ylabel('Error')

            df['error'] = error
            df.sort_values(by=column, inplace=True)
            roll = df.error.rolling(100, center=True)
            avg = roll.mean()

            if column in constants.log_params:
                ax.set_xscale('log')
            ax.scatter(df[column], df.error, s=1e-2)
            ax.plot(df[column], avg, c='r', lw=2)
            # ax.plot(df[column], avg-2*std, c='r', lw=0.5)
            # ax.plot(df[column], avg+2*std, c='r', lw=0.5)
            ax.set_yscale('log')
            ax.scatter(optimal[column], optimal.error, s=100, c='r', marker='*')
            i += 1
    fig.tight_layout()
    return fig


def param_timeseries(avg, std, c):
    p = parameters.as_dict()

    nrows = len(avg.columns)

    fig, axes = plt.subplots(nrows=nrows, figsize=(150*mm, 240*mm), sharex=True)
    fig.subplots_adjust(left=0.15, bottom=0.04, top=0.98, hspace=.2, right=0.95)
    for i in range(nrows):
        ax = axes
        if nrows > 1:
            ax = ax[i]
        ax.set_xlim((avg.index.min(), avg.index.max()))

        column = avg.columns[i]

        _p = p[column]
        unit = _p['unit'].replace('\\', '')
        ylabel = _p['math_notation']
        if unit != ' ':
            ylabel += f' [{unit}]'
        ax.set_ylabel(ylabel)

        avg[column] *= _p['conversion']
        std[column] *= _p['conversion']

        if _p['log']:
            ax.set_yscale('log')

        ax.fill_between(
            avg.index,
            avg[column]-2*std[column],
            avg[column]+2*std[column],
            lw=0, fc='r', alpha=0.5
        )
        for j in range(2):
            ax.axhline(
                float(c.model[column][j])*_p['conversion'],
                c='k', ls='--', lw=0.5
            )

        ax.plot(avg[column], lw=2, c='k')

        # ax.set_ylim(
        #     float(c.model[column][0])*_p['conversion'],
        #     float(c.model[column][1])*_p['conversion']
        # )
    return fig


def convergence(n_evals, min_err, lw=0.1):
    min_err = np.array(min_err).T
    min_err /= min_err.min(axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(n_evals, min_err, lw=lw)
    ax.set_title('Convergence')
    ax.set_yscale('log')
    return fig


if __name__ == '__main__':
    pass
