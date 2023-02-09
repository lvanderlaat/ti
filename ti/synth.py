#!/usr/bin/env python


"""
Wrapper function called by the different algorithms.
"""


# Other dependencies
import numpy as np
import ti

from obspy.signal.filter import lowpass
from scipy.fft import rfft
from scipy.signal.windows import tukey


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def synthetize(param):
    # Synthetize
    dPP0, st, A_p, Sxx = ti.model.synthetize(**param)

    # Ground velocity spectrum
    V = np.gradient(st, 1/2/param['max_freq'], axis=-1)

    # Filter waveform
    for i, tr in enumerate(V):
        tr *= tukey(tr.shape[-1], alpha=0.05)
        V[i] = lowpass(tr, 6, 2*param['max_freq'])

    # Compute spectrum
    Sxx_syn = np.abs(rfft(V))
    return Sxx_syn


def synthetize_avg(param, n=10):
    return np.array([synthetize(param) for _ in range(n)]).mean(axis=0)


if __name__ == '__main__':
    pass
