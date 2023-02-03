#!/usr/bin/env python


"""
Wrapper function called by the different algorithms.
"""


# Python Standard Library

# Other dependencies
import numpy as np

from scipy.fft import fft
from obspy.signal.filter import bandpass

# Local files
import model


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def _synthetize(param, freqmin, freqmax):
    # Synthetize
    dPP0, U, A_p, Sxx = model.synthetize(**param)

    # Ground velocity spectrum
    V = np.gradient(U, 1/param['max_freq'], axis=-1)

    # Filter waveform
    # for i, tr in enumerate(V):
    #     V[i] = bandpass(tr, freqmin, freqmax, 2*param['max_freq'])

    # Compute spectrum
    Sxx_syn = np.abs(fft(V))
    return Sxx_syn


def synthetize(param, freqmin, freqmax):
    Sxx_syns = []
    for j in range(10):
        Sxx_syn = _synthetize(param, freqmin, freqmax)
        Sxx_syns.append(Sxx_syn)
    Sxx_syn = np.array(Sxx_syns).mean(axis=0)
    return Sxx_syn


if __name__ == '__main__':
    pass
