#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
import xarray as xr
from scipy import signal

# Local files
from scipy.signal.windows import tukey


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def filter_spectra(Sx, a, b, alpha=0.05):
    Sx_filt = signal.filtfilt(b, a, Sx)
    # Sx_filt *= tukey(Sx_filt.shape[-1], alpha=0.05)
    return Sx_filt


def freq_domain(a: float, b: float, dataarray: xr.DataArray):
    _Sxx = signal.filtfilt(b, a, dataarray)
    _Sxx *= signal.windows.tukey(_Sxx.shape[2], alpha=0.05)
    data_filt = dataarray.copy()
    data_filt.data = _Sxx
    return data_filt


if __name__ == '__main__':
    pass
