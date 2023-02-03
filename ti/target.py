#!/usr/bin/env python


"""
Functions related to the target of the optimization
"""


# Python Standard Library

# Other dependencies
import numpy as np

from numba import jit

# Local files


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def weight(Sxx, top_q, wmax):
    w = np.ones(Sxx.shape)
    for i, Sx in enumerate(Sxx):
        q = np.quantile(Sx, 1-top_q)
        idx = np.where(Sx > q)[0]
        w[i, idx] = np.linspace(wmax, 1, len(idx))
    return w


@jit(nopython=True)
def spectral_angle(Sx_obs, Sx_syn):
    return np.arccos(
        np.sum(Sx_obs*Sx_syn) / (
            np.sqrt(np.sum(Sx_obs**2)) * np.sqrt(np.sum(Sx_syn**2))
        )
    )


if __name__ == '__main__':
    pass
