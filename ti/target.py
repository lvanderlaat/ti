#!/usr/bin/env python


"""
Functions related to the target of the optimization
"""


# Python Standard Library

# Other dependencies
import numpy as np

from numba import jit
from scipy.spatial import distance

# Local files


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


@jit(nopython=True)
def spectral_angle(Sx_obs, Sx_syn):
    return np.arccos(
        np.sum(Sx_obs*Sx_syn) / (
            np.sqrt(np.sum(Sx_obs**2)) * np.sqrt(np.sum(Sx_syn**2))
        )
    )


def misfit(Sx_obs, Sx_syn):
    dist = distance.euclidean(Sx_obs, Sx_syn)
    angl = spectral_angle(Sx_obs, Sx_syn)

    x = dist*np.cos(angl)
    y = dist*np.sin(angl)

    return np.sqrt(x**2 + y**2)


if __name__ == '__main__':
    pass
