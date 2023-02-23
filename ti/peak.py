#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
from numba import jit
import numpy as np
import pandas as pd

# Local files


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


@jit(nopython=True)
def _extrema(y):
    idx, label, a = [0], ['min'], [y[0]]
    for i in np.arange(1, len(y)-1):
        if y[i] < y[i-1] and y[i] <= y[i+1]:
            idx.append(i)
            label.append('min')
            a.append(y[i])
        elif y[i] > y[i-1] and y[i] >= y[i+1]:
            idx.append(i)
            label.append('max')
            a.append(y[i])
    idx.append(len(y)-1)
    label.append('min')
    a.append(y[-1])
    return idx, label, a


def extrema(y):
    idx, label, a = _extrema(y)
    df = pd.DataFrame(dict(idx=idx, label=label, a=a))
    df_min = df[df.label == 'min']
    df_max = df[df.label == 'max']
    return df, df_min, df_max


if __name__ == '__main__':
    pass
