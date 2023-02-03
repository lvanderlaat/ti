#!/usr/bin/env python


"""
"""


# Python Standard Library
from itertools import repeat
from multiprocessing import Pool

# Other dependencies
import numpy as np
import pandas as pd

# Local files


__author__ = 'Leonardo van der Laat'
__email__  = 'laat@umich.edu'


def _sample(x):
    return np.mean(np.random.choice(x, size=len(x)))


def sample(_, df, window):
    data = df.rolling(window, center=True)
    data = data.apply(_sample, raw=True, engine='numba').values
    return data


def moving_block(df, window, N, max_workers):
    args_parallel = zip(range(N), repeat(df), repeat(window))
    with Pool(max_workers) as pool:
        data = pool.starmap(sample, args_parallel)

    data = np.array(data)
    avg = pd.DataFrame(np.mean(data, axis=0), columns=df.columns, index=df.index)
    std = pd.DataFrame(np.std(data, axis=0),  columns=df.columns, index=df.index)
    return avg, std


if __name__ == '__main__':
    pass
