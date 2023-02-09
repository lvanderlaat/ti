#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
import numpy as np
import pandas as pd
import ti

from scipy.stats import loguniform

# Local files


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def get_samples(param: dict, n: int, channels: pd.DataFrame):
    _df = pd.DataFrame([param]*n)
    for key, value in param.items():
        if type(value) is list:
            low, high = tuple(param[key])
            function = np.random.uniform
            if key in ti.constants.log_params:
                function = loguniform.rvs
            _df[key] = function(low, high, size=n)
    _df['xr'] = [channels.x.tolist()]*n
    _df['yr'] = [channels.y.tolist()]*n
    _df['zr'] = [channels.z.tolist()]*n
    return _df.to_dict('records')


if __name__ == '__main__':
    pass
