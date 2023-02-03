#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
import xarray as xr

from pymoo.optimize import minimize

# Local files
import moo
import target


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def optimize(
    Sxx_obs: xr.DataArray,
    param: dict,
    keys: list,
    freqmin: float,
    freqmax: float,
    a: float,
    b: float,
    top_q: float,
    wmax: float,
    algorithm,
    termination,
    save_history: bool,
    verbose: bool,
):
    print(Sxx_obs.t.values)

    # Get weights
    w = target.weight(Sxx_obs.to_numpy(), top_q, wmax)

    problem = moo.Problem(
        Sxx_obs=Sxx_obs,
        param=param,
        keys=keys,
        freqmin=freqmin,
        freqmax=freqmax,
        a=a,
        b=b,
        w=w,
    )

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=save_history,
        verbose=verbose,
    )
    return result


if __name__ == '__main__':
    pass
