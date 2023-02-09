#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
import xarray as xr
import ti

from pymoo.optimize import minimize


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def optimize(
    Sxx_obs: xr.DataArray,
    param: dict,
    keys: list,
    a: float,
    b: float,
    n_synth: int,
    algorithm,
    termination,
    save_history: bool,
    verbose: bool,
):
    print(Sxx_obs.t.values)

    problem = ti.moo.Problem(
        Sxx_obs=Sxx_obs, param=param, keys=keys, a=a, b=b, n_synth=n_synth
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
