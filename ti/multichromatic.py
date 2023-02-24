#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ti

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.signal import peak_prominences

# Local files


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


class Problem(ElementwiseProblem):
    def __init__(
        self,
        Sx_obs=None,
        f=None,
        param=None,
        keys=None,
        fnat_range=None,
        sigma=None,
        **kwargs,
    ):
        self.Sx_obs = Sx_obs
        self.f = f
        self.param = param
        self.keys = keys
        self.fnat_range = fnat_range
        self.sigma = sigma

        self.n_peaks = len(fnat_range[0])

        self._get_limits()

        super().__init__(
            n_var=self.n_var,
            n_obj=1,
            n_ieq_constr=2*self.n_peaks,
            # n_ieq_constr=0,
            xl=self.xl,
            xu=self.xu,
            **kwargs,
        )

    def _get_limits(self):
        """
        Create the upper and lower limit arrays for the pymoo problem object
        together with a list of keys to go back to a dict form to pass to the
        model function
        """
        fnat_params = ti.model.get_fnat_params()

        self.n_var = 0
        self.effective_keys = []
        for key in self.keys:
            if key in fnat_params:
                self.n_var += self.n_peaks
                for i in range(self.n_peaks):
                    self.effective_keys.append(key)
            else:
                self.n_var += 1
                self.effective_keys.append(key)
        self.effective_keys = np.array(self.effective_keys)

        self.xl = np.empty(self.n_var)
        self.xu = np.empty(self.n_var)

        for i, key in enumerate(self.effective_keys):
            lower, upper = tuple(self.param[key])

            # Get the power for the log scaled variables
            if key in ti.constants.log_params:
                lower = np.log10(lower)
                upper = np.log10(upper)
            self.xl[i] = lower
            self.xu[i] = upper
        return

    def _evaluate(self, x, out, *args, **kwargs):
        param = self.var_to_dict(x)
        Sx_syn, fnat = ti.model_multi.synthetize(**param)
        Sx_syn = Sx_syn[0]
        Sx_syn = gaussian_filter(Sx_syn, sigma=self.sigma)

        out['F'] = [ti.target.misfit(self.Sx_obs, Sx_syn)]

        out['G'] = []
        for i in range(self.n_peaks):
            out['G'].append(self.fnat_range[0][i] - fnat[i])
            out['G'].append(fnat[i] - self.fnat_range[1][i])
        return

    def var_to_dict(self, x):
        param = self.param.copy()
        for i, key in enumerate(self.effective_keys):
            param[key] = x[np.where(self.effective_keys == key)]
            if key in ti.constants.log_params:
                param[key] = np.power(10, param[key])
        return param


def invert(
    i, f, Sx_obs, delta, param, keys, sigma, algorithm,
    termination, verbose, prominence_min=20
):
    print(i)
    df, df_min, df_max = ti.peak.extrema(Sx_obs)
    df_max.sort_values(by='a', inplace=True, ascending=False)

    prominences, _, _ = peak_prominences(Sx_obs/Sx_obs.max()*100, df_max.idx)
    df_max['prominence'] = prominences
    df_max.sort_values(by='prominence', inplace=True, ascending=False)
    df_max = df_max[:param['n']]

    if df_max.prominence.max() > prominence_min:
        df_max = df_max[df_max.prominence > prominence_min]

    Sx_obs = gaussian_filter(Sx_obs, sigma=sigma)

    fnat = f[df_max.idx]

    param['n'] = len(df_max)

    fnat_range = [fnat - delta, fnat + delta]

    problem = ti.multichromatic.Problem(
        Sx_obs=Sx_obs,
        f=f,
        param=param,
        keys=keys,
        fnat_range=fnat_range,
        sigma=sigma,
    )
    result = minimize(
        problem, algorithm, termination, seed=1, save_history=False,
        verbose=verbose,
    )
    if result.X is None:
        return None
    else:
        return problem.var_to_dict(result.X)


if __name__ == '__main__':
    pass
