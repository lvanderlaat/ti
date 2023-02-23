#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
import numpy as np
import ti

from pymoo.core.problem import ElementwiseProblem
from scipy.ndimage import gaussian_filter

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
        sigma=None,
        fnat_range=None,
        perc_diff_max=None,
        **kwargs,
    ):
        self.Sx_obs = Sx_obs
        self.f = f
        self.param = param
        self.keys = keys
        self.sigma = sigma
        self.fnat_range = fnat_range
        self.perc_diff_max = perc_diff_max

        fnat = (fnat_range[0] + fnat_range[1])/2
        self.idx = np.argmin(np.abs(self.f - fnat))

        self.n_var = len(self.keys)

        self._get_limits()

        super().__init__(
            n_var=len(self.xl),
            n_obj=1,
            n_ieq_constr=3,
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
        self.xl = np.empty(self.n_var)
        self.xu = np.empty(self.n_var)

        for i, key in enumerate(self.keys):
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
        Sx_syn, fnat = ti.synth.synthetize(param)
        Sx_syn = Sx_syn[0]
        Sx_syn = gaussian_filter(Sx_syn, sigma=self.sigma)

        perc_diff = 100*(
            np.abs(Sx_syn[self.idx] - self.Sx_obs[self.idx]) /
            np.mean([Sx_syn[self.idx], self.Sx_obs[self.idx]])
        )

        out['F'] = [ti.target.misfit(self.Sx_obs, Sx_syn)]

        out['G'] = [
            self.fnat_range[0] - fnat,
            fnat - self.fnat_range[1],
            perc_diff - self.perc_diff_max
        ]
        return

    def var_to_dict(self, x):
        param = self.param.copy()
        for i, key in enumerate(self.keys):
            if key in ti.constants.log_params:
                param[key] = 10**x[i]
            else:
                param[key] = x[i]
        return param


def var_to_dict(param, keys, x):
    opt = param.copy()
    for key, _x in zip(keys, x):
        if key in ti.constants.log_params:
            opt[key] = 10**_x
        else:
            opt[key] = _x
    return opt


if __name__ == '__main__':
    pass
