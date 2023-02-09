#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
import numpy as np
import ti

from pymoo.core.problem import ElementwiseProblem

# Local files


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


class Problem(ElementwiseProblem):
    def __init__(
        self,
        Sxx_obs=None,
        param=None,
        keys=None,
        a=None,
        b=None,
        n_synth=3,
        **kwargs,
    ):
        self.Sxx_obs = Sxx_obs
        self.param = param
        self.keys = keys
        self.a = a
        self.b = b
        self.n_synth = n_synth

        self.n_stations = self.Sxx_obs.shape[0]

        self._get_limits()

        super().__init__(
            n_var=len(self.xl),
            n_obj=self.n_stations,
            n_ieq_constr=0,
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
        xl, xu, keys_problem = [], [], []
        for key in self.keys:
            lower, upper = tuple(self.param[key])

            # Get the power for the log scaled variables
            if key in ti.constants.log_params:
                lower = np.log10(lower)
                upper = np.log10(upper)

            lower = [lower]
            upper = [upper]
            key_problem = [key]

            n = 1
            if key == 'Qf':
                n = self.n_stations

            xl.extend(n*lower)
            xu.extend(n*upper)
            keys_problem.extend(n*key_problem)
        self.xl = np.array(xl)
        self.xu = np.array(xu)
        self.keys_problem = keys_problem
        return

    def _evaluate(self, x, out, *args, **kwargs):
        self.param = self.var_to_dict(x)

        # Sxx_syn = ti.synth.synthetize_avg(self.param, n=self.n_synth)
        Sxx_syn = ti.synth.synthetize(self.param)
        Sxx_syn = ti.filt.filter_spectra(Sxx_syn, self.a, self.b)

        out['F'] = []
        for i, station in enumerate(self.Sxx_obs.station):
            Sx_obs = self.Sxx_obs.sel(station=station).to_numpy()
            Sx_syn = Sxx_syn[i]
            out['F'].append(ti.target.misfit(Sx_obs, Sx_syn))
        return

    def var_to_dict(self, x):
        param = self.param.copy()

        if 'Qf' in self.keys:
            param['Qf'] = []

        for i, _x in enumerate(x):
            key = self.keys_problem[i]

            # Back to linear scale
            if key in ti.constants.log_params:
                _x = 10**_x

            # Qf must be a list, one item per station
            if key == 'Qf':
                param['Qf'].append(_x)
            else:
                param[key] = _x
        return param


def var_to_dict(param, keys, x):
    opt = param.copy()

    if 'Qf' in keys:
        opt['Qf'] = []

    for key, _x in zip(keys, x):
        # Back to linear scale
        if key in ti.constants.log_params:
            _x = 10**_x
        # Qf must be a list, one item per station
        if key == 'Qf':
            opt['Qf'].append(_x)
        else:
            opt[key] = _x
    return opt


if __name__ == '__main__':
    pass
