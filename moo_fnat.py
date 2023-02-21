
class NaturalFrequency(ElementwiseProblem):
    def __init__(self, param, fn, **kwargs):
        self.param = param
        self.keys = [k for k, v in param.items() if type(v) is list]
        self.fn = fn
        self.n_var = len(self.keys)
        self._get_limits()

        super().__init__(
            n_var=self.n_var,
            n_obj=1,
            n_ieq_constr=0,
            xl=self.xl,
            xu=self.xu,
            **kwargs,
        )

    def _get_limits(self):
        self.xl = np.empty(self.n_var)
        self.xu = np.empty(self.n_var)

        for i in range(self.n_var):
            self.xl[i], self.xu[i] = tuple(self.param[self.keys[i]])
        return

    def var_to_dict(self, x):
        param = self.param.copy()
        for i in range(self.n_var):
            param[self.keys[i]] = x[i]
        return param

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = np.abs(
            self.fn - ti.model.natural_frequency(**self.var_to_dict(x))
        )
