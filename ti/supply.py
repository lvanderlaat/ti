#!/usr/bin/env python


"""
A function for generating peridic supply of gas, for harmonic tremor.
"""


# Python Standard Library

# Other dependencies
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import truncnorm

# Local files


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def periodic_supply(tau, period, sigma):
    n = int(tau/period)

    n_cycles = int(np.ceil(N/n))

    t0 = []
    for i in range(n_cycles):
        t0.append(np.linspace(0, tau-period, n))
    t0 = np.sort(np.array(t0).flatten())[:N]

    X = truncnorm(-0.5/sigma, 0.5/sigma, loc=0, scale=sigma)
    dev = X.rvs(N)*period
    t0 += dev + period/2
    return t0


if __name__ == '__main__':
    tau = 50
    N = 5000
    Q = 1000
    period = 10
    sigma = 0.15


    qn_ave = Q*tau/N
    qn = qn_ave + np.random.normal(0, 0.20*qn_ave, (N))
    qn[np.where(qn <= 0)] = qn_ave



    if sigma == 1:
        t0 = np.sort(np.random.rand(N)*tau)
    else:
        t0 = periodic_supply(tau, period, sigma)

    plt.scatter(t0, qn, s=0.1)
    plt.axvline(0)
    plt.axvline(tau)
    plt.show()
