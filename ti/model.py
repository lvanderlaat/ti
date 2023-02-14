#!/usr/bin/env python


"""
Script to generate synthetic seismograms according to:

    Girona, T., Caudron, C., and C. Huber (2019), "Origin of shallow
    volcanic tremor I: the dynamics of gas pockets trapped beneath thin
    permeable media", JGR.

For questions and comments regarding the model: tarsilo.girona@jpl.nasa.gov
"""

import matplotlib.pyplot as plt
import numpy as np

from numpy import pi, exp, heaviside, sqrt
from numba import jit
from scipy.fft import irfft


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


BETA = 1295
ALPHA = -0.374
RATIO = 0.73


def time_freq_arrays(tau, max_freq):
    sampling_rate = max_freq * 2
    npts = int(sampling_rate*tau)
    delta = 1 / sampling_rate
    return np.linspace(0, tau, npts), np.fft.rfftfreq(npts, delta)


@jit(nopython=True)
def auxiliary_parameters(mu_g, T, M, Rg, Q, S, D, L, kappa, phi, Pex):
    beta_a = S*phi*M/(Rg*T)
    beta_b = mu_g*phi/(kappa*(Pex-Rg*T*Q**2/(S**2*phi**2*M*Pex)))
    beta_c = Pex*M/(Rg*T*(Pex-Rg*T*Q**2/(S**2*phi**2*M*Pex)))
    beta_d = 2*Q/(S*phi*(Pex-Rg*T*Q**2/(S**2*phi**2*M*Pex)))
    beta_e = S*M*D/(Rg*T)

    P0 = Pex+mu_g*Rg*T*Q*L/(S*kappa*M*(Pex-Rg*T*Q**2/(S**2*phi**2*M*Pex)))
    return beta_a, beta_b, beta_c, beta_d, beta_e, P0


@jit(nopython=True)
def harmonic_oscillator_coefficients(
    L, beta_a, beta_b, beta_c, beta_d, beta_e
):
    GAMMA0 = 1
    GAMMA1 = (
        2*(beta_a * beta_d+beta_b * beta_e)*L+beta_a * beta_b * L**2
    )/(2*beta_a)
    GAMMA2 = (2*beta_c * beta_e * L+beta_a * beta_c * L**2)/(2*beta_a)
    gamma0 = beta_b * L/beta_a
    gamma1 = beta_c*L/beta_a
    return GAMMA0, GAMMA1, GAMMA2, gamma0, gamma1


@jit(nopython=True)
def _natural_frequency(GAMMA1, GAMMA2, gamma0, gamma1):
    return sqrt(
        (
            sqrt(
                (GAMMA2*gamma0**2 + gamma1**2)**2-GAMMA1**2*gamma0**2*gamma1**2
            ) - GAMMA2*gamma0**2
        )/(GAMMA2*gamma1**2)
    )/(2*pi)


@jit(nopython=True)
def critical_thickness(
    T, M, Rg, S, L, beta_a, beta_b, beta_c, beta_d, beta_e
):
    a0 = (beta_a*beta_b*L**2+2*beta_a*beta_d*L)**2*beta_b**2-4*beta_a**2*beta_b**2*beta_c*L**2-4*beta_a**2*beta_c**2
    a1 = 4*(beta_a*beta_b*L**2+2*beta_a*beta_d*L)*beta_b**3*L-8*beta_a*beta_b**2*beta_c*L
    a2 = 4*beta_b**4*L**2
    Dcrit = (Rg*T/(S*M))*(-a1+sqrt(a1**2-4*a0*a2))/(2*a2)
    return Dcrit


@jit(nopython=True)
def volatiles_supply(Q, tau, N):
    # instant at which impulses occur. Uniformly distributed random times
    t0 = np.random.rand(N)*tau
    t0 = np.sort(t0)

    # average mass of the bubbles that burst at the top of the magma column
    qn_ave = Q*tau/N
    # Mass contained in each bubble that bursts.
    qn = qn_ave + np.random.normal(0, .2*qn_ave, (N))
    # Avoid impulses with negative mass.
    qn[np.where(qn <= 0)] = qn_ave
    return t0, qn


def pressure_time_domain(
    Pex, max_freq, tau, N, P0, gamma0, gamma1, GAMMA0, GAMMA1, GAMMA2, t0, qn
):
    t = np.arange(0, tau, 1/max_freq)
    # Pressure evolution in the gas pocket in the time domain (equation 15)
    DP_old = 0
    GAMMA = GAMMA1/(2*GAMMA2)
    OMEGA = sqrt(np.abs((4*GAMMA0*GAMMA2-GAMMA1**2)/(4*GAMMA2**2)))
    aux = np.empty((len(t), N))
    for k in range(N):
        if 4*GAMMA0*GAMMA2 >= GAMMA1**2:
            aux[:, k] = qn[k] * heaviside(t-t0[k], 0.5) * \
                    exp(-GAMMA*(t-t0[k])) * \
                    (
                        ((gamma0-gamma1*GAMMA)/(OMEGA*GAMMA2)) *
                        np.sin(OMEGA*(t-t0[k]))+(gamma1/GAMMA2) *
                        np.cos(OMEGA*(t-t0[k]))
                    )
        else:
            aux[:, k] = qn[k] * heaviside(t-t0[k], 0.5) * \
                    exp(-GAMMA*(t-t0[k])) * \
                    (
                        ((gamma0-gamma1*GAMMA)/(OMEGA*GAMMA2)) *
                        np.sinh(OMEGA*(t-t0[k]))+(gamma1/GAMMA2) *
                        np.cosh(OMEGA*(t-t0[k]))
                    )
        DP = DP_old + aux[:, k]
        DP_old = DP
    DP = DP + Pex - P0
    return DP


@jit(nopython=True)
def pressure_freq_domain(f, t0, qn, gamma0, gamma1, GAMMA0, GAMMA1, GAMMA2):
    """
    Calculation of the pressure evolution in the gas pocket and ground
    displacement in the frequency domain [using equation (C11)]

    Note that from the frequency domain we calculate the steady-state pressure
    evolution, and thus we do not account for the transient state. That
    is why the pressure time series calculated with this approach does not
    exactly coincide with the pressure time series calculated directly in the
    time domain (equation (15)) during the first seconds of simulation.

    Note: I tried vectorizing this function, but numba.jit + for loop is faster
    """
    omega = 1j*2*pi*f
    n = len(omega)
    A_res = np.empty(n, dtype='complex')
    A_exc = np.empty(n, dtype='complex')
    A_p = np.empty(n, dtype='complex')
    for i in range(n):
        o = omega[i]
        A_res[i] = (gamma0*o**0 + gamma1*o**1) / \
                   (GAMMA0*o**0+GAMMA1*o**1+GAMMA2*o**2)
        A_exc[i] = np.sum(qn*exp(-o*t0))
        A_p[i] = A_res[i]*A_exc[i]
    return A_res, A_exc, A_p


@jit(nopython=True)
def ground_freq_domain(f, S, distance, rho_s, Qf, A_res, A_exc):
    omega = 2*pi*f
    vc = BETA*(omega/(2*pi))**ALPHA    # phase velocity
    vu = RATIO*vc                      # group velocity
    A_path = exp(1j*((omega/vc)*distance+pi/4)) * \
                (1/(8*rho_s*vc**2*vu)) * sqrt(2*vc*omega/(pi*distance)) * \
                 exp(-omega*distance/(2*vu*Qf))
    A_path[0] = 0
    u_z = S*A_res*A_exc*A_path
    return u_z


def synthetize(
    # Simulation parameters
    tau=50,         # seconds of simulation
    max_freq=25,    # maximum frequency to be reached in the simulation
    # Source location
    xs=0,
    ys=0,
    zs=-1e3,
    # Gas properties
    N=1000,         # number of mass impulses in tau seconds
    mu_g=1e-5,      # gas viscosity
    T=1000+273.15,  # gas temperature
    M=0.018,        # molecular weight of gas (water vapor)
    Rg=8.3145,      # ideal gas constant
    Q=2,            # mean gas flux
    # Conduit geometry
    R=25,           # conduit radius
    S=None,         # conduit section
    # Gas pocket
    D=0.03,         # gas pocket thickness
    # Cap properties
    L=20,           # thickness of the cap
    kappa=1e-8,     # permeability of the cap
    phi=1e-4,       # porosity of the cap
    # External pressure
    Pex=101325,     # external pressure
    # Path and receivers
    xr=[-1e3, 1e3],    # Stations x-coordinate
    yr=[0, 0],         # Stations x-coordinate
    zr=[0, 0],         # Stations x-coordinate
    Qf=[10, 100],     # Quality factors for each station
    rho_s=3000,       # density of the medium of propagation
):
    N = int(N)

    t, f = time_freq_arrays(tau, max_freq)

    if S is None or np.isnan(S):
        S = pi*R**2

    t0, qn = volatiles_supply(Q, tau, N)

    beta_a, beta_b, beta_c, beta_d, beta_e, P0 = auxiliary_parameters(
        mu_g, T, M, Rg, Q, S, D, L, kappa, phi, Pex
    )

    GAMMA0, GAMMA1, GAMMA2, gamma0, gamma1 = harmonic_oscillator_coefficients(
        L, beta_a, beta_b, beta_c, beta_d, beta_e
    )

    # print(natural_frequency(GAMMA1, GAMMA2, gamma0, gamma1))

    A_res, A_exc, A_p = pressure_freq_domain(
        f, t0, qn, gamma0, gamma1, GAMMA0, GAMMA1, GAMMA2
    )

    # Pressure differential
    # TODO why the multiplication by max_freq?
    # P = irfft(A_p)*max_freq
    P = irfft(A_p)
    DP = P + Pex - P0
    dPP0 = DP/P0

    n_stations, Sxx, st = len(xr), [], []
    for i in range(n_stations):
        _Qf = Qf
        if type(Qf) is list:
            _Qf = _Qf[i]

        d = np.sqrt((xr[i]-xs)**2 + (yr[i]-ys)**2 + (zr[i]-zs)**2)

        Sx = ground_freq_domain(f, S, d, rho_s, _Qf, A_res, A_exc)
        Sxx.append(Sx)

        # TODO why the multiplication by max_freq?
        # U_z = irfft(Sx)*max_freq
        U_z = irfft(Sx)
        st.append(U_z)
    Sxx = np.array(Sxx)
    st = np.array(st)
    return dPP0, st, A_p, Sxx


def natural_frequency(
    # Gas properties
    mu_g=1e-5,      # gas viscosity
    T=1000+273.15,  # gas temperature
    M=0.018,        # molecular weight of gas (water vapor)
    Rg=8.3145,      # ideal gas constant
    # Flux
    Q=2,            # mean gas flux
    # Conduit geometry
    R=25,           # conduit radius
    S=None,         # conduit section
    # Gas pocket
    D=0.03,         # gas pocket thickness
    # Cap properties
    L=20,           # thickness of the cap
    kappa=1e-8,     # permeability of the cap
    phi=1e-4,       # porosity of the cap
    # External pressure
    Pex=101325,     # external pressure

):
    if S is None or np.isnan(S):
        S = pi*R**2

    beta_a, beta_b, beta_c, beta_d, beta_e, P0 = auxiliary_parameters(
        mu_g, T, M, Rg, Q, S, D, L, kappa, phi, Pex
    )

    GAMMA0, GAMMA1, GAMMA2, gamma0, gamma1 = harmonic_oscillator_coefficients(
        L, beta_a, beta_b, beta_c, beta_d, beta_e
    )
    return _natural_frequency(GAMMA1, GAMMA2, gamma0, gamma1)


def figure(max_freq, tau, dPP0, st, A_p, Sxx, fn=None, lw=0.5, mm=1/25.6, alpha=0.7):
    t, f = time_freq_arrays(tau, max_freq)

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(190*mm, 180*mm), sharex='col'
    )

    axes[0, 0].plot(t, dPP0, lw=lw)

    for U_z in st:
        axes[1, 0].plot(t, U_z, lw=lw, alpha=alpha)

    axes[0, 1].plot(f, np.abs(A_p), lw=lw)

    for i, u_z in enumerate(Sxx):
        axes[1, 1].plot(
            f, np.abs(u_z), lw=lw, label=f'Station {i}', alpha=alpha
        )

    axes[1, 1].legend()

    xlabel = [
        ['', ''],
        ['Time [s]', 'Frequency [Hz]']
    ]
    ylabel = [
        ['$\Delta P/P_0$', 'Pressure [Pa]'],
        ['Ground displacement [m]', 'Ground displacement [m]']
    ]
    for i in range(2):
        if fn is not None:
            axes[i, 1].axvline(fn, lw=0.5, ls='--')
        axes[i, 1].set_xscale('log')
        # axes[i, 1].set_xlim(1e-1, max_freq)
        for j in range(2):
            axes[i, j].set_xlabel(xlabel[i][j])
            axes[i, j].set_ylabel(ylabel[i][j])

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    param = dict(
        # Simulation parameters
        tau=20,         # seconds of simulation
        max_freq=25,    # maximum frequency to be reached in the simulation
        # Source location
        xs=0,
        ys=0,
        zs=-1e3,
        # Gas properties
        N=1000,         # number of mass impulses in tau seconds
        mu_g=1e-5,      # gas viscosity
        T=1000+273.15,  # gas temperature
        M=0.018,        # molecular weight of gas (water vapor)
        Rg=8.3145,      # ideal gas constant
        Q=2,            # mean gas flux
        # Conduit geometry
        R=25,           # conduit radius
        S=None,         # conduit section
        # Gas pocket
        D=1e-3,         # gas pocket thickness
        # Cap properties
        L=20,           # thickness of the cap
        kappa=1e-8,     # permeability of the cap
        phi=1e-4,       # porosity of the cap
        # External pressure
        Pex=101325,     # external pressure
        # Path and receivers
        xr=[0, 2e3, 5e3],    # Stations x-coordinate
        yr=[0, 0, 0],         # Stations x-coordinate
        zr=[0, 0, 0],         # Stations x-coordinate
        Qf=[10, 20, 50],     # Quality factors for each station (list or number)
        rho_s=3000,       # density of the medium of propagation
    )
    dPP0, st, A_p, Sxx, fn = synthetize(**param)

    fig = figure(param['max_freq'], param['tau'], dPP0, st, A_p, Sxx)
    fig.savefig('borrar.png')
