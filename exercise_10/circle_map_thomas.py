#!/bin/bash

#############################################################################################################
#       Introduction to Complex Systems  --  Exercise-Sheet 10  --  Circle Map.                             #
#       Thomas Rittmann, Jan 2017                                                                           #
#                                                                                                           #
#       This program produces a bifurcation diagram of the "Circle map" and plots the Lyapunov exponent     #
#       and the winding number as a function of the frequency Omega (between 0 and 2), for                  #
#       different K values.                                                                                 #
#                                                                                                           #
#############################################################################################################

import numpy as np
import copy
import matplotlib.pyplot as plt
import sys


def map_step(theta, omega, K):
    theta_new = theta + omega - K / 2. / np.pi * np.sin(2. * np.pi * theta)
    return theta_new


def map_derivative(theta, K):
    d_theta = 1 - K * np.cos(2. * np.pi * theta)
    return d_theta


# ----------iteration for fixed omega -------------
# After the first transient steps, the theta values are saved in order to plot the bifurcation
# The Lyapunov exponent is determined by summing up the logarithms of the current derivative.
# The winding number is determined by building the difference of the final step and the first
# non-transient step (non modulo in this case).
# -----
def orbit(omega, K, transient_steps, plot_steps):
    thetas = []
    # reuse theta for next omega-step (therefore global)
    global theta0
    lyapunov_sum = 0
    for idx in range(plot_steps + transient_steps):
        theta0 = map_step(theta0, omega, K)
        # last transient step saved for winding number difference
        if idx == transient_steps - 1:
            theta_init = copy.copy(theta0)
        # begin saving and lyapunov calculation after enough transient steps
        if idx >= transient_steps:
            thetas.append((omega, theta0 % 1))
            lyapunov_sum += np.log(np.abs(map_derivative(theta0 % 1, K)))
    winding = [omega, (theta0 - theta_init) / plot_steps]
    lyapunov = lyapunov_sum / plot_steps
    lyapunov_tuple = [omega, lyapunov]
    return thetas, lyapunov_tuple, winding


# ------- Iterate over all Omegas for a given K  (add up results in plot vectors) ---------
def bifurcation_eval(K):
    # ----- PARAMETER SETTINGS ------
    transient_steps = 50
    plot_steps = 300
    omegas_N = 1000
    omegas = list(np.linspace(0, 2, omegas_N))
    global theta0
    theta0 = 0.2
    bifurc = []
    lyapunov = []
    winding = []
    for idx, omega in enumerate(omegas):
        bifurc_new, lyapunov_new, winding_new = orbit(omega, K, transient_steps, plot_steps)
        bifurc += bifurc_new
        lyapunov.append(lyapunov_new)
        winding.append(winding_new)
        sys.stdout.write("\r[Progress: %s/%s]" % (idx, omegas_N))
        sys.stdout.flush()
    bifurc = np.array(bifurc)
    lyapunov = np.array(lyapunov)
    winding = np.array(winding)
    return bifurc, lyapunov, winding


# ----Plot a figure for each K-value containing bifurcation, Lyapunov and winding plot ---
def plot_all(Ks):
    figu = []
    axu = []
    for idx, K in enumerate(Ks):
        bifurcation_data, lyapunov_data, winding_data = bifurcation_eval(K)
        fig, ax = plt.subplots(3, 1)
        figu.append(fig)
        axu.append(ax)
        axu[idx][0].set_title('Circle Map (K={})'.format(K))
        axu[idx][0].plot(bifurcation_data[:, 0], bifurcation_data[:, 1], "b,")
        axu[idx][0].set_ylabel(r'$\theta$ (bifurcation)')
        axu[idx][1].plot(lyapunov_data[:, 0], lyapunov_data[:, 1], "r.", markersize=1)
        axu[idx][1].set_ylabel('Lyapunov exponent')
        axu[idx][2].plot(winding_data[:, 0], winding_data[:, 1], "g.", markersize=1)
        axu[idx][2].set_xlabel(r'$\Omega$')
        axu[idx][2].set_ylabel('winding number')
    plt.show()


# ------------ MAIN -----------------
Ks = [0.1, 0.5, 0.95]
plot_all(Ks)
