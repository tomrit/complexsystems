#!/bin/bash

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# ------ set parameters --------
qs = [0, 1, 2]
epsilon_min = 1e-3
epsilon_max = .5
epsilon_N = 50
datacut_min = 10000
datacut_N = 6
linear_min = 3
linear_max = 6

# ------ load datafile and recognise min/max -------
data = np.loadtxt("data1.txt")
data_min_x = np.min(data[:, 0])
data_min_y = np.min(data[:, 1])
data_max_x = np.max(data[:, 0])
data_max_y = np.max(data[:, 1])

# ------ Visualization of data set ---------
fig0, ax0 = plt.subplots()
ax0.plot(data[:, 0], data[:, 1], ',')
ax0.set_xlabel(r'$x$')
ax0.set_ylabel(r'$y$')
ax0.set_title("Data Set")

# ------- equal spacing of epsilons for relation that shall be plotted later on -----
epsilon_log_spacing = np.linspace(np.log(1. / epsilon_max), np.log(1. / epsilon_min), epsilon_N)
epsilons = 1. / np.exp(epsilon_log_spacing)


# ------- calculate Renyi Information for given q and epsilon ------
def renyi(q, data, epsilon):
    # arrange data in boxes with epsilon side lengths...
    bins = [np.arange(data_min_x, data_max_x, epsilon), np.arange(data_min_y, data_max_y, epsilon)]
    # count how many data points are in each box...
    histo, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    # calculate probability by dividing through data size
    prob = histo / data.size
    # for q==1, use the limit case from task a) to avoid division by zero
    if q == 1:
        # filter out zero-probabilities to avoid error with log
        prob = prob[prob > 0]
        I = -np.sum(prob * np.log(prob))
    # for all other q, use the normal definition
    else:
        I = 1 / (1 - q) * np.log(np.sum(prob ** q))
    return I


# -----------create subplots for the different q-values -----------------
fig1, ax_array = plt.subplots(1, len(qs), sharey=True, figsize=(12, 8))

for idx, q in enumerate(qs):
    ax1 = ax_array[idx]

    # plot graphs for datacuts after certain lengths, keep full one for fit
    for number in np.linspace(datacut_min, data.size, datacut_N):
        I = [renyi(q, data[:number, :], e) for e in epsilons]
        x_eps = np.log(1. / epsilons)
        ax1.plot(x_eps, I, "o-", label=str(number))

    # filter out interval to perform linear fit on
    linear_interval = np.logical_and(x_eps > linear_min, x_eps < linear_max)
    linear_reg_I = np.array(I)[linear_interval]
    linear_reg_x = np.array(x_eps)[linear_interval]


    # linear fit function
    def fit(xs, ys):
        def f(x, a, b):
            return a * x + b

        params, _ = optimize.curve_fit(f, xs, ys)
        return params


    # perform linear fit and create linear curve
    a, b = fit(linear_reg_x, linear_reg_I)
    linear_fit = [a * x + b for x in x_eps]
    # plot results with proper labeling and regression parameters in title
    print("q={}: dimension from slope of linear fit: {}".format(q, a))
    ax1.plot(x_eps, linear_fit, label="linear fit")
    ax1.set_title(r'$q = {:d}; \; I \approx {:.2f} \cdot \log(1/\epsilon) +  {:.2f}$'.format(q, a, b))
    ax1.legend(loc="upper left")
    ax1.set_xlabel('$log(1/\epsilon)$')
    ax1.set_ylabel('$I(\epsilon)$')

plt.show()
