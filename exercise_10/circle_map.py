import numpy as np
import matplotlib.pylab as plt


def circle_map(theta, omega, k):
    return theta + omega - k / 2 / np.pi * np.sin(2 * np.pi * theta)


def iterate_map(theta0, omega, k, n_steps):
    thetas = np.zeros(n_steps)
    for idx, theta in enumerate(thetas):
        if idx == 0:
            theta = theta0
            continue
        theta0 = circle_map(theta0, omega, k)
        theta = theta0
    return thetas
