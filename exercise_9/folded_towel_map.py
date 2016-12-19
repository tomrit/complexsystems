import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def map_step(x):
    """
    Folded towel map
    :param x: 3D vector containing: x1, x2, x3
    :return: x(n+1): next step: x1(n+1), x2(n+1), x3(n+1)
    """
    # x(n) previous step
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    # x(n+1) next step
    x1_step = 3.8 * x1 * (1 - x1) - 0.05 * (0.35 + x2) * (1 - 2 * x3)
    x2_step = 0.1 * ((x2 + 0.35) * (1 - 2 * x3) - 1) * (1 - 1.9 * x1)
    x3_step = 3.78 * x3 * (1 - x3) + 0.2 * x2

    return [x1_step, x2_step, x3_step]


def iterate_map(N=100, x0=[1e-3, 1e-3, 1e-3]):
    xn = np.zeros((N, 3))
    for idx, iteration in enumerate(xn):
        if idx == 0:
            # set starting point
            xn[idx] = x0
        else:
            xn[idx] = map_step(xn[idx - 1])
    return xn


def visualize_map(n_steps=100000):
    transient_steps = 100
    x0 = [1, 1, 1]
    xn = iterate_map(n_steps, x0)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xn[transient_steps:, 0], xn[transient_steps:, 1], xn[transient_steps:, 2], '.', markersize=0.1)
    plt.show()


if __name__ == '__main__':
    visualize_map()
