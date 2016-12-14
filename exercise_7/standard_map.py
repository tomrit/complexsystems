import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import time
from tools import get_subplots_squared


def x_step(x, p):
    return (x + p) % 1


def p_step(x, p, k):
    return p + k / (2 * np.pi) * np.sin(2 * np.pi * x_step(x, p))


def xp_step(x, p, k):
    return [x_step(x, p), p_step(x, p, k)]


def apply_periodic_boundary(interval, value):
    """
    Applies periodic boundary conditions to value for given interval
    :param interval: tuple
    :param value:
    :return: value in interval

    >>> apply_periodic_boundary((-1, 1), -1.6)
    0.3999999999999999
    >>> apply_periodic_boundary((-1, 1), -0.6)
    -0.6
    >>> apply_periodic_boundary((-1, 1), 0.6)
    0.6000000000000001
    >>> apply_periodic_boundary((-1, 1), 1.6)
    -0.3999999999999999

    """
    interval_start, interval_end = interval
    interval_length = interval_end - interval_start
    diff = -interval_start  # shift value in positive region starting from zero
    return ((value + diff) % interval_length) - diff


def iterate_map(starting_point, k=0.1, N=1000):
    x0, p0 = starting_point
    x = np.zeros(N)
    p = np.zeros(N)

    for idx in range(0, N - 1):
        x0, p0 = xp_step(x0, p0, k)
        x[idx] = x0
        p[idx] = p0

    return [x, p]


def plot_iterated_map(x, p):
    p_interval = (-0.5, 0.5)
    p = apply_periodic_boundary(p_interval, p)
    plt.plot(x, p, '.')
    plt.show()


if __name__ == '__main__':
    import doctest

    doctest.testmod()

    p_interval = (-0.5, 0.5)
    x0s = np.linspace(0, 1, 20)
    p0s = np.linspace(-0.5, 0.5, 20)

    N = 9
    ks = np.linspace(0.1, 6, N)

    n_rows, n_cols = get_subplots_squared(N)

    fig1, ax_array = plt.subplots(n_rows, n_cols, figsize=(8, 8))
    ax_array_flat = ax_array.reshape(-1)

    markersize = 0.1
    for idx, k in enumerate(ks):
        current_axis = ax_array_flat[idx]

        for x0 in x0s:
            for p0 in p0s:
                starting_point = (x0, p0)
                [x, p] = iterate_map(starting_point, k)
                p = apply_periodic_boundary(p_interval, p)
                current_axis.plot(x, p, '.k', markersize=markersize)
        current_axis.set_xlim(0, 1)
        current_axis.set_ylim(p_interval)
        current_axis.set_title("k = {:.1f}".format(k))

    fig1.savefig("exercise7_Standard-Map.png", dpi=300, transparent=True, bbox_inches='tight')

    plt.show()
