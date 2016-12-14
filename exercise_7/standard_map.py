import matplotlib.pyplot as plt
import numpy as np
from tools import get_subplots_squared


def x_step(x, p):
    return (x + p) % 1


def p_step(x, p, k):
    return p + k / (2 * np.pi) * np.sin(2 * np.pi * x_step(x, p))


def xp_step(x, p, k):
    return [x_step(x, p), p_step(x, p, k)]


def x_backstep(x, p, k):
    return (x - p_backstep(x, p, k)) % 1


def p_backstep(x, p, k):
    return p - k / (2 * np.pi) * np.sin(2 * np.pi * x)


def xp_backstep(x, p, k):
    return [x_backstep(x, p, k), p_backstep(x, p, k)]


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


def iterate_map(starting_point, k=0.1, n_steps=1000):
    x0, p0 = starting_point
    x = np.zeros(n_steps)
    p = np.zeros(n_steps)

    for idx in range(0, n_steps - 1):
        x0, p0 = xp_step(x0, p0, k)
        x[idx] = x0
        p[idx] = p0

    return [x, p]


def iterate_grid(x0s, p0s, n_steps, k):
    p_interval = (-0.5, 0.5)
    grid_size = x0s.size * p0s.size
    result = np.zeros((grid_size, n_steps, 2))
    grid_idx = 0
    for x0 in x0s:
        for p0 in p0s:
            starting_point = (x0, p0)
            [x, p] = iterate_map(starting_point, k, n_steps)
            p = apply_periodic_boundary(p_interval, p)

            result[grid_idx, :, :] = np.column_stack((x, p))

            grid_idx += 1
    return result


def iterate_k(ks, x0s, p0s, n_steps):
    grid_size = x0s.size * p0s.size
    # result matrix: k_size x grid_size x iteration  x 2(x,p)
    k_size = ks.size
    results = np.zeros((k_size, grid_size, n_steps, 2))

    for k_idx, k in enumerate(ks):
        results[k_idx, :] = iterate_grid(x0s, p0s, n_steps, k)
    return results


if __name__ == '__main__':
    import doctest

    # test functions by running the examples from the docstring
    doctest.testmod()

    # Parameters
    n_steps = 1000
    n_grid = 20
    x0s = np.linspace(0, 1, n_grid)
    p0s = np.linspace(-0.5, 0.5, n_grid)
    p_interval = (-0.5, 0.5)

    k_size = 9
    ks = np.linspace(0.01, 2, k_size)
    # ks = np.array([0.01, 0.02, 0.5, 0.9, 0.99, 1, 2, 3, 4.5])

    # Calculation of phase space
    results = iterate_k(ks, x0s, p0s, n_steps)

    # Calculation of unstable mannifold at (0.0)
    epsilon = 0.001
    delta_x = np.linspace(-epsilon / 2, epsilon / 2, 4)
    delta_y = delta_x

    mannifold_unstable = iterate_k(ks, delta_x, delta_y, n_steps)

    # Plot
    # subplot matrix to show multiple results for different k
    n_rows, n_cols = get_subplots_squared(k_size)

    # initialize subplots
    fig1, ax_array = plt.subplots(n_rows, n_cols, figsize=(8, 8))

    # catch N=1 because then ax_array is not a list but a single ax object
    if isinstance(ax_array, np.ndarray):
        ax_array_flat = ax_array.reshape(-1)
    else:
        ax_array_flat = [ax_array]

    markersize = 0.1

    for k_idx, k in enumerate(ks):
        current_axis = ax_array_flat[k_idx]

        current_axis.plot(results[k_idx, :, :, 0], results[k_idx, :, :, 1], '.k', markersize=markersize)
        current_axis.plot(mannifold_unstable[k_idx,:,:,0], mannifold_unstable[k_idx,:,:,1], '.r', markersize=markersize)
        current_axis.set_xlim(0, 1)
        current_axis.set_ylim(p_interval)
        current_axis.set_title("k = {:.2f}".format(k))

    fig1.savefig("exercise7_Standard-Map.png", dpi=300, transparent=True, bbox_inches='tight')

    plt.show()
