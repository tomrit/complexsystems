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


def iterate_map(starting_point, k=0.1, n_steps=1000, backwards=False):
    x0, p0 = starting_point
    x = np.zeros(n_steps)
    p = np.zeros(n_steps)

    for idx in range(0, n_steps - 1):
        if backwards:
            x0, p0 = xp_backstep(x0, p0, k)
        else:
            x0, p0 = xp_step(x0, p0, k)
        x[idx] = x0
        p[idx] = p0

    return [x, p]


def iterate_grid(x0s, p0s, n_steps, k, backwards=False):
    p_interval = (-0.5, 0.5)
    grid_size = x0s.size * p0s.size
    result = np.zeros((grid_size, n_steps, 2))
    grid_idx = 0
    for x0 in x0s:
        for p0 in p0s:
            starting_point = (x0, p0)
            [x, p] = iterate_map(starting_point, k, n_steps, backwards)
            p = apply_periodic_boundary(p_interval, p)

            result[grid_idx, :, :] = np.column_stack((x, p))

            grid_idx += 1
    return result


def iterate_k(ks, x0s, p0s, n_steps, backwards=False):
    grid_size = x0s.size * p0s.size
    # result matrix: k_size x grid_size x iteration  x 2(x,p)
    k_size = ks.size
    results = np.zeros((k_size, grid_size, n_steps, 2))

    for k_idx, k in enumerate(ks):
        results[k_idx, :] = iterate_grid(x0s, p0s, n_steps, k, backwards)
    return results


def iterate_coordinates(x0s, p0s, n_steps, k, backwards=False):
    p_interval = (-0.5, 0.5)
    n_points = x0s.size
    result = np.zeros((n_points, n_steps, 2))
    for idx, x0 in enumerate(x0s):
        p0 = p0s[idx]
        starting_point = (x0, p0)
        [x, p] = iterate_map(starting_point, k, n_steps, backwards)
        p = apply_periodic_boundary(p_interval, p)

        result[idx, :, :] = np.column_stack((x, p))

    return result


def iterate_mannifold_k(ks, p0s, n_steps, backwards=False):
    n_points = p0s.size
    k_size = ks.size
    results = np.zeros((k_size, n_points, n_steps, 2))
    for k_idx, k in enumerate(ks):
        # starting point is chosen on the eigenvector
        if backwards:
            # stable
            text = " stable"
            x0s = 0.5 * (-1 - np.sqrt(k + 4) / np.sqrt(k)) * p0s
            # x0s = -0.5 * (1 + np.sqrt(k + 4) / np.sqrt(k)) * p0s
        else:
            # unstable mannifold
            text = " unstable"
            x0s = 0.5 * (-1 + np.sqrt(k + 4) / np.sqrt(k)) * p0s
        results[k_idx, :] = iterate_coordinates(x0s, p0s, n_steps, k, backwards)
    return results


if __name__ == '__main__':
    import doctest

    # test functions by running the examples from the docstring
    doctest.testmod()

    # Parameters
    n_steps = 1000
    n_grid = 10
    x0s = np.linspace(0, 1, n_grid)
    p0s = np.linspace(-0.5, 0.5, n_grid)
    p_interval = (-0.5, 0.5)

    # kicking force
    k_c = 0.971635406
    k_size = 4
    ks = np.linspace(0.7, 1, k_size)
    # ks = np.array([0.01, 0.02, 0.5, 0.9, 0.99, 1, 2, 3, 4.5])
    # ks = np.array([0.001, 0.3, k_c, 1])
    # ks = np.array([1])
    k_size = ks.size

    # Calculation of phase space
    results = iterate_k(ks, x0s, p0s, n_steps)

    # Calculation of stable and unstable mannifold at (0.0)
    epsilon = 1e-12
    n_epsilon = 1000
    delta_y = np.linspace(0, epsilon, n_epsilon)

    mannifold_steps = 40
    mannifold_unstable = iterate_mannifold_k(ks, delta_y, mannifold_steps)

    mannifold_stable = iterate_mannifold_k(ks, delta_y, mannifold_steps, backwards=True)

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

    markersize = 0.3
    markersize_mannifold = 1

    for k_idx, k in enumerate(ks):
        current_axis = ax_array_flat[k_idx]

        # plot phase space
        current_axis.plot(results[k_idx, :, :, 0], results[k_idx, :, :, 1], '.k', markersize=markersize)
        # plot unstable mannifold
        current_axis.plot(mannifold_unstable[k_idx, :, :, 0], mannifold_unstable[k_idx, :, :, 1], '.r',
                          markersize=markersize_mannifold)
        # plot stable mannifold
        current_axis.plot(mannifold_stable[k_idx, :, :, 0], mannifold_unstable[k_idx, :, :, 1], '.g',
                          markersize=markersize_mannifold)
        current_axis.set_xlim(0, 1)
        current_axis.set_ylim(p_interval)
        current_axis.set_title("k = {:.2f}".format(k))

    ks_string = "k = " + ", ".join("{:.2f}".format(k) for k in ks)
    fig1.savefig("./graphics/exercise7_Standard-Map - {}.png".format(ks_string), dpi=300, transparent=True,
                 bbox_inches='tight')

    plt.show()
