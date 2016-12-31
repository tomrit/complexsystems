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


def jacobian(x_):
    x = x_[0]
    y = x_[1]
    z = x_[2]

    d1 = [3.8 - 7.6 * x, 0.1 * z - 0.05, 0.1 * (y + 0.35)]
    d2 = [y * (0.38 * z - 0.19) + 0.133 * z + 0.1235, 0.1 * (1 - 1.9 * x) * (1 - 2 * z),
          -0.2 * (1 - 1.9 * x) * (y + 0.35)]
    d3 = [0, 0.2, -3.78 * 2 * z + 3.78]
    return [d1, d2, d3]


def jacobian_vector(xn):
    x = xn[:, 0]
    y = xn[:, 1]
    z = xn[:, 2]

    shape = xn.shape
    result_vector = np.zeros(shape + (3,))

    result_vector[:, 0, :] = np.column_stack((3.8 - 7.6 * x, 0.1 * z - 0.05, 0.1 * (y + 0.35)))
    result_vector[:, 1, :] = np.column_stack(
        (y * (0.38 * z - 0.19) + 0.133 * z + 0.1235, 0.1 * (1 - 1.9 * x) * (1 - 2 * z),
         -0.2 * (1 - 1.9 * x) * (y + 0.35)))
    result_vector[:, 2, :] = np.column_stack((np.repeat(0, shape[0]), np.repeat(0.2, shape[0]), -3.78 * 2 * z + 3.78))

    return result_vector


def calculate_largest_le(n_steps=500):
    """
    calculate the largest lyapunov exponent
    :return:

    """
    transient_steps = 100
    x0 = [1, 1, 1]
    xn = iterate_map(n_steps, x0)

    # temporal evolution of a small perturbation
    epsilon = 1e-3
    y0 = np.repeat(epsilon, 3)
    yn = np.identity(3)
    for jacobian_matrix in jacobian_vector(xn):
        yn = yn.dot(jacobian_matrix)
    yn = yn.dot(y0)
    le = 1 / n_steps * np.log(np.linalg.norm(yn) / np.linalg.norm(y0))

    return le


def calculate_le_spectrum():
    # calculate point on attractor
    # transient_steps = 50
    # x0 = [1, 1, 1]
    # xn = iterate_map(transient_steps, x0)
    # x0 = xn[-1]
    x0 = [0.49098218, 0.05363883, 0.79275178]

    # number of largest lyapunov exponents
    k = 3
    # orthogonal matrix - identity
    o_k = np.identity(k)
    # medium time step
    T = 10
    N = 1
    n_steps = T * N
    # reference orbit
    xn = iterate_map(n_steps, x0)
    # jacobians
    jacobians = jacobian_vector(xn)
    # qr decomposition
    qs = np.zeros((n_steps, 3, 3))
    rs = np.zeros((n_steps, 3, 3))
    for idx, jacobian in enumerate(jacobians):
        q, r = np.linalg.qr(jacobian)
        # diagonal matrix
        d = np.identity(3)
        d = d.dot(np.array([
            [np.sign(r[0, 0])],
            [np.sign(r[1, 1])],
            [np.sign(r[2, 2])]]))
        qs[idx] = q.dot(d)
        rs[idx] = r.dot(d)


    return None


def visualize_map(n_steps=100000):
    transient_steps = 100
    x0 = [1, 1, 1]
    markersize = 0.2
    xn = iterate_map(n_steps, x0)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xn[transient_steps:, 0], xn[transient_steps:, 1], xn[transient_steps:, 2], '.', markersize=markersize)
    plt.show()


def visualize_converging_largest_le(N=500):
    n_steps = np.linspace(1, 1000, N)
    les = np.zeros(N)
    for idx, n_step in enumerate(n_steps):
        les[idx] = calculate_largest_le(int(n_step))
    plt.plot(n_steps, les)
    plt.show()


if __name__ == '__main__':
    # visualize_map()
    # visualize_converging_largest_le()
    print(calculate_largest_le(100))
    calculate_le_spectrum()
