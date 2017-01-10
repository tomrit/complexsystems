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


def multiply_jacobian_vector(jacobian_vector):
    yn = np.identity(jacobian_vector[0].shape[0])
    for jacobian_matrix in jacobian_vector:
        yn = yn.dot(jacobian_matrix)
    return yn


def calculate_largest_le(n_steps=500):
    """
    calculate the largest lyapunov exponent
    :return:

    >>> calculate_largest_le(100)
    0.44937438976581001
    """
    transient_steps = 100
    x0 = [1, 1, 1]
    xn = iterate_map(n_steps, x0)

    # temporal evolution of a small perturbation
    epsilon = 1e-3
    y0 = np.repeat(epsilon, 3)
    jacobians = jacobian_vector(xn)
    yn = multiply_jacobian_vector(jacobians)
    yn = yn.dot(y0)
    le = 1 / n_steps * np.log(np.linalg.norm(yn) / np.linalg.norm(y0))

    return le


def calculate_le_spectrum(N=100, T=100):
    # calculate point on attractor
    # x0 = [1, 1, 1]
    # xn = iterate_map(transient_steps, x0)
    # x0 = xn[-1]
    x_0 = [0.49098218, 0.05363883, 0.79275178]

    # number of largest lyapunov exponents
    k = 3
    # orthogonal matrix - identity
    o_k = np.identity(k)
    # initialization
    x_n = x_0
    q = o_k
    qs = np.zeros((N, k, k))
    rs = np.zeros((N, k, k))
    for idx, n in enumerate(range(0, N)):
        # next step: phi_n+1
        x_ns = iterate_map(T, x_n)
        # future starting point
        x_n = x_ns[-1]
        # calculate jacobians
        jacobians = jacobian_vector(x_ns)
        # multiply jacobians
        p = np.dot(multiply_jacobian_vector(jacobians), q)
        # perform the qr decomposition
        q, r = np.linalg.qr(p)
        # diagonal matrix
        d = np.diag(np.array([
            np.sign(r[0, 0]),
            np.sign(r[1, 1]),
            np.sign(r[2, 2])]))
        q = q.dot(d)
        # make diagonal elements positive
        r = r.dot(d)
        qs[idx] = q
        rs[idx] = r

    # calculate the le specttrum
    le_spectrum = np.zeros(k)
    for r in rs:
        # sum up diagonal elements
        diagonal = np.diag(r)
        le_spectrum += np.log(diagonal)
    # normalization
    le_spectrum /= N * T
    return le_spectrum


def visualize_map(n_steps=100000):
    transient_steps = 100
    x0 = [1, 1, 1]
    markersize = 0.2
    xn = iterate_map(n_steps, x0)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xn[transient_steps:, 0], xn[transient_steps:, 1], xn[transient_steps:, 2], '.', markersize=markersize)


def visualize_converging_largest_le(N=500):
    n_steps = np.linspace(1, 800, N)
    les = np.zeros(N)
    for idx, n_step in enumerate(n_steps):
        les[idx] = calculate_largest_le(int(n_step))
    fig2, ax2 = plt.subplots()
    ax2.plot(n_steps, les)
    ax2.set_title("Largest Lyapunov exponent {:.2f}".format(les[-1]))


def visualize_le_spectrum(Ns=np.arange(1, 500)):
    result = np.zeros((Ns.size, 3))
    for idx, N in enumerate(Ns):
        spectrum = calculate_le_spectrum(N)
        result[idx] = spectrum
    fig3, ax3 = plt.subplots()
    ax3.plot(Ns, result)
    ax3.set_xlabel("N")
    ax3.set_ylabel("lyapunov exponent")

    spectrum_string = ", ".join("{:.2f}".format(le) for le in spectrum)
    ax3.set_title("Lyapunov exponent spectrum = {}".format(spectrum_string))


if __name__ == '__main__':
    visualize_map()
    visualize_converging_largest_le()
    print("largest le: ", calculate_largest_le(100))
    visualize_le_spectrum()
    print("le spectrum: ", calculate_le_spectrum(100))
    plt.show()
