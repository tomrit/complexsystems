import numpy as np
import matplotlib.pylab as plt
import time


def circle_map(theta, omega, k):
    return (theta + omega - k / 2 / np.pi * np.sin(2 * np.pi * theta)) % (2 * np.pi)


def iterate_map(theta0, omega, k, n_steps):
    thetas = np.zeros(n_steps)
    time0 = time.time()
    for idx, theta in enumerate(thetas):
        theta0 = circle_map(theta0, omega, k)
        thetas[idx] = theta0
    print("Took {:.3f} s".format(time.time() - time0))
    return thetas


def visualize_iteration():
    N = int(1e3)
    thetas = iterate_map(0.1, 0.2, 1, N)
    plt.plot(thetas / 2 / np.pi)


if __name__ == '__main__':
    visualize_iteration()
    plt.show()
