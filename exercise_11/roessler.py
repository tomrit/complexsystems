import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import ode
import copy
from mpl_toolkits.mplot3d import Axes3D


class Dgl(object):
    """
    Solve DGLs with dopri5 integrator
    """

    def __init__(self, function, r0, parameter, tolerance=1e-8, trace=False):
        self.function = function
        self.r0 = r0
        self.t0 = 0
        self.rt = []
        self.xt = []
        self.yt = []
        self.dgl = ode(self.function).set_integrator('dopri5', rtol=tolerance, nsteps=10000)
        self.dgl.set_f_params(parameter)
        if trace:
            self.dgl.set_solout(self.solout)
        self.dgl.set_initial_value(self.r0, self.t0)

    def solout(self, t, r):
        self.rt.append(copy.copy(r))
        # self.xt.append(copy.copy(r[0]))
        # self.yt.append(copy.copy(r[1]))
        # print(self.dgl.t)

    def solve(self, t_max):
        return self.dgl.integrate(t_max)


def get_subplots_squared(length):
    rows = np.floor(np.sqrt(length))
    columns = np.ceil(length / rows)
    return int(rows), int(columns)


def roessler_dgl(t, r, parameters):
    # state variables
    x = r[0]
    y = r[1]
    z = r[2]

    # Differential equations
    dx = -y - z
    dy = x + 0.1 * y
    dz = 0.1 + z * (x - 14)

    return [dx, dy, dz]


def run(r0, t_max, t_step=1, tolerance=1e-9):
    dgl = Dgl(roessler_dgl, r0, tolerance, trace=True)
    while dgl.dgl.t < t_max:
        dgl.solve(dgl.dgl.t + t_step)
    return dgl


def delay_coordinates(x, shift, d=2):
    result = np.zeros((x.size - (d - 1) * shift, d))
    if shift == 0:
        result = np.tile(x, (d, 1)).T
    else:
        for i in range(0, d):
            result[:, i] = x[i * shift:x.size - int((d - (i + 1)) * shift)]
    return result


def autocorr(x):
    """
    calculated autocorrelation of data set

    http://greenteapress.com/thinkdsp/html/thinkdsp006.html
    :param x:
    :return: autocorrelation
    """
    result = np.correlate(x, x, mode='same')
    # select one half
    N = result.size
    result = result[N // 2:]
    # correct decreasing
    lengths = range(N, N // 2, -1)
    result /= lengths
    # normalize
    result /= result[0]
    return result


def get_zero_crossings(array):
    """
    get zero crossings
    """
    return np.where(np.diff(np.sign(array)))[0]


def get_first_zero_crossing(array):
    zero_crossings = get_zero_crossings(array)
    if zero_crossings.size != 0:
        return zero_crossings[0]
    else:
        return None


def visualize_attractor(rt):
    transient_steps = 10

    markersize = 0.2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(rt[transient_steps:, 0], rt[transient_steps:, 1], rt[transient_steps:, 2], '.', markersize=markersize)

    fig2, ax_array = plt.subplots(3, 2)
    ax_array = ax_array.reshape(-1)
    ylabels = ['x', 'y', 'z']

    interval = range(transient_steps, transient_steps + 10000)
    for idx, axis in enumerate(ax_array):
        if idx % 2 == 0:
            axis.plot(rt[interval, int(idx / 2)])
            axis.set_ylabel(ylabels[int(idx / 2)])
        else:
            autocorrelation = autocorr(rt[interval, int((idx - 1) / 2)])
            zero_crossing = get_first_zero_crossing(autocorrelation)
            axis.plot(autocorrelation)
            axis.vlines(zero_crossing, 0.5, 0.5)
            if zero_crossing != None:
                axis.set_title('autocorrelation zero crossing {}'.format(zero_crossing))


def visualize_delay_coordinates(rt):
    x = rt[100:, 2]

    shifts = np.arange(0, 400, 20)
    shifts = np.arange(0, 40, 2)
    N = shifts.size

    rows, columns = get_subplots_squared(N)
    fig3, ax_array = plt.subplots(rows, columns, sharex=True, sharey=True)
    ax_flat = ax_array.reshape(-1)
    for idx, shift in enumerate(shifts):
        ax_current = ax_flat[idx]
        r_delayed = delay_coordinates(x, shift)
        x_delay = r_delayed[:, 0]
        y_delay = r_delayed[:, 1]

        ax_current.plot(x_delay, y_delay, ',')
        ax_current.set_title('delay = {}'.format(shift))


def visualize_reconstruction(rt):
    # shift = 10
    x = rt[100:, 0]
    shift_x = 81
    r_delayed_x = delay_coordinates(x, shift_x, d=3)

    y = rt[100:, 1]
    shift_y = 82
    r_delayed_y = delay_coordinates(y, shift_y, d=3)

    z = rt[100:, 2]
    shift_z = 18
    r_delayed_z = delay_coordinates(z, shift_z, d=3)

    markersize = 0.2
    fig4 = plt.figure()
    ax4 = fig4.gca(projection='3d')
    ax4.plot(r_delayed_x[:, 0], r_delayed_x[:, 1], r_delayed_x[:, 2], '.', markersize=markersize)
    ax4.set_title('reconstruction from x; shift: {}'.format(shift_x))

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.plot(r_delayed_y[:, 0], r_delayed_y[:, 1], r_delayed_y[:, 2], '.', markersize=markersize)
    ax5.set_title('reconstruction from y; shift: {}'.format(shift_y))

    fig6 = plt.figure()
    ax6 = fig6.gca(projection='3d')
    ax6.plot(r_delayed_z[:, 0], r_delayed_z[:, 1], r_delayed_z[:, 2], '.', markersize=markersize)
    ax6.set_title('reconstruction from z; shift: {}'.format(shift_z))


if __name__ == '__main__':
    r0 = [12, 12, 0]
    t_max = 1000
    dgl = run(r0, t_max)
    rt = np.array(dgl.rt)

    visualize_attractor(rt)
    visualize_delay_coordinates(rt)
    visualize_reconstruction(rt)
    plt.show()
