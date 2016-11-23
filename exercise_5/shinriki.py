import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d
import time as time

from tools import Dgl


def f_shinriki_dgl(t, r):
    # parameters
    c1 = 10e-9
    c2 = 100e-9
    l = 0.32
    r1 = 22e3
    r2 = 14.5e3
    r3 = 100
    r_nic = 6.9e3
    a = 2.295e-8
    b = 3.0038
    # state variables
    v1 = r[0]  # Voltage 1
    v2 = r[1]  # Voltage 2
    i3 = r[2]  # Current 3

    # nonlinearity
    def i_d(v):
        return a * (np.exp(b * v) - np.exp(-b * v))

    # Differential equations
    dv1 = 1 / c1 * (v1 * (1 / r_nic - 1 / r1) - i_d(v1 - v2) - (v1 - v2) / r2)
    dv2 = 1 / c2 * (i_d(v1 - v2) + (v1 - v2) / r2 - i3)
    di3 = 1 / l * (-i3 * r3 + v2)
    return [dv1, dv2, di3]


def get_zero_crossings(array):
    """
    get zero crossings in negative direction
    """
    return np.where(np.diff(np.sign(array)) == 2)[0]


def run(r0, t_max, t_step=0.001):
    start_time = time.time()
    dgl = Dgl(f_shinriki_dgl, r0, trace=True)
    while dgl.dgl.t < t_max:
        dgl.solve(dgl.dgl.t + t_step)
    print("Running dgl solver took: \t {:.2f}s".format(time.time() - start_time))
    return dgl


def plot():
    t_max = 0.5
    t_step = 1e-5
    r0 = [0, 0.5, 0.75e-3]
    dgl = run(r0, t_max, t_step)
    rt = np.array(dgl.rt)

    # start at later point
    start_idx = 300
    rt = rt[start_idx:]

    zero_crossings = get_zero_crossings(rt[:, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(rt[start_idx:, 0], rt[start_idx:, 1], rt[start_idx:, 2])

    ax.scatter(rt[zero_crossings, 0], rt[zero_crossings, 1], rt[zero_crossings, 2], marker='.', color='r')

    ax.set_xlabel(r'$V_1$ [V]')
    ax.set_ylabel(r'$V_2$ [V]')
    ax.set_zlabel(r'$I_3$ [A]')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    scale_i = 1000  # plot in mA
    ax3.plot(-rt[zero_crossings, 0], -rt[zero_crossings, 2] * scale_i, '.')
    ax3.set_title(r'Poincare cross section ($V_2=0$)')
    ax3.set_xlabel(r'$V_1$ [V]')
    ax3.set_ylabel(r'$I_3$ [mA]')
    ax3.grid()
    plt.show()

    def plot_colored():
        # http://matplotlib.org/examples/pylab_examples/multicolored_line.html
        x = np.array(dgl.xt)
        x = x[start_idx:]
        y = np.array(dgl.yt)
        y = y[start_idx:]
        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be numlines x points per line x 2 (x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create the line collection object, setting the colormapping parameters.
        # Have to set the actual values used for colormapping separately.
        lc = LineCollection(segments, cmap=plt.get_cmap('viridis'),
                            norm=plt.Normalize(0, x.size))
        lc.set_array(np.arange(1, x.size, 1))
        lc.set_linewidth(1)

        fig2 = plt.figure()
        ax = fig2.add_subplot(111)

        plt.gca().add_collection(lc)
        ax.autoscale_view()


plot()
