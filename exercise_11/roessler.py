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


def visualize_attractor(rt):
    transient_steps = 10

    markersize = 0.2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(rt[transient_steps:, 0], rt[transient_steps:, 1], rt[transient_steps:, 2], '.', markersize=markersize)

    fig2, ax_array = plt.subplots(3, 1)
    ylabels = ['x', 'y', 'z']
    for idx, axis in enumerate(ax_array):
        axis.plot(rt[transient_steps: transient_steps+10000, idx])
        axis.set_ylabel(ylabels[idx])


if __name__ == '__main__':
    r0 = [12, 12, 0]
    t_max = 10000
    dgl = run(r0, t_max)
    rt = np.array(dgl.rt)

    visualize_attractor(rt)
    plt.show()
