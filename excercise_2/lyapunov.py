#! /usr/bin/env python

# This program shall solve a Differential equation numerically and plot trajectories
# together with a 'potential' obtained as a Lyapunov function.

import copy
import numpy as np
from scipy.integrate import ode

import matplotlib.pyplot as plt


class dgl(object):
    def __init__(self, funct, r0):
        self.funct = funct
        self.r0 = r0
        self.t0 = 0
        self.xt = []
        self.yt = []
        self.dgl = ode(f).set_integrator('dopri5')
        self.dgl.set_solout(self.solout)
        self.dgl.set_initial_value(self.r0, self.t0)

    def solout(self, t, r):
        self.xt.append(copy.deepcopy(r[0]))
        self.yt.append(copy.deepcopy(r[1]))

    def solve(self, tmax):
        return self.dgl.integrate(tmax)


def f(t, y):
    dx = -2 * y[0] - y[1] ** 2
    dy = -y[1] - y[0] ** 2
    return [dx, dy]


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size
                       )


initial = [[0, 1], [1, 1], [-1, 1], [1, 0], [-1, 0], [0.5, 0.2], [-1, -1], [1, -1]]

for ini in initial:
    cur = dgl(f, ini)
    cur.solve(10)
    line, = plt.plot(cur.xt, cur.yt)
    add_arrow(line, None, 'right', 15, line.get_color())

plt.grid()
plt.show()
