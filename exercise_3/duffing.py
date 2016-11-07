#! /usr/bin/env python

# Exercises set 3:

import copy
import numpy as np
import time as time
from scipy.integrate import ode

import matplotlib.pyplot as plt


class dgl(object):
    def __init__(self, funct, r0, gamma):
        self.funct = funct
        self.r0 = r0
        self.t0 = 0
        self.xt = []
        self.yt = []
        self.dgl = ode(self.funct).set_integrator('dopri5')
        self.dgl.set_f_params(gamma)
        # self.dgl.set_solout(self.solout)
        self.dgl.set_initial_value(self.r0, self.t0)

        # def solout(self, t, r):
        #   self.xt.append(copy.deepcopy(r[0]))
        #  self.yt.append(copy.deepcopy(r[1]))

    def solve(self, tmax):
        return self.dgl.integrate(tmax)


def f_duff(t, y, gamma):
    dx = y[1]
    dy = - gamma * y[1] + y[0] - y[0] ** 3
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


starttime = time.time()

y_min = -4.
y_max = 4.
y_resolution = 0.05
x_min = -4.
x_max = 4.
x_resolution = 0.05

samples = (x_max - x_min) / x_resolution

initial = [(x, y) for x in np.arange(x_min, x_max + x_resolution, x_resolution) for y in
           np.arange(y_min, y_max + y_resolution, y_resolution)]

fig_plane = plt.figure(1)
fig_pixels = 1024
markersize = fig_pixels / samples
print markersize
ax1 = fig_plane.add_subplot(111)
plt.grid()

gamma = 1

for ini in initial:
    cur = (dgl(f_duff, ini, gamma))
    endv = [2, 3]
    print ini
    while abs(endv[1]) > 0.5 or (abs(1 - abs(endv[0])) > 0.5 and ini != (0, 0)):
        endv = cur.solve(cur.dgl.t + 0.5)
        # print endv

    # line, = ax1.plot(cur.xt, cur.yt)
    pt = ax1.plot(ini[0], ini[1])
    colors = ['r', 'b']
    fix = int(np.sign(endv[0]) + 1) / 2
    # print fix
    plt.setp(pt, marker='.', color=colors[fix], linewidth=2.0, markersize=2 * markersize)
    # x_ar = np.array(cur.xt)
    # y_ar = np.array(cur.yt)
    # add_arrow(line, None, 'right', 15, line.get_color())

insize = 12
fig_plane.set_size_inches(insize, insize)
fig_plane.set_dpi(fig_pixels / insize)
fig_plane.savefig('basin.svg')
print("Calculation took: \t {:.2f}s".format(time.time() - starttime))
plt.show()