#! /usr/bin/env python

# Exercises set 3:
import errno
import numpy as np
import time as time
import os
from scipy.integrate import ode
import sys
import copy

import matplotlib.pyplot as plt


class Dgl(object):
    def __init__(self, function, r0, gamma, trace=False):
        self.function = function
        self.r0 = r0
        self.t0 = 0
        self.xt = []
        self.yt = []
        self.dgl = ode(self.function).set_integrator('dopri5')
        self.dgl.set_f_params(gamma)
        if trace:
            self.dgl.set_solout(self.solout)
        self.dgl.set_initial_value(self.r0, self.t0)

    def solout(self, t, r):
        self.xt.append(copy.copy(r[0]))
        self.yt.append(copy.copy(r[1]))

    def solve(self, t_max):
        return self.dgl.integrate(t_max)


class MeshSize(object):
    """
    Mesh Parameters including its size and sample resolution
    """

    def __init__(self, x_min=0., x_max=1., y_min=0., y_max=1., sample=250):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.sample = sample

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= y_max


def f_duff(t, y, gamma):  # Is t needed here?
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


def find_nearest(array, value):
    idx = np.linalg.norm(array - value, axis=1).argmin()
    return array[idx]


def get_color(coordinate, color_dictionary):
    """

    :param coordinate: [x,y]
    :param color_dictionary: {fix_point (tuple): color_code}
    :return: color_code
    """
    fix_points = np.array(list(color_dictionary.keys()))
    coordinate_nearest = find_nearest(fix_points, coordinate)
    return color_dictionary[tuple(coordinate_nearest)]


def get_number(coordinate, number_dictionary):
    """

    :param coordinate: [x,y]
    :param number_dictionary: {fix_point (tuple): number}
    :return: number
    """
    fix_points = np.array(list(number_dictionary.keys()))
    coordinate_nearest = find_nearest(fix_points, coordinate)
    return number_dictionary[tuple(coordinate_nearest)]


def get_basins(mesh_size):
    start_time = time.time()

    y_min = mesh_size.y_min
    y_max = mesh_size.y_max
    N_y = mesh_size.sample
    x_min = mesh_size.x_min
    x_max = mesh_size.x_max
    N_x = mesh_size.sample

    x_range = np.linspace(x_min, x_max, N_x)
    y_range = np.linspace(y_min, y_max, N_y)

    results = np.zeros([N_x, N_y])

    # fig_plane = plt.figure(1)
    # fig_pixels = 1024
    # marker_size = fig_pixels / mesh_size.sample
    # ax1 = fig_plane.add_subplot(111)
    # plt.grid()

    fixed_points_color = {(-1, 0): 'r', (1, 0): 'b'}
    fixed_points_number = {(-1, 0): 1, (1, 0): 2}

    gamma = 1

    toolbar_width = np.size(x_range)
    sys.stdout.write("Basin\n [%s]\n" % (" " * toolbar_width))
    sys.stdout.flush()

    for idx, x in enumerate(x_range):
        for idy, y in enumerate(y_range):
            initial_point = (x, y)

            cur = (Dgl(f_duff, initial_point, gamma))
            coordinate = [2, 3]
            while abs(coordinate[1]) > 0.5 or (abs(1 - abs(coordinate[0])) > 0.5 and initial_point != (0, 0)):
                coordinate = cur.solve(cur.dgl.t + 0.5)

            results[idx, idy] = get_number(coordinate, fixed_points_number)
            # line, = ax1.plot(cur.xt, cur.yt)
            # pt = ax1.plot(initial_point[0], initial_point[1])

            # plt.setp(pt, marker='.', color=get_color(coordinate, fixed_points_color), linewidth=2.0,
            #          markersize=2 * marker_size)
            # x_ar = np.array(cur.xt)
            # y_ar = np.array(cur.yt)
            # add_arrow(line, None, 'right', 15, line.get_color())
        # Progressbar
        sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("\n")
    print("Calculating the basins took: \t {:.2f}s".format(time.time() - start_time))
    return results


def get_manifold(mesh_size, N):
    gamma = 1
    saddle = [0, 0]
    epsilon = 0.01
    x_range = np.linspace(saddle[0] - epsilon, saddle[0] + epsilon, N)
    y_range = np.linspace(saddle[1] - epsilon, saddle[1] + epsilon, N)

    xt_array = []
    yt_array = []
    for x in x_range:
        for y in y_range:
            if x == 0 and y == 0:
                continue
            initial_point = (x, y)
            cur = (Dgl(f_duff, initial_point, gamma, True))
            while mesh_size.contains(initial_point[0], initial_point[1]):
                initial_point = cur.solve(cur.dgl.t - 0.5)
            xt_array.append(cur.xt)
            yt_array.append(cur.yt)

    return xt_array, yt_array

y_min = -10.
y_max = 10.
x_min = -5.
x_max = 5.

N = 64
mesh_size = MeshSize(x_min, x_max, y_min, y_max, N)

xt_array, yt_array = get_manifold(mesh_size, 3)

results = get_basins(mesh_size)

filename_figure = os.path.join('graphics', 'basin.svg')

fig_plane2 = plt.figure(2)
fig_size_inch = 13
fig_pixels = 1024
fig_plane2.set_size_inches(fig_size_inch, fig_size_inch)
fig_plane2.set_dpi(fig_pixels / fig_size_inch)

ax2 = fig_plane2.add_subplot(111)

ax2.imshow(results.T, extent=[x_min, x_max, y_min, y_max], interpolation='none')

for idx, xt in enumerate(xt_array):
    ax2.plot(np.array(xt) * (-1), np.array(yt_array[idx]) * (-1))
ax2.set_xlim(mesh_size.x_min, mesh_size.x_max)
ax2.set_ylim(mesh_size.y_min, mesh_size.y_max)

if not os.path.exists(os.path.dirname(filename_figure)):
    try:
        os.makedirs(os.path.dirname(filename_figure))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# Edit: In Python 3.2+, there is a more elegant way that avoids the race condition above:
# os.makedirs(os.path.dirname(filename_figure), exist_ok=True)

# fig_plane.savefig(filename_figure)

plt.show()
