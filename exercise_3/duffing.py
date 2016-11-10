#! /usr/bin/env python

# Exercises set 3:
import errno
import numpy as np
import time as time
import os
from scipy.integrate import ode

import matplotlib.pyplot as plt


class Dgl(object):
    def __init__(self, function, r0, gamma):
        self.function = function
        self.r0 = r0
        self.t0 = 0
        self.xt = []
        self.yt = []
        self.dgl = ode(self.function).set_integrator('dopri5')
        self.dgl.set_f_params(gamma)
        # self.dgl.set_solout(self.solout)
        self.dgl.set_initial_value(self.r0, self.t0)

    def solout(self, t, r):
        self.xt.append(np.copy.deepcopy(r[0]))
        self.yt.append(np.copy.deepcopy(r[1]))

    def solve(self, t_max):
        return self.dgl.integrate(t_max)


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


start_time = time.time()

y_min = -4.
y_max = 4.
y_resolution = 0.1
N_y = int((y_max - y_min) / y_resolution)
x_min = -4.
x_max = 4.
x_resolution = 0.1
N_x = int((x_max - x_min) / x_resolution)

samples = (x_max - x_min) / x_resolution

x_range = np.linspace(x_min, x_max, N_x)
y_range = np.linspace(y_min, y_max, N_y)
initial_coordinates = [(x, y) for x in x_range for y in y_range]

results = np.zeros([N_x, N_y])

fig_plane = plt.figure(1)
fig_pixels = 1024
marker_size = fig_pixels / samples
print(marker_size)
ax1 = fig_plane.add_subplot(111)
plt.grid()

fixed_points = {(-1, 0): 'r', (1, 0): 'b'}

gamma = 1


for idx, initial_point in enumerate(initial_coordinates):

    cur = (Dgl(f_duff, initial_point, gamma))
    coordinate = [2, 3]
    # print(initial_point)
    while abs(coordinate[1]) > 0.5 or (abs(1 - abs(coordinate[0])) > 0.5 and initial_point != (0, 0)):
        coordinate = cur.solve(cur.dgl.t + 0.5)
        # print endv

    # line, = ax1.plot(cur.xt, cur.yt)
    pt = ax1.plot(initial_point[0], initial_point[1])

    plt.setp(pt, marker='.', color=get_color(coordinate, fixed_points), linewidth=2.0, markersize=2 * marker_size)
    # x_ar = np.array(cur.xt)
    # y_ar = np.array(cur.yt)
    # add_arrow(line, None, 'right', 15, line.get_color())

fig_size_inch = 13
fig_plane.set_size_inches(fig_size_inch, fig_size_inch)
fig_plane.set_dpi(fig_pixels / fig_size_inch)

filename_figure = os.path.join('graphics', 'basin.svg')

if not os.path.exists(os.path.dirname(filename_figure)):
    try:
        os.makedirs(os.path.dirname(filename_figure))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# Edit: In Python 3.2+, there is a more elegant way that avoids the race condition above:
# os.makedirs(os.path.dirname(filename_figure), exist_ok=True)

fig_plane.savefig(filename_figure)

print("Calculation took: \t {:.2f}s".format(time.time() - start_time))
plt.show()
