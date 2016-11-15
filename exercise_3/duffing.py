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
from matplotlib.backends.backend_pdf import PdfPages

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
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


def f_duff(t, y,
           gamma):  # t is needed here: If it had just two arguments, dgl.ode would think of y as time and gamma as location
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


def get_basins(mesh_size, end_treshold, gamma):
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

    fixed_points_number = {(-1, 0): 1, (1, 0): 2}

    toolbar_width = np.size(x_range)
    sys.stdout.write("Basin\n [%s]\n" % (" " * toolbar_width))
    sys.stdout.flush()

    for idx, x in enumerate(x_range):
        for idy, y in enumerate(y_range):
            initial_point = (x, y)
            cur = (Dgl(f_duff, initial_point, gamma))
            coordinate = [2, 3]

            while (abs(coordinate[1]) > end_treshold or abs(
                        1 - abs(coordinate[0]))) > end_treshold and initial_point != (0, 0):
                coordinate = cur.solve(cur.dgl.t + 0.5)

            results[idx, idy] = get_number(coordinate, fixed_points_number)

        # Progressbar
        sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("\n")
    print("Calculating the basins took: \t {:.2f}s".format(time.time() - start_time))
    return results


def get_manifold(mesh_size, N, gamma):
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


y_min = -20.
y_max = 20.
x_min = -10.
x_max = 10.

N = 120

gamma_list = [0.5, 0.75, 1, 1.5, 2, 3, 5]
pp = PdfPages("basins_overview.pdf")

for i_gamma, gamma in enumerate(gamma_list):

    mesh_size = MeshSize(x_min, x_max, y_min, y_max, N)

    xt_array, yt_array = get_manifold(mesh_size, 5, gamma)

    results = get_basins(mesh_size, 0.1, gamma)

    fig_plane2 = plt.figure(i_gamma)
    fig_size_inch = 5.
    fig_pixels = 1600.
    fig_ratio = (y_max - y_min) / (x_max - x_min)
    fig_plane2.set_size_inches(fig_size_inch, fig_ratio * fig_size_inch)
    fig_plane2.set_dpi(fig_pixels / fig_size_inch)

    ax2 = fig_plane2.add_subplot(111)
    ax2.imshow(np.flipud(results.T), extent=[x_min, x_max, y_min, y_max], interpolation='none')

    for idx, xt in enumerate(xt_array):
        manifold_plot = ax2.plot(np.array(xt) * (-1), np.array(yt_array[idx]) * (-1))
        plt.setp(manifold_plot, color='y', linewidth=1.5)

    ax2.set_xlim(mesh_size.x_min, mesh_size.x_max)
    ax2.set_ylim(mesh_size.y_min, mesh_size.y_max)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'd$x$/d$t$')
    ax2.set_title(r'for $\gamma= ${}'.format(gamma))
    ax2.plot(-1, 0, 'o', color='#00BFFF')
    ax2.plot(1, 0, 'or', )
    ax2.plot(0, 0, 'o', color='#01DF01')

    pp_single = PdfPages("basins_gamma_{}.pdf".format(gamma))
    pp_single.savefig(fig_plane2)
    pp_single.close()
    pp.savefig(fig_plane2)

# plt.show()
pp.close()
