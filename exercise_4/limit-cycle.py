#! /usr/bin/env python

from matplotlib.backends.backend_pdf import PdfPages

from tools import Dgl, Mesh, get_subplots_squared, get_a4_width
import matplotlib.pyplot as plt
import numpy as np


def f_limit_cycle(t, y, a):
    x1 = y[0]
    x2 = y[1]
    dx = x2
    dy = - a * x2 * (x1 ** 2 + x2 ** 2 - 1) - x1
    return [dx, dy]


def f_limit_cycle_xy(x_, y_, a_):
    return f_limit_cycle(t=None, y=[x_, y_], a=a_)


x_min = - 3
x_max = - x_min
y_min = x_min
y_max = - y_min
# a = 1.
# a_range = np.linspace(0.9, 1.1, 16)
a_range = np.array([0, 0.5, 1.06, 3])

mesh = Mesh(x_min, x_max, y_min, y_max)
n_rows, n_cols = get_subplots_squared(a_range.size)

a4_width = get_a4_width()

pp = PdfPages("exercise4_Limit-Cycle.pdf")
fig1, ax_array = plt.subplots(n_rows, n_cols, figsize=(a4_width, a4_width))
ax_array_flat = ax_array.reshape(-1)

fig2, ax1 = plt.subplots(1, 1, figsize=(a4_width, a4_width))

# Solving differential equations with Runge-Kutta - dopri5
for a in a_range:
    x0 = (0.5, 0.7)
    dgl = (Dgl(f_limit_cycle, x0, a, True))
    while len(dgl.xt) < 50000:
        dgl.solve(50)
    
    ax1.plot(dgl.xt, dgl.yt, label="a = {:.1f}".format(a))
    length_traject = len(dgl.xt)
    print(length_traject)
    ax1.set_xlim(mesh.x_min, mesh.x_max)
    ax1.set_ylim(mesh.y_min, mesh.y_max)
    ax1.legend(loc="best")

pp.savefig(fig2, transparent=True, bbox_inches='tight')
pp.close()
plt.show()
