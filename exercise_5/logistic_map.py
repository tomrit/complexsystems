import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d
import time as time
import copy
from matplotlib.backends.backend_pdf import PdfPages
from tools import get_a4_width

global x_start


def map_step(x, r):
    x_new = r * x * (1 - x)
    return x_new


# find the end orbit points / attractor for a given parameter r:
def attractor(r, steps_total, steps_discard=0, plot_show=False):
    time_series = np.zeros(steps_total)
    x = x_start

    for step in range(0, steps_total):
        time_series[step] = x
        x = map_step(x, r)

    if plot_show:
        fig_series = plt.figure()
        ax3 = fig_series.add_subplot(111)
        ax3.plot(range(0, steps_total), time_series)
        plt.show()

    stable_series = time_series[steps_discard:steps_total]
    return stable_series


steps_plotted_list = [1000]
markersizes = [0.01]

ax_bifurc = [0] * len(steps_plotted_list) * len(markersizes)
fig_bifurc = [0] * len(steps_plotted_list) * len(markersizes)
for column, steps_plotted in enumerate(steps_plotted_list):
    start_time = time.time()
    x_start = 0.6
    r_min = 2
    r_max = 4
    r_step = 0.0001
    x_bifurc = []
    r_bifurc = []
    steps_discard = 1000
    steps_total = steps_discard + steps_plotted
    for r in np.arange(r_min, r_max, r_step):
        at = attractor(r, steps_total, steps_discard, False)
        x_bifurc.append(copy.copy(at))
        r_bifurc.append([r] * len(at))

    print("Bifurcation diagram took: \t {:.2f}s".format(time.time() - start_time))
    for row, markersize in enumerate(markersizes):
        fignr = len(steps_plotted_list) * row + column
        fig_bifurc[fignr] = plt.figure()

        fig_bifurc[fignr].set_size_inches(10, 7)
        fig_bifurc[fignr].set_dpi = 300
        print fignr
        # ax_bifurc[fignr]=fig_bifurc.add_subplot(len(markersizes),len(steps_plotted_list),fignr+1)
        ax_bifurc[fignr] = fig_bifurc[fignr].add_subplot(1, 1, 1)
        ax_bifurc[fignr].plot(r_bifurc, x_bifurc, 'k.', markersize=markersize)
        ax_bifurc[fignr].set_xlabel(r'$r$')
        ax_bifurc[fignr].set_ylabel(r'$x_s$')
        ax_bifurc[fignr].set_title(r'Exercise 5.1.c)  Bifurcation Diagram of the Logistic Map')
        ax_bifurc[fignr].set_xlim(2, r_max)
        fig_bifurc[fignr].savefig(
            "logistic_map_m{}_d{}_r{}_p{}.png".format(markersize, steps_discard, r_step, steps_plotted), dpi=500)

        # pp = PdfPages("logistic_map.pdf")
        # pp = PdfPages("logistic_map_m{}_d{}_r{}_p{}.pdf".format(markersize,steps_discard,r_step,steps_plotted))
        # pp.savefig(fig_bifurc[fignr])
        # pp.close()
        # plt.xlim(2.9,3.1)
        # plt.ylim(0.64,0.68)
        # plt.show()
