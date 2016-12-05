import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d
import time as time
import copy
import multiprocessing as mp

from tools import Dgl


def f_shinriki_dgl(t, r, r1=22e3):
    # parameters
    c1 = 10e-9
    c2 = 100e-9
    l = 0.32
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


def run(r0, t_max, t_step=0.001, r1=22e3):
    start_time = time.time()
    dgl = Dgl(f_shinriki_dgl, r0, r1, trace=True)
    while dgl.dgl.t < t_max:
        dgl.solve(dgl.dgl.t + t_step)
    print("Running dgl solver took: \t {:.2f}s".format(time.time() - start_time))
    return dgl


def get_zero_crossings(array):
    """
    get zero crossings in negative direction
    """
    return np.where(np.diff(np.sign(array)) == 2)[0]


cores = mp.cpu_count()

# Parameter settinge:
t_max = 0.2
t_discard = 0.05
discard_frac = t_discard / t_max
t_step = 5e-5  # 1e-5 is nicer
N = 8
r1s = np.linspace(20.62e3, 20.75e3, N)
markersize = 1

# r1s = np.linspace(19e3, 22e3,N)
# r1s=[20.699e3]
r0 = [0, 0.5, 0.75e-3]
v1s = []
rrls = []


def parameter_swipe():
    # t_max = 1.5  # better 1.0
    # t_discard=1.0


    # initialize output figure:
    fig_bifurc = plt.figure()
    ax_bifurc = fig_bifurc.add_subplot(111)

    pool = mp.Pool(processes=cores)

    res = pool.map(main_eval, (1, 2, 3, 4))
    # print(res[1][1][0])

    for idx in range(0, cores):
        for jdx in range(0, N / cores):
            ax_bifurc.plot(res[idx][0][jdx] / 1000, res[idx][1][jdx], '.r', markersize=markersize)

    ax_bifurc.set_xlim(20.5, 20.8)
    # ax_bifurc.set_xlim(18.5,19.5)
    # ax_bifurc.set_xlim(20.690,20.710)
    # ax_bifurc.set_ylim(0.24, 0.32)
    ax_bifurc.set_xlabel(r'$R_1$ [k$\Omega$]')
    ax_bifurc.set_ylabel(r'$V_1$ [V]')
    ax_bifurc.set_title(r'Bifurcation Diagram of the Shinriki Oscillator ($V_2=0$)')
    fig_bifurc.set_size_inches(10, 7)
    fig_bifurc.set_dpi = 500
    fig_bifurc.savefig("shinriki_bifurcation_t.png", dpi=500)

    plt.show()


def main_eval(thread_nr):
    v1_poincare = []
    r1_vec = []
    for idx_r1, r1 in enumerate(r1s[N / cores * (thread_nr - 1):N / cores * thread_nr]):
        print("[{}/{}]: Running solver for \t r1 = {:.0f} Ohm".format(idx_r1 + 1, N, r1))
        dgl = run(r0, t_max, t_step, r1)
        traject = np.array(dgl.rt)
        # change start_idx depending on resolution --> maybe better throw away fixed number of zero_crossings(prob:slower)
        # start_idx = 300000
        start_idx = np.floor(discard_frac * len(traject))
        traject = traject[start_idx:]
        zero_crossings = get_zero_crossings(traject[:, 1])

        # linear interpolation: very essential here! brings much more than finer resolution!
        v2dif = traject[zero_crossings + 1, 1] - traject[zero_crossings, 1]
        v1dif = traject[zero_crossings + 1, 0] - traject[zero_crossings, 0]
        v2part = -traject[zero_crossings, 1]
        v1add = v2part / v2dif * v1dif

        v1_poincare.append((copy.copy(traject[zero_crossings, 0] + v1add)))
        nr_crossings = len(v1_poincare[idx_r1])
        r1_vec.append((copy.copy(np.array([copy.copy(r1)] * nr_crossings))))
        print("Saved {} crossings trough cross section (V2=0)\n\r".format(nr_crossings))
    return r1_vec, v1_poincare
    # ax_bifurc.plot(r1_vec / 1000, v1_poincare, '.r',markersize=markersize)
    # qual=np.max(v1)-np.min(v1)
    # print(qual)


parameter_swipe()
