import numpy as np
import matplotlib
import gc
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d
import time as time
import copy
import multiprocessing as mp

from tools import Dgl
from tools import plot_1Dfile


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
    return dgl


def get_zero_crossings(array):
    """
    get zero crossings in negative direction
    """
    return np.where(np.diff(np.sign(array)) == 2)[0]


zero_time = time.time()
cores = mp.cpu_count()
# for running without X-Server

# Parameter settinge:
t_max = 0.2
t_discard = 0.05
discard_frac = t_discard / t_max
t_step = 1e-5  # 1e-5 is nicer
N = 8
rmin = 20.62e3
rmax = 20.74e3
r1s = np.linspace(rmin, rmax, N)
markersize = 1

# r1s = np.linspace(19e3, 22e3,N)
# r1s=[20.699e3]
r0 = [0, 0.5, 0.75e-3]

def parameter_swipe():

    outputname = "shinriki_bifurc__rmin_{}__rmax_{}__N_{}__tmax_{}__tdis_{}__tstep_{}".format(rmin, rmax, N, t_max,
                                                                                              t_discard, t_step)
    print(
    "This simulation evaluates the bifurcation diagram of the Shinriki Oscillator for the following parameters:\n")
    print("r_min={:.0f}".format(rmin))
    print("r_max={:.0f}".format(rmax))
    print("r_steps={}".format(N))
    print("dgl_tmax={}".format(t_max))
    print("dgl_t_discarded={}".format(t_discard))
    print("dgl_t_stepsize={}\n".format(t_step))

    # initialize output figure:
    fig_bifurc = plt.figure()
    ax_bifurc = fig_bifurc.add_subplot(111)

    pool = mp.Pool(processes=cores)


    res = pool.map(main_eval, (1, 2, 3, 4))
    print("Generating plots and output files...")
    # print(res[1][1][0])
    datafile_path = outputname + ".txt"
    datafile_id = open(datafile_path, 'w+')
    datafile_id.write("r" + " " * 9 + "V1\n\n")
    for idx in range(0, cores):
        for jdx in range(0, N / cores):
            ax_bifurc.plot(res[idx][0][jdx] / 1000, res[idx][1][jdx], '.r', markersize=markersize)
            temp_data = np.array([np.array(res[idx][0][jdx] / 1000), np.array(res[idx][1][jdx])])
            temp_data = temp_data.T
            np.savetxt(datafile_id, temp_data, fmt=['%f', '%f'])
    datafile_id.close()

    ax_bifurc.set_xlim(rmin / 1000 - 0.02, rmax / 1000 + 0.02)
    ax_bifurc.set_xlabel(r'$R_1$ [k$\Omega$]')
    ax_bifurc.set_ylabel(r'$V_1$ [V]')
    ax_bifurc.set_title(r'Bifurcation Diagram of the Shinriki Oscillator ($V_2=0$)')
    fig_bifurc.set_size_inches(10, 7)
    fig_bifurc.set_dpi = 500
    fig_bifurc.savefig(outputname + ".png", dpi=500)

    print("\nThe simulation took {:.2f} s".format(time.time() - zero_time))
    # plt.show()


gc.enable()
def main_eval(thread_nr):
    v1_poincare = []
    r1_vec = []
    for idx_r1, r1 in enumerate(r1s[N / cores * (thread_nr - 1):N / cores * thread_nr]):
        print("[{}/{}]: starting for r1 = {:.0f} Ohm".format(idx_r1 * cores + thread_nr, N, r1))
        dgl_time = time.time()
        dgl = run(r0, t_max, t_step, r1)
        gc.collect()  # free all unallocated memory (otherwise RAM usage rises to 10GB in some minutes) don't know reason
        print("[{}/{}]: solving took {:.2f}s".format(idx_r1 * cores + thread_nr, N, time.time() - dgl_time))
        traject = np.array(dgl.rt)
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
        print("[{}/{}]: Saved {} crossings trough cross section (V2=0)\n\r".format(idx_r1 * cores + thread_nr, N,
                                                                                   nr_crossings))
    return r1_vec, v1_poincare
    # ax_bifurc.plot(r1_vec / 1000, v1_poincare, '.r',markersize=markersize)
    # qual=np.max(v1)-np.min(v1)
    # print(qual)


parameter_swipe()
# plot_1Dfile("test.txt")
