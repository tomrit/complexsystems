import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tools import Dgl


def f_shinriki_dgl(t, r):
    # parameters
    c1 = 10e-9
    c2 = 100e-9
    l = 0.32
    r1 = 22e3
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


def run(r0):
    dgl = Dgl(f_shinriki_dgl, r0, trace=True)
    dgl.solve(1000)
    return dgl.rt


r0 = [0, 0.5, 0.5]
rt = run(r0)
rt = np.array(rt)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rt[:, 0], rt[:, 1], rt[:, 2])
plt.show()
