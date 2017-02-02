#!/bin/bash
######################################################################################
#    Intro to Complex Systems  - Exercise-Sheet 12 -  PCA-Attractor Reconstruction   #
#    Thomas Rittmann, Jan 2017                                                       #
#                                                                                    #
#    This program simulates the chaotic Lorenz System and performs a SVD on it.      #
#    The singular values are then compared for a noisy and a pure signal, which      #
#    shows a cut-off.                                                                #
#    The system is then reconstructed (in 2D and 3D) via this SVD, which is compared #
#    to a reconstruction from the delay coordinates.                                 #
######################################################################################

import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode


def Lor(t, x):
    return ([10 * (x[1] - x[0]), x[0] * (28 - x[2]) - x[1], x[0] * x[1] - 8 / 3 * x[2]])


# ---- Integration parameters ----
t0 = 0
dt = 0.01
Nt = 10000
t1 = Nt * dt - dt
t = np.linspace(t0, t1, Nt)

# ----- Integrate/Iterate the system -----
y0 = np.array([0.1, 0.1, 0.1])
solver = ode(Lor)
solver.set_integrator('dopri5')
solver.set_initial_value(y0, t0)
sol = np.empty([Nt, 3])
sol[0, :] = y0
k = 1

while solver.successful() and solver.t < t1:
    solver.integrate(t[k])
    sol[k, :] = np.asarray(solver.y)
    k += 1

# ----- Plot the Lorenz Attractor -----
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(sol[:, 0], sol[:, 1], sol[:, 2])
plt.savefig('Lorentzattraktor')

# ----- Add the Gaussian Noise -----
x = sol[:, 0]
noise = np.random.normal(0, 0.2, Nt)
s = x + noise

xb = x - np.mean(x)
sb = s - np.mean(s)

d = 20
Xx = np.zeros((Nt - d, d))
Xs = np.zeros((Nt - d, d))

for i in range(d):
    Xx[:, i] = x[i:Nt - d + i]
    Xs[:, i] = s[i:Nt - d + i]

# ---- Perform the single value decomposition algorithm ----
Ux, sx, Vx = np.linalg.svd(Xx)
Us, ss, Vs = np.linalg.svd(Xs)

# ---- plot sigma_k vs k --> compare noisy signal to pure one ----
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.semilogy(sx, label='signal')
ax2.semilogy(ss, label='noisy signal')
ax2.legend()
ax2.set_xlabel(r"$k$")
ax2.set_ylabel(r"$\sigma_k$")
fig2.savefig('singular-values')

# ---- get coordinates reconstructed from SVD -----
xu = Us[:, 0]
yu = Us[:, 1]
zu = Us[:, 2]

# ---- plot 2D reconstruction via SVD ----
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(xu, yu)
fig3.savefig('2d-svd-reconstruction')

# ---- plot 3D reconstruction via SVD ----
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.plot(xu, yu, zu)
fig4.savefig('3d-svd-reconstruction')

# ----- reconstruction via delay -----
T_d = 10
S_d = np.zeros((Nt - 2 * T_d, 3))
S_d[:, 0] = s[:Nt - 2 * T_d]
t = np.roll(s, -T_d)
S_d[:, 1] = t[:Nt - 2 * T_d]
t = np.roll(s, -2 * T_d)
S_d[:, 2] = t[:Nt - 2 * T_d]

# ----- plot 2D reconstruction via delay -----
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.plot(S_d[:, 0], S_d[:, 1])
fig5.savefig('2d-delay-coordinates')

# ----- plot 3D reconstruction via delay ----
fig6 = plt.figure()
ax6 = fig6.add_subplot(111, projection='3d')
ax6.plot(S_d[:, 0], S_d[:, 1], S_d[:, 2])
fig6.savefig('3d-delay-coordinates')

plt.show()
