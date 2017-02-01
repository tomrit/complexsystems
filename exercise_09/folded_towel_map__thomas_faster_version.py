#!/bin/bash

#############################################################################################################
#       Introduction to Complex Systems  --  Exercise-Sheet 9  --  The folded Towel Map.                    #
#       Thomas Rittmann, Dec 2016                                                                           #
#                                                                                                           #
#       This program visualizes the "Folded Towel Map" and calculates the largest Lyapunov exponent of      #
#       the system (using the limit method of a small pertubation) as well as the whole Lyapunov spectrum   #
#       (all 3 Lyapunov exponents), by using the Q-R-decomposition for re-orthonomalization.                #
#       It returns convergence plots of these exponents as well as the final values                         #
#                                                                                                           #
#############################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D

# -------------------- Simulation Step Parameters --------------------------
plotSteps = 100000  # for Map Visualization
spektrumSteps = 3000  # for convergence of Lyapunov Spectrum
largestSteps = 700  # for convergence of largest L. Exponent (a) --> choose <1000 to avoid overflow in that method!
transientSteps = 1000  # transient Steps to discard in every run (let system evolve onto attractor)


def map_step(x):
    x_new = 3.8 * x[0] * (1 - x[0]) - 0.05 * (x[1] + 0.35) * (1 - 2 * x[2])
    y_new = 0.1 * ((x[1] + 0.35) * (1 - 2 * x[2]) - 1) * (1 - 1.9 * x[0])
    z_new = 3.78 * x[2] * (1 - x[2]) + 0.2 * x[1]
    return [x_new, y_new, z_new]


def jacobian(x):
    ja = np.array(
        [[3.8 * (1 - x[0]) - 3.8 * x[0], -.05 * (1 - 2 * x[2]), .1 * (x[1] + 0.35)],
         [.19 * ((x[1] + .35) * (1 - 2 * x[2]) - 1), .1 * (1 - 2 * x[2]) * (1 - 1.9 * x[0]),
          -.2 * (x[1] + .35) * (1 - 1.9 * x[0])],
         [0, .2, 3.78 * (1 - x[2]) - 3.78 * x[2]]])
    return ja


xVec = np.zeros((plotSteps + 1, 3))
# Random initial coordinates
xVec[0,] = [np.random.random(), np.random.random(), np.random.random()]

# small disturbance used for method (a)
epsilon = 1e-10
y0 = np.array([[epsilon], [epsilon], [epsilon]])
yVec = y0
largestExp = np.zeros(largestSteps)

# Random initial Q-Matrix
qMat = np.random.rand(3, 3)
currentSpec = np.zeros((spektrumSteps, 3))
totalSpec = np.zeros((3,))

# -------------------- Transient Iterations (let attractor establish) --------------------------
for i in range(transientSteps):
    xVec[i + 1] = map_step(xVec[i])
    jacob = jacobian(xVec[i + 1])
    yMat = np.dot(jacob, qMat)
    qMat, rMat = np.linalg.qr(yMat)

# dicard transient steps and start over
xVec[0, :] = xVec[transientSteps, :]

# --------------------- Iterates on attractor --------------------------------
for i in range(plotSteps):
    xVec[i + 1] = map_step(xVec[i])

    # for given number of steps, calculate evolving Lyapunov Spectrum using QR-decomposition
    if (i < spektrumSteps):
        jacob = jacobian(xVec[i + 1])
        yMat = np.dot(jacob, qMat)
        # qr decomposition from linalg-lib  ("re-orthogonalize")
        qMat, rMat = np.linalg.qr(yMat)
        # correct signs (need positiv diagonal elements in qMat)
        qMat = np.dot(qMat, np.diag(np.sign(np.diag(rMat))))
        # add diagonal elements of r-Matrix for Spectrum calculation
        addSpec = np.log(np.abs(np.diag(rMat)))
        # summing up ...
        totalSpec += addSpec
        # save current spectrum (after i steps) for plotting the convergence
        currentSpec[i] = copy.copy(totalSpec / (i + 1))

        # calculate the largest lyapunov exponent in the way required in task a)
        if (i < largestSteps):
            yVec = np.dot(jacob, yVec)
            # save current largest Exp (after i steps) for plotting the convergence
            largestExp[i] = 1. / (i + 1) * (np.log(np.linalg.norm(yVec)) - np.log(np.linalg.norm(y0)))

# final Lyapunov spectrum:
lyapunovSpec = currentSpec[spektrumSteps - 1]

print('Final largest Lyapunov Exponent (task a):  {}'.format(largestExp[largestSteps - 1]))
print('Final Lyapunov spectrum (task b): {}'.format(lyapunovSpec))

# -------------- plot towel map / attractor in 3d ------------------
markersize = 0.2
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.plot(xVec[:, 0], xVec[:, 1], xVec[:, 2], '.', markersize=markersize)
ax1.set_xlabel(r"$x_1$")
ax1.set_ylabel(r"$x_2$")
ax1.set_zlabel(r"$x_3$")
ax1.set_title("Visualization of the towel map attractor")

# -------------- plot towel map / attractor in 2d ------------------
fig2, ax2 = plt.subplots()
ax2.plot(xVec[:, 0], xVec[:, 1], '.', markersize=markersize)
ax2.set_xlabel(r"$x_1$")
ax2.set_ylabel(r"$x_2$")
ax2.set_title("2D-Visualization of the towel map")

# -------------- plot convergence of Lyapunov Spectrum ------------------
fig3, ax3 = plt.subplots()
ax3.plot(currentSpec)
ax3.set_xlabel("N")
ax3.set_ylabel("lyapunov exponent")
ax3.set_title("Convergence of Lyapunov Spectrum.   Final exponents: {:.3f}, {:.3f}, {:.3f}".format(lyapunovSpec[0],
                                                                                                   lyapunovSpec[1],
                                                                                                   lyapunovSpec[2]))

# -------------- plot convergence of largest Lyapunov Exponent (a) ------------------
fig4, ax4 = plt.subplots()
ax4.plot(largestExp)
ax4.set_xlabel("N")
ax4.set_ylabel("lyapunov exponent")
ax4.set_title("Convergence of the largest Lyapunov Exponent.   Final L.E.: {:.3f}".format(largestExp[largestSteps - 1]))

plt.show()
