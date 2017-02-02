#!/bin/bash
######################################################################################
#    Introduction to Complex Systems  --  Exercise-Sheet 11 --  Hilbert Transform.   #
#    Thomas Rittmann, Jan 2017                                                       #
#                                                                                    #
#    This program computes the Hilbert Transform of a given signal function          #
#    via its relation to the Fourier Transform (numerical FFT).                      #
#    The result is compared to the analytical solution calculated in the assignment. #
#                                                                                    #
######################################################################################

import numpy as np
import matplotlib.pyplot as plt


def hilbert(sig):
    # Fourier Transform
    fourier = np.fft.fft(sig)
    # Sign (minus for negative frequencies = 2nd half in FFT)
    fourier[len(fourier) / 2 + 1:] *= -1
    fourier *= -1j
    # Get Hilbert Transform by Inverse Fourier Transformation
    hilb = np.fft.ifft(fourier)
    return hilb


a = 2
b = 0.5
omega = 1.5

t_max = 20
# choose samplesize that is 2**x for FFT algorithm to work properly.
# at least very important: must be dividable by 2!
t = np.linspace(0, 20, 4096)
signal = a * np.sin(omega * t) + b * np.cos(2 * omega * t)
analyt = -a * np.cos(omega * t) + b * np.sin(2 * omega * t)

plt.plot(t, signal, label=r"signal function $2 \,\sin (1.5\, t) + 0.5\, \cos(1.5\, t)$")
plt.plot(t, analyt, 'g--', label="analytical hilbert transform")
plt.plot(t, hilbert(signal), 'r', label="numerical hilbert transform (via FFT)")
plt.legend()
plt.xlabel(r"$t$")
plt.ylim(-3, 4)
plt.show()
