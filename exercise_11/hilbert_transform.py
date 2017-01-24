import numpy as np
import matplotlib.pylab as plt

from scipy.signal import hilbert


def ex_1d():
    N = 2 ** 16
    t_max = 40
    t = np.linspace(0, t_max, N)
    time_step = t_max / (N - 1)
    t_rad = t / (2 * np.pi)
    a = 2
    b = 0.5
    omega = 1.5

    s = a * np.sin(omega * t) + b * np.cos(2 * omega * t)
    s_analytic = hilbert(s)

    sp = np.fft.fft(s) / t_max / N
    freq = np.fft.fftfreq(t.shape[-1], d=time_step) * 2 * np.pi

    fig1, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(6, 12))
    ax1.set_title("Signal")
    ax1.plot(t_rad, s)
    ax1.set_ylim([-3, 3])
    ax1.set_xlabel("$t/2\pi$")

    ax2.set_title("Fourier transformation")
    ax2.plot(freq, sp.real, freq, sp.imag)
    ax2.set_xlim([-5, 5])
    ax2.set_xlabel("$\omega $")

    ax3.set_title("Hilbert transform")
    ax3.plot(t_rad, s_analytic.real, t_rad, s_analytic.imag)
    ax3.set_ylim([-3, 3])
    ax3.set_xlabel("$t/2\pi$")

    fig2, ax = plt.subplots()
    ax.plot(s_analytic.real, s_analytic.imag, ',')


if __name__ == '__main__':
    ex_1d()
    plt.show()
