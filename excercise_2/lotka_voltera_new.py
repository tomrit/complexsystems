#! /usr/bin/env python


import numpy as np
import time as time

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

t = int(time.time())
print t
sample = 50

a = 200.
b = 30.
c = 200.
d = 40.

dis = 1 - 4 * d / a * (d / b - 1)
print dis
xf = b / d
print xf
yf = a / c * (1 - b / d)
print yf

x, y = np.meshgrid(np.linspace(xf - 0.1, xf + 0.1, 200), np.linspace(yf - 0.1, yf + 0.1, 200))
# x,y=np.meshgrid(np.linspace(0,1.5,200), np.linspace(0,1.5,200))
dx = a * (1 - x) * x - c * x * y
dy = -b * y + d * x * y
speed = np.sqrt(dx * dx + dy * dy)

fig1 = plt.quiver(x, y, dx, dy)

fig = plt.figure(figsize=(13, 10))
plt.title('Extended Lotka-Volterra model')
plt.xlabel('Prey population')
plt.ylabel('Predator population')
plt.streamplot(x, y, dx, dy, density=2)
# linewidth=5*speed/speed.max()

pp = PdfPages("./test.pdf")
plt.show()
pp.savefig(fig)
pp.close()
