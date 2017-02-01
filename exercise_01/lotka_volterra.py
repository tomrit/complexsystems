###############################################################################################
# This is the program to plot the vector field and phase portrait of the Lotka Volterra model # 
#											      #
###############################################################################################

import numpy as np
import matplotlib.streamplot as streams
import matplotlib.pyplot as plt

sample = 50
# a=2 for stable node
a = 2
b = 0.3
c = 2
d = 0.2
dis = 1 - 4 * d / a * (d / b - 1)
print dis
xf = b / d
print xf
yf = a / c * (1 - b / d)
print yf
x, y = np.meshgrid(np.linspace(xf - 0.2, xf + 0.2, 200), np.linspace(yf - 0.2, yf + 0.2, 200))
# x,y=np.meshgrid(linspace(-1.5,1.5,200), linspace(-1,1,200))
dx = a * (1 - x) * x - c * x * y
dy = -b * y + d * x * y
dxlin = (-2 * a * xf + a - c * yf) * x - c * xf * y
dylin = d * yf * x + (-b + d * xf) * y
# fig=plt.quiver(x,y,dxlin,dylin)
# plt.show()

fig1, ax1 = plt.subplots()
strm = ax1.streamplot(x, y, dx, dy)
plt.show()
