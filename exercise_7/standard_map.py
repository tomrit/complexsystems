import numpy as np
import matplotlib.pyplot as plt


def bernoulli_shift(x, b):
    x_new = (b * x) % 1
    # x_new=4*x*(1-x)     #that would be the natural density for logistic r=4
    return x_new


iterates = 100000
x = np.random.random(1)[0]
print(x)

b = 3.0
xs = []
for idx in xrange(iterates):
    x = bernoulli_shift(x, b)
    xs.append(x)
print(x)
fig = plt.figure()
fig.set_size_inches(8, 6)
ax = fig.add_subplot(111)
ax.hist(xs, 100)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$H(x)$', rotation=0)
ax.set_title(r'Natural density of the generalized Bernoulli shift for $b={}$'.format(b))
fig.savefig("Bernoulli_density__b_{:.2f}.png".format(b), dpi=300)
plt.show()
