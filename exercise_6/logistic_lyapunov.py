import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d
import time as time
import sys
import copy
from matplotlib.backends.backend_pdf import PdfPages
from tools import get_a4_width

x0=0.5
iterates=10000
discard=1000
r_min=2.001
r_max=3.999
r_N=12000
rs=np.linspace(r_min,r_max,r_N)
lyapunov=[]

start_time=time.time()

def map_step(x, r):
    x_new = r * x * (1 - x)
    return x_new



def calc_lyapunov(r):
    xn=x0
    lyapunov_sum=0
    for step in xrange(iterates):
        xn=map_step(xn,r)
        if step >= discard:
            log_arg=np.abs(r - 2 * r * xn)
            if log_arg==0:
                lyapunov_sum=-1e12
                break
            summand = np.log(log_arg)
            lyapunov_sum+=summand
    return lyapunov_sum/(iterates-discard)


for idx,r in enumerate(rs):
    lyapunov.append(calc_lyapunov(r))
    sys.stdout.write("\r[Progress: %s/%s]" % (idx+1, r_N))
    sys.stdout.flush()

outputname = "logistic_lyapunov__r_min_{}__r_max_{}__r_N_{}__iter_{}__disc_{}".format(r_min,r_max,r_N,iterates,discard)
fig=plt.figure()
ax=fig.add_subplot(111)
fig.set_size_inches(10, 6)
ax.plot(rs,np.array(lyapunov),'darkblue',linewidth=0.4)
ax.axhline(0, color='black')
ax.set_ylim(-2,0.8)
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$\lambda$',rotation=0)
ax.set_title(r'Lyapnuov exponent for the logistic map')
fig.savefig(outputname + ".png", dpi=500)

print('\n\nNumerical simulation took {:.2f}s'.format(time.time()-start_time))
plt.show()