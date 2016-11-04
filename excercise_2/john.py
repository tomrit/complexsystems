#! /usr/bin/env python
# a bit of editing just to check out the branch functions

import numpy as np
import time as time

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Parameters(object):
    """
    Parameters to run the Lotka Volterra simulation
    """

    def __init__(self, a, b, c, d, description="tbd"):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.description = description

    def get_dis(self):
        return 1 - 4 * self.d / self.a * (self.d / self.b - 1)

    def get_xf(self):
        return self.b / self.d

    def get_yf(self):
        return self.a / self.c * (1 - self.b / self.d)

    def get_fp(self):
        xf = self.get_xf()
        yf = self.get_yf()
        return xf, yf

    def get_filename(self):
        return "{}-a{:.0f}b{:.0f}c{:.0f}d{:.0f}".format(self.description, self.a, self.b, self.c, self.d)

    def get_title(self):
        return r'{}: a = {}, b = {}, c = {}, d = {}'.format(self.description, self.a, self.b, self.c, self.d)

    def __repr__(self, *args, **kwargs):
        return "Parameters(a = {}, b = {}, c = {}, d = {}, FP = ({}, {}), dis = {:.2f})" \
            .format(self.a, self.b, self.c, self.d, self.get_xf(), self.get_yf(), self.get_dis())


class MeshParameters(object):
    """
    Mesh Parameters including its size and sample resolution
    """

    def __init__(self, x_min=0., x_max=1., y_min=0., y_max=1., sample=250):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.sample = sample

    @classmethod
    def from_fix_points(cls, parameters):
        xf, yf = parameters.get_fp()
        x_min = xf - 0.1
        x_max = xf + 0.1
        y_min = yf - 0.1
        y_max = yf + 0.1
        mesh_parameters = cls(x_min, x_max, y_min, y_max)
        return mesh_parameters

    def __repr__(self):
        return "MeshParameters(xMin = {}, xMax = {},yMin = {}, yMax = {}, sample = {})" \
            .format(self.x_min, self.x_max, self.y_min, self.y_max, self.sample)

    def create_mesh(self, parameters):
        x, y = np.meshgrid(np.linspace(self.x_min, self.x_max, self.sample),
                           np.linspace(self.y_min, self.y_max, self.sample))
        dx = parameters.a * (1 - x) * x - parameters.c * x * y
        dy = -parameters.b * y + parameters.d * x * y
        return x, y, dx, dy


def create_plots(parameters, show_plots=False, save_plots=False):
    start_time = time.time()
    mesh_params_zoom = MeshParameters.from_fix_points(parameters)
    mesh_params_overview = MeshParameters(0, 1.1, 0, 1.3)

    print(parameters)
    xf = parameters.get_xf()
    yf = parameters.get_yf()

    x, y, dx, dy = mesh_params_zoom.create_mesh(parameters)

    plt.rc('text')
    plt.rc('font', family='serif')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey='row', figsize=(15, 10))

    ax3.set_title('Zoom to fix point')
    ax3.plot(xf, yf, 'or')
    ax3.set_xlabel('Prey population')
    ax3.set_ylabel('Predator population')
    ax3.set_xlim(xf - 0.1, xf + 0.1)
    ax3.set_ylim(yf - 0.1, yf + 0.1)
    ax3.streamplot(x, y, dx, dy, density=2)

    mesh_params_zoom.sample = 20
    x, y, dx, dy = mesh_params_zoom.create_mesh(parameters)

    ax4.set_title('Zoom to fix point')
    ax4.quiver(x, y, dx, dy)
    ax4.plot(xf, yf, 'or')
    ax4.set_xlabel('Prey population')
    ax4.set_xlim(xf - 0.1, xf + 0.1)
    ax4.set_ylim(yf - 0.1, yf + 0.1)

    x, y, dx, dy = mesh_params_overview.create_mesh(parameters)

    ax1.set_title('Streamplot')

    ax1.streamplot(x, y, dx, dy, density=2)
    ax1.set_xlim(mesh_params_overview.x_min - 0.02, mesh_params_overview.x_max)
    ax1.set_ylim(mesh_params_overview.y_min - 0.02, mesh_params_overview.y_max)
    ax1.set_ylabel('Predator population')
    ax1.plot(xf, yf, 'or')
    ax1.plot(0, 0, 'or')

    mesh_params_overview.sample = 20
    x, y, dx, dy = mesh_params_overview.create_mesh(parameters)

    ax2.set_title('Quiverplot')

    ax2.quiver(x, y, dx, dy)
    ax2.plot(xf, yf, 'or')
    ax2.plot(0, 0, 'or')
    ax2.set_xlim(-0.02, 1.1)
    ax2.set_ylim(-0.02, 1.3)

    fig.suptitle(parameters.get_title(), size='x-large')

    if save_plots:
        pp = PdfPages("./{}.pdf".format(parameters.get_filename()))
        pp.savefig(fig)
        pp.close()

    print("Plotting {} took: \t {:.2f}s".format(parameters.description, time.time() - start_time))

    if show_plots:
        plt.show()


stableSpiral = Parameters(2., 15., 2., 20., "stable-spiral")
stableNode = Parameters(20., 3., 20., 4., "stable-node")
degenerateNode = Parameters(16., 2., 16., 4., "degenerate-node")

parameter_arr = [stableNode, stableSpiral, degenerateNode]
for parameter in parameter_arr:
    create_plots(parameter, True, True)
    # create_plots(parameter, True, True)
    # test
