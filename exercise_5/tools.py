from scipy.integrate import ode
import numpy as np
import copy


class Dgl(object):
    """
    Solve DGLs with dopri5 integrator
    """
    def __init__(self, function, r0, parameter, trace=False):
        self.function = function
        self.r0 = r0
        self.t0 = 0
        self.rt = []
        self.xt = []
        self.yt = []
        self.dgl = ode(self.function).set_integrator('dopri5')
        self.dgl.set_f_params(parameter)
        if trace:
            self.dgl.set_solout(self.solout)
        self.dgl.set_initial_value(self.r0, self.t0)

    def solout(self, t, r):
        self.rt.append(copy.copy(r))
        self.xt.append(copy.copy(r[0]))
        self.yt.append(copy.copy(r[1]))

    def solve(self, t_max):
        return self.dgl.integrate(t_max)


class Mesh(object):
    """
    Mesh Parameters including its size and sample resolution
    """

    def __init__(self, x_min=0., x_max=1., y_min=0., y_max=1., sample=250):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.sample = sample

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def __repr__(self):
        return "Mesh(xMin = {}, xMax = {},yMin = {}, yMax = {}, sample = {})" \
            .format(self.x_min, self.x_max, self.y_min, self.y_max, self.sample)

    def create_phase_space(self, function, function_params):
        """
        Create a mesh from differential equation function on current Mesh
        :param function: Calculate dx, dy from x, y - [dx, dy] = function(x,y)
        :return: x, y, dx, dy

        """
        x, y = np.meshgrid(np.linspace(self.x_min, self.x_max, self.sample),
                           np.linspace(self.y_min, self.y_max, self.sample))
        [dx, dy] = function(x, y, function_params)
        return x, y, dx, dy


def get_subplots_squared(length):
    rows = np.floor(np.sqrt(length))
    columns = np.ceil(length / rows)
    return int(rows), int(columns)


def get_a4_width():
    a4_width = 448.13095 / 72.27
    return a4_width
