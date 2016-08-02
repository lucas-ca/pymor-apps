from pymor.grids.tria import TriaGrid

from stokes.analyticalproblems.cavity import CavityProblem

import numpy as np
from matplotlib import pyplot as plt


def setup_analytical_solution():
    u = lambda x: x[..., 0]**2 *\
                  (np.ones_like(x[..., 0]) - x[..., 0])**2 *\
                  2*x[..., 1] *\
                  (np.ones_like(x[..., 0]) - x[..., 1]) *\
                  (2*x[..., 1] - np.ones_like(x[..., 0]))

    v = lambda x: x[..., 1]**2 *\
                  (np.ones_like(x[..., 0]) - x[..., 1])**2 *\
                  2*x[..., 0] *\
                  (np.ones_like(x[..., 0]) - x[..., 0]) *\
                  (np.ones_like(x[..., 0]) - 2*x[..., 0])
    p = lambda x: x[..., 0] *\
                  x[..., 1] *\
                  (np.ones_like(x[..., 0]) - x[..., 0]) *\
                  (np.ones_like(x[..., 0]) - x[..., 1])

    return {'u': u, 'v': v, 'p': p}

# constants
viscosity = 1
num_intervals = (80, 80)
quiver_intervals = (12, 12)

problem = CavityProblem(viscosity)
domain = problem.domain.domain

plot_grid = TriaGrid(num_intervals, domain)
x = plot_grid.centers(2)[..., 0]
y = plot_grid.centers(2)[..., 1]
triangles = plot_grid.subentities(0, 2)

quiver_grid = TriaGrid(quiver_intervals, domain)
x_quiver = quiver_grid.centers(2)[..., 0]
y_quiver = quiver_grid.centers(2)[..., 1]

solution = setup_analytical_solution()

u = solution['u'](quiver_grid.centers(2))
v = solution['v'](quiver_grid.centers(2))
p = solution['p'](plot_grid.centers(2))

plt.figure('Cavity analytisch p')
plt.tripcolor(x, y, triangles, p)
plt.colorbar()

plt.figure('Cavity analytisch u')
plt.quiver(x_quiver, y_quiver, u, v)

# END
z = 0



