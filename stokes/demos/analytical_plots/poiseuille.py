from pymor.grids.tria import TriaGrid

from stokes.analyticalproblems.poiseuille import PoiseuilleProblem

import numpy as np
from matplotlib import pyplot as plt


def setup_analytical_solution(width, height):
    assert isinstance(width, (int, float))
    assert isinstance(height, (int, float))

    u = lambda x: -4.0/(height**2)*x[..., 1]**2 + 4.0/height*x[..., 1]
    v = lambda x: np.zeros_like(x[..., 0])
    p = lambda x: 8.0/(height**2) * (width - x[..., 0])

    return {'u': u, 'v': v, 'p': p}

# constants
width = 4
height = 1
viscosity = 1
num_intervals = (80, 80)
quiver_intervals = (12, 12)

problem = PoiseuilleProblem(width, height, viscosity)
domain = problem.domain.domain

plot_grid = TriaGrid(num_intervals, domain)
x = plot_grid.centers(2)[..., 0]
y = plot_grid.centers(2)[..., 1]
triangles = plot_grid.subentities(0, 2)

quiver_grid = TriaGrid(quiver_intervals, domain)
x_quiver = quiver_grid.centers(2)[..., 0]
y_quiver = quiver_grid.centers(2)[..., 1]

solution = setup_analytical_solution(width, height)

u = solution['u'](quiver_grid.centers(2))
v = solution['v'](quiver_grid.centers(2))
p = solution['p'](plot_grid.centers(2))

plt.figure('Poiseuille analytisch p')
plt.tripcolor(x, y, triangles, p)
plt.colorbar()

plt.figure('Poiseuille analytisch u')
plt.quiver(x_quiver, y_quiver, u, v)

z = 0



