from stokes.analyticalproblems.poiseuille import PoiseuilleProblem
from stokes.analyticalproblems.affine_transformed_stokes import AffineTransformedStokes
from stokes.discretizers.stationary_incompressible_stokes import discretize_stationary_incompressible_stokes
from pymor.parameters.base import Parameter

import numpy as np

PROBLEM = 1
FEM_ORDER = 2
PLOT_TYPE = 1
RESOLUTION = 20
DIA = True
H = 20

trans = -1.*np.eye(2)
trans = np.array([[1., 1.], [0., 1.]])
trans = np.array([[0., -1.], [1., 0.]])

mu = Parameter({'transformation': trans})

if PROBLEM == 1:
    problem = PoiseuilleProblem()
    problem_trans = AffineTransformedStokes(problem)
else:
    raise NotImplementedError

if DIA:
    disc, g = discretize_stationary_incompressible_stokes(problem, 1./H, fem_order=FEM_ORDER, plot_type=PLOT_TYPE,
                                                          resolution=RESOLUTION)
    disc_trans, g_trans = discretize_stationary_incompressible_stokes(problem_trans, 1./H, fem_order=FEM_ORDER,
                                                                      plot_type=PLOT_TYPE, resolution=RESOLUTION)
else:
    raise NotImplementedError

sol = disc.solve()
sol_trans = disc_trans.solve(mu)
sol_alt = np.genfromtxt('/home/lucas/sol_alt')
sol_neu = sol_trans._array
#disc.visualize(sol)
disc.visualize(sol_trans)

