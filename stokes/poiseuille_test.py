from stokes.analyticalproblems.poiseuille import PoiseuilleProblem
from stokes.discretizers.stationary_incompressible_stokes import discretize_stationary_incompressible_stokes


PROBLEM = 1
FEM_ORDER = 2
PLOT_TYPE = 1
RESOLUTION = 20
DIA = True
H = 20

if PROBLEM == 1:
    problem = PoiseuilleProblem()
else:
    raise NotImplementedError

if DIA:
    disc, g = discretize_stationary_incompressible_stokes(problem, 1./H, fem_order=FEM_ORDER, plot_type=PLOT_TYPE,
                                                          resolution=RESOLUTION)
else:
    raise NotImplementedError

sol = disc.solve()

disc.visualize(sol)

