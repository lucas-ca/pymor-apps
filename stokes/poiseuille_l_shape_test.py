from stokes.analyticalproblems.poiseuille import PoiseuilleProblem
from stokes.analyticalproblems.cavity_factorized import CavityProblem
from stokes.analyticalproblems.affine_transformed_stokes_new import AffineTransformedStokes
from stokes.analyticalproblems.poiseuille_l_shape import PoiseuilleLShapeProblem
from stokes.discretizers.stationary_incompressible_stokes_dirichlet_direct import discretize_stationary_incompressible_stokes
from stokes.domaindiscretizers.gmsh import discretize_gmsh
from stokes.functions.affine_transformation import AffineTransformation
from pymor.parameters.base import Parameter

import numpy_ as np

PROBLEM = 1
FEM_ORDER = 2
PLOT_TYPE = 1
RESOLUTION = 10
DIA = True
H = 10

def rotate(angle):
    s = np.sin(angle)
    c = np.cos(angle)

    return np.array([[c, -s], [s, c]])

# 180
trans = -1.*np.eye(2)
#trans = np.array([[1., 1.], [0., 1.]])
#trans = np.array([[1., -1.], [1., 0.]])
# 45
trans = rotate(3*np.pi/4.)

mu = Parameter({'transformation': trans})

if PROBLEM == 1:
    problem = PoiseuilleLShapeProblem()
    transformation = AffineTransformation(parameter_name='transformation', mu_min=0.1, mu_max=1.0, ranges=None,
                                          name='AffineTransformation')
    problem_trans = AffineTransformedStokes(problem, transformation)
#elif PROBLEM == 2:
#    problem = CavityProblem()
#    transformation = AffineTransformation(parameter_name='transformation', mu_min=0.1, mu_max=1.0, ranges=None,
#                                          name='AffineTransformation')
#    problem_trans = AffineTransformedStokes(problem, transformation)
else:
    raise ValueError

#if DIA:
#    disc, g = discretize_stationary_incompressible_stokes(problem, 1./H, fem_order=FEM_ORDER, plot_type=PLOT_TYPE,
#                                                          resolution=RESOLUTION)
#    disc_trans, g_trans = discretize_stationary_incompressible_stokes(problem_trans, 1./H, fem_order=FEM_ORDER,
#                                                                      plot_type=PLOT_TYPE, resolution=RESOLUTION)
#else:
#    raise NotImplementedError

g, bi = discretize_gmsh(problem.domain, geo_file_path='/home/lucas/test42.geo', refinement_steps=2)

disc, grid = discretize_stationary_incompressible_stokes(analytical_problem=problem,
                                                         diameter=None,
                                                         domain_discretizer=None,
                                                         grid=g,
                                                         boundary_info=bi,
                                                         fem_order=2,
                                                         plot_type=1)

sol = disc.solve()

c0 = grid['grid'].centers(2)[..., 0]
c1 = grid['grid'].centers(2)[..., 1]
indi = grid['grid'].subentities(0, 2)

np1 = grid['grid'].size(2)
np2 = grid['grid'].size(2) + grid['grid'].size(1)

su = sol._array[0][0:np2]
sv = sol._array[0][np2:2*np2]
sp = sol._array[0][2*np2:]

from matplotlib import pyplot as plt

plt.triplot(c0, c1, indi)

plt.figure()
plt.title('u')
plt.tripcolor(c0, c1, indi, su[0:np1])
plt.colorbar()

plt.figure()
plt.title('v')
plt.tripcolor(c0, c1, indi, sv[0:np1])
plt.colorbar()

plt.figure()
plt.title('p')
plt.tripcolor(c0, c1, indi, sp)
plt.colorbar()

disc.visualize(sol)
z = 0


#sol = disc.solve()
#sol_trans = disc_trans.solve(mu)
#sol_alt = np.genfromtxt('/home/lucas/sol_alt')
#sol_neu = sol_trans._array
#disc.visualize(sol)
#disc.visualize(sol, mu=None)
#disc_trans.visualize(sol_trans, mu=mu)


