from stokes.analyticalproblems.poiseuille import PoiseuilleProblem
from stokes.analyticalproblems.cavity_factorized import CavityProblem
from stokes.analyticalproblems.affine_transformed_stokes_new import AffineTransformedStokes
from stokes.analyticalproblems.poiseuille_l_shape import PoiseuilleLShapeProblem
from stokes.discretizers.stationary_incompressible_stokes_dirichlet_direct import discretize_stationary_incompressible_stokes
from stokes.domaindiscretizers.gmsh import discretize_gmsh
from stokes.functions.affine_transformation import AffineTransformation
from pymor.parameters.base import Parameter

import numpy_ as np

from stokes.functions.piecewise_affine_transformation import PiecewiseAffineTransformation
from stokes.analyticalproblems.poiseuille_2x1 import Poiseuille2x1Problem

PROBLEM = 1
FEM_ORDER = 2
PLOT_TYPE = 1
RESOLUTION = 10
DIA = True
H = 10


def build_parameter(parameter):
    assert isinstance(parameter, Parameter)

    parameter_list = parameter['transformation']

    assert len(parameter_list) == 4

    w0, h0, w1, s1 = parameter_list

    a0 = np.array([[w0, 0.0], [0.0, h0]])
    b0 = np.array([0.0, 0.0])
    a1 = np.array([[w1, 0.0], [s1, h0]])
    b1 = np.array([w0 - 1.0, 0.0])

    p0 = Parameter({'matrix': a0, 'translation': b0})
    p1 = Parameter({'matrix': a1, 'translation': b1})

    return [p0, p1]


trans = [1.0, 2.0, 1.0, 0.0]

mu = Parameter({'transformation': trans})

if PROBLEM == 1:
    problem = Poiseuille2x1Problem()
    #transformation = AffineTransformation(parameter_name='transformation', mu_min=0.1, mu_max=1.0, ranges=None,
    #                                      name='AffineTransformation')
    transformation = PiecewiseAffineTransformation(problem.domain, build_parameter)
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

g, bi = discretize_gmsh(problem.domain, geo_file_path='/home/lucas/test42.geo', refinement_steps=1)

disc, grid = discretize_stationary_incompressible_stokes(analytical_problem=problem,
                                                         diameter=None,
                                                         domain_discretizer=None,
                                                         grid=g,
                                                         boundary_info=bi,
                                                         fem_order=2,
                                                         plot_type=1)

sol = disc.solve(mu)

c = grid['grid'].centers(2)
c0 = grid['grid'].centers(2)[..., 0]
c1 = grid['grid'].centers(2)[..., 1]
indi = grid['grid'].subentities(0, 2)

ct = transformation.evaluate(c, mu)
c0 = ct[..., 0]
c1 = ct[..., 1]

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


