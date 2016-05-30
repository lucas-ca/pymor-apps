from pymor.vectorarrays.numpy import NumpyVectorArray

from stokes.analyticalproblems.poiseuille import PoiseuilleProblem
from stokes.analyticalproblems.cavity import CavityProblem
from stokes.analyticalproblems.affine_transformed_stokes_new import AffineTransformedStokes
from stokes.discretizers.stationary_incompressible_stokes import discretize_stationary_incompressible_stokes
from stokes.functions.affine_transformation import AffineTransformation
from pymor.parameters.base import Parameter

import numpy as np

PROBLEM = 1
FEM_ORDER = 1
PLOT_TYPE = 0
RESOLUTION = 25
DIA = True
H = 50

def rotate(angle):
    s = np.sin(angle)
    c = np.cos(angle)

    return np.array([[c, -s], [s, c]])

# 180
trans = np.array([[1.0, 0.0], [0.0, 1.0]])
#trans = np.array([[1., 1.], [0., 1.]])
#trans = np.array([[1., -1.], [1., 0.]]
# 45
#trans = np.dot(rotate(1*np.pi/4.), trans1)
#trans = rotate(6*np.pi/4.)

mu = Parameter({'transformation': trans})

if PROBLEM == 1:
    problem = PoiseuilleProblem()
    transformation = AffineTransformation(parameter_name='transformation', mu_min=0.1, mu_max=1.0, ranges=None,
                                          name='AffineTransformation')
    problem_trans = AffineTransformedStokes(problem, transformation)
elif PROBLEM == 2:
    problem = CavityProblem()
    transformation = AffineTransformation(parameter_name='transformation', mu_min=0.1, mu_max=1.0, ranges=None,
                                          name='AffineTransformation')
    problem_trans = AffineTransformedStokes(problem, transformation)

if DIA:
    #disc, g = discretize_stationary_incompressible_stokes(problem, 1./H, fem_order=FEM_ORDER, plot_type=PLOT_TYPE,
    #                                                      resolution=RESOLUTION)
    disc_trans, g_trans = discretize_stationary_incompressible_stokes(problem_trans, 1./H, fem_order=FEM_ORDER,
                                                                      plot_type=PLOT_TYPE, resolution=RESOLUTION)
else:
    raise NotImplementedError

#sol = disc.solve()
sol_trans = disc_trans.solve(mu)
#sol_alt = np.genfromtxt('/home/lucas/sol_alt')
#sol_neu = sol_trans._array
#disc.visualize(sol)
#disc.visualize(sol, mu=None)

#det = np.linalg.det(trans)
#sol_trans_det = NumpyVectorArray(det*sol_trans._array)

from stokes.algorithms.rb_generation import reduce_naive

ts = [Parameter({'transformation': a}) for a in [np.array([[1.5, 0], [0, 3]]),
                                                 np.array([[0.5, 0], [0, 4.3]]),
                                                 np.array([[3.8, 0], [0, 0.4]]),
                                                 np.array([[2.6, 0], [0, 1.112]]),
                                                 np.array([[0.667, 0], [0, 1.478]])]]

ts = Parameter({'transformation': np.eye(2)})

disc_red, rec = reduce_naive(discretization=disc_trans,
                                basis_size=200,
                                training_set=None,
                                add_supremizer=False,
                                grid=g_trans['grid'],
                                element_type='P1P1')

sol_red = disc_red.solve(mu)
sol_rec = rec.reconstruct(sol_red)
disc_trans.visualize(sol_rec, mu=ts)
disc_trans.visualize(sol_trans, mu=ts)


