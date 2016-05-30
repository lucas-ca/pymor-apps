from scipy.sparse import bmat, csc_matrix

from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.functions.basic import ConstantFunction
from pymor.grids.tria import TriaGrid

from stokes.analyticalproblems.poiseuille import PoiseuilleProblem
from stokes.analyticalproblems.cavity_factorized import CavityProblem
from stokes.analyticalproblems.affine_transformed_stokes_new import AffineTransformedStokes
from stokes.discretizers.stationary_incompressible_stokes import discretize_stationary_incompressible_stokes
from stokes.functions.affine_transformation import AffineTransformation
from pymor.parameters.base import Parameter

import numpy_ as np

from stokes.operators.cg import DiffusionOperatorP2, AdvectionOperatorP2, ZeroOperator, L2VectorProductFunctionalP2

from pymor.grids.boundaryinfos import BoundaryInfoFromIndicators, EmptyBoundaryInfo

PROBLEM = 1
FEM_ORDER = 2
PLOT_TYPE = 0
RESOLUTION = 10
DIA = True
H = 10

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


if PROBLEM == 1:
    problem = PoiseuilleProblem()

    BoundaryType.register_type('do_nothing')

    dirichlet_ind = lambda X: np.logical_or(np.isclose(X[..., 0], 0.0),
                                            np.logical_or(np.isclose(X[..., 1], 0.0), np.isclose(X[..., 1], 1.0)))
    do_nothing_ind = lambda X: np.isclose(X[..., 0], 1.0)

    indicators = {BoundaryType('dirichlet'): dirichlet_ind, BoundaryType('do_nothing'): do_nothing_ind}

    transformation = AffineTransformation(parameter_name='transformation', mu_min=0.1, mu_max=1.0, ranges=None,
                                          name='AffineTransformation')
    problem_trans = AffineTransformedStokes(problem, transformation)
elif PROBLEM == 2:
    problem = CavityProblem()
    transformation = AffineTransformation(parameter_name='transformation', mu_min=0.1, mu_max=1.0, ranges=None,
                                          name='AffineTransformation')
    problem_trans = AffineTransformedStokes(problem, transformation)

jac = trans
jac_inv = np.linalg.inv(trans)
jac_inv_t = jac_inv.T
det = np.abs(np.linalg.det(trans))
det_inv = 1.0/det

diffusion_matrix = ConstantFunction(value=det*np.dot(jac_inv, jac_inv_t), dim_domain=2)
advection_matrix = ConstantFunction(value=det*jac_inv_t, dim_domain=2)
rhs_matrix = ConstantFunction(value=det_inv * det * jac, dim_domain=2)
dirichlet_matrix = ConstantFunction(value=det_inv * jac, dim_domain=2)

grid = TriaGrid(num_intervals=(20, 20))

empty_boundary_info = EmptyBoundaryInfo(grid)
boundary_info = BoundaryInfoFromIndicators(grid, indicators)

a = DiffusionOperatorP2(grid=grid,
                        boundary_info=boundary_info,
                        diffusion_function=diffusion_matrix,
                        diffusion_constant=1.0,
                        dirichlet_clear_columns=False,
                        dirichlet_clear_diag=True,
                        solver_options=None,
                        name='Diffusion')
b1 = AdvectionOperatorP2(grid=grid,
                         boundary_info=boundary_info,
                         advection_function=advection_matrix,
                         dirichlet_clear_rows=True,
                         name='Advection1')
b2 = AdvectionOperatorP2(grid=grid,
                         boundary_info=empty_boundary_info,
                         advection_function=advection_matrix,
                         dirichlet_clear_rows=False,
                         name='Advection2')
c = ZeroOperator(source=grid.size(grid.dim),
                 range=grid.size(grid.dim),
                 sparse=False,
                 name='Relaxation')
f1 = L2VectorProductFunctionalP2(grid=grid,
                                 function=problem.rhs,
                                 boundary_info=boundary_info,
                                 dirichlet_data=problem.dirichlet_data,
                                 neumann_data=None,
                                 robin_data=None,
                                 order=2,
                                 transformation_function=rhs_matrix,
                                 dirichlet_transformation=dirichlet_matrix,
                                 clear_dirichlet_dofs=False,
                                 clear_non_dirichlet_dofs=False,
                                 name='Functional')
A_ = a._assemble(mu=None)
A = bmat([[A_, None], [None, A_]])
B1 = -1.0 * b1._assemble(mu=None)
B2 = b2._assemble(mu=None).T
C = c._assemble(mu=None)

F1 = f1._assemble(mu=None)
F2 = np.zeros((1, grid.size(grid.dim)))

S = bmat([[A, B1], [B2, C]])
F = np.hstack((F1, F2))

from scipy.sparse.linalg import spsolve

U = spsolve(S, F, None, False)

np1 = grid.size(grid.dim)
np2 = grid.size(grid.dim) + grid.size(grid.dim - 1)
u = U[0:np1]
v = U[np2:np2+np1]
p = U[2*np2:]

from matplotlib import pyplot as plt

x = grid.centers(2)[..., 0]
y = grid.centers(2)[..., 1]
i = grid.subentities(0, 2)

plt.tripcolor(x, y, u, i)
Z = 0


