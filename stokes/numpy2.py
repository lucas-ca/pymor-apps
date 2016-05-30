from __future__ import absolute_import, division, print_function

from pymor.operators.cg import L2ProductFunctionalP1, DiffusionOperatorP1

from stokes.analyticalproblems.poiseuille import PoiseuilleProblem
from stokes.operators.cg_testing import L2VectorProductFunctionalP2
from stokes.analyticalproblems.cavity import CavityProblem
from stokes.operators.cg_testing import AdvectionOperatorP1, DiffusionOperatorP2, AdvectionOperatorP2
from pymor.functions.basic import ConstantFunction, GenericFunction
from pymor.domaindescriptions.basic import RectDomain, BoundaryType
from pymor.grids.tria import TriaGrid
from pymor.grids.boundaryinfos import BoundaryInfoFromIndicators, EmptyBoundaryInfo

import numpy as np
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import spsolve


def slice_solution(solution, num_v):

    u = solution[0:num_v]
    v = solution[num_v:2*num_v]
    p = solution[2*num_v:]
    uv = solution[0:2*num_v ]

    return {'u': u, 'v': v, 'p': p, 'uv': uv}

def transformation_functionals(mu):
    assert isinstance(mu, np.ndarray)
    assert mu.ndim == 2
    assert mu.shape == (2, 2)

    jac = mu
    jac_inv = np.linalg.inv(mu)
    jac_inv_t = jac_inv.T
    det = np.abs(np.linalg.det(mu))
    det_inv = 1.0/det

    df = det * np.dot(jac_inv, jac_inv_t)
    af = det * jac_inv_t
    rhsf = det_inv * det * jac
    ddf = det_inv * jac

    return {'diffusion': df,
            'advection': af,
            'rhs': rhsf,
            'dirichlet_data': ddf}

def calculate_transformed_solution(problem, num_intervals, fem_order, transformation):

    # problem
    if problem == 1:
        # problem
        p = PoiseuilleProblem()
        # dirichlet_boundary
        db = lambda X: np.isclose(X[..., 0], np.zeros_like(X[..., 0])) +\
        np.isclose(X[..., 1], np.zeros_like(X[..., 0])) +\
        np.isclose(X[..., 1], np.ones_like(X[..., 0]))
    elif problem == 2:
        # problem
        p = CavityProblem()
        # dirichlet boundary
        db = lambda X: np.isclose(X[..., 0], np.zeros_like(X[..., 0])) +\
        np.isclose(X[..., 0], np.ones_like(X[..., 0])) +\
        np.isclose(X[..., 1], np.zeros_like(X[..., 1])) +\
        np.isclose(X[..., 1], np.ones_like(X[..., 1]))
    else:
        raise NotImplementedError

    # dirichlet indicator
    dirichlet_indicator = GenericFunction(mapping=db,
                                          dim_domain=2)

    # domain
    domain = [[0, 0], [1, 1]]

    # grid
    grid = TriaGrid((num_intervals, num_intervals), domain)

    # number of dofs
    num_p1_dofs = grid.size(grid.dim)
    num_p2_dofs = grid.size(grid.dim) + grid.size(grid.dim - 1)

    # problem data
    diffusion_functions = p.diffusion_functions
    rhs = p.rhs
    dirichlet_data = p.dirichlet_data
    functionals = transformation_functionals(transformation)

    # boundary infos
    boundary_info = BoundaryInfoFromIndicators(grid=grid,
                                               indicators={BoundaryType('dirichlet'): dirichlet_indicator})
    empty_boundary_info = EmptyBoundaryInfo(grid=grid)

    # operators
    if fem_order == 1:
        raise NotImplementedError
    elif fem_order == 2:
        # diffusion
        A = DiffusionOperatorP2(grid=g,
                                boundary_info=bi,
                                diffusion_function=functionals['diffusion'],
                                diffusion_constant=viscosity,
                                dirichlet_clear_diag=False,
                                dirichlet_clear_columns=False,
                                solver_options=None,
                                name='Diffusion',
                                direct=True)._assemble()

        # advection 1
        B = AdvectionOperatorP2(grid=g,
                                boundary_info=bi,
                                advection_function=functionals['advection'],
                                dirichlet_clear_rows=True,
                                name='Advection_1')._assemble()

        # advection 2
        BT = AdvectionOperatorP2(grid=g,
                                 boundary_info=ebi,
                                 advection_function=functionals['advection'],
                                 dirichlet_clear_rows=False,
                                 name='Advection_2')._assemble()

        # functional 1
        F_ges = L2VectorProductFunctionalP2(grid=g,
                                            function=rhs,
                                            boundary_info=bi,
                                            dirichlet_data=dirichlet_data,
                                            neumann_data=None,
                                            robin_data=None,
                                            order=2,
                                            transformation_function=functionals['rhs'],
                                            dirichlet_transformation=functionals['dirichlet_data'],
                                            clear_dirichlet_dofs=False,
                                            clear_non_dirichlet_dofs=False,
                                            name='Functional')._assemble()

        # functional 2
        Z = np.zeros((1, npsf))
    else:
        raise NotImplementedError

    # linear system
    if fem_order == 1:
        A2 = bmat([[A, None], [None, A]])
        S = bmat([[A2, -B], [BT.T, C]])
        F = np.hstack((Fx, Fy, Z))
    elif fem_order == 2:
        A2 = bmat([[A, None], [None, A]])
        S = bmat([[A2, -B], [BT.T, None]])
        F = np.hstack((F_ges, Z))

    # solve
    solution = spsolve(S, F)

    return solution


def calculate_reduced_solution(problem, snapshots, num_intervals, fem_order, transformation):

    # problem
    if problem == 1:
        # problem
        p = PoiseuilleProblem()
        # dirichlet_boundary
        db = lambda X: np.isclose(X[..., 0], np.zeros_like(X[..., 0])) +\
        np.isclose(X[..., 1], np.zeros_like(X[..., 0])) +\
        np.isclose(X[..., 1], np.ones_like(X[..., 0]))
    elif problem == 2:
        # problem
        p = CavityProblem()
        # dirichlet boundary
        db = lambda X: np.isclose(X[..., 0], np.zeros_like(X[..., 0])) +\
        np.isclose(X[..., 0], np.ones_like(X[..., 0])) +\
        np.isclose(X[..., 1], np.zeros_like(X[..., 1])) +\
        np.isclose(X[..., 1], np.ones_like(X[..., 1]))
    else:
        raise NotImplementedError

    # dirichlet indicator
    dirichlet_indicator = GenericFunction(mapping=db,
                                          dim_domain=2)

    # domain
    domain = [[0, 0], [1, 1]]

    # grid
    grid = TriaGrid((num_intervals, num_intervals), domain)

    # number of dofs
    num_p1_dofs = grid.size(grid.dim)
    num_p2_dofs = grid.size(grid.dim) + grid.size(grid.dim - 1)

    # problem data
    diffusion_functions = p.diffusion_functions
    rhs = p.rhs
    dirichlet_data = p.dirichlet_data
    functionals = transformation_functionals(transformation)

    # boundary infos
    boundary_info = BoundaryInfoFromIndicators(grid=grid,
                                               indicators={BoundaryType('dirichlet'): dirichlet_indicator})
    empty_boundary_info = EmptyBoundaryInfo(grid=grid)

    # operators
    if fem_order == 1:
        raise NotImplementedError
    elif fem_order == 2:
        num_v = num_p2_dofs
        # diffusion
        A = DiffusionOperatorP2(grid=g,
                                boundary_info=boundary_info,
                                diffusion_function=functionals['diffusion'],
                                diffusion_constant=viscosity,
                                dirichlet_clear_diag=False,
                                dirichlet_clear_columns=False,
                                solver_options=None,
                                name='Diffusion',
                                direct=True)._assemble()

        #A = DiffusionOperatorP2(grid=g,
        #                        boundary_info=empty_boundary_info,
        #                        diffusion_function=functionals['diffusion'],
        #                        diffusion_constant=viscosity,
        #                        dirichlet_clear_diag=False,
        #                        dirichlet_clear_columns=False,
        #                        solver_options=None,
        #                        name='Diffusion',
        #                        direct=True)
        # advection 1
        B = AdvectionOperatorP2(grid=g,
                                boundary_info=boundary_info,
                                advection_function=functionals['advection'],
                                dirichlet_clear_rows=True,
                                name='Advection_1')._assemble()

        #B = AdvectionOperatorP2(grid=g,
        #                        boundary_info=empty_boundary_info,
        #                        advection_function=functionals['advection'],
        #                        dirichlet_clear_rows=True,
        #                        name='Advection_1')._assemble()

        # advection 2
        BT = AdvectionOperatorP2(grid=g,
                                 boundary_info=empty_boundary_info,
                                 advection_function=functionals['advection'],
                                 dirichlet_clear_rows=False,
                                 name='Advection_2')._assemble()

        C = np.zeros((grid.size(grid.dim), grid.size(grid.dim)))

        # functional 1
        F_ges = L2VectorProductFunctionalP2(grid=g,
                                            function=rhs,
                                            boundary_info=boundary_info,
                                            dirichlet_data=dirichlet_data,
                                            neumann_data=None,
                                            robin_data=None,
                                            order=2,
                                            transformation_function=functionals['rhs'],
                                            dirichlet_transformation=functionals['dirichlet_data'],
                                            clear_dirichlet_dofs=False,
                                            clear_non_dirichlet_dofs=False,
                                            name='Functional')._assemble()

        #F_ges = L2VectorProductFunctionalP2(grid=g,
        #                                    function=rhs,
        #                                    boundary_info=empty_boundary_info,
        #                                    dirichlet_data=dirichlet_data,
        #                                    neumann_data=None,
        #                                    robin_data=None,
        #                                    order=2,
        #                                    transformation_function=functionals['rhs'],
        #                                    dirichlet_transformation=functionals['dirichlet_data'],
        #                                    clear_dirichlet_dofs=False,
        #                                    clear_non_dirichlet_dofs=False,
        #                                    name='Functional')._assemble()

        # functional 2
        Z = np.zeros((1, npsf))
    else:
        raise NotImplementedError

    A2 = bmat([[A, None], [None, A]])

    # reduced operators
    basis_size = len(snapshots)
    A_red = np.zeros((basis_size, basis_size))
    B_red_1 = np.zeros((basis_size, basis_size))
    B_red_2 = np.zeros((basis_size, basis_size))
    C_red = np.zeros((basis_size, basis_size))
    F1_red = np.zeros(basis_size)
    F2_red = np.zeros(basis_size)

    for i in range(basis_size):
        for j in range(basis_size):
            # Diffusion reduced
            uv_i = slice_solution(snapshots[i], num_v)['uv']
            uv_j = slice_solution(snapshots[j], num_v)['uv']
            #A_red[j, i] = A2.dot(uv_i).dot(uv_j)
            A_red[i, j] = uv_i.dot(A2.dot(uv_j))
            # Advection 1 reduced
            uv_i = slice_solution(snapshots[i], num_v)['uv']
            p_j = slice_solution(snapshots[j], num_v)['p']
            #B_red_1[i, j] = B.dot(p_j).dot(uv_i)
            B_red_1[i, j] = uv_i.dot(B.dot(p_j))
            # Advection  2 reduced
            uv_j = slice_solution(snapshots[j], num_v)['uv']
            p_i = slice_solution(snapshots[i], num_v)['p']
            #B_red_2[i, j] = BT.T.dot(uv_j).dot(p_i)
            B_red_2[i, j] = uv_i.dot(BT.dot(p_j))
            # C reduced
            p_i = slice_solution(snapshots[i], num_v)['p']
            p_j = slice_solution(snapshots[j], num_v)['p']
            C_red[i, j] = C.dot(p_j).dot(p_i)
        # F1 reduced
        uv_i = slice_solution(snapshots[i], num_v)['uv']
        F1_red[i] = F_ges.dot(uv_i)
        # F2 reduced
        p_i = slice_solution(snapshots[i], num_v)['p']
        F2_red[i] = Z.dot(p_i)

    # reduced system
    S = np.bmat([[A_red, -1.0*B_red_1], [B_red_2, C_red]])
    F = np.hstack((F1_red, F2_red))

    sol_reduced = np.linalg.solve(S, F)
    Z=0



### PROBLEM

## general
PROBLEM = 1
ni = 10
fem_order = 2
plot = True
width = 1
height = 1
viscosity = 1.0
alpha = 0.001
h_square = (1/ni)**2
trans = np.array([[1.0, 0.0], [0.0, 1.0]])
#trans = 1.0*np.eye(2)
functionals = transformation_functionals(trans)


## DOMAIN
d = [[0, 0], [width, height]]

## GRID
g = TriaGrid((ni, ni), d)
npsf = g.size(g.dim)
nvsf = g.size(g.dim) + g.size(g.dim - 1)


## BOUNDARY
if PROBLEM == 1:
    p = PoiseuilleProblem()
    db = lambda X: np.isclose(X[..., 0], np.zeros_like(X[..., 0])) +\
        np.isclose(X[..., 1], np.zeros_like(X[..., 0])) +\
        np.isclose(X[..., 1], np.ones_like(X[..., 0]))
elif PROBLEM == 2:
    p = CavityProblem()
    db = lambda X: np.isclose(X[..., 0], np.zeros_like(X[..., 0])) +\
        np.isclose(X[..., 0], np.ones_like(X[..., 0])) +\
        np.isclose(X[..., 1], np.zeros_like(X[..., 1])) +\
        np.isclose(X[..., 1], np.ones_like(X[..., 1]))
else:
    raise NotImplementedError

dirichlet_indicator = GenericFunction(db, dim_domain=2)

#BoundaryType.register_type('do_nothing')

#domain = RectDomain(domain=([0, 0], [width, height]),
#                    left = BoundaryType('dirichlet'),
#                    bottom = BoundaryType('dirichlet'),
#                    top=BoundaryType('dirichlet'),
#                    right=BoundaryType('do_nothing'))

diffusion_functions = [ConstantFunction(value=1., dim_domain=2)]
diffusion_functions = p.diffusion_functions

rhs = p.rhs
dirichlet_data = p.dirichlet_data

if PROBLEM == 1:
    ddx = GenericFunction(lambda X: (-4./(height**2) * X[..., 1]**2 + 4./height * X[..., 1]) * np.isclose(X[..., 0], np.zeros_like(X[..., 0])), dim_domain=2)
    ddy = ConstantFunction(value=0., dim_domain=2)
elif PROBLEM == 2:
    ddx = GenericFunction(lambda X: (np.ones_like(X[..., 0])) * np.isclose(X[..., 1], np.ones_like(X[..., 0])), dim_domain=2)
    ddy = ConstantFunction(value=0., dim_domain=2)
else:
    raise NotImplementedError

bi = BoundaryInfoFromIndicators(g, {BoundaryType('dirichlet'): dirichlet_indicator})
ebi = EmptyBoundaryInfo(g)


## RHS

"""
## OPERATORS
if fem_order == 1:
    A = DiffusionOperatorP1(grid=g, boundary_info=bi, diffusion_constant=viscosity, dirichlet_clear_diag=False,
                        dirichlet_clear_columns=False)._assemble()
    Bx, By = AdvectionOperatorP1(grid=g, boundary_info=bi, dirichlet_clear_rows=True)._assemble()
    Bx2, By2 = AdvectionOperatorP1(grid=g, boundary_info=ebi, dirichlet_clear_rows=False)._assemble()
    C = alpha * h_square * DiffusionOperatorP1(grid=g, boundary_info=ebi)._assemble()
    Fx = L2ProductFunctionalP1(grid=g, function=rhs, boundary_info=bi, dirichlet_data=ddx)._assemble()
    Fy = L2ProductFunctionalP1(grid=g, function=rhs, boundary_info=bi, dirichlet_data=ddy)._assemble()
    Z = np.zeros((1, npsf))
elif fem_order == 2:
    A = DiffusionOperatorP2(grid=g, boundary_info=bi, diffusion_constant=viscosity, dirichlet_clear_diag=False,
                            dirichlet_clear_columns=False)._assemble()
    Bx, By = AdvectionOperatorP2(grid=g, boundary_info=bi, dirichlet_clear_rows=True)._assemble()
    Bx2, By2 = AdvectionOperatorP2(grid=g, boundary_info=ebi, dirichlet_clear_rows=False)._assemble()
    Fx = L2ProductFunctionalP2(grid=g, function=rhs, boundary_info=bi, dirichlet_data=ddx)._assemble()
    Fy = L2ProductFunctionalP2(grid=g, function=rhs, boundary_info=bi, dirichlet_data=ddy)._assemble()
    Z = np.zeros((1, npsf))

#Axm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Ax', delimiter=',')
#Aym = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Ay', delimiter=',')
##Bxm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Bx', delimiter=',')
#Bym = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/By', delimiter=',')
#BxTm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/BxT', delimiter=',')
#ByTm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/ByT', delimiter=',')
#Cm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/C', delimiter=',')
#Fxm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Fx', delimiter=',')
#Fym = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Fy', delimiter=',')
#Zm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Z', delimiter=',')
#Zxm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Zx', delimiter=',')
#Zym = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Zy', delimiter=',')

if fem_order == 1:
    #A = bmat([[A, None], [None, A]])
    #S = bmat([[A, -B], [B.T, C]])
    S = bmat([[A, None, -Bx], [None, A, -By], [Bx2.T, By2.T, C]])
    F = np.hstack((Fx, Fy, Z))
    #Zx = np.zeros((g.size(g.dim), g.size(g.dim)))
elif fem_order == 2:
    S = bmat([[A, None, -Bx], [None, A, -By], [Bx2.T, By2.T, None]])
    F = np.hstack((Fx, Fy, Z))
"""
## OPERATORS
if fem_order == 1:
    A = DiffusionOperatorP1(grid=g, boundary_info=bi, diffusion_constant=viscosity, dirichlet_clear_diag=False,
                        dirichlet_clear_columns=False)._assemble()
    #Bx, By = AdvectionOperatorP1(grid=g, boundary_info=bi, dirichlet_clear_rows=True)._assemble()
    B = AdvectionOperatorP1(grid=g, boundary_info=bi, dirichlet_clear_rows=True)._assemble()
    #Bx2, By2 = AdvectionOperatorP1(grid=g, boundary_info=ebi, dirichlet_clear_rows=False)._assemble()
    BT = AdvectionOperatorP1(grid=g, boundary_info=ebi, dirichlet_clear_rows=False)._assemble()
    C = alpha * h_square * DiffusionOperatorP1(grid=g, boundary_info=ebi)._assemble()
    Fx = L2ProductFunctionalP1(grid=g, function=rhs, boundary_info=bi, dirichlet_data=ddx)._assemble()
    Fy = L2ProductFunctionalP1(grid=g, function=rhs, boundary_info=bi, dirichlet_data=ddy)._assemble()
    Z = np.zeros((1, npsf))
elif fem_order == 2:
    A = DiffusionOperatorP2(grid=g,
                            boundary_info=bi,
                            diffusion_function=functionals['diffusion'],
                            diffusion_constant=viscosity,
                            dirichlet_clear_diag=False,
                            dirichlet_clear_columns=False,
                            solver_options=None,
                            name='Diffusion',
                            direct=True)._assemble()
    #Bx, By = AdvectionOperatorP2(grid=g, boundary_info=bi, dirichlet_clear_rows=True)._assemble()
    B = AdvectionOperatorP2(grid=g,
                            boundary_info=bi,
                            advection_function=functionals['advection'],
                            dirichlet_clear_rows=True,
                            name='Advection_1')._assemble()
    #Bx2, By2 = AdvectionOperatorP2(grid=g, boundary_info=ebi, dirichlet_clear_rows=False)._assemble()
    BT = AdvectionOperatorP2(grid=g,
                             boundary_info=ebi,
                             advection_function=functionals['advection'],
                             dirichlet_clear_rows=False,
                             name='Advection_2')._assemble()
    #Fx = L2ProductFunctionalP2(grid=g, function=rhs, boundary_info=bi, dirichlet_data=ddx)._assemble()
    #Fy = L2ProductFunctionalP2(grid=g, function=rhs, boundary_info=bi, dirichlet_data=ddy)._assemble()
    F_ges = L2VectorProductFunctionalP2(grid=g,
                                        function=rhs,
                                        boundary_info=bi,
                                        dirichlet_data=dirichlet_data,
                                        neumann_data=None,
                                        robin_data=None,
                                        order=2,
                                        transformation_function=functionals['rhs'],
                                        dirichlet_transformation=functionals['dirichlet_data'],
                                        clear_dirichlet_dofs=False,
                                        clear_non_dirichlet_dofs=False,
                                        name='Functional')._assemble()
    Z = np.zeros((1, npsf))

#Axm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Ax', delimiter=',')
#Aym = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Ay', delimiter=',')
##Bxm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Bx', delimiter=',')
#Bym = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/By', delimiter=',')
#BxTm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/BxT', delimiter=',')
#ByTm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/ByT', delimiter=',')
#Cm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/C', delimiter=',')
#Fxm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Fx', delimiter=',')
#Fym = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Fy', delimiter=',')
#Zm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Z', delimiter=',')
#Zxm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Zx', delimiter=',')
#Zym = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/Matlab Code P1 P1/Zy', delimiter=',')

if fem_order == 1:
    A2 = bmat([[A, None], [None, A]])
    S = bmat([[A2, -B], [BT.T, C]])
    #S2 = bmat([[A, None, -Bx], [None, A, -By], [Bx2.T, By2.T, C]])
    F = np.hstack((Fx, Fy, Z))
    #Zx = np.zeros((g.size(g.dim), g.size(g.dim)))
elif fem_order == 2:
    A2 = bmat([[A, None], [None, A]])
    S = bmat([[A2, -B], [BT.T, None]])
    #S = bmat([[A, None, -Bx], [None, A, -By], [Bx2.T, By2.T, None]])
    F = np.hstack((F_ges, Z))

#S_matlab = csc_matrix(np.bmat([[Axm, Zxm, Bxm], [Zym, Aym, Bym], [BxTm, ByTm, Cm]]))
#F_matlab = np.hstack((Fxm, Fym, Zm))
#S_mix_1 = np.bmat([A.todense() , Zx, Bxm])
#S_mix_2 = np.bmat([Zx, A.todense(), Bym])
#S_mix_3 = np.bmat([BxTm, ByTm, C.todense()])
#S_mix = np.bmat([[S_mix_1], [S_mix_2], [S_mix_3]])
#S_mix = csc_matrix(S_mix)
#F_mix = F

#F = F.ravel()


###### TESTS


## SOLVE
#solution_alt = spsolve(S, F)
#solution = np.linalg.solve()
solution1 = calculate_transformed_solution(PROBLEM, ni, fem_order, 1*np.eye(2))
solution2 = calculate_transformed_solution(PROBLEM, ni, fem_order, 2*np.eye(2))
#solution3 = calculate_transformed_solution(PROBLEM, ni, fem_order, 2*np.eye(2))
sol_red = calculate_reduced_solution(PROBLEM, [solution1, solution2], ni, fem_order, trans)

if fem_order == 1:
    u = solution[0:npsf]
    u2 = u
    v = solution[npsf:2*npsf]
    v2 = v
    p = solution[2*npsf:]
elif fem_order == 2:
    u = solution[0:nvsf]
    u2 = u[0:npsf]
    v = solution[nvsf:2*nvsf]
    v2 = v[0:npsf]
    p = solution[2*nvsf:]

## PLOT
if plot:
    #g.visualize(u)
    #g.visualize(v)
    #g.visualize(p)
    from matplotlib import pyplot as plt
    X = g.centers(2)[..., 0]
    Y = g.centers(2)[..., 1]
    XY_trans = np.einsum('ij,ej->ei', trans, g.centers(2))
    X_trans = XY_trans[..., 0]
    Y_trans = XY_trans[..., 1]
    plt.figure('Quiver')
    plt.quiver(X_trans, Y_trans, u2, v2)
    plt.figure('u')
    plt.tripcolor(X_trans, Y_trans, g.subentities(0, 2), u2)
    plt.colorbar()
    plt.figure('v')
    plt.tripcolor(X_trans, Y_trans, g.subentities(0, 2), v2)
    plt.colorbar()
    plt.figure('p')
    plt.tripcolor(X_trans, Y_trans, g.subentities(0, 2), p)
    plt.colorbar()

    A

    z = 0



