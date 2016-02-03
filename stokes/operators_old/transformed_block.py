# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Module containing a block operator."""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from matplotlib import pyplot as plt

from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.functions.basic import ConstantFunction
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.operators.cg import DiffusionOperatorP1, L2ProductP1
from pymor.vectorarrays.numpy import NumpyVectorArray
from stokes.analyticalproblems_old.cavity import CavityProblem
from stokes.analyticalproblems_old.poiseuille import PoiseuilleProblem
from stokes.grids_old.transformed_tria import AffineTransformedGrid

from stokes.operators_old.cg import DiffusionOperatorP2, AdvectionOperatorP1, AdvectionOperatorP2,\
    L2TensorProductFunctionalP1, L2TensorProductFunctionalP2, TwoDimensionalL2ProductFunctionalP2

height = 1
width = 1

def rotate(alpha):
    a = np.deg2rad(alpha)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def scale(xscale, yscale):
    return np.array([[xscale, 0.], [0., yscale]])

def compute_fem(problem, grid, boundary_info, mu, fem_order):
    g = grid
    bi = boundary_info
    ebi = EmptyBoundaryInfo(g)
    
    num_p1 = g.size(g.dim)
    num_p2 = g.size(g.dim) + g.size(g.dim - 1)
    
    inv = np.linalg.inv(mu)
    det = np.linalg.det(mu)

    t_A = ConstantFunction(value=inv.dot(inv.T), dim_domain=2)
    t_B = ConstantFunction(value=inv.T, dim_domain=2)

    # operators
    if fem_order == 1:
        raise NotImplementedError
    elif fem_order == 2:
        a = DiffusionOperatorP2(g, bi, t_A)._assemble()
        b = AdvectionOperatorP2(g, bi, t_B)._assemble()
        b_t = AdvectionOperatorP2(g, ebi, t_B, False)._assemble()
        c = np.zeros((num_p1, num_p1))
        f = TwoDimensionalL2ProductFunctionalP2(grid, problem.rhs, bi, problem.dirichlet_data,
                                                transformation_matrix=mu)._assemble()
    else:
        raise NotImplementedError

    A = bmat(([[a, None], [None, a]]))
    L = bmat([[A, -1.0*b], [b_t.T, c]])
    R = np.hstack((f, np.zeros((1, num_p1))))

    # solve
    S = spsolve(L, R)

    if fem_order == 1:
        raise NotImplementedError
        shift = num_p1
        u = S[0:shift]
        v = S[shift:2*shift]
        p = S[2*shift:]
    elif fem_order == 2:
        shift = num_p2
        u = S[0:shift]
        v = S[shift:2*shift]
        p = S[2*shift:]
    
    return u, v, p

def compute_rb(mus, problem, grid, boundary_info, fem_order):
    print('Generating reduced basis space...')
    g = grid
    bi = boundary_info
    ebi = EmptyBoundaryInfo(g)

    num_p1 = g.size(g.dim)
    num_p2 = g.size(g.dim) + g.size(g.dim - 1)

    us = []
    ps = []

    for i, mu in enumerate(mus):
        print('Solving {} of {}'.format(i, len(mus) - 1))
        inv = np.linalg.inv(mu)
        det = np.linalg.det(mu)

        t_A = ConstantFunction(value=inv.dot(inv.T), dim_domain=2)
        t_B = ConstantFunction(value=inv.T, dim_domain=2)

        # operators
        if fem_order == 1:
            raise NotImplementedError
        elif fem_order == 2:
            a = DiffusionOperatorP2(g, bi, t_A)._assemble()
            b = AdvectionOperatorP2(g, bi, t_B)._assemble()
            b_t = AdvectionOperatorP2(g, ebi, t_B, False)._assemble()
            c = np.zeros((num_p1, num_p1))
            f = TwoDimensionalL2ProductFunctionalP2(grid, problem.rhs, bi, problem.dirichlet_data,
                                                    transformation_matrix=mu)._assemble()
        else:
            raise NotImplementedError

        A = bmat(([[a, None], [None, a]]))
        L = bmat([[A, -1.0*b], [b_t.T, c]])
        R = np.hstack((f, np.zeros((1, num_p1))))

        # solve
        S = spsolve(L, R)

        if fem_order == 1:
            raise NotImplementedError
            shift = num_p1
            u = S[0:shift]
            v = S[shift:2*shift]
            p = S[2*shift:]
        elif fem_order == 2:
            shift = num_p2
            u = S[0:shift]
            v = S[shift:2*shift]
            p = S[2*shift:]

        #u = NumpyVectorArray(sol[0:2*num_p2])
        #p = NumpyVectorArray(sol[2*num_p2:])
        u = S[0:2*num_p2]
        p = S[2*num_p2:]

        us.append(u)
        ps.append(p)

    return us, ps, {'grid': g, 'boundary_info': bi, 'problem': problem}

def compute_reduced(problem, us, ps, grid, boundary_info, mu, fem_order):
    g = grid
    bi = boundary_info
    
    num_p1 = g.size(g.dim)
    num_p2 = g.size(g.dim) + g.size(g.dim - 1)
    
    inv = np.linalg.inv(mu)
    det = np.linalg.det(mu)

    t_A = ConstantFunction(value=inv.dot(inv.T), dim_domain=2)
    t_B = ConstantFunction(value=inv.T, dim_domain=2)

    # operators
    if fem_order == 1:
        raise NotImplementedError
    elif fem_order == 2:
        a = DiffusionOperatorP2(g, bi, t_A)._assemble()
        b = AdvectionOperatorP2(g, bi, t_B)._assemble()
        b_t = AdvectionOperatorP2(g, EmptyBoundaryInfo(g), t_B, False)._assemble()
        c = np.zeros((num_p1, num_p1))
        f = TwoDimensionalL2ProductFunctionalP2(grid, problem.rhs, bi, problem.dirichlet_data, transformation_matrix=mu)._assemble()
    else:
        raise NotImplementedError

    A = bmat(([[a, None], [None, a]]))
    #L = bmat([[A, -1.0*b], [b_t.T, c]])

    a_red = np.zeros((len(us), len(us)))
    b_red = np.zeros((len(us), len(us)))
    b_t_red = np.zeros((len(us), len(us)))
    f_red = np.zeros((1, len(us)))

    for i in xrange(len(us)):
        for j in xrange(len(us)):
            a_red[i, j] = us[i].dot(A.dot(us[j]))
            b_red[i, j] = us[i].dot(b.dot(ps[j]))
            b_t_red[i, j] = us[i].dot(b_t.dot(ps[j]))
        f_red[0, i] = f.dot(us[i])[0]
    
    L_red = np.bmat([[a_red, -1.0*b_red], [b_t_red.T, np.zeros((len(us), len(us)))]])
    R_red = np.hstack((f_red[0], np.zeros(len(us))))
    
    S_red = np.linalg.solve(L_red, R_red)
    
    u_red = np.zeros_like(us[0])
    p_red = np.zeros_like(ps[0])
    
    for i in xrange(len(us)):
        u_red += S_red[i]*us[i]
        p_red += S_red[i+len(us)]*ps[i]
    
    if fem_order == 1:
        raise NotImplementedError
    elif fem_order == 2:
        u2_red = u_red[0:num_p1]
        v2_red = u_red[num_p2:num_p2+num_p1]
    
    return u2_red, v2_red, p_red

def evaluate_exact_solution(problem_number, grid):
    if problem_number == 1:
        # poiseuille problem
        exact_u = lambda X: -4./(height**2)*X[..., 1]**2 + 4./height*X[..., 1]
        exact_v = lambda X: np.zeros_like(X[..., 0])
        exact_p = lambda X: 8./(height**2) * (width - X[..., 0])
    elif problem_number == 2:
        # cavity problem
        exact_u = lambda X: X[..., 0]**2 * (np.ones_like(X[..., 0]) - X[..., 0])**2 * 2*X[..., 1] *\
            (np.ones_like(X[..., 1]) - X[..., 1]) * (2*X[..., 1] - np.ones_like(X[..., 1]))
        exact_v = lambda X: X[..., 1]**2 * (np.ones_like(X[..., 1]) - X[..., 1])**2 * 2*X[..., 0] *\
            (np.ones_like(X[..., 1]) - X[..., 0]) * (np.ones_like(X[..., 0]) - 2*X[..., 0])
        exact_p = lambda X: X[..., 0] * (np.ones_like(X[..., 0]) - X[..., 0]) * (np.ones_like(X[..., 1]) - X[..., 1])
    else:
        raise NotImplementedError
    
    ref_points = grid.embeddings(2)[1]
    
    u = exact_u(ref_points)
    v = exact_v(ref_points)
    p = exact_p(ref_points)
    
    return u, v, p

def plot_uv_fem_red_ref(grid, mu, u_ref, v_ref, u_fem, v_fem, u_red, v_red):
    
    g_t = AffineTransformedGrid(grid, mu)
    x_t = g_t.centers(2)[..., 0]
    y_t = g_t.centers(2)[..., 1]
    
    # transform exact solution with piola transformation
    exact_uv_trans = np.einsum('ij,ej->ei', mu, np.vstack((u_ref, v_ref)).T)
    exact_uv_trans /= np.linalg.det(mu)
    
    u_ref = exact_uv_trans[..., 0]
    v_ref = exact_uv_trans[..., 1]

    grid_size = (1, 3)
    
    plt.subplot2grid(grid_size, (0, 0), rowspan=1, colspan=1)
    plt.title('analytical')
    plt.quiver(x_t, y_t, u_ref, v_ref, color='b')
    plt.subplot2grid(grid_size, (0, 1), rowspan=1, colspan=1)
    plt.title('FEM')
    plt.quiver(x_t, y_t, u_fem, v_fem, color='r')
    plt.subplot2grid(grid_size, (0, 2), rowspan=1, colspan=1)
    plt.title('FEM reduced')
    plt.quiver(x_t, y_t, u_red, v_red)#, pivot='mid', color='r')
    #plt.tight_layout()
    plt.show()

def main():
    DIAMETER = 1./12
    FEM_ORDER = 2
    
    PROBLEM = 2
    
    if PROBLEM == 1:
        problem = PoiseuilleProblem()
    elif PROBLEM == 2:
        problem = CavityProblem()
    else:
        raise NotImplementedError
    
    grid, boundary_info = discretize_domain_default(problem.domain, diameter=DIAMETER)

    num_p1 = grid.size(grid.dim)
    num_p2 = grid.size(grid.dim) + grid.size(grid.dim - 1)
    
    mu_test = np.eye(2)  # id
    mu_test = rotate(45)   # rotate 45
    mu_test = rotate(90)   # rotate 90
    mu_test = rotate(135)  # rotate 135
    mu_test = rotate(180)  # rotate 180
    mu_test = rotate(22.5)  # rotate 225
    mu_test = rotate(270)  # rotate 270
    mu_test = np.array([[1, 1], [0, 1]])  # rotate 315
    
    train_set = [np.eye(2), rotate(45), rotate(90), rotate(135), rotate(180), rotate(225), rotate(270), rotate(315)]
    
    # compute rb
    us, ps, _ = compute_rb(train_set, problem, grid, boundary_info, FEM_ORDER)
    
    # get analytical solution
    u_ref, v_ref, p_ref = evaluate_exact_solution(PROBLEM, grid)
    
    # get fem solution
    u_fem, v_fem, p_fem = compute_fem(problem, grid, boundary_info, mu_test, FEM_ORDER)

    # solve for test parameter
    u_red, v_red, p_red = compute_reduced(problem, us, ps, grid, boundary_info, mu_test, FEM_ORDER)

    # plot
    plot_uv_fem_red_ref(grid, mu_test, u_ref, v_ref, u_fem[0:num_p1], v_fem[0:num_p1], u_red, v_red)

    print(u_fem[0:num_p1]/u_red)
    
    
if __name__ == '__main__':
    main()

    A = 0


"""  
# exact solution poiseuille
exact_poiseuille_reference_u = lambda X: -4./(height**2)*X[..., 1]**2 + 4./height*X[..., 1]
exact_poiseuille_reference_v = lambda X: np.zeros_like(X[..., 0])
exact_poiseuille_reference_p = lambda X: 8./(height**2) * (width - X[..., 0])

# exact solution cavity
exact_cavity_reference_u = lambda X: X[..., 0]**2 * (np.ones_like(X[..., 0]) - X[..., 0])**2 * 2*X[..., 1] *\
    (np.ones_like(X[..., 1]) - X[..., 1]) * (2*X[..., 1] - np.ones_like(X[..., 1]))
exact_cavity_reference_v = lambda X: X[..., 1]**2 * (np.ones_like(X[..., 1]) - X[..., 1])**2 * 2*X[..., 0] *\
                               (np.ones_like(X[..., 1]) - X[..., 0]) * (np.ones_like(X[..., 0]) - 2*X[..., 0])
exact_cavity_reference_p = lambda X: X[..., 0] * (np.ones_like(X[..., 0]) - X[..., 0]) * (np.ones_like(X[..., 1]) - X[..., 1])


b = rotate(45)

diameter = 1./12
fem_order = 2

PROBLEM = 2
PLOT_SEPARATE = False

if PROBLEM == 1:
    problem = PoiseuilleProblem()
elif PROBLEM == 2:
    problem = CavityProblem()
else:
    raise NotImplementedError

grid, boundary_info = discretize_domain_default(problem.domain, diameter=diameter)

ref_points = grid.embeddings(2)[1]

if PROBLEM == 1:
    exact_u = exact_poiseuille_reference_u(ref_points)
    exact_v = exact_poiseuille_reference_v(ref_points)
    exact_p = exact_poiseuille_reference_p(ref_points)
elif PROBLEM == 2:
    exact_u = exact_cavity_reference_u(ref_points)
    exact_v = exact_cavity_reference_v(ref_points)
    exact_p = exact_cavity_reference_p(ref_points)
else:
    raise NotImplementedError

num_p1 = grid.size(2)
num_p2 = grid.size(2) + grid.size(1)

bi_A = boundary_info
bi_B = boundary_info
bi_BT = EmptyBoundaryInfo(grid)
bi_F = boundary_info
bi = boundary_info

#trafo = np.array([[1.0, 0.0], [0.0, 1.0]])
#trafo = np.array([[0., -1.], [1., 0.]])
#trafo = np.array([[-1.0, 0.0], [0.0, -1.0]])
#trafo = np.array([[1./np.sqrt(2), -1./np.sqrt(2)], [1./np.sqrt(2), 1./np.sqrt(2)]])
#trafo = 2*np.eye(2)
#trafo = np.array([[1.0, 1.0], [0.0, 1.0]])

trafo = np.eye(2)
trafo = rotate(45)   # rotate 45
#trafo = rotate(90)   # rotate 90
#trafo = rotate(135)  # rotate 135
#trafo = rotate(180)  # rotate 180
#trafo = rotate(22.5)  # rotate 225
trafo = rotate(270)  # rotate 270
#trafo = rotate(315)  # rotate 315
#trafo = scale(2, 2)  # scale 2, 2
#trafo = scale(2, 1)  # scale 2, 1
#trafo = scale(1, 2)  # scale 1, 2
#trafo = np.array([[1,1],[0,1]])

mus = [np.eye(2), rotate(45), rotate(90), rotate(135), rotate(180), rotate(225), rotate(270), rotate(315)]
us = []
ps = []

for mu in mus:
    t_inv = np.linalg.inv(mu)
    t_det = np.linalg.det(mu)

    t_A = t_inv.dot(t_inv.T)
    t_B = t_inv.T

    t_A = ConstantFunction(value=t_A, dim_domain=2)
    t_B = ConstantFunction(value=t_B, dim_domain=2)

    ebi = EmptyBoundaryInfo(grid)

    #rhs = FunctionComposition(problem.rhs, trafo)
    rhs = problem.rhs
    #dirichlet_data = FunctionComposition(problem.dirichlet_data, trafo)
    dirichlet_data = problem.dirichlet_data

    if fem_order == 1:
        raise NotImplementedError
        op_A = DiffusionOperatorP1(grid, bi, trafo_matrix_A)._assemble()
        op_B = AdvectionOperatorP1(grid, bi, trafo_matrix_B)._assemble()
        op_BT = AdvectionOperatorP1(grid, ebi, trafo_matrix_B, False)._assemble()
        op_C = None
        op_F = L2TensorProductFunctionalP1(grid, rhs, bi, dirichlet_data)._assemble()
    elif fem_order == 2:
        op_A = DiffusionOperatorP2(grid, bi, t_A)._assemble()
        op_B = AdvectionOperatorP2(grid, bi, t_B)._assemble()
        op_BT = AdvectionOperatorP2(grid, ebi, t_B, False)._assemble()
        op_C = np.zeros((num_p1, num_p1))
        op_F_alt = L2TensorProductFunctionalP2(grid, rhs, bi, dirichlet_data)._assemble()
        op_F = TwoDimensionalL2ProductFunctionalP2(grid, rhs, bi, dirichlet_data, transformation_matrix=mu)._assemble()
    else:
        raise NotImplementedError

    A = bmat([[op_A, None], [None, op_A]])
    L = bmat([[A, -1.*op_B], [op_BT.T, op_C]])
    R = np.hstack((op_F, np.zeros((1, num_p1))))
    #R = trafo_det * R

    sol = spsolve(L, R)

    if fem_order == 1:
        raise NotImplementedError
        shift = num_p1
        u = sol[0:shift]
        v = sol[shift:2*shift]
        p = sol[2*shift:]
    elif fem_order == 2:
        shift = num_p2
        u = sol[0:shift]
        v = sol[shift:2*shift]
        p = sol[2*shift:]

    u = NumpyVectorArray(sol[0:2*num_p2])
    p = NumpyVectorArray(sol[2*num_p2:])
    u = sol[0:2*num_p2]
    p = sol[2*num_p2:]

    us.append(u)
    ps.append(p)

mu_test = trafo
t_inv = np.linalg.inv(mu_test)
t_det = np.linalg.det(mu_test)

t_A = t_inv.dot(t_inv.T)
t_B = t_inv.T

t_A = ConstantFunction(value=t_A, dim_domain=2)
t_B = ConstantFunction(value=t_B, dim_domain=2)

#rhs = FunctionComposition(problem.rhs, trafo)
rhs = problem.rhs
#dirichlet_data = FunctionComposition(problem.dirichlet_data, trafo)
dirichlet_data = problem.dirichlet_data
op_A = DiffusionOperatorP2(grid, bi, t_A)._assemble()
op_B = AdvectionOperatorP2(grid, bi, t_B)._assemble()
op_BT = AdvectionOperatorP2(grid, ebi, t_B, False)._assemble()
op_C = np.zeros((num_p1, num_p1))
#op_F_alt = L2TensorProductFunctionalP2(grid, rhs, bi_F, dirichlet_data)._assemble()
op_F = TwoDimensionalL2ProductFunctionalP2(grid, rhs, bi, dirichlet_data, transformation_matrix=mu)._assemble()

A = bmat([[op_A, None], [None, op_A]])
A_red = np.zeros((len(mus), len(mus)))
#A_red_v = np.zeros((len(mus), len(mus)))
B_red = np.zeros((len(mus), len(mus)))
#B_red_y = np.zeros((len(mus), len(mus)))
B_t_red = np.zeros((len(mus), len(mus)))
#B_t_red_y = np.zeros((len(mus), len(mus)))
F_red = np.zeros((1, len(mus)))
#F_red_y = np.zeros((1, len(mus)))

u2s = [us[i][0:num_p2] for i in xrange(len(mus))]
v2s = [us[i][num_p2:2*num_p2] for i in xrange(len(mus))]

for i in xrange(len(mus)):
    for j in xrange(len(mus)):
        A_red[i][j] = us[i].dot(A.dot(us[j]))
        #A_red_v[i][j] = v2s[i].dot(op_A.dot(v2s[j]))
        B_red[i][j] = us[i].dot(op_B.dot(ps[j]))
        #B_red_y[i][j] = v2s[i].dot(op_B.dot(ps[j]))
        B_t_red[i][j] = us[i].dot(op_BT.dot(ps[j]))
        #B_t_red_y[i][j] = v2s[i].dot(op_BT.dot(ps[j]))
    F_red[0][i] = op_F.dot(us[i])[0]
    #F_red_y[0][i] = op_F.dot(v2s[i])[0]

L_red = np.bmat([[A_red, -1.0*B_red], [B_t_red.T, np.zeros((B_red.shape[1], B_red.shape[1]))]])
F_red = np.hstack((F_red[0], np.zeros(B_red.shape[1])))
#L_red = np.bmat([[A_red, None, -1.0*B_red], [None, A_red, -1.0*B_red], [B_t_red.T, np.zeros((B_red.shape[1], B_red.shape[1]))]])
#F_red = np.hstack((F_red[0], np.zeros(B_red.shape[1])))

sol_red = np.linalg.solve(L_red, F_red)

u_red = np.zeros_like(us[0])
p_red = np.zeros_like(ps[0])

for i in xrange(len(mus)):
    u_red += sol_red[i]*us[i]
    p_red += sol_red[i+len(mus)]*ps[i]

u2_red = u_red[0:shift]
v2_red = u_red[shift:2*shift]

uv_red_t = np.einsum('ij,ej->ei', mu_test, np.vstack((u2_red, v2_red)).T)
uv_red_t /= t_det
u_red_t = uv_red_t[..., 0]
v_red_t = uv_red_t[..., 1]
#uv_trans = np.einsum('ij,ej->ei', trafo, np.vstack((u, v)).T)
#uv_trans /= trafo_det
#u2 = uv_trans[..., 0]
#v2 = uv_trans[..., 1]

u_plot = u_red_t[0:num_p1]
v_plot = v_red_t[0:num_p1]

test_grid = AffineTransformedGrid(grid, mu_test)
test_grid.visualize(NumpyVectorArray(p_red))

x_t = test_grid.centers(2)[..., 0]
y_t = test_grid.centers(2)[..., 1]

#from matplotlib import pyplot as plt

#plt.figure(1)
#plt.title('FEM reduced')
#plt.quiver(x_t, y_t, u2_red[0:num_p1], v2_red[0:num_p1])#, pivot='mid', color='r')
#plt.show()



trafo_inv = np.linalg.inv(trafo)
trafo_det = np.linalg.det(trafo)

# transform exact solution with piola transformation
exact_uv_trans = np.einsum('ij,ej->ei', trafo, np.vstack((exact_u, exact_v)).T)
exact_uv_trans /= trafo_det

exact_u2 = exact_uv_trans[..., 0]
exact_v2 = exact_uv_trans[..., 1]

#trafo_matrix_A = trafo_det * trafo_inv.dot(trafo_inv.T)
#trafo_matrix_B = trafo_det * trafo_inv.T

trafo_matrix_A = trafo_inv.dot(trafo_inv.T)
trafo_matrix_B = trafo_inv.T

trafo_matrix_A = ConstantFunction(value=trafo_matrix_A, dim_domain=2)
trafo_matrix_B = ConstantFunction(value=trafo_matrix_B, dim_domain=2)

trafo_grid = AffineTransformedGrid(grid, trafo)


if fem_order == 1:
    op_A = DiffusionOperatorP1(grid, bi_A, trafo_matrix_A)._assemble()
    op_B = AdvectionOperatorP1(grid, bi_B, trafo_matrix_B)._assemble()
    op_BT = AdvectionOperatorP1(grid, bi_BT, trafo_matrix_B, False)._assemble()
    op_C = None
    op_F = L2TensorProductFunctionalP1(grid, rhs, bi_F, dirichlet_data)._assemble()
elif fem_order == 2:
    op_A = DiffusionOperatorP2(grid, bi_A, trafo_matrix_A)._assemble()
    op_B = AdvectionOperatorP2(grid, bi_B, trafo_matrix_B)._assemble()
    op_BT = AdvectionOperatorP2(grid, bi_BT, trafo_matrix_B, False)._assemble()
    op_C = np.zeros((num_p1, num_p1))
    op_F_alt = L2TensorProductFunctionalP2(grid, rhs, bi_F, dirichlet_data)._assemble()
    op_F = TwoDimensionalL2ProductFunctionalP2(grid, rhs, bi_F, dirichlet_data, transformation_matrix=trafo)._assemble()
else:
    raise NotImplementedError

A = bmat([[op_A, None], [None, op_A]])
L = bmat([[A, -1.*op_B], [op_BT.T, op_C]])
R = np.hstack((op_F, np.zeros((1, num_p1))))
#R = trafo_det * R

sol = spsolve(L, R)
if fem_order == 1:
    shift = num_p1
    u = sol[0:shift]
    v = sol[shift:2*shift]
    p = sol[2*shift:]
elif fem_order == 2:
    shift = num_p2
    u = sol[0:shift][0:num_p1]
    v = sol[shift:2*shift][0:num_p1]
    p = sol[2*shift:]

uv_trans = np.einsum('ij,ej->ei', trafo, np.vstack((u, v)).T)
uv_trans /= trafo_det

u2 = uv_trans[..., 0]
v2 = uv_trans[..., 1]

U = NumpyVectorArray(u2)
V = NumpyVectorArray(v2)
P = NumpyVectorArray(p)

#trafo_grid.visualize(U)
#trafo_grid.visualize(V)

from matplotlib import pyplot as plt

x_t = trafo_grid.centers(2)[..., 0]
y_t = trafo_grid.centers(2)[..., 1]
x = grid.centers(2)[..., 0]
y = grid.centers(2)[..., 1]

fp2 = lambda X: X[..., 0]*(2. - X[..., 0])*(2. - X[..., 1])
p2 = fp2(trafo_grid.centers(2))
P2 = NumpyVectorArray(p2)

#plt_transform = mplt.Affine2D(matrix = None)#np.array([[1,0,0], [0,1,0], [0,0,1]]))

#plt.title('Transformed cavity')
#plt.plot(x, y, transform=plt_transform)

#plt.quiver(x_t, y_t, u2, v2)
"""
fig = plt.figure()
ax = plt.subplot(111)
base_trans = ax.transData
plt_trafo = np.hstack((trafo, np.array([[0], [0]])))
plt_trafo = np.vstack((plt_trafo, np.array([0, 0, 1])))
tr =  mplt.Affine2D(matrix = plt_trafo) + base_trans

Q = plt.quiver(x, y, u, v,
            pivot='mid', color='r', units='inches', transform = tr )
qk = plt.quiverkey(Q, 0.5, 0.03, 1, r'', fontproperties={'weight': 'bold'})
#plt.plot(x, y, 'k.', transform = tr)

plt.axis([-2,2,-2,2])
#fig.savefig('demo.png')
plt.show()
"""
trafo_centers = np.einsum('ij,ej->ei', trafo, grid.centers(2))

#plt.title('u')
#plot = plt.tripcolor(trafo_centers[..., 0], trafo_centers[..., 1], grid.subentities(0, 2), u2)
#plt.legend(handles=[plot])
#plt.show()

if PLOT_SEPARATE:
    trafo_grid.visualize(U, title='u')
    trafo_grid.visualize(V, title='v')
    trafo_grid.visualize(P, title='p')

e1 = trafo_inv[:,0]
e2 = trafo_inv[:,1]

u3 = np.zeros_like(u2)
v3 = np.zeros_like(v2)

for i in xrange(len(u)):
    u3[i] = u2[i] * e1[0] + v2[i] * e2[0]
    v3[i] = u2[i] * e1[1] + v2[i] * e2[1]

# plot
grid_size = (1, 3)
plt.subplot2grid(grid_size, (0, 0), rowspan=1, colspan=1)
plt.title('FEM reduced')
plt.quiver(x_t, y_t, u2_red[0:num_p1], v2_red[0:num_p1])#, pivot='mid', color='r')
plt.subplot2grid(grid_size, (0, 1), rowspan=1, colspan=1)
plt.title('FEM')
plt.quiver(x_t, y_t, u, v, color='r')
plt.subplot2grid(grid_size, (0, 2), rowspan=1, colspan=1)
plt.title('analytical')
plt.quiver(x_t, y_t, exact_u2, exact_v2, color='b')
#plt.tight_layout()
plt.show()

plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
plt.title('FEM reduced')
plt.tripcolor(trafo_grid.centers(2)[..., 0], trafo_grid.centers(2)[..., 1], trafo_grid.subentities(0, 2), p_red)
plt.colorbar()
plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
plt.title('FEM')
plt.tripcolor(trafo_grid.centers(2)[..., 0], trafo_grid.centers(2)[..., 1], trafo_grid.subentities(0, 2), p)
plt.colorbar()
plt.show()


plt.figure(1)
plt.title('FEM reduced')
plt.quiver(x_t, y_t, u2_red[0:num_p1], v2_red[0:num_p1])#, pivot='mid', color='r')
plt.show()

plt.figure(2)
plt.title('FEM')
plt.quiver(x_t, y_t, u, v)#, pivot='mid', color='r')
plt.show()


plt.figure(3)
plt.title('Analytical')
plt.quiver(x_t, y_t, exact_u2, exact_v2)
plt.show()

err_u = u - exact_u2
ERR_U = NumpyVectorArray(err_u)
err_u_trans = u2 - exact_u2
ERR_U_TRANS = NumpyVectorArray(err_u_trans)

bi_ref = EmptyBoundaryInfo(grid)

# NORMS
h1 = DiffusionOperatorP1(grid, bi_ref).apply2
l2 = L2ProductP1(grid, bi_ref).apply2

e = h1(ERR_U, ERR_U)**0.5
e_trans = h1(ERR_U_TRANS, ERR_U_TRANS)**0.5

z=0
"""
