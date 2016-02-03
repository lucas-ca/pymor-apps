# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Module containing a block operator."""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

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

diameter = 1./40
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

#trafo = np.array([[1.0, 0.0], [0.0, 1.0]])
#trafo = np.array([[0., -1.], [1., 0.]])
#trafo = np.array([[-1.0, 0.0], [0.0, -1.0]])
#trafo = np.array([[1./np.sqrt(2), -1./np.sqrt(2)], [1./np.sqrt(2), 1./np.sqrt(2)]])
#trafo = 2*np.eye(2)
#trafo = np.array([[1.0, 1.0], [0.0, 1.0]])

trafo = np.eye(2)
trafo = rotate(45)   # rotate 45
trafo = rotate(90)   # rotate 90
trafo = rotate(135)  # rotate 135
trafo = rotate(180)  # rotate 180
#trafo = rotate(225)  # rotate 225
#trafo = rotate(270)  # rotate 270
#trafo = rotate(315)  # rotate 315
#trafo = scale(2, 2)  # scale 2, 2
#trafo = scale(2, 1)  # scale 2, 1
#trafo = scale(1, 2)  # scale 1, 2
#trafo = np.array([[1,1],[0,1]])


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

#rhs = FunctionComposition(problem.rhs, trafo)
rhs = problem.rhs
#dirichlet_data = FunctionComposition(problem.dirichlet_data, trafo)
dirichlet_data = problem.dirichlet_data


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

plt.figure(1)
plt.title('FEM')
plt.quiver(x_t, y_t, u, v)#, pivot='mid', color='r')
plt.show()

plt.figure(2)
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
