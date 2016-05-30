import numpy_ as np
from matplotlib import pyplot as plt

from pymor.grids.tria import TriaGrid
from stokes.grids.affine_transformed_tria import AffineTransformedTriaGrid
from stokes.functions.affine_transformation import AffineTransformation
from stokes_alt.analyticalproblems.cavity import CavityProblem


def rotate(angle):
    s = np.sin(angle)
    c = np.cos(angle)

    return np.array([[c, -s], [s, c]])

r = 1.0
problem = 2

trans = np.array([[1.0, 0.0], [0.0, 1.2]])
#trans = rotate(np.pi/4.0)
nx = 11
h = 1.0/(nx-1)

grid = TriaGrid((nx, nx))

cavity_u = lambda X: X[..., 0]**2 * (np.ones_like(X[..., 0]) - X[..., 0])**2 * 2 * X[..., 1] * (np.ones_like(X[..., 1]) - X[..., 1]) * (2 * X[..., 1] - np.ones_like(X[..., 1]))
cavity_v = lambda X: X[..., 1]**2 * (np.ones_like(X[..., 1]) - X[..., 1])**2 * 2 * X[..., 0] * (np.ones_like(X[..., 0]) - X[..., 0]) * (np.ones_like(X[..., 0]) - 2 * X[..., 0])

poiseuille_u = lambda X: -4.0 * X[..., 1]**2 + 4.0 * X[..., 1]
poiseuille_v = lambda X: np.zeros_like(X[..., 0])

X = grid.centers(2)[..., 0]
Y = grid.centers(2)[..., 1]
if problem == 1:
    U = poiseuille_u(grid.centers(2))
    V = poiseuille_v(grid.centers(2))
elif problem == 2:
    U = cavity_u(grid.centers(2))
    V = cavity_v(grid.centers(2))

XY_trans = np.einsum('ij,ej->ei', trans, grid.centers(2))
X_trans = XY_trans[..., 0]
Y_trans = XY_trans[..., 1]

UV_trans = np.einsum('ij, ej->ei', trans, np.concatenate((U[..., np.newaxis], V[..., np.newaxis]), axis=-1))
UV_trans *= 1.0/np.linalg.det(trans)

U_trans = UV_trans[..., 0]
V_trans = UV_trans[..., 1]

def r(X):
    x = X[..., 0]
    y = X[..., 1]
    one = np.ones_like(X[..., 0])

    res = np.dstack([
        4.0 * (
            x**4 * (6.0 * y - 3.0*one) +
            x**3 * (6.0*one - 12.0 * y) +
            3.0 * x**2 * (4.0 * y**3 - 6.0 * y**2 + 4.0 * y - 1.0 * one) -
            6.0 * x * y * (2.0 * y**2 - 3.0 * y + 1.0 * one) +
            y * (2.0*y**2 - 3.0 * y + 1.0 * one)
        ) + (2.0 * x - 1.0 * one) * (y - 1.0 * one) * y
    ,
        -4.0 * (
            2.0 * x**3 * (6.0 * y**2 - 6.0 * y + 1.0 * one) -
            3.0 * x**2 * (6.0 * y**2 - 6.0 * y + 1.0 * one) +
            x * (6.0 * y**4 - 12.0 * y**3 + 12.0 * y**2 - 6.0*y + 1.0 * one) -
            3.0 * (y - 1.0 * one)**2 * y**2
        ) + (2.0 * y - 1.0 * one) * (x - 1.0 * one) * x
    ])

    return res

U_r = r(grid.centers(2))[0][..., 0]
V_r = r(grid.centers(2))[0][..., 1]

UV_r_trans = np.einsum('ej, ij->ei', np.concatenate((U_r[..., np.newaxis], V_r[..., np.newaxis]), axis=-1), trans)
UV_r_trans *= 1.0/np.linalg.det(trans)

U_r_trans = UV_r_trans[..., 0]
V_r_trans = UV_r_trans[..., 1]

#plt.figure('Cavity Quiver')
#plt.quiver(X, Y, U, V)
#plt.figure('Cavity u')
#plt.tripcolor(X, Y, grid.subentities(0, 2), U)
#plt.colorbar()
#plt.figure('Cavity v')
#plt.tripcolor(X, Y, grid.subentities(0, 2), V)
#plt.colorbar()
plt.figure('Cavity Quiver')
plt.quiver(X, Y, U_r, V_r)
plt.figure('Cavity Quiver transformed')
plt.quiver(X_trans, Y_trans, U_r_trans, V_r_trans)
plt.figure('Cavity transformed u')
plt.tripcolor(X_trans, Y_trans, grid.subentities(0, 2), U_trans)
plt.colorbar()
plt.figure('Cavity transformed v')
plt.tripcolor(X_trans, Y_trans, grid.subentities(0, 2), V_trans)
plt.colorbar()
print('Min u: {}, Min u_trans: {}, ratio: {}'.format(np.min(U), np.min(U_trans), np.min(U)/np.min(U_trans)))
print('Max u: {}, Max u_trans: {}, ratio: {}'.format(np.max(U), np.max(U_trans), np.max(U)/np.max(U_trans)))
print('Min v: {}, Min v_trans: {}, ratio: {}'.format(np.min(V), np.min(V_trans), np.min(V)/np.min(V_trans)))
print('Max v: {}, Max v_trans: {}, ratio: {}'.format(np.max(V), np.max(V_trans), np.max(V)/np.max(V_trans)))
plt.plot(xx2[..., 0], yy2[..., 0])
z = 0