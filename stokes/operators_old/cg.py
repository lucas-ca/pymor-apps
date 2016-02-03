# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Schaefer <michael.schaefer@uni-muenster.de>
#               lucas-ca <lucascamp@web.de>

""" This module provides some operators for continuous finite element discretizations."""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, vstack

from pymor.functions.interfaces import FunctionInterface
from pymor.functions.basic import GenericFunction
from pymor.grids.referenceelements import triangle, line
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.vectorarrays.interfaces import VectorSpace
from pymor.operators.cg import L2ProductFunctionalP1

from stokes.functions.finite_elements import P1ShapeFunctions, P1ShapeFunctionGradients, P2ShapeFunctions, P2ShapeFunctionGradients


class DiffusionOperatorP2(NumpyMatrixBasedOperator):
    """Diffusion |Operator| for quadratic finite elements und simplex grid.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ d(x) ∇ u(x) ]

    The function `d` can be scalar- or matrix-valued.
    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    diffusion_function
        The |Function| `d(x)` with ``shape_range == tuple()`` or
        ``shape_range = (grid.dim_outer, grid.dim_outer)``. If `None`, constant one is
        assumed.
    diffusion_constant
        The constant `c`. If `None`, `c` is set to one.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero to obtain a symmetric system matrix. Otherwise, only the rows will
        be set to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero (e.g. for affine decomposition). Otherwise they are set to one.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False, name=None):
        assert grid.reference_element(0) in {triangle, line}, 'A simplicial grid is expected!'
        assert diffusion_function is None \
            or (isinstance(diffusion_function, FunctionInterface) and
                diffusion_function.dim_domain == grid.dim_outer and
                diffusion_function.shape_range == tuple() or diffusion_function.shape_range == (grid.dim_outer,)*2)
        self.source = self.range = NumpyVectorSpace(grid.size(grid.dim) + grid.size(grid.dim-1))
        self.grid = grid
        self.boundary_info = boundary_info
        self.diffusion_constant = diffusion_constant
        self.diffusion_function = diffusion_function
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.name = name
        #if diffusion_function is not None:
        #    self.build_parameter_type(inherits=(diffusion_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # number of local and global shape functions
        if g.dim == 1:
            num_local_sf = 3
            num_global_sf = g.size(g.dim) + g.size(g.dim - 1)
        elif g.dim == 2:
            num_local_sf = 6
            num_global_sf = g.size(g.dim) + g.size(g.dim - 1)
        else:
            raise NotImplementedError

        # get quadrature points and weights
        q, w = g.reference_element.quadrature(order=2)

        # gradients of shape functions
        SF_GRAD = P2ShapeFunctionGradients(g.dim)(q)
        #SF_GRAD = np.array((
        #        [4.*q[..., 0] + 4.*q[..., 1] - 3., 4.*q[..., 0] + 4.*q[..., 1] - 3.],
        #        [4.*q[..., 0] - 1., np.zeros_like(q[..., 0])],
        #        [np.zeros_like(q[..., 0]), 4.*q[..., 1] - 1.],
        #        [4.*q[..., 1], 4.*q[..., 0]],
        #        [-4.*q[..., 1], -8.*q[..., 1] - 4.*q[..., 0] + 4.],
        #        [-8.*q[..., 0] - 4.*q[..., 1] + 4., -4.*q[..., 0]]
        #    ))

        # transformed gradients
        self.logger.info('Calulate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pjc->epic', g.jacobian_inverse_transposed(0), SF_GRAD)
        #del SF_GRAD
        #gradm = np.genfromtxt('/home/lucas/Dropbox/Uni/Master/Masterarbeit/MatlabCodeP2P1/grads', delimiter=',')

        # scalar products
        self.logger.info('Calculate all local scalar products beween gradients ...')
        if self.diffusion_function is not None and self.diffusion_function.shape_range == tuple():
            D = self.diffusion_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('epic,eqic,c,e,e->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D
        elif self.diffusion_function is not None:
            #D = self.diffusion_function(self.grid.centers(0), mu=mu)
            D = self.diffusion_function(self.grid.quadrature_points(0, order=2), mu=mu)
            SF_INTS = np.einsum('epic,eqjc,c,e,ecij->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D
        else:
            SF_INTS = np.einsum('epic,eqic,c,e->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0)).ravel()

        del SF_GRADS

        if self.diffusion_constant is not None:
            SF_INTS *= self.diffusion_constant

        # global dofs
        self.logger.info('Determine global dofs ...')
        VN = g.subentities(0, g.dim)
        EN = g.subentities(0, g.dim - 1) + g.size(g.dim)
        N = np.concatenate((VN, EN), axis=-1)
        del VN, EN

        SF_I0 = np.repeat(N, num_local_sf, axis=1).ravel()
        SF_I1 = np.tile(N, [1, num_local_sf]).ravel()
        del N

        # boundary
        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            VDM = bi.dirichlet_mask(g.dim)
            if g.dim == 1:
                EDM = np.zeros(g.size(g.dim - 1), dtype=bool)
            elif g.dim == 2:
                EDM = bi.dirichlet_mask(g.dim - 1)
            else:
                raise NotImplementedError
            DM = np.concatenate((VDM, EDM), axis=-1)
            del VDM, EDM

            SF_INTS = np.where(DM[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(DM[SF_I1], 0, SF_INTS)
            del DM

            if not self.dirichlet_clear_diag:
                if g.dim == 1:
                    DN = bi.dirichlet_boundaries(g.dim)
                elif g.dim == 2:
                    VDN = bi.dirichlet_boundaries(g.dim)
                    EDN = bi.dirichlet_boundaries(g.dim - 1) + g.size(2)
                    DN = np.concatenate((VDN, EDN), axis=-1)
                    del VDN, EDN
                else:
                    raise NotImplementedError

                SF_INTS = np.hstack((SF_INTS, np.ones(DN.size)))
                SF_I0 = np.hstack((SF_I0, DN))
                SF_I1 = np.hstack((SF_I1, DN))
                del DN

        # assemble global matrix
        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(num_global_sf, num_global_sf))
        del SF_INTS, SF_I0, SF_I1

        A = csc_matrix(A).copy()

        return A


class L2ProductFunctionalP2(NumpyMatrixBasedOperator):
    """|Functional| representing the scalar product with an L2-|Function| for quadratic finite elements.

    In one d the global node indices are given as follows ::

        |----------o----------|----------o----------|
        0          3          1          4          2

    In two d the global node indices are given as follows ::

        6---------23----------7---------24----------8
        | \                 / | \                 / |
        |    35         31    |    36         32    |
        |       \     /       |       \     /       |
       16         11         17         12          18
        |       /     \       |       /     \       |
        |    27         39    |    28         40    |
        | /                 \ | /                 \ |
        3---------21----------4---------22----------5
        | \                 / | \                 / |
        |    33         29    |    34         30    |
        |       \     /       |       \     /       |
       13          9         14         10          15
        |       /     \       |       /     \       |
        |    25         37    |    26         38    |
        | /                 \ | /                 \ |
        0---------19----------1---------20----------2

    Boundary treatment can be performed by providing `boundary_info` and `dirichlet_data`,
    in which case the DOFs corresponding to Dirichlet boundaries are set to the values
    provided by `dirichlet_data`. Neumann boundaries are handled by providing a
    `neumann_data` function, Robin boundaries by providing a `robin_data` tuple.

    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the scalar product.
    boundary_info
        |BoundaryInfo| determining the Dirichlet and Neumann boundaries or `None`.
        If `None`, no boundary treatment is performed.
    dirichlet_data
        |Function| providing the Dirichlet boundary values. If `None`,
        constant-zero boundary is assumed.
    neumann_data
        |Function| providing the Neumann boundary values. If `None`,
        constant-zero is assumed.
    robin_data
        Tuple of two |Functions| providing the Robin parameter and boundary values, see `RobinBoundaryOperator`.
        If `None`, constant-zero for both functions is assumed.
    order
        Order of the Gauss quadrature to use for numerical integration.
    name
        The name of the functional.
    """

    sparse = False
    range = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, neumann_data=None, robin_data=None,
                 order=2, name=None):
        assert grid.reference_element(0) in {line, triangle}
        #assert function.shape_range == tuple()
        self.source = NumpyVectorSpace(grid.size(grid.dim) + grid.size(grid.dim - 1))
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.order = order
        self.name = name
        self.build_parameter_type(inherits=(function, dirichlet_data, neumann_data))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        if g.dim == 1:
            num_local_sf = 3
            num_global_sf = g.size(g.dim) + g.size(g.dim - 1)
        elif g.dim == 2:
            num_local_sf = 6
            num_global_sf = g.size(g.dim) + g.size(g.dim - 1)

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points)
        g.quadrature_points(0, order=self.order)
        F = self.function(g.quadrature_points(0, order=self.order), mu=mu)

        # quadrature points and weights on reference element
        q, w = g.reference_element.quadrature(order=self.order)

        # local shape functions
        SF = P2ShapeFunctions(g.dim)(q)

        #if g.dim == 1:
        #    # phi_0 = 2x² - 3x + 1
        #    # phi_1 = 2x² - x
        #    # phi_2 = 2x² - x
        #    SF_old = np.array((
        #        2.*q[..., 0]**2 - 3.*q[..., 0] + 1.,
        #        2.*q[..., 0]**2 - q[..., 0],
        #        -4.*q[..., 0]**2 + 4.*q[..., 0]
        #    ))
        #elif g.dim == 2:
        #    # phi_0 = 2x² + 2y² + 4xy - 3x - 3y + 1
        #    # phi_1 = 2x² - x
        #    # phi_2 = 2y² - y
        #    # phi_3 = 4xy
        #    # phi_4 = -4y² - 4xy + 4y
        #    # phi_5 = -4x² - 4xy + 4x
        #    SF_old = np.array((
        #        2.*q[..., 0]**2 + 2.*q[..., 1]**2 + 4.*q[..., 0]*q[..., 1] - 3.*q[..., 0] - 3.*q[..., 1] + 1.,
        #        2.*q[..., 0]**2 - q[..., 0],
        #        2.*q[..., 1]**2 - q[..., 1],
        #        4.*q[..., 0]*q[..., 1],
        #        -4.*q[..., 1]**2 - 4.*q[..., 0]*q[..., 1] + 4.*q[..., 1],
        #        -4.*q[..., 0]**2 - 4.*q[..., 0]*q[..., 1] + 4.*q[..., 0]
        #    ))
        #else:
        #    raise NotImplementedError

        # integrate the products of the function with the shape functions on each element
        SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(0), w).ravel()
        del F, SF

        # map local DOFs to global DOFs
        # FIXME This implementation is horrible, find a better way!
        VN = g.subentities(0, g.dim)
        EN = g.subentities(0, g.dim - 1) + g.size(g.dim)
        N = np.concatenate((VN, EN), axis=-1).ravel()
        del VN, EN

        I = np.array(coo_matrix((SF_INTS, (np.zeros_like(N), N)), shape=(1, num_global_sf)).todense()).ravel()
        del SF_INTS

        # neumann boundary treatment
        if bi is not None and bi.has_neumann and self.neumann_data is not None:
            NI = bi.neumann_boundaries(1)
            if g.dim == 1:
                I[NI] -= self.neumann_data(g.centers(1)[NI])
            else:
                VN = g.subentities(g.dim - 1, g.dim)[NI]
                EN = NI + g.size(g.dim)
                N = np.concatenate((VN, EN[..., np.newaxis]), axis=-1).ravel()
                del VN, EN

                F = -self.neumann_data(g.quadrature_points(1, order=self.order)[NI], mu=mu)
                q, w = line.quadrature(order=self.order)
                SF = P2ShapeFunctions(g.dim - 1)(q)

                #SF = np.array((
                #    2.*q[..., 0]**2 - 3.*q[..., 0] + 1.,
                #    -4.*q[..., 0]**2 + 4.*q[..., 0],
                #    2.*q[..., 0]**2 - q[..., 0]
                #))

                SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(1)[NI], w).ravel()
                del F, SF, NI

                I += np.array(coo_matrix((SF_INTS, (np.zeros_like(N), N)), shape=(1, num_global_sf))
                                        .todense()).ravel()
                del SF_INTS

        # robin boundary treatment
        if bi is not None and bi.has_robin and self.robin_data is not None:
            raise NotImplementedError
            # TODO: not updated yet should be like neumann
            RI = bi.robin_boundaries(1)
            if g.dim == 1:
                xref = g.centers(1)[RI]
                I[RI] += (self.robin_data[0](xref) * self.robin_data[1](xref))
            else:
                xref = g.quadrature_points(1, order=self.order)[RI]
                F = (self.robin_data[0](xref, mu=mu) * self.robin_data[1](xref, mu=mu))
                q, w = line.quadrature(order=self.order)
                SF = np.squeeze(np.array([1 - q, q]))
                SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(1)[RI], w).ravel()
                N = g.subentities(1, 2)[RI].ravel()
                I += np.array(coo_matrix((SF_INTS, (np.zeros_like(N), N)), shape=(1, g.size(g.dim)))
                                        .todense()).ravel()

        if bi is not None and bi.has_dirichlet:
            if g.dim == 1:
                DI = bi.dirichlet_boundaries(g.dim)
            if g.dim == 2:
                VN = bi.dirichlet_boundaries(g.dim)
                EN = bi.dirichlet_boundaries(g.dim-1) + g.size(g.dim)
                DI = np.concatenate((VN, EN))
                del VN, EN
            if self.dirichlet_data is not None:
                DC = np.concatenate((g.centers(g.dim), g.centers(g.dim - 1)))[DI]
                I[DI] = self.dirichlet_data(DC, mu=mu)
                del DC
            else:
                I[DI] = 0
            del DI

        return I.reshape((1, -1))


class L2TensorProductFunctionalP1(NumpyMatrixBasedOperator):
    """|Functional| representing the scalar product with an L2-|Function| for linear finite elements.

    The L2-|Function| f is of the form:

        f:   R^m -> R^n
        f_i: R^m -> R

    The individual operators for f_i are stacked. The operator returns the following matrix:

        int (f_1 * v)
        ...
        int (f_n * v)
    """

    sparse = False
    range = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, neumann_data=None, robin_data=None,
                 order=2, name=None):
        assert grid.reference_element(0) in {line, triangle}
        self.source = NumpyVectorSpace(grid.size(grid.dim) * function.shape_range[0])
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.order = order
        self.name = name
        self.build_parameter_type(inherits=(function, dirichlet_data, neumann_data))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info
        sr = self.function.shape_range[0]

        assert self.dirichlet_data is None or self.dirichlet_data.shape_range == self.function.shape_range
        assert self.neumann_data is None or self.neumann_data.shape_range == self.function.shape_range
        assert self.robin_data is None or self.robin_data.shape_range == self.function.shape_range

        Fs = [GenericFunction(mapping=(lambda X: self.function.evaluate(X)[..., i]),
                              dim_domain=self.function.dim_domain,
                              shape_range=tuple(),
                              parameter_type=self.function.parameter_type,
                              name="{}_{}".format(self.function.name, i))
              for i in xrange(sr)]

        Ds = [GenericFunction(mapping=(lambda X: self.dirichlet_data.evaluate(X)[..., i]),
                              dim_domain=self.dirichlet_data.dim_domain,
                              shape_range=tuple(),
                              parameter_type=self.dirichlet_data.parameter_type,
                              name="{}_{}".format(self.dirichlet_data.name, i))
              if self.dirichlet_data else None for i in xrange(sr)]

        Ns = [GenericFunction(mapping=(lambda X: self.neumann_data.evaluate(X)[..., i]),
                              dim_domain=self.neumann_data.dim_domain,
                              shape_range=tuple(),
                              parameter_type=self.neumann_data.parameter_type,
                              name="{}_{}".format(self.neumann_data.name, i))
              if self.neumann_data else None for i in xrange(sr)]

        Rs = [GenericFunction(mapping=(lambda X: self.robin_data.evaluate(X)[..., i]),
                              dim_domain=self.robin_data.dim_domain,
                              shape_range=tuple(),
                              parameter_type=self.robin_data.parameter_type,
                              name="{}_{}".format(self.robin_data.name, i))
              if self.robin_data else None for i in xrange(sr)]

        Is = [L2ProductFunctionalP1(grid=g,
                                    function=Fs[i],
                                    boundary_info=bi,
                                    dirichlet_data=Ds[i],
                                    neumann_data=Ns[i],
                                    robin_data=Rs[i],
                                    order=self.order,
                                    name="{}_{}".format(self.name, i))._assemble()
              for i in xrange(sr)]

        return np.hstack(Is)


class L2TensorProductFunctionalP2(NumpyMatrixBasedOperator):
    """|Functional| representing the scalar product with an L2-|Function| for linear finite elements.

    The L2-|Function| f is of the form:

        f:   R^m -> R^n
        f_i: R^m -> R

    The individual operators for f_i are stacked. The operator returns the following matrix:

        int (f_1 * v)
        ...
        int (f_n * v)
    """

    sparse = False
    range = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, neumann_data=None, robin_data=None,
                 order=2, name=None):
        assert grid.reference_element(0) in {line, triangle}
        self.source = NumpyVectorSpace((grid.size(grid.dim) + grid.size(grid.dim - 1)) * function.shape_range[0])
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.order = order
        self.name = name
        self.build_parameter_type(inherits=(function, dirichlet_data, neumann_data))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info
        sr = self.function.shape_range[0]

        assert self.dirichlet_data is None or self.dirichlet_data.shape_range == self.function.shape_range
        assert self.neumann_data is None or self.neumann_data.shape_range == self.function.shape_range
        assert self.robin_data is None or self.robin_data.shape_range == self.function.shape_range

        Fs = [GenericFunction(mapping=(lambda X: self.function.evaluate(X)[..., i]),
                              dim_domain=self.function.dim_domain,
                              shape_range=tuple(),
                              parameter_type=self.function.parameter_type,
                              name="{}_{}".format(self.function.name, i))
              for i in xrange(sr)]

        Ds = [GenericFunction(mapping=(lambda X: self.dirichlet_data.evaluate(X)[..., i]),
                              dim_domain=self.dirichlet_data.dim_domain,
                              shape_range=tuple(),
                              parameter_type=self.dirichlet_data.parameter_type,
                              name="{}_{}".format(self.dirichlet_data.name, i))
              if self.dirichlet_data else None for i in xrange(sr)]

        Ns = [GenericFunction(mapping=(lambda X: self.neumann_data.evaluate(X)[..., i]),
                              dim_domain=self.neumann_data.dim_domain,
                              shape_range=tuple(),
                              parameter_type=self.neumann_data.parameter_type,
                              name="{}_{}".format(self.neumann_data.name, i))
              if self.neumann_data else None for i in xrange(sr)]

        Rs = [GenericFunction(mapping=(lambda X: self.robin_data.evaluate(X)[..., i]),
                              dim_domain=self.robin_data.dim_domain,
                              shape_range=tuple(),
                              parameter_type=self.robin_data.parameter_type,
                              name="{}_{}".format(self.robin_data.name, i))
              if self.robin_data else None for i in xrange(sr)]

        Is = [L2ProductFunctionalP2(grid=g,
                                    function=Fs[i],
                                    boundary_info=bi,
                                    dirichlet_data=Ds[i],
                                    neumann_data=Ns[i],
                                    robin_data=Rs[i],
                                    order=self.order,
                                    name="{}_{}".format(self.name, i))._assemble()
              for i in xrange(sr)]

        return np.hstack(Is)


class L2ProductP2(NumpyMatrixBasedOperator):
    pass


class AdvectionOperatorP1(NumpyMatrixBasedOperator):
    """
    An advection operator. Stacks x_i components.
    """

    def __init__(self, grid, boundary_info, advection_function=None, dirichlet_clear_rows=True, name=None):
        self.grid = grid
        self.boundary_info = boundary_info
        self.advection_function = advection_function
        self.dirichlet_clear_rows = dirichlet_clear_rows
        self.name = name
        # TODO ???
        self.range = NumpyVectorSpace(2*grid.size(grid.dim))
        self.source = NumpyVectorSpace(grid.size(grid.dim))
        if advection_function is not None:
            self.build_parameter_type(inherits=(advection_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        if g.dim == 1:
            raise NotImplementedError
        elif g.dim == 2:
            num_local_psf = 3
            num_local_vsf = 3
            num_global_psf = g.size(g.dim)
            num_global_vsf = g.size(g.dim)
            PN = g.subentities(0, g.dim)
            VN = g.subentities(0, g.dim)
        else:
            raise NotImplementedError

        q, w = g.reference_element.quadrature(order=2)

        PSF = P1ShapeFunctions(g.dim)(q)
        VSF_GRAD = P1ShapeFunctionGradients(g.dim)(q)
        del q


        #transform gradients
        VSF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), VSF_GRAD)
        del VSF_GRAD

        # evaluate advection_function

        # calculate scalar products
        if self.advection_function is not None and self.advection_function.shape_range == tuple():
            A = self.advection_function(self.grid.centers(0), mu=mu)
            INTS = np.einsum('pq,evi,e,q,e->evpi', PSF, VSF_GRADS, g.integration_elements(0), w, A)
            del A
        elif self.advection_function is not None:
            A = self.advection_function(self.grid.centers(0), mu=mu)
            INTS = np.einsum('pq,evi,e,q, eji->evpj', PSF, VSF_GRADS, g.integration_elements(0), w, A)
            del A
        else:
            INTS = np.einsum('pq,evi,e,q->evpi', PSF, VSF_GRADS, g.integration_elements(0), w)
        del PSF, VSF_GRADS, w

        INTS_X = INTS[..., 0].ravel()
        INTS_Y = INTS[..., 1].ravel()
        del INTS

        SF_I0 = np.repeat(VN, num_local_psf, axis=1).ravel()
        SF_I1 = np.tile(PN, [1, num_local_vsf]).ravel()
        del PN, VN

        if bi is not None and bi.has_dirichlet:
            # set whole row to zero on boundary nodes
            # d_m = bi.dirichlet_mask(g.dim)
            if self.dirichlet_clear_rows:
                INTS_X = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, INTS_X)
                INTS_Y = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, INTS_Y)

        B_X = coo_matrix((INTS_X, (SF_I0, SF_I1)), shape=(num_global_vsf, num_global_psf))
        B_Y = coo_matrix((INTS_Y, (SF_I0, SF_I1)), shape=(num_global_vsf, num_global_psf))
        del SF_I0, SF_I1, INTS_X, INTS_Y

        return vstack((csc_matrix(B_X).copy(), csc_matrix(B_Y).copy()))


class AdvectionOperatorP2(NumpyMatrixBasedOperator):
    """
    An advection operator. Stacks d_x and d_y.
    """

    def __init__(self, grid, boundary_info, advection_function=None, dirichlet_clear_rows=True, name=None):
        self.grid = grid
        self.boundary_info = boundary_info
        self.advection_function = advection_function
        self.dirichlet_clear_rows = dirichlet_clear_rows
        self.name = None
        # TODO ???
        self.range = NumpyVectorSpace(2*(grid.size(grid.dim) + grid.size(grid.dim - 1)))
        self.source = NumpyVectorSpace(grid.size(grid.dim))
        if advection_function is not None:
            self.build_parameter_type(inherits=(advection_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        if g.dim == 1:
            raise NotImplementedError
        elif g.dim == 2:
            num_local_psf = 3
            num_local_vsf = 6
            num_global_psf = g.size(g.dim)
            num_global_vsf = g.size(g.dim) + g.size(g.dim - 1)
            PN = g.subentities(0, g.dim)
            VVN = g.subentities(0, g.dim)
            VEN = g.subentities(0, g.dim - 1) + g.size(g.dim)
            VN = np.concatenate((VVN, VEN), axis=-1)
            del VVN, VEN
        else:
            raise NotImplementedError

        q, w = g.reference_element.quadrature(order=2)

        PSF = P1ShapeFunctions(g.dim)(q)
        VSF_GRAD = P2ShapeFunctionGradients(g.dim)(q)
        del q

        #if g.dim == 2:
        #    PSF = np.array((
        #        1. - q[..., 0] - q[..., 1],
        #        q[..., 0],
        #        q[..., 1]
        #    ))

        #    VSF_GRAD = np.array((
        #        [-1., -1.],
        #        [1., 0.],
        #        [0., 1.]
        #    ))
        #else:
        #    raise NotImplementedError

        #transform gradients
        VSF_GRADS = np.einsum('eij,vjc->evic', g.jacobian_inverse_transposed(0), VSF_GRAD)
        del VSF_GRAD

        # calculate scalar products
        if self.advection_function is not None and self.advection_function.shape_range == tuple():
            A = self.advection_function(self.grid.centers(0), mu=mu)
            INTS = np.einsum('pq,eviq,e,q,e->evpi', PSF, VSF_GRADS, g.integration_elements(0), w, A)
            del A
        elif self.advection_function is not None:
            A = self.advection_function(self.grid.centers(0), mu=mu)
            INTS = np.einsum('pq,eviq,e,q,eji->evpj', PSF, VSF_GRADS, g.integration_elements(0), w, A)
            del A
        else:
            INTS = np.einsum('pq,eviq,e,q->evpi', PSF, VSF_GRADS, g.integration_elements(0), w)
        del PSF, VSF_GRADS, w

        INTS_X = INTS[..., 0].ravel()
        INTS_Y = INTS[..., 1].ravel()
        del INTS

        SF_I0 = np.repeat(VN, num_local_psf, axis=1).ravel()
        SF_I1 = np.tile(PN, [1, num_local_vsf]).ravel()
        del PN, VN

        if bi is not None and bi.has_dirichlet:
            # set whole row to zero on boundary nodes
            # d_m = bi.dirichlet_mask(g.dim)
            if self.dirichlet_clear_rows:
                VDM = bi.dirichlet_mask(g.dim)
                EDM = bi.dirichlet_mask(g.dim - 1)
                DM = np.concatenate((VDM, EDM), axis=-1)
                del VDM, EDM

                INTS_X = np.where(DM[SF_I0], 0, INTS_X)
                INTS_Y = np.where(DM[SF_I0], 0, INTS_Y)


        B_X = coo_matrix((INTS_X, (SF_I0, SF_I1)), shape=(num_global_vsf, num_global_psf))
        B_Y = coo_matrix((INTS_Y, (SF_I0, SF_I1)), shape=(num_global_vsf, num_global_psf))
        del SF_I0, SF_I1, INTS_X, INTS_Y

        return vstack((csc_matrix(B_X).copy(), csc_matrix(B_Y).copy()))


class StabilizationOperatorP1(NumpyMatrixBasedOperator):
    """An operator for stabilization of stokes problem.
    It has the form:
        Σ h_K² ∫ ∇ p_h ∇ q_h.
        K      K

    Parameters:
    -----------
    grid
        The |Grid| for which to assemble the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, alpha=0.001, name=None):
        self.grid = grid
        self.alpha = alpha
        self.name = name

        self.source = NumpyVectorSpace(grid.size(grid.dim))
        self.range = NumpyVectorSpace(grid.size(grid.dim))

    def _assemble(self, mu=None):
        g = self.grid

        t = g.centers(g.dim)[g.subentities(0, g.dim)]
        T = np.abs(t[:,[0,0,1],:] - t[:,[1,2,2],:])
        diameters = self.alpha * np.max(np.linalg.norm(T, 2, 2)**2, axis=1)

        # quadrature rule
        q, w = g.reference_element.quadrature(order=2)

        # graddients of the shape functions
        SF_GRAD = P1ShapeFunctionGradients(g.dim)(q)

        # gradients of shape functions transformed by reference map
        SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)

        SF_INTS = np.einsum('epi,eqi,e,e->epq', SF_GRADS, SF_GRADS, g.volumes(0), diameters).ravel()

        # determine global dofs
        SF_I0 = np.repeat(g.subentities(0, g.dim), g.dim + 1, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, g.dim + 1]).ravel()

        # assemble system matrix
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        del SF_INTS, SF_I0, SF_I1

        return csc_matrix(A).copy()


class ZeroOperator(NumpyMatrixBasedOperator):

    """An operator represented by range x source zero matrix."""

    def __init__(self, source, range,  sparse=False, name=None):
        assert isinstance(source, (VectorSpace, int)), "source must be a NumpyVectorSpace or int"
        assert isinstance(range, (VectorSpace, int)), "range must be a NumpyVectorSpace or int"
        self.source = NumpyVectorSpace(source) if isinstance(source, int) else source
        self.range = NumpyVectorSpace(range) if isinstance(range, int) else range
        self.sparse = sparse
        self.name = name

    def _assemble(self, mu=None):
        s = self.source.dim
        r = self.range.dim

        if self.sparse:
            # return a sparse zero matrix in csc format
            return csc_matrix(shape=(r, s))
        else:
            # return dense zero matrix
            return np.zeros((r, s))


class TransposedOperator(NumpyMatrixBasedOperator):
    """Represents the transposed of an MatrixBasedOperator."""

    def __init__(self, op):
        self.op = op
        self.source = op.range
        self.range = op.source
        self.name = "{}_transposed".format(op.name)

    def _assemble(self, mu=None):
        return self.op._assemble(mu).T


class TwoDimensionalL2ProductFunctionalP2(NumpyMatrixBasedOperator):

    """
    |Functional| representing the scalar product with an L2-|Function| f for quadratic finite elements.
    f is as |Function|
        f: R^2 -> R^2.
    A 2x2 transformation matrix T can be applied, with
        T: R^2 -> R^2,
    The following Integral is computed:
         (A11 A12) (f1) (v1)   ∫ A11 f1 + A12 f2 v1
        ∫(       )           =
         (A21 A22) (f2) (v2)   ∫ A21 f1 + A22 f2 v2

    Parameters:
    ===========
    grid
        The grid over which to assemble the operator.
    function
        The |Function| to take the scalar product with.
    boundary_info

    dirichlet_data

    neumann_data

    robin_data

    order

    name
        The name of the operator.
    transformation_matrix
        The 2x2 transformation matrix T to perform the piola transformation with.
    """

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, neumann_data=None, robin_data=None,
                 order=2, name=None, transformation_matrix=None):
        assert grid.reference_element is triangle
        assert function.shape_range[0] == 2

        self.source = NumpyVectorSpace(2*(grid.size(grid.dim) + grid.size(grid.dim - 1)))
        self.range = NumpyVectorSpace(1)

        self.grid = grid
        self.function = function
        self.boundary_info = boundary_info
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.order = order
        self.name = name
        self.transformation_matrix = transformation_matrix
        self.build_parameter_type(inherits=(function, dirichlet_data, neumann_data))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # number of global shape functions
        num_global_sf = g.size(g.dim) + g.size(g.dim - 1)

        # evaluate f in all quadrature points
        # shape: (g.shape[0], num_quadrature_points, 2)
        f = self.function(g.quadrature_points(0, order=self.order), mu=mu)
        assert f.shape[-1] == 2

        # transform f with piola transformation
        if self.transformation_matrix is not None:
            #f2 = np.einsum('ij, ecj->eci', np.linalg.inv(self.transformation_matrix), f)
            #f2 *= np.linalg.det(self.transformation_matrix)
            f2 = np.einsum('ij,ecj->eci', self.transformation_matrix, f)
            f2 /= np.linalg.det(self.transformation_matrix)
        else:
            f2 = f

        # quadrature on reference element
        q, w = g.reference_element.quadrature(order=self.order)

        # shape functions on refenrence element
        SF = P2ShapeFunctions(2)(q)

        # split F into F1 and F2
        F = [f2[..., i] for i in xrange(2)]
        Is = []

        # calculate F1 and F2 separately
        for sr in xrange(2):
            SF_INTS = np.einsum('ec,pc,e,c->ep', F[sr], SF, g.integration_elements(0), w).ravel()

            # vertex nodes
            VN = g.subentities(0, g.dim)
            # edge nodes
            EN = g.subentities(0, g.dim - 1) + g.size(g.dim)
            # all nodes
            N = np.concatenate((VN, EN), axis=-1).ravel()
            del VN, EN

            # build vector
            i = np.array(coo_matrix((SF_INTS, (np.zeros_like(N), N)), shape=(1, num_global_sf)).todense()).ravel()
            del SF_INTS

            # neumann boundary
            if bi is not None and bi.has_neumann:
                raise NotImplementedError

            # robin boundary
            if bi is not None and bi.has_robin:
                raise NotImplementedError

            if bi is not None and bi.has_dirichlet:
                # vertex nodes
                VN = bi.dirichlet_boundaries(g.dim)
                # edge nodes
                EN = bi.dirichlet_boundaries(g.dim-1) + g.size(g.dim)
                # all nodes
                DI = np.concatenate((VN, EN))
                del VN, EN

                if self.dirichlet_data is not None:
                    # points to evaluate dirichlet function in
                    DC = np.concatenate((g.centers(g.dim), g.centers(g.dim - 1)))[DI]
                    # transform dirichlet function with piola transformation
                    D = self.dirichlet_data(DC, mu=mu)
                    #D = np.einsum('ij, ej->ei', np.linalg.inv(self.transformation_matrix), D)
                    #D *= np.linalg.det(self.transformation_matrix)
                    D = np.einsum('ij, ej->ei', self.transformation_matrix, D)
                    D /= np.linalg.det(self.transformation_matrix)

                    i[DI] = D[..., sr]
                    del DC
                else:
                    i[DI] = 0
                del DI

                Is.append(i.reshape((1, -1)))

        I = np.hstack(Is)

        return I



if __name__ == '__main__':
    from pymor.grids.tria import TriaGrid
    from pymor.functions.basic import ConstantFunction
    from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo

    g = TriaGrid((2,2))
    bi = AllDirichletBoundaryInfo(g)
    rhs = ConstantFunction(value=np.array([0., 0.]), dim_domain=2)
    r = lambda X: np.dstack(
            [
                24.*X[...,0]**4*X[...,1] -\
                12.*X[...,0]**4 -\
                48.*X[...,0]**3*X[...,1] +\
                24.*X[...,0]**3 +\
                48.*X[...,0]**2*X[...,1]**3 -\
                72.*X[...,0]**2*X[...,1]**2 +\
                48.*X[...,0]**2*X[...,1] -\
                12.*X[...,0]**2 -\
                48.*X[...,0]*X[...,1]**3 +\
                72.*X[...,0]*X[...,1]**2 -\
                22.*X[...,0]*X[...,1] -\
                2.*X[...,0] +\
                8.*X[...,1]**3 -\
                12.*X[...,1]**2 +\
                3.*X[...,1] +\
                1.
            ,
                -48.*X[...,0]**3*X[...,1]**2 +\
                48.*X[...,0]**3*X[...,1] -\
                8.*X[...,0]**3 +\
                72.*X[...,0]**2*X[...,1]**2 -\
                72.*X[...,0]**2*X[...,1] +\
                13.*X[...,0]**2 -\
                24.*X[...,0]*X[...,1]**4 +\
                48.*X[...,0]*X[...,1]**3 -\
                48.*X[...,0]*X[...,1]**2 +\
                24.*X[...,0]*X[...,1] -\
                5.*X[...,0] +\
                12.*X[...,1]**4 -\
                24.*X[...,1]**3 +\
                12.*X[...,1]**2
            ])
    rhs = GenericFunction(mapping=r, dim_domain=2, shape_range=(2,), name='force')

    f1 = L2TensorProductFunctionalP2(g, rhs, bi)._assemble()
    f2 = TwoDimensionalL2ProductFunctionalP2(g, rhs, bi, transformation_matrix=-1.*np.eye(2))._assemble()

    Z = 0