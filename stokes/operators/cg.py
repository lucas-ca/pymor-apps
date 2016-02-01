# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from pymor.functions.interfaces import FunctionInterface
from pymor.grids.referenceelements import line, triangle
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.vectorarrays.interfaces import VectorSpace

from stokes.functions.finite_elements import P2ShapeFunctions, P2ShapeFunctionGradients


class DiffusionOperatorP2(NumpyMatrixBasedOperator):
    """Diffusion |Operator| for quadratic finite elements.
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
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False, solver_options=None,
                 name=None):
        assert grid.reference_element(0) in {triangle, line}
        assert diffusion_function is None \
            or (isinstance(diffusion_function, FunctionInterface) and
                diffusion_function.dim_domain == grid.dim_outer and
                diffusion_function.shape_range == tuple() or
                diffusion_function.shape_range == (grid.dim_outer,)*2)
        self.source = NumpyVectorSpace(grid.size(grid.dim) + grid.size(grid.dim - 1))
        self.grid = grid
        self.boundary_info = boundary_info
        self.diffusion_constant = diffusion_constant
        self.diffusion_function = diffusion_function
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.solver_options = solver_options
        self.name = name
        if diffusion_function is not None:
            self.build_parameter_type(inherits=(diffusion_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # quadrature rule on reference element
        q, w = g.reference_element.quadrature(order=2)

        # gradients of shape functions
        SF_GRAD = P2ShapeFunctionGradients(g.dim)
        # TODO: check number of local shape functions
        num_local_sf = SF_GRAD.shape[-2]
        num_global_sf = g.size(g.dim) + g.size(g.dim - 1)

        # calculate gradients of shape functions transformed by reference map
        SF_GRADS = np.einsum('eij,pjc->epic', g.jacobian_inverse_transposed(0), SF_GRAD)
        del SF_GRAD

        # calculate all local scalar products between gradients
        if self.diffusion_function is not None and self.diffusion_function.shape_range == tuple():
            # evaluate diffusion function
            D = self.diffusion_function(g.quadrature_points(0, order=2), mu=mu)
            SF_INTS = np.einsum('epic,eqic,c,e,ec->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D
        elif self.diffusion_function is not None:
            # evaluate diffusion function
            D = self.diffusion_function(g.quadrature_points(0, order=2), mu=mu)
            SF_INTS = np.einsum('epic,eqjc,c,e,ecij->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D
        else:
            SF_INTS = np.einsum('epic,eqic,c,e->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0)).ravel()

        del SF_GRADS

        if self.diffusion_constant is not None:
            SF_INTS *= self.diffusion_constant

        # determine global dofs
        # vertex nodes
        VN = self.subentities(0, g.dim)
        # edge nodes
        EN = self.subentities(0, g.dim - 1) + g.size(g.dim)
        # all nodes
        N = np.concatenate((VN, EN), axis=-1)
        del VN, EN

        SF_I0 = np.repeat(N, num_local_sf, axis=1).ravel()
        SF_I1 = np.tile(N, [1, num_local_sf]).ravel()

        # boundary treatment
        if bi.has_dirichlet:
            # dirichlet mask
            if g.dim == 1:
                # vertex dirichlet mask
                VDM = bi.dirichlet_mask(g.dim)
                # edge dirichlet mask
                EDM = np.np.zeros(g.size(g.dim - 1), dtype=bool)
                # dirichlet mask
                DM = np.concatenate((VDM, EDM), axis=-1)
                del VDM, EDM
            elif g.dim == 2:
                # vertex dirichlet mask
                VDM = bi.dirichlet_mask(g.dim)
                # edge dirichlet mask
                EDM = bi.dirichlet_mask(g.dim - 1) + g.size(g.dim)
                # dirichlet mask
                DM = np.concatenate((VDM, EDM), axis=-1)
                del VDM, EDM
            else:
                raise NotImplementedError

            # clear dirichlet rows
            SF_INTS = np.where(DM[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                # clear dirichlet columns
                SF_INTS = np.where(DM[SF_I1], 0, SF_INTS)
            del DM

            if not self.dirichlet_clear_diag:
                # dirichlet nodes
                if g.dim == 1:
                    # dirichlet nodes
                    DN = bi.dirichlet_boundaries(g.dim)
                elif g.dim == 2:
                    # vertex dirichlet nodes
                    VDN = bi.dirichlet_boundaries(g.dim)
                    # edge dirichlet nodes
                    EDN = bi.dirichlet_boundaries(g.dim - 1) + g.size(g.dim)
                    # dirichlet nodes
                    DN = np.concatenate((VDN, EDN), axis=-1)
                    del VDN, EDN
                else:
                    raise NotImplementedError

                # force one on diagonal
                SF_INTS = np.hstack((SF_INTS, np.ones(DN.size)))
                SF_I0 = np.hstack((SF_I0, DN))
                SF_I1 = np.hstack((SF_I1, DN))
                del DN

            # assemble global stiffness matrix
            A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(num_global_sf, num_global_sf))
            del SF_INTS, SF_I0, SF_I1

            return csc_matrix(A).copy()


class L2ProductP2(NumpyMatrixBasedOperator):
    """|Operator| representing the L2-product between linear finite element functions.
    To evaluate the product use the :meth:`~pymor.operators.interfaces.OperatorInterface.apply2`
    method.
    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.
    Parameters
    ----------
    grid
        The |Grid| for which to assemble the product.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    dirichlet_clear_rows
        If `True`, set the rows of the system matrix corresponding to Dirichlet boundary
        DOFs to zero. (Useful when used as mass matrix in time-stepping schemes.)
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero (to obtain a symmetric matrix).
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero (e.g. for affine decomposition). Otherwise, if either `dirichlet_clear_rows` or
        `dirichlet_clear_columns` is `True`, the diagonal entries are set to one.
    coefficient_function
        Coefficient |Function| for product with ``shape_range == tuple()``.
        If `None`, constant one is assumed.
    name
        The name of the product.
    """

    sparse = True

    def __init__(self, grid, boundary_info, dirichlet_clear_rows=True, dirichlet_clear_columns=False,
                 dirichlet_clear_diag=False, coefficient_function=None, solver_options=None, name=None):
        assert grid.reference_element in (line, triangle)
        self.source = NumpyVectorSpace(grid.size(grid.dim) + grid.size(grid.dim - 1))
        self.range = NumpyVectorSpace(grid.size(grid.dim) + grid.size(grid.dim - 1))
        self.grid = grid
        self.boundary_info = boundary_info
        self.dirichlet_clear_rows = dirichlet_clear_rows
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.coefficient_function = coefficient_function
        self.solver_options = solver_options
        self.name = name

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # quadrature rule
        q, w = g.reference_element.quadrature(order=2)

        # shape functions evaluated in quadrature points
        SFQ = P2ShapeFunctionGradients(g.dim, q)
        # TODO: check number of local shape functions
        num_local_sf = SFQ.shape[-2]
        num_global_sf = g.size(g.dim) + g.size(g.dim - 1)

        # integrate the products of the shape functions on each element
        if self.coefficient_function is not None:
            C = self.coefficient_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('iq,jq,q,e,e->eij', SFQ, SFQ, w, g.integration_elements(0), C).ravel()
            del C
        else:
            SF_INTS = np.einsum('iq,jq,q,e->eij', SFQ, SFQ, w, g.integration_elements(0)).ravel()

        del SFQ

        # determine global dofs
        # vertex nodes
        VN = self.subentities(0, g.dim)
        # edge nodes
        EN = self.subentities(0, g.dim - 1) + g.size(g.dim)
        # all nodes
        N = np.concatenate((VN, EN), axis=-1)
        del VN, EN

        SF_I0 = np.repeat(N, num_local_sf, axis=1).ravel()
        SF_I1 = np.tile(N, [1, num_local_sf]).ravel()

        # boundary treatment
        if bi.has_dirichlet:
            # dirichlet mask
            if g.dim == 1:
                # vertex dirichlet mask
                VDM = bi.dirichlet_mask(g.dim)
                # edge dirichlet mask
                EDM = np.np.zeros(g.size(g.dim - 1), dtype=bool)
                # dirichlet mask
                DM = np.concatenate((VDM, EDM), axis=-1)
                del VDM, EDM
            elif g.dim == 2:
                # vertex dirichlet mask
                VDM = bi.dirichlet_mask(g.dim)
                # edge dirichlet mask
                EDM = bi.dirichlet_mask(g.dim - 1) + g.size(g.dim)
                # dirichlet mask
                DM = np.concatenate((VDM, EDM), axis=-1)
                del VDM, EDM
            else:
                raise NotImplementedError
            if self.dirichlet_clear_rows:
                SF_INTS = np.where(DM[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(DM[SF_I1], 0, SF_INTS)
            del DM

            # dirichlet nodes
            if g.dim == 1:
                DN = bi.dirichlet_boundaries(g.dim)
            elif g.dim == 2:
                # vertex dirichlet nodes
                VDN = bi.dirichlet_boundaries(g.dim)
                # edge dirichlet nodes
                EDN = bi.dirichlet_boundaries(g.dim - 1) + g.size(g.dim)
                # dirichlet nodes
                DN = np.concatenate((VDN, EDN), axis=-1)
                del VDN, EDN
            else:
                raise NotImplementedError

            if not self.dirichlet_clear_diag and (self.dirichlet_clear_rows or self.dirichlet_clear_columns):
                SF_INTS = np.hstack((SF_INTS, np.ones(DN.size)))
                SF_I0 = np.hstack((SF_I0,DN))
                SF_I1 = np.hstack((SF_I1, DN))
            del DN

        # assemble global mass matrix
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(num_global_sf, num_global_sf))
        del SF_INTS, SF_I0, SF_I1

        return csc_matrix(A).copy()


class ZeroOperator(NumpyMatrixBasedOperator):
    """An operator represented by a range x source zero matrix.
    """

    def __init__(self, source, range, sparse=False, name=None):
        assert isinstance(source, (VectorSpace, int))
        assert isinstance(range, (VectorSpace, int))

        self.source = NumpyVectorSpace(source) if isinstance(source, int) else source
        self.range = NumpyVectorSpace(range) if isinstance(range, int) else range
        self.sparse = sparse
        self.name = name

    def _assemble(self, mu=None):
        s = self.source.dim
        r = self.range.dim

        if self.sparse:
            # return a sparse matrix in csc format
            return csc_matrix(shape=(r, s))
        else:
            # return a dense matrix
            return np.zeros(shape=(r, s))


class TransposedOperator(NumpyMatrixBasedOperator):
    """Represents the transposed of an MatrixBasedOperator."""

    def __init__(self, operator):
        self.operator = operator
        self.source = operator.range
        self.range = operator.source
        self.name = '{0}_transposed'.format(operator.name)

    def _assemble(self, mu=None):
        return self.operator._assemble(mu).T