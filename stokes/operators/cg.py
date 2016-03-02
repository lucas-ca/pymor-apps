# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, vstack

from pymor.functions.interfaces import FunctionInterface
from pymor.grids.referenceelements import line, triangle
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.vectorarrays.interfaces import VectorSpace

from stokes.functions.finite_elements import P1ShapeFunctions, P1ShapeFunctionGradients
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
        SFQ = P2ShapeFunctions(g.dim)(q)
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


class AdvectionOperatorP1(NumpyMatrixBasedOperator):
    """An advection operator. P1 shape functions for pressure and velocity nodes. Computes the following integral::

    """

    def __init__(self, grid, boundary_info, advection_function=None, dirichlet_clear_rows=False, name=None):
        assert grid.reference_element is triangle
        assert advection_function is None or \
            (advection_function is not None and \
                advection_function.dim_domain == grid.dim_outer and \
                advection_function.shape_range == (grid.dim_outer,)*2)
        self.source = NumpyVectorSpace(grid.size(grid.dim))
        self.range = NumpyVectorSpace(grid.dim * grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.advection_function = advection_function
        self.dirichlet_clear_rows = dirichlet_clear_rows
        self.name = name
        if advection_function is not None:
            self.build_parameter_type(inherits=(advection_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # quadrature rule on reference element
        q, w = g.reference_element.quadrature(order=2)

        if g.dim == 1:
            raise NotImplementedError
        elif g.dim == 2:
            # pressure shape functions
            num_local_psf = 3
            num_global_psf = g.size(g.dim)
            # pressure nodes
            PN = g.subentities(0, g.dim)
            # velocity shape functions
            num_local_vsf = 3
            num_global_vsf = g.size(g.dim)
            # velocity nodes
            VN = g.subentities(0, g.dim)
        else:
            raise NotImplementedError

        # evaluate pressure shape functions in all quadrature points
        PSF = P1ShapeFunctions(g.dim)(q)

        # evaluate gradients of velocity shape functions in all quadrature points
        VSF_GRAD = P1ShapeFunctionGradients(g.dim)(q)
        del q

        # transform gradients
        VSF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), VSF_GRAD)
        del VSF_GRAD

        # calculate scalar products
        if self.advection_function is not None:
            A = self.advection_function(g.centers(0), mu=mu)
            INT = np.einsum('pq,evi,e,q,eji->evpj', PSF, VSF_GRADS, g.intergration_elements(0), w, A)
        else:
            INT = np.einsum('pq,evi,e,q->evpi', PSF, VSF_GRADS, g.integration_elements(0), w)
        del PSF, VSF_GRADS, w

        INTS = [INT[..., i].ravel() for i in xrange(g.dim)]
        del INT

        # determine global dofs
        SF_I0 = np.repeat(VN, num_local_psf, axis=1).ravel()
        SF_I1 = np.tile(PN, [1, num_local_vsf]).ravel()

        # boundary treatment
        if bi is not None and bi.has_dirichlet:
            if self.dirichlet_clear_rows:
                # set row to 0 on dirichlet boundary nodes
                INTS = [np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, INTS[i]) for i in xrange(g.dim)]

        # build global system matrix
        B = [coo_matrix((INTS[i], (SF_I0, SF_I1)), shape=(num_global_vsf, num_global_psf)) for i in xrange(g.dim)]
        del SF_I0, SF_I1, INTS

        # convert to csc matrix
        B = vstack([csc_matrix(B[i].copy()) for i in xrange(g.dim)])

        return B


class AdvectionOperatorP2(NumpyMatrixBasedOperator):
    """An advection operator. P1 shape functions for pressure and P2 shape functions for velocity nodes.
    Computes the following integral::

    """

    def __init__(self, grid, boundary_info, advection_function=None, dirichlet_clear_rows=False, name=None):
        assert grid.reference_element is triangle
        assert advection_function is None or \
            (advection_function is not None and \
                advection_function.dim_domain == grid.dim_outer and \
                advection_function.shape_range == (grid.dim_outer,)*2)
        self.source = NumpyVectorSpace(grid.size(grid.dim))
        self.range = NumpyVectorSpace(grid.dim * (grid.size(grid.dim) + grid.size(grid.dim - 1)))
        self.grid = grid
        self.boundary_info = boundary_info
        self.advection_function = advection_function
        self.dirichlet_clear_rows = dirichlet_clear_rows
        self.name = name
        if advection_function is not None:
            self.build_parameter_type(inherits=(advection_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # quadrature rule on reference element
        q, w = g.reference_element.quadrature(order=2)

        if g.dim == 1:
            raise NotImplementedError
        elif g.dim == 2:
            # pressure shape functions
            num_local_psf = 3
            num_global_psf = g.size(g.dim)
            # pressure nodes
            PN = g.subentities(0, g.dim)

            # velocity shape functions
            num_local_vsf = 6
            num_global_vsf = g.size(g.dim)+ g.size(g.dim - 1)
            # velocity vertex nodes
            VVN = g.subentities(0, g.dim)
            # velocity edge nodes
            VEN = g.subentities(0, g.dim - 1) + g.size(g.dim)
            # velocity nodes
            VN = np.concatenate((VVN, VEN), axis=-1)
            del VVN, VEN
        else:
            raise NotImplementedError

        # evaluate pressure shape functions in all quadrature points
        PSF = P1ShapeFunctions(g.dim)(q)

        # evaluate gradients of velocity shape functions in all quadrature points
        VSF_GRAD = P2ShapeFunctionGradients(g.dim)(q)
        del q

        # calculate transformed gradients of velocity shape functions by reference map
        VSF_GRADS = np.einsum('eij,vjc->evic', g.jacobian_inverse_transposed(0), VSF_GRAD)
        del VSF_GRAD

        # calculate products between all pressure shape functions and gradients of velocity shape functions
        if self.advection_function is not None:
            A = self.advection_function(self.grid.centers(0), mu=mu)
            INT = np.einsum('pq,eviq,e,q,eji->evpj', PSF, VSF_GRADS, g.integration_elements(0), w, A)
        else:
            INT = np.einsum('pq,eviq,e,q->evpi', PSF, VSF_GRADS, g.intergration_elements(0), w)
        del PSF, VSF_GRADS, w

        INTS = [INT[..., i].ravel() for i in xrange(g.dim)]
        del INT

        # determine global dofs
        SF_I0 = np.repeat(VN, num_local_psf, axis=1).ravel()
        SF_I1 = np.tile(PN, [1, num_local_vsf]).ravel()
        del PN, VN

        # boundary treatment
        if bi is not None and bi.has_dirichlet:
            if self.dirichlet_clear_rows:
                # dirichlet vertex mask
                DVM = bi.dirichlet_mask(g.dim)
                # dirichlet edge mask
                DEM = bi.dirichlet_mask(g.dim - 1)
                # dirichlet mask
                DM = np.concatenate((DVM, DEM), axis=-1)
                del DVM, DEM
                # set row to 0 on dirichlet nodes
                INTS = [np.where(DM[SF_I0], 0, INTS[i]) for i in xrange(g.dim)]

        # build global system matrix
            B = [coo_matrix((INTS[i], (SF_I0, SF_I1)), shape=(num_global_vsf, num_global_psf)) for i in xrange(g.dim)]
            del SF_I0, SF_I1, INTS

            # convert to csc matrix
            B = vstack([csc_matrix(B[i].copy()) for i in xrange(g.dim)])

            return B


class RelaxationOperatorP1(NumpyMatrixBasedOperator):
    """An operator for stabilization of stokes problem. It has the form ::
        Σ h_K² ∫ ∇ p_h ∇ q_h.
        K      K
    """

    def  __init__(self, grid, relaxation_parameter=0.001, name=None):
        self.grid = grid
        self.relaxation_parameter = relaxation_parameter
        self.name = name

        self.source = NumpyVectorSpace(grid.size(grid.dim))
        self.range = NumpyVectorSpace(grid.size(grid.dim))
        self.sparse = True

    def _assemble(self, mu=None):
        g = self.grid

        # diameters
        # TODO: try pymors diameter
        t = g.centers(g.dim)[g.subentities(0, g.dim)]
        T = np.abs(t[:, [0, 0, 1], :] - t[:, [1, 2, 2], :])
        diameters = self.relaxation_parameter * np.max(np.linalg.norm(T, 2, 2)**2, axis=1)

        # quadrature rule on reference element
        q, w = g.reference_element.quadrature(order=2)

        # gradients of shape functions
        SF_GRAD = P1ShapeFunctionGradients(g.dim)

        # gradients of shape functions transformed by refeeerence map
        SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)

        # calculate scalar products between transformed gradients of shape functions
        SF_INTS = np.einsum('epi,eqi,e->epq', SF_GRADS, SF_GRADS, g.volumes(0), diameters).ravel()

        # determine global dofs
        SF_I0 = np.repeat(g.subentities(0, g.dim), g.dim + 1, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, g.dim + 1]).ravel()

        # assemble system matrix
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        del SF_INTS, SF_I0, SF_I1

        # convert to csc format
        A = csc_matrix(A).copy()

        return A


class L2VectorProductFunctionalP1(NumpyMatrixBasedOperator):
    """An operator to calculate integral f*v for vector valued functions f with linear shape functions. A transformation
    matrix can be applied.
    """

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, neumann_data=None, robin_data=None,
                 order=2, transformation_function=None, clear_dirichlet_dofs=False,
                 clear_non_dirichlet_dofs = False, name=None):
        assert grid.reference_element is triangle
        assert isinstance(function, FunctionInterface)
        assert transformation_function is None or \
               (transformation_function.dim_domain == grid.dim_outer and \
                   transformation_function.shape_range == (grid.dim_outer,)*2)
        self.source = NumpyVectorSpace(2*grid.size(grid.dim))
        self.range = NumpyVectorSpace(1)

        self.grid = grid
        self.function = function
        self.boundary_info = boundary_info
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.order = order
        self.transformation_function = transformation_function
        self.clear_dirichlet_dofs = clear_dirichlet_dofs
        self.clear_non_dirichlet_dofs = clear_non_dirichlet_dofs
        self.name = name

        if neumann_data is not None:
            raise NotImplementedError
        if robin_data is not None:
            raise NotImplementedError

        if transformation_function is not None:
            self.build_parameter_type(inherits=(transformation_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # quadrature rule on reference element
        q, w = g.reference_element.quadrature(order=2)

        # shape functions on reference element
        SF = P1ShapeFunctions(g.dim)(q)
        num_global_sf = g.size(g.dim)
        del q

        # evaluate function in all quadrature points on domain
        F_0 = self.function(g.quadrature_points(0, order=self.order), mu=mu)

        if self.transformation_function is not None:
            # apply piola transformation
            T = self.transformation_function(g.centers(g.dim), mu=mu)
            F_0 = np.einsum('ij,ecj->eci', T, F_0)

        # calculate integrals for f_i separately
        F = [F_0[..., i] for i in xrange(g.dim)]

        # calculate products
        SF_INT = [(np.einsum('ec,pc,e,c->ep', F[i], SF, g.integration_elements(0), w).ravel()) for i in xrange(g.dim)]
        del w

        # determine dofs
        # all nodes
        N = g.subentities(0, g.dim)

        SF_I0 = np.zeros_like(N)
        SF_I1 = N
        del N

        # build global vector
        I = [np.array(coo_matrix((SF_INT[i], (SF_I0, SF_I1)), shape=(1, num_global_sf)).todense()).ravel()
             for i in xrange(g.dim)]
        del SF_INT, SF_I0, SF_I1

        # boundary treatment
        # neumann
        if bi is not None and bi.has_neumann:
            raise NotImplementedError

        # robin
        if bi is not None and bi.has_robin:
            raise NotImplementedError

        # dirichlet
        if bi is not None and bi.has_dirichlet:
            # dirichlet nodes
            DN = bi.dirichlet_boundaries(g.dim)

            if self.dirichlet_data is not None:
                # points to evaluate dirichlet function in
                DC = g.centers(g.dim)[DN]

                # evaluate dirichlet function
                D = self.dirichlet_data(DC, mu=mu)

                if self.transformation_function is not None:
                    # apply piola transformation
                    T = self.transformation_function(DC, mu=mu)
                    D = np.einsum('eij,ej->ei', T, D)
                for i in xrange(g.dim):
                    I[i][DC] = D[..., i]
            else:
                for i in xrange(g.dim):
                    I[i][DN] = 0
            del DN

        I = np.hstack([I[i].reshape((1, -1))])

        return I


class L2VectorProductFunctionalP2(NumpyMatrixBasedOperator):
    """An operator to calculate integral f*v for vector valued functions f with linear shape functions. A transformation
    matrix can be applied.
    """

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, neumann_data=None, robin_data=None,
                 order=2, transformation_function=None, clear_dirichlet_dofs=False,
                 clear_non_dirichlet_dofs = False, name=None):
        assert grid.reference_element is triangle
        assert isinstance(function, FunctionInterface)
        assert transformation_function is None or \
               (transformation_function.dim_domain == grid.dim_outer and \
                   transformation_function.shape_range == (grid.dim_outer,)*2)
        self.source = NumpyVectorSpace(2*(grid.size(grid.dim) + grid.size(grid.dim)))
        self.range = NumpyVectorSpace(1)

        self.grid = grid
        self.function = function
        self.boundary_info = boundary_info
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.order = order
        self.transformation_function = transformation_function
        self.clear_dirichlet_dofs = clear_dirichlet_dofs
        self.clear_non_dirichlet_dofs = clear_non_dirichlet_dofs
        self.name = name

        if neumann_data is not None:
            raise NotImplementedError
        if robin_data is not None:
            raise NotImplementedError

        if transformation_function is not None:
            self.build_parameter_type(inherits=(transformation_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # quadrature rule on reference element
        q, w = g.reference_element.quadrature(order=2)

        # shape functions on reference element
        SF = P2ShapeFunctions(g.dim)(q)
        num_global_sf = g.size(g.dim) + g.size(g.dim - 1)
        del q

        # evaluate function in all quadrature points on domain
        F_0 = self.function(g.quadrature_points(0, order=self.order), mu=mu)

        if self.transformation_function is not None:
            # apply piola transformation
            T = self.transformation_function(g.centers(g.dim), mu=mu)
            F_0 = np.einsum('ij,ecj->eci', T, F_0)

        # calculate integrals for f_i separately
        F = [F_0[..., i] for i in xrange(g.dim)]

        # calculate products
        SF_INT = [(np.einsum('ec,pc,e,c->ep', F[i], SF, g.integration_elements(0), w).ravel()) for i in xrange(g.dim)]
        del w

        # determine dofs
        # vertex nodes
        VN = g.subentities(0, g.dim)
        # edge nodes
        EN = g.subentities(0, g.dim - 1) + g.size(g.dim)
        # all nodes
        N = np.concatenate((VN, EN), axis=-1).ravel()
        del VN, EN

        SF_I0 = np.zeros_like(N)
        SF_I1 = N
        del N

        # build global vector
        I = [np.array(coo_matrix((SF_INT[i], (SF_I0, SF_I1)), shape=(1, num_global_sf)).todense()).ravel()
             for i in xrange(g.dim)]
        del SF_INT, SF_I0, SF_I1

        # boundary treatment
        # neumann
        if bi is not None and bi.has_neumann:
            raise NotImplementedError

        # robin
        if bi is not None and bi.has_robin:
            raise NotImplementedError

        # dirichlet
        if bi is not None and bi.has_dirichlet:
            # vertex dirchlet nodes
            VDN = bi.dirichlet_boundaries(g.dim)
            # edge dirichlet nodes
            EDN = bi.dirichlet_boundaries(g.dim - 1) + g.size(g.dim)
            # dirichlet nodes
            DN = np.concatenate((VDN, EDN))
            del VDN, EDN

            if self.dirichlet_data is not None:
                # points to evaluate dirichlet function in
                DC = np.concatenate((g.centers(g.dim), g.centers(g.dim - 1)))[DN]

                # evaluate dirichlet function
                D = self.dirichlet_data(DC, mu=mu)

                if self.transformation_function is not None:
                    # apply piola transformation
                    T = self.transformation_function(DC, mu=mu)
                    D = np.einsum('eij,ej->ei', T, D)
                for i in xrange(g.dim):
                    I[i][DC] = D[..., i]
            else:
                for i in xrange(g.dim):
                    I[i][DN] = 0
            del DN

        I = np.hstack([I[i].reshape((1, -1))])

        return I


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
        assert isinstance(operator, NumpyMatrixBasedOperator)
        self.operator = operator
        self.source = operator.range
        self.range = operator.source
        self.name = '{0}_transposed'.format(operator.name)

    def _assemble(self, mu=None):
        return self.operator._assemble(mu).T