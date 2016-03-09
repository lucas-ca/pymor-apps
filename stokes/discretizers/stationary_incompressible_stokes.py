# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from stokes.analyticalproblems.stokes import StokesProblem
from pymor.discretizations.basic import StationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.grids.referenceelements import line, triangle, square
from pymor.gui.qt import PatchVisualizer, Matplotlib1DVisualizer
from pymor.operators import cg
from pymor.operators.constructions import LincombOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

# operators
from pymor.operators.cg import DiffusionOperatorP1
from stokes.operators.cg import DiffusionOperatorP2, AdvectionOperatorP1, AdvectionOperatorP2, RelaxationOperatorP1,\
    TransposedOperator, ZeroOperator, L2VectorProductFunctionalP1, L2VectorProductFunctionalP2
from stokes.operators.block import StokesLhsBlockOperator, StokesRhsBlockOperator

# visualizer
from stokes.gui.stokes_visualizer import StokesVisualizer

def discretize_stationary_incompressible_stokes(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None, fem_order=2, plot_type=None, resolution=None, mu=None):
    """Discretizes an |EllipticProblem| using finite elements.
    Parameters
    ----------
    analytical_problem
        The |EllipticProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If further arguments should be passed to the discretizer, use
        :func:`functools.partial`. If `None`, |discretize_domain_default| is used.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.
    fem_order
        The order of finite element method to be used.
    plot_type
        Type of velocity plots:
            0: separate u_1 and u_2 plots
            1: quiver plot
            2: streamline plot
    resolution
        Number of vectors in quiver plot.
    Returns
    -------
    discretization
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:
            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
    """

    assert isinstance(analytical_problem, StokesProblem)

    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    # triangle grid
    assert grid.reference_element in (triangle,)

    # operators and functionals
    if fem_order == 1:
        DiffusionOperator = DiffusionOperatorP1
        AdvectionOperator = AdvectionOperatorP1
        #RelaxationOperator = RelaxationOperatorP1
        Functional = L2VectorProductFunctionalP1
    elif fem_order == 2:
        DiffusionOperator = DiffusionOperatorP2
        AdvectionOperator = AdvectionOperatorP2
        #RelaxationOperator = ZeroOperator
        Functional = L2VectorProductFunctionalP2
    else:
        raise NotImplementedError

    p = analytical_problem

    if p.diffusion_functionals is not None:
        if p.advection_functionals is not None:
            if p.rhs_transformation_functionals is not None:
                if p.dirichlet_data_transformation_functionals is not None:
                    # diffusion operator
                    # boundary part
                    A0 = DiffusionOperator(grid, boundary_info, diffusion_constant=0., name='diffusion_boundary_part')
                    # non boundary part
                    Ai = [DiffusionOperator(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                                            name='diffusion_{0}'.format(i))
                          for i, df in enumerate(p.diffusion_functions)]
                    A = LincombOperator(operators=[A0] + Ai, coefficients=[1.] + list(p.diffusion_functionals),
                                        name='diffusion')

                    # advection operator
                    # boundary part
                    # ---
                    # non boundary part
                    Bi = [AdvectionOperator(grid, boundary_info, advection_function=af, dirichlet_clear_rows=True,
                                            name='advection_{0}'.format(i))
                          for i, af in enumerate(p.advection_functions)]
                    B = LincombOperator(
                        operators=[LincombOperator(
                            operators=Bi, coefficients=list(p.advection_functionals))],
                        coefficients=[-1.], name='advection')

                    # transposed advection operator
                    # boundary part
                    # ---
                    # non boundary part
                    Bti = [AdvectionOperator(grid, boundary_info, advection_function=af, dirichlet_clear_rows=False,
                                            name='advection_{0}'.format(i))
                          for i, af in enumerate(p.advection_functions)]
                    Bt = TransposedOperator(LincombOperator(operators=Bti, coefficients=list(p.advection_functionals)),
                                            name='advection')

                    # relaxation part
                    if fem_order == 1:
                        # boundary part
                        # ---
                        # non boundary part
                        Ci = [RelaxationOperatorP1(grid, name='diffusion_{0}'.format(i))
                              for i, df in enumerate(p.diffusion_functions)]
                        C = LincombOperator(Ci, list(p.diffusion_functionals), name='relaxation')
                    elif fem_order == 2:
                        C = ZeroOperator(source=grid.size(grid.dim), range=grid.size(grid.dim), name='relaxation')
                    else:
                        raise NotImplementedError

                    # functional
                    # boundary part
                    Fi0 = [Functional(grid=grid,
                                      function=p.rhs,
                                      boundary_info=boundary_info,
                                      dirichlet_data=p.dirichlet_data,
                                      neumann_data=p.neumann_data,
                                      robin_data=p.robin_data,
                                      transformation_function=tf,
                                      clear_dirichlet_dofs=False,
                                      clear_non_dirichlet_dofs=True,
                                      name='Function_boundary_part_{0}'.format(i))
                           for i, tf in enumerate(p.dirichlet_data_transformation_functions)]
                    # non boundary part
                    Fi = [Functional(grid=grid,
                                     function=p.rhs,
                                     boundary_info=boundary_info,
                                     dirichlet_data=p.dirichlet_data,
                                     neumann_data=p.neumann_data,
                                     robin_data=p.robin_data,
                                     transformation_function=tf,
                                     clear_dirichlet_dofs=True,
                                     clear_non_dirichlet_dofs=False,
                                     name='Function_{0}'.format(i))
                          for i, tf in enumerate(p.rhs_transformation_functions)]
                    F1 = LincombOperator(operators=Fi0 + Fi,
                                        coefficients=list(p.dirichlet_data_transformation_functionals) +
                                                     list(p.rhs_transformation_functionals))
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            raise ValueError
    else:
        assert len(p.diffusion_functions == 1)
        A = DiffusionOperator(grid=grid, boundary_info=boundary_info, diffusion_function=p.diffusion_functions[0],
                              diffusion_constant=p.viscosity, dirichlet_clear_diag=False, name='diffusion')
        B = LincombOperator(operators=[AdvectionOperator(grid=grid, boundary_info=boundary_info, advection_function=None,
                              dirichlet_clear_rows=True, name='advection')], coefficients=[-1.0], name='advection')
        Bt = TransposedOperator(AdvectionOperator(grid=grid, boundary_info=boundary_info, advection_function=None,
                              dirichlet_clear_rows=False, name='advection'))
        if fem_order == 1:
            C = RelaxationOperatorP1(grid=grid, name='relaxation')
        elif fem_order == 2:
            C = ZeroOperator(source=NumpyVectorSpace(grid.size(grid.dim)), range=NumpyVectorSpace(grid.size(grid.dim)),
                             sparse=False, name='relaxation')
        F1 = Functional(grid=grid, rhs=p.rhs, boundary_info=boundary_info, dirchlet_data=p.dirichlet_data,
                       neumann_data=p.neumann_data, robin_data=p.robin_data, transformation_function=None,
                       clear_dirichlet_dofs=False, clear_non_dirichlet_dofs=False)

    # zero component on rhs
    Fz = ZeroOperator(source=NumpyVectorSpace(grid.size(grid.dim)), range=NumpyVectorSpace(1), sparse=False)

    # build complete stokes operator
    L = StokesLhsBlockOperator([A, B, Bt, C])
    F = StokesRhsBlockOperator([F1, Fz])

    visualizer = StokesVisualizer(grid=grid, bounding_box=grid.bounding_box(), codim=2, plot_type=plot_type,
                                  resolution=resolution, mu=mu)

    products = None

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                              parameter_space=parameter_space, name='{}_CG'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}