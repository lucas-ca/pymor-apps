from __future__ import absolute_import, division, print_function

from pymor.discretizations.basic import StationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.grids.referenceelements import line, triangle, square
from pymor.operators.cg import DiffusionOperatorP1, L2ProductP1
from pymor.operators.constructions import LincombOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from stokes.analyticalproblems.stokes import StokesProblem
from stokes.operators.block import StokesLhsBlockOperator, StokesRhsBlockOperator, DiagonalBlockOperator
from stokes.operators.cg import DiffusionOperatorP2, AdvectionOperatorP1, AdvectionOperatorP2, RelaxationOperatorP1,\
    TransposedOperator, ZeroOperator, L2VectorProductFunctionalP1, L2VectorProductFunctionalP2, L2ProductP2, \
    RelaxationOperator2


def discretize_stationary_incompressible_stokes(analytical_problem, diameter=None, domain_discretizer=None,
                                                grid=None, boundary_info=None, fem_order=2):
    """Discretizes an Stokes problem using finite elements.

    Parameters
    ----------
    analytical_problem
        The Stokes problem to discretize.
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

    Returns
    -------
    discretization
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:
            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
    """

    # stokes problem
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
    else:
        grid = grid
        boundary_info = boundary_info

    empty_boundary_info = EmptyBoundaryInfo(grid)
    # triangle grid
    assert grid.reference_element in (triangle,)

    # operators and functionals
    if fem_order == 1:
        DiffusionOperator = DiffusionOperatorP1
        AdvectionOperator = AdvectionOperatorP1
        #RelaxationOperator = RelaxationOperatorP1
        Functional = L2VectorProductFunctionalP1
        MassOperator_velocity = L2ProductP1
        MassOperator_pressure = L2ProductP1
    elif fem_order == 2:
        DiffusionOperator = DiffusionOperatorP2
        AdvectionOperator = AdvectionOperatorP2
        #RelaxationOperator = ZeroOperator
        Functional = L2VectorProductFunctionalP2
        MassOperator_velocity = L2ProductP2
        MassOperator_pressure = L2ProductP1
    else:
        raise NotImplementedError

    p = analytical_problem

    if p.diffusion_functionals is not None:
        if p.advection_functionals is not None:
            if p.rhs_functionals is not None:
                if p.dirichlet_data_functionals is not None:
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
                    Bti = [AdvectionOperator(grid, empty_boundary_info, advection_function=af, dirichlet_clear_rows=False,
                                            name='advection_{0}'.format(i))
                          for i, af in enumerate(p.advection_functions)]
                    Bt = TransposedOperator(LincombOperator(operators=Bti, coefficients=list(p.advection_functionals),
                                                            name='advection'))

                    # relaxation part
                    if fem_order == 1:
                        # boundary part
                        # ---
                        # non boundary part
                        # TODO new relaxation operator
                        Ci = [RelaxationOperator2(grid=grid, boundary_info=empty_boundary_info, diffusion_function=df,
                                                  name='relaxation_{0}'.format(i))
                              for i, df in enumerate(p.diffusion_functions)]
                        C = LincombOperator(Ci, list(p.diffusion_functionals), name='relaxation')
                    elif fem_order == 2:
                        C = ZeroOperator(source=grid.size(grid.dim), range=grid.size(grid.dim), sparse=True,
                                         name='relaxation')
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
                                      transformation_function=None,
                                      dirichlet_transformation=tf,
                                      clear_dirichlet_dofs=False,
                                      clear_non_dirichlet_dofs=True,
                                      name='Function_boundary_part_{0}'.format(i))
                           for i, tf in enumerate(p.dirichlet_data_functions)]
                    # non boundary part
                    Fi = [Functional(grid=grid,
                                     function=p.rhs,
                                     boundary_info=boundary_info,
                                     dirichlet_data=p.dirichlet_data,
                                     neumann_data=p.neumann_data,
                                     robin_data=p.robin_data,
                                     transformation_function=tf,
                                     dirichlet_transformation=None,
                                     clear_dirichlet_dofs=True,
                                     clear_non_dirichlet_dofs=False,
                                     name='Function_{0}'.format(i))
                          for i, tf in enumerate(p.rhs_functions)]
                    F1 = LincombOperator(operators=Fi0 + Fi,
                                         coefficients=list(p.dirichlet_data_functionals) + list(p.rhs_functionals))

                    # supremizer operators
                    # supremizer operator mass
                    supremizer_operator_mass_1 = MassOperator_velocity(grid=grid, boundary_info=empty_boundary_info,
                                                              dirichlet_clear_rows=False, dirichlet_clear_columns=False,
                                                              dirichlet_clear_diag=False, coefficient_function=None,
                                                              solver_options=None, name='supremizer_mass')
                    supremizer_operator_mass = DiagonalBlockOperator([supremizer_operator_mass_1,
                                                                      supremizer_operator_mass_1])
                    # supremizer operator advection
                    # boundary part
                    # ---
                    # non boundary part
                    supremizer_operator_advection_i = [AdvectionOperator(grid=grid, boundary_info=empty_boundary_info,
                                                                         advection_function=af,
                                                                         dirichlet_clear_rows=False,
                                                                         name='supremizer_advection_{0}'.format(i))
                                                       for i, af in enumerate(p.advection_functions)]
                    supremizer_operator_advection = LincombOperator(operators=supremizer_operator_advection_i,
                                                                    coefficients=list(p.advection_functionals),
                                                                    name='supremizer_advection')

                    ### operators for calculation inf sup constant

                    # mass matrix of pressure
                    mass_pressure = MassOperator_pressure(grid=grid, boundary_info=empty_boundary_info,
                                                          dirichlet_clear_rows=False, dirichlet_clear_columns=False,
                                                          dirichlet_clear_diag=False, coefficient_function=None,
                                                          solver_options=None, name='pressure_mass_matrix')

                    stiffness_velocity = DiffusionOperator(grid=grid, boundary_info=empty_boundary_info,
                                                           diffusion_function=None, diffusion_constant=None,
                                                           dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                                                           name='velocity_stiffness_matrix')

                    velocity_divergence = AdvectionOperator(grid=grid, boundary_info=empty_boundary_info,
                                                            advection_function=None, dirichlet_clear_rows=False,
                                                            name='velocity_divergence_matrix')
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            raise ValueError
    else:
        assert len(p.diffusion_functions) == 1
        A = DiffusionOperator(grid=grid, boundary_info=boundary_info, diffusion_function=p.diffusion_functions[0],
                              diffusion_constant=p.viscosity, dirichlet_clear_diag=False, name='diffusion')
        B = LincombOperator(operators=[AdvectionOperator(grid=grid, boundary_info=boundary_info, advection_function=None,
                              dirichlet_clear_rows=True, name='advection')], coefficients=[-1.0], name='advection')
        Bt = TransposedOperator(AdvectionOperator(grid=grid, boundary_info=empty_boundary_info, advection_function=None,
                              dirichlet_clear_rows=False, name='advection'))

        if fem_order == 1:
            # TODO new relaxation operator
            #C = RelaxationOperatorP1(grid=grid,  name='relaxation')
            C = LincombOperator([RelaxationOperator2(grid=grid, boundary_info=empty_boundary_info, name='relaxation')],
                                [1.0])
        elif fem_order == 2:
            C = ZeroOperator(source=NumpyVectorSpace(grid.size(grid.dim)), range=NumpyVectorSpace(grid.size(grid.dim)),
                             sparse=True, name='relaxation')
        F1 = Functional(grid=grid, function=p.rhs, boundary_info=boundary_info, dirichlet_data=p.dirichlet_data,
                       neumann_data=p.neumann_data, robin_data=p.robin_data, transformation_function=None,
                       clear_dirichlet_dofs=False, clear_non_dirichlet_dofs=False)

        # supremizer operators
        # supremizer operator mass
        supremizer_operator_mass_1 = MassOperator_velocity(grid=grid, boundary_info=empty_boundary_info,
                                                  dirichlet_clear_rows=False, dirichlet_clear_columns=False,
                                                  dirichlet_clear_diag=False, coefficient_function=None,
                                                  solver_options=None, name='supremizer_mass')
        supremizer_operator_mass = DiagonalBlockOperator([supremizer_operator_mass_1, supremizer_operator_mass_1])
        # supremizer operator advection
        supremizer_operator_advection = AdvectionOperator(grid=grid, boundary_info=empty_boundary_info,
                                                          advection_function=None, dirichlet_clear_rows=False,
                                                          name='supremizer_advection')

        # operators for calculation inf sup constant
        mass_pressure = MassOperator_pressure(grid=grid, boundary_info=empty_boundary_info, dirichlet_clear_rows=False,
                                              dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                                              coefficient_function=None, solver_options=None,
                                              name='pressure_mass_matrix')
        stiffness_velocity = DiffusionOperator(grid=grid, boundary_info=empty_boundary_info, diffusion_function=None,
                                               diffusion_constant=None, dirichlet_clear_columns=False,
                                               dirichlet_clear_diag=False, name='velocity_stiffness_matrix')
        velocity_divergence = AdvectionOperator(grid=grid, boundary_info=empty_boundary_info, advection_function=None,
                                                dirichlet_clear_rows=False, name='velocity_divergence_matrix')

    # zero component on rhs
    Fz = ZeroOperator(source=NumpyVectorSpace(grid.size(grid.dim)), range=NumpyVectorSpace(1), sparse=False)

    # build complete stiffness matrix
    A2 = DiagonalBlockOperator([A, A])

    # operator
    L = StokesLhsBlockOperator([A2, B, Bt, C])

    # functional
    F = StokesRhsBlockOperator([F1, Fz])

    #if None:#isinstance(p, AffineTransformedStokes):
    #    plot_grid = AffineTransformedTriaGrid(grid, p.transformation)
    #else:
    #    plot_grid = grid

    visualizer = None

    # products
    if fem_order == 1:
        l2_prod_velocity = L2ProductP1
        h1_semi_prod_velocity = DiffusionOperatorP1
        zero_velocity = ZeroOperator(source=grid.size(grid.dim), range=grid.size(grid.dim), sparse=True)
        l2_prod_pressure = L2ProductP1
        h1_semi_prod_pressure = DiffusionOperatorP1
        zero_pressure = ZeroOperator(source=grid.size(grid.dim), range=grid.size(grid.dim), sparse=True)
        energy_c = RelaxationOperatorP1(grid=grid)
    elif fem_order == 2:
        l2_prod_velocity = L2ProductP2
        h1_semi_prod_velocity = DiffusionOperatorP2
        zero_velocity = ZeroOperator(source=grid.size(grid.dim) + grid.size(grid.dim - 1),
                                     range=grid.size(grid.dim) + grid.size(grid.dim - 1), sparse=True)
        l2_prod_pressure = L2ProductP1
        h1_semi_prod_pressure = DiffusionOperatorP1
        zero_pressure = ZeroOperator(source=grid.size(grid.dim), range=grid.size(grid.dim), sparse=True)
        energy_c = zero_pressure
    else:
        raise NotImplementedError

    # L2 products
    l2_product_vel = l2_prod_velocity(grid=grid, boundary_info=empty_boundary_info, dirichlet_clear_rows=False,
                                      name='l2_vel')
    l2_product_pre = l2_prod_pressure(grid=grid, boundary_info=empty_boundary_info, dirichlet_clear_rows=False,
                                      name='l2_pre')
    l2_product_u = DiagonalBlockOperator([l2_product_vel, zero_velocity, zero_pressure])
    l2_product_v = DiagonalBlockOperator([zero_velocity, l2_product_vel, zero_pressure])
    l2_product_p = DiagonalBlockOperator([zero_velocity, zero_velocity, l2_product_pre])
    l2_product_uv = DiagonalBlockOperator([l2_product_vel, l2_product_vel, zero_pressure])
    l2_product_uvp = DiagonalBlockOperator([l2_product_vel, l2_product_vel, l2_product_pre])
    # for gram schmidt
    l2_product_uv_single = DiagonalBlockOperator([l2_product_vel, l2_product_vel])

    # H1 semi products
    h1_semi_product_vel = h1_semi_prod_velocity(grid=grid, boundary_info=empty_boundary_info, name='h1_semi_vel')
    h1_semi_product_pre = h1_semi_prod_pressure(grid=grid, boundary_info=empty_boundary_info, name='h1_semi_pre')
    h1_semi_product_u = DiagonalBlockOperator([h1_semi_product_vel, zero_velocity, zero_pressure])
    h1_semi_product_v = DiagonalBlockOperator([zero_velocity, h1_semi_product_vel, zero_pressure])
    h1_semi_product_p = DiagonalBlockOperator([zero_velocity, zero_velocity, h1_semi_product_pre])
    h1_semi_product_uv = DiagonalBlockOperator([h1_semi_product_vel, h1_semi_product_vel, zero_pressure])
    h1_semi_product_uvp = DiagonalBlockOperator([h1_semi_product_vel, h1_semi_product_vel, h1_semi_product_pre])
    # for gram schmidt
    h1_semi_product_uv_single = DiagonalBlockOperator([h1_semi_product_vel, h1_semi_product_vel])

    # energy norm
    energy_a = DiagonalBlockOperator([h1_semi_product_vel, h1_semi_product_vel])
    energy_b1 = LincombOperator(operators=[AdvectionOperator(grid=grid, boundary_info=empty_boundary_info,
                                                             advection_function=None, dirichlet_clear_rows=False,
                                                             name='energy_b1')], coefficients=[-1.0], name='energy_b1')
    energy_b2 = TransposedOperator(AdvectionOperator(grid=grid, boundary_info=empty_boundary_info,
                                                     advection_function=None, dirichlet_clear_rows=False,
                                                     name='energy_b1'))
    energy = StokesLhsBlockOperator([energy_a, energy_b1, energy_b2, energy_c])

    products = {'h1_u': h1_semi_product_u + l2_product_u,
                'h1_v': h1_semi_product_v + l2_product_v,
                'h1_p': h1_semi_product_p + l2_product_p,
                'h1_uv': h1_semi_product_uv + l2_product_uv,
                'h1_uvp': h1_semi_product_uvp + l2_product_uvp,
                'h1_uv_single': h1_semi_product_uv_single + l2_product_uv_single,
                'h1_semi_u': h1_semi_product_u,
                'h1_semi_v': h1_semi_product_v,
                'h1_semi_p': h1_semi_product_p,
                'h1_semi_uv': h1_semi_product_uv,
                'h1_semi_uvp': h1_semi_product_uvp,
                'l2_u': l2_product_u,
                'l2_v': l2_product_v,
                'l2_p': l2_product_p,
                'l2_uv': l2_product_uv,
                'l2_uvp': l2_product_uvp,
                'l2_p_single': l2_product_pre,
                'energy': energy}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization1 = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                               parameter_space=parameter_space, name='{}_CG'.format(p.name))
    ops = {'operator': L,
           'supremizer_mass': supremizer_operator_mass,
           'supremizer_advection': supremizer_operator_advection}
    discretization2 = StationaryDiscretization(rhs=F, products=products, operators=ops, visualizer=visualizer,
                                               parameter_space=parameter_space, name='{}_CG'.format(p.name))

    return discretization2, {'grid': grid, 'boundary_info': boundary_info}
