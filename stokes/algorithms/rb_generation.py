from __future__ import absolute_import, division, print_function

from pymor.discretizations.basic import StationaryDiscretization
from pymor.grids.tria import TriaGrid
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.operators.cg import L2ProductP1
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace

from stokes.grids.affine_transformed_tria import AffineTransformedTriaGrid
from stokes.operators.block import StokesLhsBlockOperator, StokesRhsBlockOperator
from stokes.recuctors.basic import GenericStokesRBReconstructor
from stokes.operators.cg import TransposedOperator, L2ProductP2, AdvectionOperatorP2, AdvectionOperatorP1


def slice_solution(solution, num_velocity_knots, num_pressure_knots):
    """
    Slices a stokes solution into velocity and pressure solution.
    Parameters
    ----------
    solution: The solution to be sliced.
    num_velocity_knots: The number of velocity knots (per dimension).
    num_pressure_knots: The number of pressure knots.

    Returns
    -------

    """
    assert isinstance(solution, VectorArrayInterface)
    assert isinstance(num_velocity_knots, int)
    assert isinstance(num_pressure_knots, int)
    assert num_velocity_knots > 0
    assert num_pressure_knots > 0


    array = solution._array[0]

    u = array[0:2*num_velocity_knots]
    p = array[2*num_velocity_knots:]

    U = NumpyVectorArray(u)
    P = NumpyVectorArray(p)

    return {'velocity': U, 'pressure': P}


def reduce_generic_rb_stokes(discretization, velocity_rb, pressure_rb, vector_product=None, disable_caching=True, extends=None):
    """
    Replaces each |Operator| of the given |Discretization| with the projection
    onto the span of the given reduced basis. Just for Stokes problems.
    Parameters
    ----------
    discretization
    velocity_rb: THe reduced basis for velocity.
    pressure_rb: The reduced basis for pressure.
    vector_product
    disable_caching
    extends

    Returns
    -------

    """
    assert extends is None or len(extends) == 3

    assert velocity_rb is not None
    assert pressure_rb is not None

    lhs_operator = discretization.operators['operator']
    rhs_functional = discretization.functionals['rhs']
    a = lhs_operator.blocks[0]
    b = lhs_operator.blocks[1]
    bt = lhs_operator.blocks[2]
    c = lhs_operator.blocks[3]
    f1 = rhs_functional.blocks[0]
    f2 = rhs_functional.blocks[1]

    # operators
    projected_a = a.projected(range_basis=velocity_rb, source_basis=velocity_rb, product=None)
    projected_b = b.projected(range_basis=velocity_rb, source_basis=pressure_rb, product=None)
    projected_bt = TransposedOperator(bt.operator.projected(range_basis=velocity_rb, source_basis=pressure_rb, product=None))
    projected_c = c.projected(range_basis=pressure_rb, source_basis=pressure_rb, product=None)

    projected_lhs = StokesLhsBlockOperator([projected_a, projected_b, projected_bt, projected_c])

    # functionals
    projected_f1 = f1.projected(range_basis=None, source_basis=velocity_rb, product=None)
    projected_f2 = f2.projected(range_basis=None, source_basis=pressure_rb, product=None)

    projected_rhs = StokesRhsBlockOperator([projected_f1, projected_f2])

    # products
    # TODO implement product projection

    cache_region = None if disable_caching else discretization.caching

    rd = discretization.with_(operator=projected_lhs,
                              rhs=projected_rhs,
                              vector_operators=None,
                              products=None,
                              visualizer=None,
                              estimator=None,
                              cache_region=cache_region,
                              name=discretization.name + '_reduced')

    rc = GenericStokesRBReconstructor(velocity_rb=velocity_rb,
                                      pressure_rb=pressure_rb)

    return rd, rc, {}


def reduce_naive(discretization, basis_size=None, training_set=None, add_supremizer=False, grid=None,
                 element_type=None):
    """
    Takes a discretization, a reductor and either a basis_size or a list of parameters and returns a reduced
    discretization and a reconstructor.
    Parameters
    ----------
    discretization: The discretization to be reduced.
    reductor: The reductor to be used.
    basis_size: The size of the training set. Must be None if training_set is given.
    training_set: The training set to be used. Must be None if basis_size is given.
    add_supremizer: Whether to add supremizer solutions to the reduced basis.
    grid: The grid, which is used to calculate snapshots. Must be provided to split the stokes solutions.
    element_type: Either 'P1P1' for P1P1 FEM or 'P2P1' for P2P1 FEM.

    Returns
    -------
    rd The reduced discretization.
    rc The reconstructor for the reduced discretization.

    """
    assert isinstance(discretization, StationaryDiscretization)
    assert isinstance(basis_size, int) or basis_size is None
    assert isinstance(training_set, list) or training_set is None
    assert (basis_size is not None and training_set is None) or (basis_size is None and training_set is not None)

    assert isinstance(grid, (TriaGrid, AffineTransformedTriaGrid))

    assert isinstance(element_type, str)
    assert element_type == 'P1P1' or element_type == 'P2P1'

    supremizer_bi = EmptyBoundaryInfo(grid)
    clear_rows = False
    mu_sup = None

    if element_type == 'P2P1':
        num_velocity_knots = grid.size(grid.dim) + grid.size(grid.dim - 1)
        num_pressure_knots = grid.size(grid.dim)
        supremizerOperator = L2ProductP2(grid=grid, boundary_info=supremizer_bi, dirichlet_clear_rows=clear_rows)
        # mu
        advectionOperator = AdvectionOperatorP2(grid=grid, boundary_info=supremizer_bi)
    elif element_type == 'P1P1':
        num_velocity_knots = grid.size(grid.dim)
        num_pressure_knots = grid.size(grid.dim)
        supremizerOperator = L2ProductP1(grid=grid, boundary_info=supremizer_bi, dirichlet_clear_rows=clear_rows)
        # mu
        advectionOperator = AdvectionOperatorP1(grid=grid, boundary_info=supremizer_bi)
    else:
        raise ValueError

    # training set
    if training_set is None:
        training_set = discretization.parameter_space.sample_randomly(basis_size)
    else:
        training_set = training_set

    velocity_snapshots = NumpyVectorSpace(2*num_velocity_knots).empty()
    pressure_snapshots = NumpyVectorSpace(num_pressure_knots).empty()

    # generate snapshots
    for i, mu in enumerate(training_set):
        print('Calculating snapshot {} of {} with mu={}'.format(i, len(training_set), str(mu)))
        sol = discretization.solve(mu)
        d = slice_solution(sol, num_velocity_knots, num_pressure_knots)
        velocity_snapshots.append(d['velocity'])
        if add_supremizer:
            # TODO implement calculation of supremizer
            #sup = supremizerOperator.apply_inverse(advectionOperator.apply(d['pressure'], mu=mu), mu=mu_sup)
            #velocity_snapshots.append(sup)
            # assemble?
            #X = supremizerOperator.assemble(mu).data
            raise NotImplementedError
        pressure_snapshots.append(d['pressure'])

    # reduced discretization and reconstructor
    rd, rc, _ = reduce_generic_rb_stokes(discretization=discretization,
                                         velocity_rb=velocity_snapshots,
                                         pressure_rb=pressure_snapshots)

    return rd, rc
