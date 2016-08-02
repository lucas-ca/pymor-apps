from __future__ import absolute_import, division, print_function

from pymor.grids.gmsh import GmshGrid
from pymor.grids.tria import TriaGrid
from pymor.parameters.base import Parameter
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace
from pymor.algorithms.gram_schmidt import gram_schmidt

from stokes.analyticalproblems.affine_transformed_stokes import AffineTransformedStokes

import time


def lift_pressure_solution(solution, grid, fem_order):
    """
    Lift a pressure FEM solution to zero at minimum.

    Parameters
    ----------
    solution
        The FEM solution to be lifted.
    grid
        The grid, which was used to compute the solution.
    fem_order
        The order of velocity FEM space. Either 1 or 2.

    Returns
    -------
    lifted_solution
        The solution with lifted pressure to zero minimum.

    """
    assert isinstance(solution, NumpyVectorArray)
    assert isinstance(grid, (TriaGrid, GmshGrid))
    assert isinstance(fem_order, int)

    if fem_order == 1:
        p = solution.data[0][2*grid.size(2):]
        assert p.shape == (grid.size(2), )
        solution.data[0][2*grid.size(2):] -= p.min()
    elif fem_order == 2:
        p = solution.data[0][2*(grid.size(2) + grid.size(1)):]
        assert p.shape == (grid.size(2), )
        solution.data[0][2*(grid.size(2) + grid.size(1)):] -= p.min()

    return solution


def sample_training_set_randomly(problem, basis_size):
    """
    Samples a problems parameter space randomly.

    Parameters
    ----------
    problem
        An AffineTransformedStokes problem for which to sample parameters.
    basis_size
        The number of randomly generated parameters.

    Returns
    -------
    training_set
        The sampled parameters.
    data
        Dictionary with additional data, like sampling time.
    """
    assert isinstance(problem, AffineTransformedStokes)
    assert isinstance(basis_size, int)

    print('Sampling parameter space randomly with basis size {}'.format(basis_size))
    tic = time.time()
    training_set = problem.parameter_space.sample_randomly(basis_size)
    toc = time.time()
    print('Sampling took {} s'.format(toc - tic))

    data = {'time': toc - tic}

    return training_set, data


def sample_training_set_uniformly(problem, basis_size):
    """
    Samples a problems parameter space uniformly.

    Parameters
    ----------
    problem
        An AffineTransformedStokes problem for which to sample parameters.
    basis_size
        Number of parameters per parameter space interval.

    Returns
    -------
    training_set
        The sampled parameters.
    data
        Dictionary with additional data, like sampling time.
    """
    assert isinstance(problem, AffineTransformedStokes)
    assert isinstance(basis_size, int)

    print('Sampling parameter space randomly with basis size {}'.format(basis_size))
    tic = time.time()
    training_set = problem.parameter_space.sample_uniformly(basis_size)
    toc = time.time()
    print('Sampling took {} s'.format(toc - tic))

    data = {'time': toc - tic}

    return training_set, data


def generate_snapshots(discretization, training_set, grid=None, element_type=None, orthonormalize=True,
                       products=None, lift_pressure=True):
    """
    Generate velocity and pressure reduced basis from given discretization and training set.

    Parameters
    ----------
    discretization
        The discretization for which to solve the snapshots.
    training_set
        The snapshot parameters.
    grid
        The grid, which was used to compute the solution.
    element_type
        The order of velocity FEM space. Either 1 or 2.
    orthonormalize
        If True, orthonormalize the reduced bases with gram schmidt.
    products
        The products to be used for orthonormalization. H1 for velocity and L2 for pressure are used automatically.
    lift_pressure
        Whether to lift the pressure solution to minimum zero.

    Returns
    -------
    velocity_snapshots
        The velocity snapshots.
    pressure_snapshots
        The pressure snapshots.

    """

    assert isinstance(grid, (TriaGrid, GmshGrid))

    assert isinstance(element_type, str)
    assert element_type == 'P1P1' or element_type == 'P2P1'

    assert products is None or isinstance(products, dict)

    if isinstance(products, dict):
        assert len(products) == 2
        assert 'velocity' in products.keys()
        assert 'pressure' in products.keys()

        velocity_product = products['velocity']
        pressure_product = products['pressure']
    elif products is None:
        velocity_product = discretization.products['h1_uv_single']
        pressure_product = discretization.products['l2_p_single']
    else:
        raise ValueError

    if element_type == 'P2P1':
        num_velocity_knots = grid.size(grid.dim) + grid.size(grid.dim - 1)
        num_pressure_knots = grid.size(grid.dim)
    elif element_type == 'P1P1':
        num_velocity_knots = grid.size(grid.dim)
        num_pressure_knots = grid.size(grid.dim)
    else:
        raise ValueError

    velocity_snapshots = NumpyVectorSpace(2*num_velocity_knots).empty()
    pressure_snapshots = NumpyVectorSpace(num_pressure_knots).empty()

    # generate snapshots
    for i, mu in enumerate(training_set):
        print('Calculating snapshot {} of {} with mu={}'.format(i+1, len(training_set), str(mu)))
        sol = discretization.solve(mu)
        d = slice_solution(sol, num_velocity_knots, num_pressure_knots)
        velocity_snapshots.append(d['velocity'])
        p = d['pressure']
        if lift_pressure:
            p.data[0] -= p.data[0].min()
        pressure_snapshots.append(p)

    if orthonormalize:
        velocity_snapshots = gram_schmidt(velocity_snapshots, product=velocity_product)
        pressure_snapshots = gram_schmidt(pressure_snapshots, product=pressure_product)

    return velocity_snapshots, pressure_snapshots


def slice_solution(solution, num_velocity_knots, num_pressure_knots):
    """
    Slices a stokes solution into velocity and pressure solution.

    Parameters
    ----------
    solution
        The solution to be sliced.
    num_velocity_knots
        The number of velocity knots (per dimension).
    num_pressure_knots
        The number of pressure knots.

    Returns
    -------
    sliced_solution
        Dictionary with keys 'velocity' and 'pressure' for the separated velocity and pressure solution.

    """
    assert isinstance(solution, NumpyVectorArray)
    assert isinstance(num_velocity_knots, int)
    assert isinstance(num_pressure_knots, int)
    assert num_velocity_knots > 0
    assert num_pressure_knots > 0

    array = solution.data[0]

    u = array[0:2*num_velocity_knots]
    p = array[2*num_velocity_knots:]

    U = NumpyVectorArray(u)
    P = NumpyVectorArray(p)

    sliced_solution = {'velocity': U, 'pressure': P}

    return sliced_solution


def offline_supremizer(discretization, velocity_rb, pressure_rb, test_parameters):
    """
    Calculate offline supremizer.

    Parameters
    ----------
    discretization
        The discretization which was used to calculate pressure snapshots.
    velocity_rb
        The velocity basis.
    pressure_rb
        The pressure basis.
    test_parameters
        The snapshot parameters which were used to compute the velocity and pressure bases.

    Returns
    -------
    supremizer_rb
        The basis of offline calculated supremizer solutions.

    """

    mu_id = Parameter({'scale_x': 1, 'scale_y': 1, 'shear': 0})

    mass_supremizer_operator = discretization.operators['supremizer_mass']
    advection_supremizer_operator = discretization.operators['supremizer_advection']

    supremizer_rb = NumpyVectorSpace(velocity_rb.dim).empty()

    for i, mu_ in enumerate(test_parameters):
        mu = mu_
        p = NumpyVectorArray(pressure_rb.data[i, :])
        supremizer = mass_supremizer_operator.apply_inverse(advection_supremizer_operator.apply(p, mu=mu), mu=mu_id)
        supremizer_rb.append(supremizer)

    return supremizer_rb
