from __future__ import absolute_import, division, print_function

from stokes.analyticalproblems.stokes import StokesProblem
from stokes.discretizers.stationary_incompressible_stokes import discretize_stationary_incompressible_stokes

from pymor.grids.tria import TriaGrid
from pymor.vectorarrays.numpy import NumpyVectorArray

import numpy as np
import time


def setup_discretization(problem, diameter, fem_order):
    discretization, data = discretize_stationary_incompressible_stokes(problem, diameter=diameter, fem_order=fem_order)

    return discretization, data


def get_evaluation_points(grid, fem_order):
    assert isinstance(grid, TriaGrid)

    assert isinstance(fem_order, int)
    assert 0 < fem_order
    assert fem_order <= 2

    if fem_order == 1:
        c = grid.centers(2)
        evaluation_points = {'u': c, 'v': c, 'p': c}
    elif fem_order == 2:
        c1 = grid.centers(2)
        c2 = np.concatenate((grid.centers(2), grid.centers(1)))
        evaluation_points = {'u': c2, 'v': c2, 'p': c1}
    else:
        raise ValueError

    return evaluation_points


def evaluate_analytical_solution(analytical_solution, grid, fem_order):
    assert isinstance(analytical_solution, dict)
    assert 'u' in analytical_solution.keys()
    assert 'v' in analytical_solution.keys()
    assert 'p' in analytical_solution.keys()

    assert isinstance(grid, TriaGrid)

    assert isinstance(fem_order, int)
    assert 0 < fem_order
    assert fem_order <= 2

    if fem_order == 1:
        c = grid.centers(2)
        evaluation_points = {'u': c, 'v': c, 'p': c}
    elif fem_order == 2:
        c1 = grid.centers(2)
        c2 = np.concatenate((grid.centers(2), grid.centers(1)))
        evaluation_points = {'u': c2, 'v': c2, 'p': c1}
    else:
        raise ValueError

    analytical_u = analytical_solution['u'](evaluation_points['u'])
    analytical_v = analytical_solution['v'](evaluation_points['v'])
    analytical_p = analytical_solution['p'](evaluation_points['p'])

    analytical_evaluation = np.concatenate((analytical_u, analytical_v, analytical_p))

    return NumpyVectorArray(analytical_evaluation)


def error_for_diameter(problem, analytical_solution, diameter, fem_order, products, lift_p_dof=True):
    assert isinstance(problem, StokesProblem)

    assert isinstance(analytical_solution, dict)
    assert 'u' in analytical_solution.keys()
    assert 'v' in analytical_solution.keys()
    assert 'p' in analytical_solution.keys()

    assert isinstance(diameter, (int, float))

    assert isinstance(fem_order, int)
    assert 0 < fem_order
    assert fem_order <= 2

    assert isinstance(products, list)

    for prod in products:
        assert prod in ('h1_uv', 'l2_uv', 'h1_p', 'l2_p')

    assert isinstance(lift_p_dof, bool)

    data = {}

    tic = time.time()

    discretization_fem, data_fem = setup_discretization(problem, diameter, fem_order)

    # solve fem
    solution_fem = discretization_fem.solve()

    toc = time.time()

    data['time'] = toc-tic

    # grid
    grid_fem = data_fem['grid']

    # num_dofs
    if fem_order == 1:
        num_dofs = 3*grid_fem.size(2)
    elif fem_order == 2:
        num_dofs = 3*grid_fem.size(2) + 2*grid_fem.size(1)
    else:
        raise ValueError

    data['num_dofs'] = num_dofs

    # evaluate analytical solution in fem dofs
    analytical_evaluation = evaluate_analytical_solution(analytical_solution, grid_fem, fem_order)

    if lift_p_dof:
        if fem_order == 1:
            p = solution_fem.data[0][2*grid_fem.size(2):]
            solution_fem.data[0][2*grid_fem.size(2):] -= p.min()
        elif fem_order == 2:
            p = solution_fem.data[0][2*(grid_fem.size(2) + grid_fem.size(1)):]
            solution_fem.data[0][2*(grid_fem.size(2) + grid_fem.size(1)):] -= p.min()

    # error
    assert analytical_evaluation.dim == solution_fem.dim
    e = analytical_evaluation - solution_fem

    # products
    errors = {}
    for k in products:
        product = discretization_fem.products[k]

        ae = np.sqrt(product.apply2(e, e)[0, 0])
        re = ae / np.sqrt(product.apply2(analytical_evaluation, analytical_evaluation)[0, 0])
        errors[k] = {'abs': ae, 'rel': re}

    return errors, data
