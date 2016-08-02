from __future__ import absolute_import, division, print_function

import time
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.vectorarrays.numpy import NumpyVectorArray

from stokes.algorithms.rb_generation import reduce_generic_rb_stokes
from stokes.analyticalproblems.bypass_a import BypassAProblem
from stokes.algorithms.algorithms import sample_training_set_randomly, sample_training_set_uniformly,\
    lift_pressure_solution, generate_snapshots, offline_supremizer

from stokes.analyticalproblems.affine_transformed_stokes import AffineTransformedStokes
from stokes.discretizations.reduced_supremizer_stokes import ReducedSupremizerStokesDiscretization
from stokes.discretizers.stationary_incompressible_stokes import discretize_stationary_incompressible_stokes
from stokes.domaindiscretizers.gmsh import discretize_gmsh
from stokes.functions.errors import absolute_error, relative_error
from stokes.functions.piecewise_affine_transformation import PiecewiseAffineTransformation
from pymor.parameters.base import Parameter

from stokes.gui.plotting import plot_transformed_pyplot, plot_multiple_transformed_pyplot

from matplotlib import pyplot as plt
import numpy as np


def setup_problem(viscosity=1):

    problem = BypassAProblem(viscosity)

    return problem


def setup_transformation(domain, parameter_mapping, ranges):
    parameter_type = {'tau': 0}
    transformation = PiecewiseAffineTransformation(domain, parameter_mapping, parameter_type, ranges)

    return transformation


def setup_transformed_problem(problem, transformation):
    transformed_problem = AffineTransformedStokes(problem, transformation)

    return transformed_problem


def build_parameter_from_parametrization(parameter):
    assert isinstance(parameter, Parameter)

    tau = parameter['tau']
    s = tau
    l = tau
    t = 3.0 - 2.0*tau

    a_1 = np.array([[1.0, 0.0], [0.0, t]])
    a_2 = np.array([[1.0, 0.0], [0.0, s]])
    a_3 = np.array([[1.0, 0.0], [0.0, t]])
    a_4 = np.array([[1.0, 0.0], [0.0, l]])

    b_1 = np.array([0.0, -1.0 + l])
    b_2 = np.array([-1.0, -1.0 + l + t])
    b_3 = np.array([-1.0, -1.0 + l])
    b_4 = np.array([-1.0, -1.0])

    p_1 = Parameter({'matrix': a_1, 'translation': b_1})
    p_2 = Parameter({'matrix': a_2, 'translation': b_2})
    p_3 = Parameter({'matrix': a_3, 'translation': b_3})
    p_4 = Parameter({'matrix': a_4, 'translation': b_4})

    p = [p_1, p_2, p_3, p_4]

    return p


def main():

    viscosity = 1
    fem_order = 1
    refinement_steps = 2
    basis_size = 100
    range_size = 200
    test_size = 20
    max_rb = 80
    pod_tol = 4e-10

    orthonormalize = True
    supremizer = True
    online_supremizer = False
    pod = True

    plot_ref_grid = False
    plot_examples = False
    plot_example_rb = False
    test_rb_size_1 = 2
    test_rb_size_2 = 5

    sample_strategy = 'uniformly'

    ranges = {'tau': (0.1, 1.45)}

    grid_mu = Parameter({'tau': 1.3})
    example_mus = [Parameter({'tau': 1.0}), Parameter({'tau': 0.1}), Parameter({'tau': 1.45})]
    test_parameter_rb = Parameter({'tau': 0.7})

    problem = setup_problem(viscosity)
    transformation = setup_transformation(problem.domain, build_parameter_from_parametrization, ranges)
    transformed_problem = setup_transformed_problem(problem, transformation)

    if fem_order == 1:
        element_type = 'P1P1'
    elif fem_order == 2:
        element_type = 'P2P1'
    else:
        raise ValueError

    g, bi = discretize_gmsh(transformed_problem.domain, refinement_steps=refinement_steps)

    if plot_ref_grid:
        plt.figure('Reference grid')
        plt.triplot(g.centers(2)[..., 0], g.centers(2)[..., 1], g.subentities(0, 2), color='blue')
        xy = transformation.evaluate(g.centers(2), mu=grid_mu)
        plt.figure('Grid for mu = {}'.format(grid_mu))
        plt.triplot(xy[..., 0], xy[..., 1], g.subentities(0, 2), color='blue')

    discretization, data = discretize_stationary_incompressible_stokes(analytical_problem=transformed_problem,
                                                                       diameter=None, domain_discretizer=None,
                                                                       grid=g, boundary_info=bi, fem_order=fem_order)

    grid = data['grid']

    example_sols = []

    if plot_examples:
        for i, mu in enumerate(example_mus):
            print('Solve reference solution for example parameter {} of {} with mu={}'.format(i+1, test_size, mu))
            sol = discretization.solve(mu)
            sol_lift = lift_pressure_solution(sol, grid, fem_order)
            example_sols.append(sol_lift)

        plot_multiple_transformed_pyplot(example_sols, grid, transformation, Parameter({'tau': 1.0}), fem_order,
                                         velocity='absolute', rescale_colorbars=False)

    test_parameters, sampling_time = sample_training_set_randomly(transformed_problem, test_size)

    products = {'h1_uv': discretization.products['h1_uv'],
                'l2_p': discretization.products['l2_p']}

    test_solutions = []
    for i, p in enumerate(test_parameters):
        print('Solve reference solution for test parameter {} of {} with mu={}'.format(i+1, test_size, p))
        sol = discretization.solve(p)
        sol_lift = lift_pressure_solution(sol, grid, fem_order)
        test_solutions.append(sol_lift)

    if sample_strategy == 'randomly':
        training_set, sampling_time = sample_training_set_randomly(transformed_problem, basis_size)
    elif sample_strategy == 'uniformly':
        training_set, sampling_time = sample_training_set_uniformly(transformed_problem, range_size)
    else:
        raise ValueError

    # build snapshots
    velocity_rb, pressure_rb = generate_snapshots(discretization, training_set, grid, element_type, False)

    if supremizer:
        # offline supremizer
        if not online_supremizer:
            supremizer_rb = offline_supremizer(discretization, velocity_rb, pressure_rb, training_set)

    # POD
    if pod:
        from pymor.algorithms.pod import pod
        print('Performing POD for velocity...')
        vel_rb, vel_svals = pod(velocity_rb, len(training_set), product=discretization.products['h1_uv_single'],
                                rtol=pod_tol)

        print('Performing POD for pressure...')
        pre_rb, pre_svals = pod(pressure_rb, len(training_set), product=discretization.products['l2_p_single'],
                                rtol=pod_tol)

        if supremizer:
            if not online_supremizer:
                print('Performing POD for offline supremizer...')
                sup_rb, sup_svals = pod(supremizer_rb, len(training_set),
                                        product=discretization.products['h1_uv_single'], rtol=pod_tol)
    else:
        if orthonormalize:
            vel_rb = gram_schmidt(velocity_rb, product=discretization.products['h1_uv_single'])
            pre_rb = gram_schmidt(pressure_rb, product=discretization.products['l2_p_single'])

    # maximal rb size
    if supremizer:
        if not online_supremizer:
            max_rb_size = min(len(vel_rb), len(pre_rb), len(sup_rb), max_rb)
        else:
            max_rb_size = min(len(vel_rb), len(pre_rb), max_rb)
    else:
        max_rb_size = min(len(vel_rb), len(pre_rb), max_rb)

    # dictionary for errors
    abs_errors = {}
    rel_errors = {}
    # array for times measurement
    times = np.zeros((max_rb_size, test_size))

    for k in products.keys():
        abs_errors[k] = np.zeros((max_rb_size, test_size))
        rel_errors[k] = np.zeros((max_rb_size, test_size))

    for i in xrange(max_rb_size):
        # slice rb; first i snapshots
        v_rb = NumpyVectorArray(vel_rb.data[0:i+1, :])
        p_rb = NumpyVectorArray(pre_rb.data[0:i+1, :])

        if supremizer:
            # offline supremizer
            if not online_supremizer:
                s_rb = NumpyVectorArray(sup_rb.data[0:i+1, :])
                v_rb.append(s_rb)

        # reduced discretization
        if supremizer:
            if online_supremizer:
                print('Generating reduced discretization with online supremizers ')
                reduced_discretization = ReducedSupremizerStokesDiscretization(discretization, v_rb, p_rb,
                                                                               orthonormalize=True)
            else:
                print('Generating reduced discretization with offline supremizers ')
                reduced_discretization, reconstructor, reduced_data = reduce_generic_rb_stokes(discretization,
                                                                                               v_rb, p_rb,
                                                                                               None, None, None)
        else:
            print('Generating reduced discretization without supremizers')
            reduced_discretization, reconstructor, reduced_data = reduce_generic_rb_stokes(discretization,
                                                                                           v_rb, p_rb,
                                                                                           None, None, None)

        for i_p, test_p in enumerate(test_parameters):
            print('Solving reduced discretization for test parameter {}/{} with rb size {}/{}'.format(i_p+1, test_size,
                                                                                                      len(p_rb),
                                                                                                      max_rb_size))
            tic = time.time()
            if supremizer:
                if online_supremizer:
                    #print('Using online supremizer')
                    reduced_solution, reconstructor = reduced_discretization.solve(test_p)
                else:
                    #print('Using offline supremizer')
                    reduced_solution = reduced_discretization.solve(test_p)
            else:
                #print('No supremizer')
                reduced_solution = reduced_discretization.solve(test_p)
            toc = time.time()

            times[i, i_p] = (toc-tic)
            print('Solving reduced discretization took {} s'.format(toc - tic))

            # calculating error
            for k, product in products.iteritems():
                rec_sol = reconstructor.reconstruct(reduced_solution)
                rec_sol_lift = lift_pressure_solution(rec_sol, grid, fem_order)
                ae = absolute_error(test_solutions[i_p], rec_sol_lift, product)[0]
                re = relative_error(test_solutions[i_p], rec_sol_lift, product)[0]
                abs_errors[k][i, i_p] = ae
                rel_errors[k][i, i_p] = re

    if plot_example_rb:
        if test_rb_size_1 > max_rb_size or test_rb_size_2 > max_rb_size:
            raise ValueError
        u_ref = discretization.solve(test_parameter_rb)
        u_ref = lift_pressure_solution(u_ref, grid, fem_order)

        # rb 1
        v_rb_1 = NumpyVectorArray(vel_rb.data[0:test_rb_size_1+1, :])
        p_rb_1 = NumpyVectorArray(pre_rb.data[0:test_rb_size_1+1, :])

        if supremizer:
            # offline supremizer
            if not online_supremizer:
                s_rb_1 = NumpyVectorArray(sup_rb.data[0:test_rb_size_1+1, :])
                v_rb_1.append(s_rb_1)

        # reduced discretization
        if supremizer:
            if online_supremizer:
                print('Generating reduced discretization with online supremizers ')
                reduced_discretization_1 = ReducedSupremizerStokesDiscretization(discretization, v_rb_1, p_rb_1,
                                                                                 orthonormalize=True)
            else:
                print('Generating reduced discretization with offline supremizers ')
                reduced_discretization_1, reconstructor_1, reduced_data_1 = reduce_generic_rb_stokes(discretization,
                                                                                                     v_rb_1, p_rb_1,
                                                                                                     None, None, None)
        else:
            print('Generating reduced discretization without supremizers')
            reduced_discretization_1, reconstructor_1, reduced_data_1 = reduce_generic_rb_stokes(discretization,
                                                                                                 v_rb_1, p_rb_1,
                                                                                                 None, None, None)
        u_rb_1 = reduced_discretization_1.solve(test_parameter_rb)
        u_rb_rec_1 = reconstructor_1.reconstruct(u_rb_1)
        u_rb_rec_lift_1 = lift_pressure_solution(u_rb_rec_1, grid, fem_order)
        e1 = u_ref - u_rb_rec_lift_1

        # rb 2
        v_rb_2 = NumpyVectorArray(vel_rb.data[0:test_rb_size_2+1, :])
        p_rb_2 = NumpyVectorArray(pre_rb.data[0:test_rb_size_2+1, :])

        if supremizer:
            # offline supremizer
            if not online_supremizer:
                s_rb_2 = NumpyVectorArray(sup_rb.data[0:test_rb_size_2+1, :])
                v_rb_2.append(s_rb_2)

        # reduced discretization
        if supremizer:
            if online_supremizer:
                print('Generating reduced discretization with online supremizers ')
                reduced_discretization_2 = ReducedSupremizerStokesDiscretization(discretization, v_rb_2, p_rb_2,
                                                                                 orthonormalize=True)
            else:
                print('Generating reduced discretization with offline supremizers ')
                reduced_discretization_2, reconstructor_2, reduced_data_2 = reduce_generic_rb_stokes(discretization,
                                                                                                     v_rb_2, p_rb_2,
                                                                                                     None, None, None)
        else:
            print('Generating reduced discretization without supremizers')
            reduced_discretization_2, reconstructor_2, reduced_data_2 = reduce_generic_rb_stokes(discretization,
                                                                                                 v_rb_2, p_rb_2,
                                                                                                 None, None, None)
        u_rb_2 = reduced_discretization_2.solve(test_parameter_rb)
        u_rb_rec_2 = reconstructor_2.reconstruct(u_rb_2)
        u_rb_rec_lift_2 = lift_pressure_solution(u_rb_rec_2, grid, fem_order)
        e2 = u_ref - u_rb_rec_lift_2

        rb_sols = [u_ref, u_rb_rec_lift_1, u_rb_rec_lift_2]
        es = [e1, e2]

        plot_multiple_transformed_pyplot(rb_sols, grid, transformation, Parameter({'tau': 1.0}), fem_order, 'absolute',
                                         True)
        plot_multiple_transformed_pyplot(es, grid, transformation, Parameter({'tau': 1.0}), fem_order, 'absolute',
                                         True)



    # u h1
    plt.figure('relative_error for u in h1')
    plt.semilogy(np.arange(1, max_rb_size+1), np.max(rel_errors['h1_uv'], axis=1), label='max')
    plt.semilogy(np.arange(1, max_rb_size+1), np.mean(rel_errors['h1_uv'], axis=1), label='mean')
    plt.semilogy(np.arange(1, max_rb_size+1), vel_svals[0:max_rb_size], label='$\sigma_i^u$')
    plt.xlabel('$N$')
    plt.ylabel('$||e_u||_{H^1}$')
    plt.legend()

    # p l2
    plt.figure('relative_error for p in l2')
    plt.semilogy(np.arange(1, max_rb_size+1), np.max(rel_errors['l2_p'], axis=1), label='max')
    plt.semilogy(np.arange(1, max_rb_size+1), np.mean(rel_errors['l2_p'], axis=1), label='mean')
    plt.semilogy(np.arange(1, max_rb_size+1), pre_svals[0:max_rb_size], label='$\sigma_i^p$')
    plt.xlabel('$N$')
    plt.ylabel('$||e_p||_{L^2}$')
    plt.legend()
    # END
    z = 0


if __name__ == '__main__':
    main()
