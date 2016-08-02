from __future__ import absolute_import, division, print_function

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.parameters.base import Parameter
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace

from stokes.algorithms.algorithms import generate_snapshots, lift_pressure_solution, sample_training_set_randomly, \
    sample_training_set_uniformly
from stokes.algorithms.rb_generation import reduce_generic_rb_stokes
from stokes.analyticalproblems.affine_transformed_stokes import AffineTransformedStokes
from stokes.analyticalproblems.poiseuille import PoiseuilleProblem
from stokes.discretizations.reduced_supremizer_stokes import ReducedSupremizerStokesDiscretization
from stokes.discretizers.stationary_incompressible_stokes import discretize_stationary_incompressible_stokes
from stokes.functions.affine_transformation import AffineTransformation

import numpy as np
from matplotlib import pyplot as plt

import time


def setup_problem(width, height, viscosity=1):
    assert isinstance(width, (int, float))
    assert isinstance(height, (int, float))

    problem = PoiseuilleProblem(width, height, viscosity)

    return problem


def setup_transformation(ranges, parameter_mapping):
    parameter_type = {'scale_x': 0, 'scale_y': 0, 'shear': 0}
    transformation = AffineTransformation(parameter_type=parameter_type, ranges=ranges,
                                          parameter_mapping=parameter_mapping,
                                          name='AffineTransformation')

    return transformation


def setup_transformed_problem(problem, transformation):
    transformed_problem = AffineTransformedStokes(problem, transformation)

    return transformed_problem


def setup_discretization(problem, diameter, fem_order):
    discretization, data = discretize_stationary_incompressible_stokes(problem, diameter=diameter, fem_order=fem_order)

    return discretization, data


def build_parameter_from_parametrization(parameter):
    assert isinstance(parameter, Parameter)

    scale_x = parameter['scale_x']
    scale_y = parameter['scale_y']
    shear = parameter['shear']

    a = np.array([[scale_x, 0.0], [shear, scale_y]])
    b = np.array([0.0, 0.0])

    p = Parameter({'matrix': a, 'translation': b})

    return p


def offline_supremizer(discretization, velocity_rb, pressure_rb, test_parameters):

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


def main():
    # PARAMETERS
    width = 1
    height = 1
    viscosity = 1
    fem_order = 1
    diameter = 1./50.
    basis_size = 100
    range_size = 8
    test_size = 100
    max_rb = 60

    orthonormalize = True
    supremizer = True
    online_supremizer = True
    pod = True

    sample_strategy = 'uniformly'

    ranges = {'scale_x': (4.0, 8.0), 'scale_y': (0.5, 2.0), 'shear': (0.0, 1.0)}
    test_parameter = Parameter({'scale_x': 4.0, 'scale_y': 1.0, 'shear': 1.0})

    if fem_order == 1:
        element_type = 'P1P1'
    elif fem_order == 2:
        element_type = 'P2P1'
    else:
        raise ValueError

    problem = setup_problem(width, height, viscosity)
    transformation = setup_transformation(ranges, build_parameter_from_parametrization)
    transformed_problem = setup_transformed_problem(problem, transformation)

    test_parameters, sampling_time = sample_training_set_randomly(transformed_problem, test_size)

    discretization, data = setup_discretization(transformed_problem, diameter, fem_order)
    grid = data['grid']

    products = {'h1_uv': discretization.products['h1_uv'],
                'l2_p': discretization.products['l2_p']}

    # calculate test solutions
    test_solutions = []
    for i, p in enumerate(test_parameters):
        print('Solve reference solution for test parameter {} of {} with mu={}'.format(i+1, test_size, p))
        sol = discretization.solve(p)
        sol_lift = lift_pressure_solution(sol, grid, fem_order)
        test_solutions.append(sol_lift)

    # sample training set
    if sample_strategy == 'randomly':
        training_set, sampling_time = sample_training_set_randomly(transformed_problem, basis_size)
    elif sample_strategy == 'uniformly':
        training_set, sampling_time = sample_training_set_uniformly(transformed_problem, range_size)
    else:
        raise ValueError

    # build snapshots
    velocity_rb, pressure_rb = generate_snapshots(discretization, training_set, grid, element_type, False)

    # offline supremizer
    if supremizer:
        if not online_supremizer:
            supremizer_rb = offline_supremizer(discretization, velocity_rb, pressure_rb, training_set)

    # pod
    if pod:
        from pymor.algorithms.pod import pod
        # velocity
        print('Performing POD for velocity...')
        vel_rb, vel_svals = pod(velocity_rb, len(training_set), product=discretization.products['h1_uv_single'])

        # pressure
        print('Performing POD for pressure...')
        pre_rb, pre_svals = pod(pressure_rb, len(training_set), product=discretization.products['l2_p_single'])

        # offline supremizer
        if supremizer:
            if not online_supremizer:
                print('Performing POD for offline supremizer...')
                sup_rb, sup_svals = pod(supremizer_rb, len(training_set), product=discretization.products['h1_uv_single'])
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
    times = np.zeros((max_rb_size, test_size))

    for k in products.keys():
        abs_errors[k] = np.zeros((max_rb_size, test_size))
        rel_errors[k] = np.zeros((max_rb_size, test_size))

    for i in xrange(max_rb_size):
        # slice rb; first i snapshots
        v_rb = NumpyVectorArray(vel_rb.data[0:i+1, :])
        p_rb = NumpyVectorArray(pre_rb.data[0:i+1, :])

        # append offline supremizer to velocity rb
        if supremizer:
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
            print('Generating reduced discretization without online supremizers')
            reduced_discretization, reconstructor, reduced_data = reduce_generic_rb_stokes(discretization,
                                                                                           v_rb, p_rb,
                                                                                           None, None, None)
        for i_p, test_p in enumerate(test_parameters):
            print('Solving reduced discretization for test parameter {} of {} with rb size {}'.format(i_p+1, test_size,
                                                                                                      len(p_rb)))
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
                #if k == "h1_uv":
                #    norm = discretization.h1_uv_norm
                #elif k == "l2_p":
                #    norm = discretization.l2_p_norm
                #ae_ = absolute_error_disc(test_solutions[i_p], rec_sol_lift, norm)
                #re_ = relative_error_disc(test_solutions[i_p], rec_sol_lift, norm)
                abs_errors[k][i, i_p] = ae
                rel_errors[k][i, i_p] = re
                #if not ae == ae_ or not re == re_:
                #    z = 0
                #if k in absolute_errors.keys():
                #    absolute_errors[k].append(ae)
                #else:
                #    absolute_errors[k] = [ae]
                #if k in relative_errors.keys():
                #    relative_errors[k].append(re)
                #else:
                #    relative_errors[k] = [re]

    # u h1
    plt.figure('relative_error for u in h1')
    plt.semilogy(np.arange(1, max_rb_size+1), np.max(rel_errors['h1_uv'], axis=1), label='max')
    plt.semilogy(np.arange(1, max_rb_size+1), np.mean(rel_errors['h1_uv'], axis=1), label='mean')
    plt.xlabel('$N$')
    plt.ylabel('$||e_u||_{H^1}$')
    plt.legend()

    # p l2
    plt.figure('relative_error for p in l2')
    plt.semilogy(np.arange(1, max_rb_size+1), np.max(rel_errors['l2_p'], axis=1), label='max')
    plt.semilogy(np.arange(1, max_rb_size+1), np.mean(rel_errors['l2_p'], axis=1), label='mean')
    plt.xlabel('$N$')
    plt.ylabel('$||e_p||_{L^2}$')
    plt.legend()

    #rb_vel = NumpyVectorArray(vel_rb.data[0:24, :])
    #rb_pre = NumpyVectorArray(pre_rb.data[0:24, :])
    #len(rb_vel)
    #len(rb_pre)
    #test_mu = test_parameters[24]
    #reduced_discretization = ReducedSupremizerStokesDiscretization(discretization, rb_vel, rb_pre)
    #u = discretization.solve(test_mu)
    #u_rb, rec = reduced_discretization.solve(test_mu)
    #u_rb_rec = rec.reconstruct(u_rb)
    #e = u-u_rb_rec
    #rel_e = discretization.h1_uv_norm(e)/discretization.h1_uv_norm(u)

    # END
    z = 0


if __name__ == '__main__':
    main()
