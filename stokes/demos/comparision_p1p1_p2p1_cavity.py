from stokes.analyticalproblems.cavity import CavityProblem
from stokes.demos.error_calculation import error_for_diameter

import numpy as np
from matplotlib import pyplot as plt


def setup_problem(viscosity):
    assert isinstance(viscosity, (int, float))

    problem = CavityProblem(viscosity)

    return problem


def setup_analytical_solution():
    u = lambda x: x[..., 0]**2 *\
                  (np.ones_like(x[..., 0]) - x[..., 0])**2 *\
                  2*x[..., 1] *\
                  (np.ones_like(x[..., 0]) - x[..., 1]) *\
                  (2*x[..., 1] - np.ones_like(x[..., 0]))

    v = lambda x: x[..., 1]**2 *\
                  (np.ones_like(x[..., 0]) - x[..., 1])**2 *\
                  2*x[..., 0] *\
                  (np.ones_like(x[..., 0]) - x[..., 0]) *\
                  (np.ones_like(x[..., 0]) - 2*x[..., 0])
    p = lambda x: x[..., 0] *\
                  x[..., 1] *\
                  (np.ones_like(x[..., 0]) - x[..., 0]) *\
                  (np.ones_like(x[..., 0]) - x[..., 1])

    return {'u': u, 'v': v, 'p': p}


def main():
    print("Calculating errors for FEM solution of Cavity problem...")

    viscosity = 1.0
    fem_order = 2
    lift_pressure = True

    runs = 5

    diameters = [1./20., 1./40., 1.0/60.0, 1.0/80.0]

    product = 'h1_uv'
    products = [product]

    problem = setup_problem(viscosity)
    analytical_solution = setup_analytical_solution()

    absolute_errors = {}
    relative_errors = {}

    p1p1 = {}
    p2p1 = {}

    for i, diameter in enumerate(diameters):
        print('Calculating error {} of {} for diameter {} with P1P1 FEM'.format(i+1, len(diameters), diameter))

        p1p1[i] = {}
        for j in xrange(runs):
            print('Run {} of {}'.format(j+1, runs))
            errors, data = error_for_diameter(problem, analytical_solution, diameter, 1, products, lift_pressure)
            if 'time' in p1p1[i].keys():
                p1p1[i]['time'].append(data['time'])
            else:
                p1p1[i]['time'] = [data['time']]
            p1p1[i]['num_dofs'] = data['num_dofs']
            p1p1[i]['error'] = errors[product]['rel']

        print('Calculating error {} of {} for diameter {} with P2P1 FEM'.format(i+1, len(diameters), diameter))

        p2p1[i] = {}
        for j in xrange(runs):
            print('Run {} of {}'.format(j+1, runs))
            errors, data = error_for_diameter(problem, analytical_solution, diameter, 2, products, lift_pressure)
            if 'time' in p2p1.keys():
                p2p1[i]['time'].append(data['time'])
            else:
                p2p1[i]['time'] = [data['time']]
            p2p1[i]['num_dofs'] = data['num_dofs']
            p2p1[i]['error'] = errors[product]['rel']


    # P1P1 results
    for i, dia in enumerate(diameters):
        print('P1P1 results for diamater {}:'.format(dia))
        print(' num_dofs: {}'.format(p1p1[i]['num_dofs']))
        print(' error: {}'.format(p1p1[i]['error']))
        print(' time: {}'.format(np.mean(p1p1[i]['time'])))

    # P2P1 results
    for i, dia in enumerate(diameters):
        print('P2P1 results for diamater {}:'.format(dia))
        print(' num_dofs: {}'.format(p2p1[i]['num_dofs']))
        print(' error: {}'.format(p2p1[i]['error']))
        print(' time: {}'.format(np.mean(p2p1[i]['time'])))

    # END
    z = 0


if __name__ == '__main__':
    main()
