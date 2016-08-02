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

    fontsize = 16

    if fem_order == 1:
        diameters = [1./10., 1./20., 1./30., 1./40., 1.0/50.0, 1.0/60.0, 1.0/70.0, 1.0/80.0]#, 1.0/90.0, 1.0/100.0, 1./110., 1./120.]
    elif fem_order == 2:
        diameters = [1./10., 1./20., 1./30., 1./40., 1./50., 1.0/60.0, 1.0/70.0, 1.0/80.0]
    else:
        raise ValueError

    products = ['h1_uv', 'l2_uv', 'h1_p', 'l2_p']

    problem = setup_problem(viscosity)
    analytical_solution = setup_analytical_solution()

    absolute_errors = {}
    relative_errors = {}

    for i, diameter in enumerate(diameters):
        print('Calculating error {} of {} for diameter {} with P{}P1 FEM'.format(i+1, len(diameters), diameter,
                                                                                 fem_order))

        errors, _ = error_for_diameter(problem, analytical_solution, diameter, fem_order, products, lift_pressure)

        for k in products:
            if k in absolute_errors.keys():
                absolute_errors[k].append(errors[k]['abs'])
            else:
                absolute_errors[k] = [errors[k]['abs']]
            if k in relative_errors.keys():
                relative_errors[k].append(errors[k]['rel'])
            else:
                relative_errors[k] = [errors[k]['rel']]

    inverse_diameters = [1.0/d for d in diameters]

    # h1 u
    plt.figure('relative_error_cavity_p1p1_h1_u')
    plt.loglog(inverse_diameters, relative_errors['h1_uv'], label='error in u')
    if fem_order == 1:
        plt.loglog(inverse_diameters, [d**1*np.max(relative_errors['h1_uv'])*1.5/diameters[0]**1
                                       for d in diameters], linestyle='--', color='black', label='order 1')
    elif fem_order == 2:
        plt.loglog(inverse_diameters, [d**2*np.max(relative_errors['h1_uv'])*1.5/diameters[0]**2
                                       for d in diameters], linestyle='--', color='black', label='order 2')
    else:
        raise ValueError
    plt.legend()
    plt.xlabel('$1/h$', fontsize=fontsize)
    plt.ylabel('$||e_u||_{H^1}$', fontsize=fontsize)

    # l2 u
    plt.figure('relative_error_cavity_p1p1_l2_u')
    plt.loglog(inverse_diameters, relative_errors['l2_uv'], label='error in u')
    if fem_order == 1:
        plt.loglog(inverse_diameters, [d**2*np.max(relative_errors['l2_uv'])*1.5/diameters[0]**2 for d in diameters],
                   linestyle='--', color='black', label='order 2')
    elif fem_order == 2:
        plt.loglog(inverse_diameters, [d**3*np.max(relative_errors['l2_uv'])*1.5/diameters[0]**3 for d in diameters],
                   linestyle='--', color='black', label='order 3')
    else:
        raise ValueError
    plt.legend()
    plt.xlabel('$1/h$', fontsize=fontsize)
    plt.ylabel('$||e_u||_{L^2}$', fontsize=fontsize)

    # h1 p
    plt.figure('relative_error_cavity_p1p1_h1_p')
    plt.loglog(inverse_diameters, relative_errors['h1_p'], label='error in p')
    if fem_order == 1:
        plt.loglog(inverse_diameters, [d**0.5*np.max(relative_errors['h1_p'])*1.5/diameters[0]**0.5 for d in diameters],
                   linestyle='--', color='black', label='order 0.5')
    elif fem_order == 2:
        plt.loglog(inverse_diameters, [d**1*np.max(relative_errors['h1_p'])*1.5/diameters[0]**1 for d in diameters],
                   linestyle='--', color='black', label='order 1')
    else:
        raise ValueError
    plt.legend()
    plt.xlabel('$1/h$', fontsize=fontsize)
    plt.ylabel('$||e_p||_{H^1}$', fontsize=fontsize)

    # l2 p
    plt.figure('relative_error_cavity_p1p1_l2_p')
    plt.loglog(inverse_diameters, relative_errors['l2_p'], label='error in p')
    if fem_order == 1:
        plt.loglog(inverse_diameters, [d**1*np.max(relative_errors['l2_p'])*1.5/diameters[0]**1 for d in diameters],
                   linestyle='--', color='black', label='order 1')
    elif fem_order == 2:
        plt.loglog(inverse_diameters, [d**2*np.max(relative_errors['l2_p'])*1.5/diameters[0]**2 for d in diameters],
                   linestyle='--', color='black', label='order 2')
    else:
        raise ValueError
    plt.legend()
    plt.xlabel('$1/h$', fontsize=fontsize)
    plt.ylabel('$||e_p||_{L^2}$', fontsize=fontsize)
    # END
    z = 0


if __name__ == '__main__':
    main()
