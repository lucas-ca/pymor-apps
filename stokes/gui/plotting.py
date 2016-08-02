from __future__ import absolute_import, division, print_function

from matplotlib import pyplot as plt
import numpy as np


def plot_pyplot(solution, grid, fem_order, velocity='quiver'):
    assert isinstance(velocity, str)
    assert velocity in ('quiver', 'absolute')
    n_p1 = grid.size(2)
    n_p2 = grid.size(2) + grid.size(1)

    if fem_order == 1:
        u = solution.data[0, 0:n_p1]
        v = solution.data[0, n_p1:2*n_p1]
        p = solution.data[0, 2*n_p1:]
    elif fem_order == 2:
        u = solution.data[0, 0:n_p1]
        v = solution.data[0, n_p2:n_p2+n_p1]
        p = solution.data[0, 2*n_p2:]
    else:
        raise ValueError

    # lift p
    p -= p.min()

    X = grid.centers(2)

    x = X[..., 0]
    y = X[..., 1]

    # plot p
    plt.figure('pressure')
    plt.tripcolor(x, y, grid.subentities(0, 2), p)
    plt.colorbar()

    # plot u
    if velocity == 'quiver':
        plt.figure('velocity')
        plt.quiver(x, y, u, v)
    elif velocity == 'absolute':
        plt.figure('absolute velocity')
        plt.tripcolor(x, y, grid.subentities(0, 2), np.sqrt(u**2 + v**2))
        plt.colorbar()


def plot_transformed_pyplot(solution, grid, transformation, parameter, fem_order, velocity='quiver'):

    assert isinstance(velocity, str)
    assert velocity in ('quiver', 'absolute')
    n_p1 = grid.size(2)
    n_p2 = grid.size(2) + grid.size(1)

    if fem_order == 1:
        u = solution.data[0, 0:n_p1]
        v = solution.data[0, n_p1:2*n_p1]
        p = solution.data[0, 2*n_p1:]
    elif fem_order == 2:
        u = solution.data[0, 0:n_p1]
        v = solution.data[0, n_p2:n_p2+n_p1]
        p = solution.data[0, 2*n_p2:]
    else:
        raise ValueError

    # lift p
    p -= p.min()

    X = transformation.evaluate(grid.centers(2), mu=parameter)

    x = X[..., 0]
    y = X[..., 1]

    # plot p
    plt.figure('pressure for mu = {}'.format(parameter))
    plt.tripcolor(x, y, grid.subentities(0, 2), p)
    plt.colorbar()

    # plot u
    if velocity == 'quiver':
        plt.figure('velocity for mu = {}'.format(parameter))
        plt.quiver(x, y, u, v)
    elif velocity == 'absolute':
        plt.figure('absolute velocity for mu = {}'.format(parameter))
        plt.tripcolor(x, y, grid.subentities(0, 2), np.sqrt(u**2 + v**2))
        plt.colorbar()


def plot_multiple_transformed_pyplot(solutions, grid, transformation, parameter, fem_order, velocity='quiver',
                                     rescale_colorbars=True):

    assert isinstance(velocity, str)
    assert velocity in ('quiver', 'absolute', 'quiver_absolute')
    n_p1 = grid.size(2)
    n_p2 = grid.size(2) + grid.size(1)

    if fem_order == 1:
        us = [solution.data[0, 0:n_p1] for solution in solutions]
        vs = [solution.data[0, n_p1:2*n_p1] for solution in solutions]
        ps = [solution.data[0, 2*n_p1:] for solution in solutions]
    elif fem_order == 2:
        us = [solution.data[0, 0:n_p1] for solution in solutions]
        vs = [solution.data[0, n_p2:n_p2+n_p1] for solution in solutions]
        ps = [solution.data[0, 2*n_p2:] for solution in solutions]
    else:
        raise ValueError

    # lift p
    for p in ps:
        p -= p.min()

    if rescale_colorbars:
        p_min = min([p.min() for p in ps])
        p_max = max([p.max() for p in ps])
        if velocity == 'absolute':
            u_min = min([np.sqrt(u**2 + v**2).min() for u, v in zip(us, vs)])
            u_max = max([np.sqrt(u**2 + v**2).max() for u, v in zip(us, vs)])

    X = transformation.evaluate(grid.centers(2), mu=parameter)

    x = X[..., 0]
    y = X[..., 1]

    for i in xrange(len(solutions)):
        u = us[i]
        v = vs[i]
        p = ps[i]
        # plot p
        plt.figure('pressure {} for mu = {}'.format(i, parameter))
        plt.tripcolor(x, y, grid.subentities(0, 2), p)
        if rescale_colorbars:
            plt.clim(p_min, p_max)
            plt.colorbar()
        else:
            plt.colorbar()

        # plot u
        if velocity == 'quiver':
            plt.figure('velocity {} for mu = {}'.format(i, parameter))
            plt.quiver(x, y, u, v)
        elif velocity == 'absolute':
            plt.figure('absolute velocity {} for mu = {}'.format(i, parameter))
            plt.tripcolor(x, y, grid.subentities(0, 2), np.sqrt(u**2 + v**2))
            if rescale_colorbars:
                plt.clim(u_min, u_max)
                plt.colorbar()
            else:
                plt.colorbar()
        elif velocity == 'quiver_absolute':
            plt.figure('velocity quiver and absolute {} for mu = {}'.format(i, parameter))
            plt.tripcolor(x, y, grid.subentities(0, 2), np.sqrt(u**2 + v**2))
            if rescale_colorbars:
                plt.clim(u_min, u_max)
                plt.colorbar()
            else:
                plt.colorbar()
            plt.quiver(x, y, u, v)
