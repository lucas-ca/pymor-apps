from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction, GenericFunction

from stokes.analyticalproblems.stokes import StokesProblem


class CavityProblem(StokesProblem):
    """
    The Cavity Problem in unit square.
    Parameters
    ----------
    viscosity
        Kinematic viscosity.
    """

    def __init__(self, viscosity=1):

        domain = RectDomain(domain=([0, 0], [1, 1]))

        f = lambda X: np.dstack(
            [
                4.0 * (6 * X[..., 0]**2 - 6 * X[..., 0] + 1) *
                (X[..., 1] - np.ones_like(X[..., 1])) * X[..., 1] * (2 * X[..., 1] - np.zeros_like(X[..., 1])) +
                12.0 * (np.ones_like(X[..., 0]) - X[..., 0])**2 * X[..., 0]**2 * (2.0 * X[..., 1] - np.ones_like(X[..., 1])) +
                (np.ones_like(X[..., 0]) - 2.0 * X[..., 0]) * (np.ones_like(X[..., 1]) - X[..., 1])
            ,
                -4.0 * (6 * X[..., 1]**2 - 6 * X[..., 1] + 1) *
                (X[..., 0] - np.ones_like(X[..., 0])) * X[..., 0] * (2 * X[..., 0] - np.zeros_like(X[..., 0])) -
                12.0 * (np.ones_like(X[..., 1]) - X[..., 1])**2 * X[..., 1]**2 * (2.0 * X[..., 0] - np.ones_like(X[..., 0])) -
                X[..., 0] * (np.ones_like(X[..., 0]) - X[..., 0])
            ])

        rhs = GenericFunction(mapping=f, dim_domain=2, shape_range=(2,), name='force')
        diffusion_functions = (ConstantFunction(value=viscosity, dim_domain=2),)
        dirichlet_data = ConstantFunction(value=np.array([0., 0.]), dim_domain=2, name='dirichlet_data')

        self.viscosity = viscosity

        super(CavityProblem, self).__init__(domain=domain, rhs=rhs, diffusion_functions=diffusion_functions,
                                            dirichlet_data=dirichlet_data)
