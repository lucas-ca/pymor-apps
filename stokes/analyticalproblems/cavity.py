from __future__ import absolute_import, division, print_function

from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction, GenericFunction

from stokes.analyticalproblems.stokes import StokesProblem

import numpy as np


class CavityProblem(StokesProblem):
    """
    The Cavity Problem in unit square.

    Parameters
    ----------
    viscosity
        Kinematic viscosity.
    """

    def __init__(self, viscosity=1):

        # domain
        domain = RectDomain(domain=([0, 0], [1, 1]))

        # force
        def r(X):
            x = X[..., 0]
            y = X[..., 1]
            one = np.ones_like(X[..., 0])

            res = np.dstack([
                4.0 * (
                    x**4 * (6.0 * y - 3.0*one) +
                    x**3 * (6.0*one - 12.0 * y) +
                    3.0 * x**2 * (4.0 * y**3 - 6.0 * y**2 + 4.0 * y - 1.0 * one) -
                    6.0 * x * y * (2.0 * y**2 - 3.0 * y + 1.0 * one) +
                    y * (2.0*y**2 - 3.0 * y + 1.0 * one)
                ) + (2.0 * x - 1.0 * one) * (y - 1.0 * one) * y
            ,
                -4.0 * (
                    2.0 * x**3 * (6.0 * y**2 - 6.0 * y + 1.0 * one) -
                    3.0 * x**2 * (6.0 * y**2 - 6.0 * y + 1.0 * one) +
                    x * (6.0 * y**4 - 12.0 * y**3 + 12.0 * y**2 - 6.0*y + 1.0 * one) -
                    3.0 * (y - 1.0 * one)**2 * y**2
                ) + (2.0 * y - 1.0 * one) * (x - 1.0 * one) * x
            ])

            return res

        rhs = GenericFunction(mapping=r, dim_domain=2, shape_range=(2,), name='force')

        # dirichlet data
        dirichlet_data = ConstantFunction(value=np.array([0., 0.]), dim_domain=2, name='dirichlet_data')

        # constant 1 diffusion
        diffusion_functions = (ConstantFunction(value=viscosity, dim_domain=2),)

        # kinematic viscosity
        self.viscosity = viscosity

        super(CavityProblem, self).__init__(domain=domain, rhs=rhs, diffusion_functions=diffusion_functions,
                                            dirichlet_data=dirichlet_data)
