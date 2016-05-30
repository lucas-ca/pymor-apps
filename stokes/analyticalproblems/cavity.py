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

        #f = lambda X: np.dstack(
        #    [
        #        24. * X[..., 0]**4 * X[..., 1] -
        #        12. * X[..., 0]**4 -
        #        48. * X[..., 0]**3 * X[..., 1] +
        #        24. * X[..., 0]**3 +
        #        48. * X[..., 0]**2 * X[..., 1]**3 -
        #        72. * X[..., 0]**2 * X[..., 1]**2 +
        #        48. * X[..., 0]**2 * X[...,  1] -
        #        12. * X[..., 0]**2 -
        #        48. * X[..., 0] * X[..., 1]**3 +
        #        72. * X[..., 0] * X[..., 1]**2 -
        #        22. * X[..., 0] * X[..., 1] -
        #        2. * X[..., 0] +
        #        8. * X[..., 1]**3 -
        #        12. * X[..., 1]**2 +
        #        3. * X[..., 1] +
        #        1.
        #    ,
        #        -48. * X[..., 0]**3 * X[..., 1]**2 +
        #        48. * X[..., 0]**3 * X[..., 1] -
        #        8. * X[..., 0]**3 +
        #        72. * X[..., 0]**2 * X[..., 1]**2 -
        #        72. * X[..., 0]**2 * X[..., 1] +
        #        13. * X[..., 0]**2 -
        #        24. * X[..., 0] * X[..., 1]**4 +
        #        48. * X[..., 0] * X[..., 1]**3 -
        #        48. * X[..., 0] * X[..., 1]**2 +
        #        24. * X[..., 0] * X[..., 1] -
        #        5. * X[..., 0] +
        #        12. * X[..., 1]**4 -
        #        24. * X[..., 1]**3 +
        #        12. * X[..., 1]**2
        #    ])

        f = lambda X: np.dstack(
            [
                4*(
                    X[..., 0]**4 * (6.0*X[..., 1] - 3.0 * np.ones_like(X[..., 0])) +
                    X[..., 0]**3 * (6.0 * np.ones_like(X[..., 0]) - 12.0*X[..., 1]) +
                    3.0 * X[..., 0]**2 * (4.0*X[..., 1]**3 - 6.0*X[..., 1]**2 + 4.0 * X[..., 1] - 1.0 * np.ones_like(X[..., 0])) -
                    6.0 * X[..., 0]*X[..., 1] * (2.0 * X[..., 1]**2 - 3.0 * X[..., 1] + 1.0 * np.ones_like(X[..., 0])) +
                    X[..., 1] * (2.0 * X[..., 1]**2 - 3.0 * X[..., 1] + np.ones_like(X[..., 0]))
                ) + (2.0 * X[..., 0] - 1.0 * np.ones_like(X[..., 0])) * (X[..., 1] - 1.0 * np.ones_like(X[..., 0])) * X[..., 1]
            ,
                -4.0 * (
                    2.0 * X[..., 0]**3 * (6.0 * X[..., 1]**2 - 6.0 * X[..., 1] + 1.0 * np.ones_like(X[..., 0])) -
                    3.0 * X[..., 0]**2 * (6.0 * X[..., 1]**2 - 6.0 * X[..., 1] + 1.0 * np.ones_like(X[..., 0])) +
                    X[..., 0] * (6.0 * X[..., 1]**4 - 12.0 * X[..., 1]**3 + 12.0 * X[..., 1]**2 - 6.0*X[..., 1] + 1.0 * np.ones_like(X[..., 0])) -
                    3.0 * (X[..., 1] - 1.0 * np.ones_like(X[..., 0]))**2 * X[..., 1]**2
                ) + (2.0 * X[..., 1] - 1.0 * np.ones_like(X[..., 1])) * (X[..., 0] - 1.0 * np.ones_like(X[..., 1])) * X[..., 0]
            ])

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
        diffusion_functions = (ConstantFunction(value=viscosity, dim_domain=2),)
        dirichlet_data = ConstantFunction(value=np.array([0., 0.]), dim_domain=2, name='dirichlet_data')

        self.viscosity = viscosity

        super(CavityProblem, self).__init__(domain=domain, rhs=rhs, diffusion_functions=diffusion_functions,
                                            dirichlet_data=dirichlet_data)
