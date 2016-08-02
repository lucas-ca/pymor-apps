from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction, GenericFunction

from pymor.domaindescriptions.boundarytypes import BoundaryType

from stokes.analyticalproblems.stokes import StokesProblem
from stokes.domaindescriptions.polygonal import PolygonalDomain


class BypassAProblem(StokesProblem):
    """
    Bypass configuration with dirichlet 0 boundary and force term.

    Parameters
    ----------
    viscosity
        Kinematic viscosity.
    """

    def __init__(self, viscosity=1):

        # domain
        domain = PolygonalDomain(points=[[0, 0], [1, 0], [1, 1], [0, 1], [0, 2],
                                         [-1, 2], [-1, 1], [-1, 0], [-1, -1], [0, -1]],
                                 boundary_types={BoundaryType('dirichlet'): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
                                 holes=[],
                                 inner_edges=[[0, 3], [3, 6], [0, 7]],
                                 subdomains=[[0, 1, 2, 3],
                                             [6, 3, 4, 5],
                                             [7, 0, 3, 6],
                                             [8, 9, 0, 7]])

        # force
        def r(X):
            x = X[..., 0]
            y = X[..., 1]
            zero = np.zeros_like(X[..., 0])

            res = np.dstack([
                zero
            ,
                -10.0*x
            ])

            return res

        rhs = GenericFunction(mapping=r, dim_domain=2, shape_range=(2,), name='force')

        # constant one diffusion
        diffusion_functions = (ConstantFunction(dim_domain=2),)

        # dirichlet data
        dirichlet_data = ConstantFunction(value=np.array((0., 0.)), dim_domain=2, name='dirichlet_data')

        super(BypassAProblem, self).__init__(domain=domain, rhs=rhs, diffusion_functions=diffusion_functions,
                                             dirichlet_data=dirichlet_data, viscosity=viscosity)
