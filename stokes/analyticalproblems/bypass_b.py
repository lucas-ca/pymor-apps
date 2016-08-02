from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction, GenericFunction

from pymor.domaindescriptions.boundarytypes import BoundaryType

from stokes.analyticalproblems.stokes import StokesProblem
from stokes.domaindescriptions.polygonal import PolygonalDomain


class BypassBProblem(StokesProblem):
    """
    Bypass configuration with inhomogeneous dirichlet boundary and force term 0.

    Parameters
    ----------
    viscosity
        Kinematic viscosity.
    """

    def __init__(self, viscosity=1):

        # do nothing boundary type must be registered
        BoundaryType.register_type('do_nothing')

        # domain
        domain = PolygonalDomain(points=[[0, 0], [1, 0], [1, 1], [0, 1], [0, 2],
                                         [-1, 2], [-1, 1], [-1, 0], [-1, -1], [0, -1]],
                                 boundary_types={BoundaryType('dirichlet'): [0, 1, 2, 3, 4, 5, 6, 7, 9],
                                                 BoundaryType('do_nothing'): [8]},
                                 holes=[],
                                 inner_edges=[[0, 3], [3, 6], [0, 7]],
                                 subdomains=[[0, 1, 2, 3],
                                             [6, 3, 4, 5],
                                             [7, 0, 3, 6],
                                             [8, 9, 0, 7]])

        # force
        rhs = ConstantFunction(value=np.array((0., 0.)), dim_domain=2, name='force')

        # dirichlet data
        dir_dat = lambda X: np.array([-4.0 * X[..., 1] * (1.0 - X[..., 1]),
                                      np.zeros_like(X[..., 0])]).T *\
            np.isclose(X[..., 0], 1.)[..., np.newaxis]

        dirichlet_data = GenericFunction(mapping=dir_dat, dim_domain=2, shape_range=(2,), name='dirichlet_data')

        # constant 1 diffusion
        diffusion_functions = (ConstantFunction(dim_domain=2),)

        super(BypassBProblem, self).__init__(domain=domain, rhs=rhs, diffusion_functions=diffusion_functions,
                                             dirichlet_data=dirichlet_data, viscosity=viscosity)
