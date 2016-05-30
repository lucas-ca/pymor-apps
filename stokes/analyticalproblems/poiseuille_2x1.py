from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction, GenericFunction

from pymor.domaindescriptions.boundarytypes import BoundaryType

from stokes.analyticalproblems.stokes import StokesProblem
from stokes.domaindescriptions.polygonal import PolygonalDomain


class Poiseuille2x1Problem(StokesProblem):
    """
    The Poiseuille Problem in a rectangle.
    Parameters
    ----------
    viscosity
        Kinematic viscosity.
    """

    def __init__(self, viscosity=1):

        BoundaryType.register_type('do_nothing')

        #domain = RectDomain(domain=([0, 0], [width, height]), right=BoundaryType('do_nothing'))

        #domain = PolygonalDomain([[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2], [0, 1]],
        #                 {BoundaryType('dirichlet'): [0, 1, 3, 4, 5, 6, 7],
        #                  BoundaryType('do_nothing'): [2]},
        #                 [],
        #                 [[1, 4], [4, 7]])

        domain = PolygonalDomain(points=[[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1]],
                                 boundary_types={BoundaryType('dirichlet'): [0, 1, 3, 4, 5],
                                                 BoundaryType('do_nothing'): [2]},
                                 holes=[],
                                 inner_edges=[[1, 4]],
                                 subdomains=[[0, 1, 4, 5], [1, 2, 3, 4]])

        rhs = ConstantFunction(value=np.array((0., 0.)), dim_domain=2, name='force')

        diffusion_functions = (ConstantFunction(dim_domain=2),)

        #dir_dat = lambda X: np.array([-4./(height**2) * X[..., 1] ** 2 + 4./height * X[..., 1],
        #                                       np.zeros_like(X[..., 0])]).T *\
        #                             np.isclose(X[..., 0], 0.)[..., np.newaxis]

        dir_dat = lambda X: np.array([-4./(1**2) * X[..., 1] ** 2 + 4./1 * X[..., 1],
                                      np.zeros_like(X[..., 0])]).T * np.isclose(X[..., 0], 0.)[..., np.newaxis]

        dirichlet_data = GenericFunction(mapping=dir_dat, dim_domain=2, shape_range=(2,), name='dirichlet_data')

        super(Poiseuille2x1Problem, self).__init__(domain=domain, rhs=rhs, diffusion_functions=diffusion_functions,
                                                      dirichlet_data=dirichlet_data, viscosity=viscosity)
