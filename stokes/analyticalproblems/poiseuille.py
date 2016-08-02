from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction, GenericFunction

from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.parameters.spaces import CubicParameterSpace

from stokes.analyticalproblems.stokes import StokesProblem


class PoiseuilleProblem(StokesProblem):
    """
    The Poiseuille Problem in a rectangle.

    Parameters
    ----------
    width
        Width of domain.
    height
        Height of domain.
    viscosity
        Kinematic viscosity.
    """

    def __init__(self, width=1, height=1, viscosity=1):

        # do nothing boundary type must be registered
        BoundaryType.register_type('do_nothing')

        # domain
        domain = RectDomain(domain=([0, 0], [width, height]), right=BoundaryType('do_nothing'))

        # force
        rhs = ConstantFunction(value=np.array((0., 0.)), dim_domain=2, name='force')

        # dirichlet data
        dir_dat = lambda X: np.array([-4./(height**2) * X[..., 1] ** 2 + 4./height * X[..., 1],
                                      np.zeros_like(X[..., 0])]).T *\
            np.isclose(X[..., 0], 0.)[..., np.newaxis]

        dirichlet_data = GenericFunction(mapping=dir_dat, dim_domain=2, shape_range=(2,), name='dirichlet_data')

        # constant 1 diffusion
        diffusion_functions = (ConstantFunction(dim_domain=2),)

        parameter_space = None

        super(PoiseuilleProblem, self).__init__(domain=domain, rhs=rhs, diffusion_functions=diffusion_functions,
                                                dirichlet_data=dirichlet_data, viscosity=viscosity,
                                                parameter_space=parameter_space)
