from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction, GenericFunction


class CavityProblem(EllipticProblem):
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
                24.*X[...,0]**4*X[...,1] -\
                12.*X[...,0]**4 -\
                48.*X[...,0]**3*X[...,1] +\
                24.*X[...,0]**3 +\
                48.*X[...,0]**2*X[...,1]**3 -\
                72.*X[...,0]**2*X[...,1]**2 +\
                48.*X[...,0]**2*X[...,1] -\
                12.*X[...,0]**2 -\
                48.*X[...,0]*X[...,1]**3 +\
                72.*X[...,0]*X[...,1]**2 -\
                22.*X[...,0]*X[...,1] -\
                2.*X[...,0] +\
                8.*X[...,1]**3 -\
                12.*X[...,1]**2 +\
                3.*X[...,1] +\
                1.
            ,
                -48.*X[...,0]**3*X[...,1]**2 +\
                48.*X[...,0]**3*X[...,1] -\
                8.*X[...,0]**3 +\
                72.*X[...,0]**2*X[...,1]**2 -\
                72.*X[...,0]**2*X[...,1] +\
                13.*X[...,0]**2 -\
                24.*X[...,0]*X[...,1]**4 +\
                48.*X[...,0]*X[...,1]**3 -\
                48.*X[...,0]*X[...,1]**2 +\
                24.*X[...,0]*X[...,1] -\
                5.*X[...,0] +\
                12.*X[...,1]**4 -\
                24.*X[...,1]**3 +\
                12.*X[...,1]**2
            ])


        #rhs = ConstantFunction(value=np.array((0., 0.)), dim_domain=2, name='force')
        rhs = GenericFunction(mapping=f, dim_domain=2, shape_range=(2,), name='force')
        diffusion_functions = (ConstantFunction(value=viscosity, dim_domain=2),)
        #dir_dat = lambda X: np.array([-4./(height**2) * X[..., 1] ** 2 + 4./height * X[..., 1],
        #                                       np.zeros_like(X[..., 0])]).T *\
        #                             np.isclose(X[..., 0], 0.)[..., np.newaxis]
        #dir_dat = lambda X: np.array([np.ones_like(X[..., 0]), np.zeros_like(X[..., 0])]).T *\
        #                             np.isclose(X[..., 1], 1.)[..., np.newaxis]

        #dirichlet_data = GenericFunction(mapping=dir_dat, dim_domain=2, shape_range=(2,), name='dirichlet_data')
        dirichlet_data = ConstantFunction(value=np.array([0., 0.]), dim_domain=2, name='dirichlet_data')

        self.viscosity = viscosity

        super(CavityProblem, self).__init__(domain=domain, rhs=rhs, diffusion_functions=diffusion_functions,
                                                dirichlet_data=dirichlet_data)