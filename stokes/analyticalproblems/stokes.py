# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction


class StokesProblem(EllipticProblem):
    """ Incompressible stationary stokes equation.

    The problem consists in solving ::

    |  -ν Δ u(x, μ) + ∇ p(x, μ) = f(x, μ) in Ω
    |               ∇ ⋅ u(x, μ) = 0       in Ω
    |                   u(x, μ) = u_D     on Γ_D

    for (u, p).

    """

    def __init__(self,
                 domain=RectDomain(),
                 rhs=ConstantFunction(value=np.array([[0.0], [0.0]]), dim_domain=2),
                 rhs_functions=None,
                 rhs_functionals=None,
                 diffusion_functions=(ConstantFunction(dim_domain=2),),
                 diffusion_functionals=None,
                 advection_functions=None,
                 advection_functionals=None,
                 dirichlet_data=ConstantFunction(value=np.array([[0.0], [0.0]]), dim_domain=2),
                 dirichlet_data_functions=None,
                 dirichlet_data_functionals=None,
                 neumann_data=None,
                 robin_data=None,
                 viscosity=1.0,
                 parameter_space=None,
                 name='StokesProblem'):

        if neumann_data is not None:
            raise NotImplementedError
        if robin_data is not None:
            raise NotImplementedError

        super(StokesProblem, self).__init__(domain=domain,
                                            rhs=rhs,
                                            diffusion_functions=diffusion_functions,
                                            diffusion_functionals=diffusion_functionals,
                                            advection_functions=advection_functions,
                                            advection_functionals=advection_functionals,
                                            dirichlet_data=dirichlet_data,
                                            neumann_data=neumann_data,
                                            robin_data=robin_data,
                                            parameter_space=parameter_space,
                                            name=name)
        self.rhs_functions = rhs_functions
        self.rhs_functionals = rhs_functionals
        self.dirichlet_data_functions = dirichlet_data_functions
        self.dirichlet_data_functionals = dirichlet_data_functionals
        self.viscosity = viscosity
