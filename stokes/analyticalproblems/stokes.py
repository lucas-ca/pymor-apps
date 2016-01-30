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

    |  -ν ⋅ Δ u(x, μ) + ∇ p(x,μ) = f(x, μ) in Ω
    |                   ∇ u(x,μ) = 0       in Ω
    |                     u(x,μ) = u_D     on Γ_D

    for (u, p).

    """

    def __init__(self,
                 domain = RectDomain(),
                 rhs = ConstantFunction(value = np.array([[0.0], [0.0]]), dim_domain=2),
                 diffusion_function = ConstantFunction(dim_domain=2),
                 dirichlet_data = ConstantFunction(value = np.array([[0.0], [0.0]]), dim_domain=2),
                 viscosity = 1.0):
        self.domain = domain
        self.rhs = rhs
        self.diffusion_function = diffusion_function
        self.dirichlet_data = dirichlet_data
        self.viscosity = viscosity
