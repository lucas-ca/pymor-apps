# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction


class InstationaryAdvectionDiffusionProblem(ImmutableInterface):

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 flux_function=ConstantFunction(value=np.array([0, 0]), dim_domain=2),
                 flux_function_derivative=ConstantFunction(value=np.array([0, 0]), dim_domain=2),
                 dirichlet_data=ConstantFunction(value=0, dim_domain=2),
                 neumann_data=ConstantFunction(value=0, dim_domain=2),
                 initial_data=ConstantFunction(dim_domain=2), T=1, diffusion=1.0, name=None):
        self.domain = domain
        self.rhs = rhs
        self.flux_function = flux_function
        self.flux_function_derivative = flux_function_derivative
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.initial_data = initial_data
        self.T = T
        self.diffusion = diffusion
        self.name = name