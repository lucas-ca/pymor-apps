# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from functools import partial

from stokes.analyticalproblems.stokes import StokesProblem
from pymor.functions.basic import ConstantFunction
from pymor.parameters.functionals import GenericParameterFunctional


class AffineTransformedStokes(StokesProblem):

    def __init__(self, problem, affine_transformation):
        assert isinstance(problem, StokesProblem)
        
        # rhs
        assert problem.rhs_functions is None
        assert problem.rhs_functionals is None
        
        # diffusion
        assert len(problem.diffusion_functions) == 1
        assert problem.diffusion_functionals is None
        
        # advection
        assert problem.advection_functions is None
        assert problem.advection_functionals is None
        
        # dirichlet data
        assert problem.dirichlet_data_functions is None
        assert problem.dirichlet_data_functionals is None
        
        # diffusion
        def evaluate_diffusion_functional(mu, entry):

            A = mu['transformation']

            det = np.linalg.det(A)
            inv = np.linalg.inv(A)

            assert not det == 0

            res = inv.dot(inv.T) * det

            return res[entry]

        # advection
        def evaluate_advection_functional(mu, entry):

            A = mu['transformation']

            det = np.linalg.det(A)
            inv = np.linalg.inv(A)

            assert not det == 0

            res = inv.T * det

            return res[entry]

        # rhs
        def evaluate_rhs_functional(mu, entry):

            A = mu['transformation']

            det = np.linalg.det(A)
            
            assert not det == 0

            res = 1.0/det * A

            return res[entry]

        # dirichlet data
        def evaluate_dirichlet_data_functional(mu, entry):

            A = mu['transformation']

            det = np.linalg.det(A)

            assert not det == 0

            res = 1.0/det * A

            return res[entry]

        functions = [ConstantFunction(np.array([[1.0, 0.0], [0.0, 0.0]]), dim_domain=2),
                     ConstantFunction(np.array([[0.0, 1.0], [0.0, 0.0]]), dim_domain=2),
                     ConstantFunction(np.array([[0.0, 0.0], [1.0, 0.0]]), dim_domain=2),
                     ConstantFunction(np.array([[0.0, 0.0], [0.0, 1.0]]), dim_domain=2)]

        self.diffusion_functions = functions
        self.advection_functions = functions
        self.rhs_transformation_functions = functions
        self.dirichlet_data_transformation_functions = functions

        # diffusion
        self.diffusion_functionals = [GenericParameterFunctional(mapping=partial(evaluate_diffusion_functional,
                                                                                 entry=(0, 0)),
                                                                 parameter_type={'transformation': (2, 2)},
                                                                 name='Transformation'),
                                      GenericParameterFunctional(mapping=partial(evaluate_diffusion_functional,
                                                                                 entry=(0, 1)),
                                                                 parameter_type={'transformation': (2, 2)},
                                                                 name='Transformation'),
                                      GenericParameterFunctional(mapping=partial(evaluate_diffusion_functional,
                                                                                 entry=(1, 0)),
                                                                 parameter_type={'transformation': (2, 2)},
                                                                 name='Transformation'),
                                      GenericParameterFunctional(mapping=partial(evaluate_diffusion_functional,
                                                                                 entry=(1, 1)),
                                                                 parameter_type={'transformation': (2, 2)},
                                                                 name='Transformation')]

        # advection
        self.advection_functionals = [GenericParameterFunctional(mapping=partial(evaluate_advection_functional,
                                                                                 entry=(0, 0)),
                                                                 parameter_type={'transformation': (2, 2)},
                                                                 name='Transformation'),
                                      GenericParameterFunctional(mapping=partial(evaluate_advection_functional,
                                                                                 entry=(0, 1)),
                                                                 parameter_type={'transformation': (2, 2)},
                                                                 name='Transformation'),
                                      GenericParameterFunctional(mapping=partial(evaluate_advection_functional,
                                                                                 entry=(1, 0)),
                                                                 parameter_type={'transformation': (2, 2)},
                                                                 name='Transformation'),
                                      GenericParameterFunctional(mapping=partial(evaluate_advection_functional,
                                                                                 entry=(1, 1)),
                                                                 parameter_type={'transformation': (2, 2)},
                                                                 name='Transformation')]

        # rhs
        self.rhs_functionals = [GenericParameterFunctional(mapping=partial(evaluate_rhs_functional, entry=(0, 0)),
                                                           parameter_type={'transformation': (2, 2)},
                                                           name='Transformation'),
                                GenericParameterFunctional(mapping=partial(evaluate_rhs_functional, entry=(0, 1)),
                                                           parameter_type={'transformation': (2, 2)},
                                                           name='Transformation'),
                                GenericParameterFunctional(mapping=partial(evaluate_rhs_functional, entry=(1, 0)),
                                                           parameter_type={'transformation': (2, 2)},
                                                           name='Transformation'),
                                GenericParameterFunctional(mapping=partial(evaluate_rhs_functional, entry=(1, 1)),
                                                           parameter_type={'transformation': (2, 2)},
                                                           name='Transformation')]

        # dirichlet data
        self.dirichlet_data_functionals = [GenericParameterFunctional(mapping=
                                                                      partial(evaluate_dirichlet_data_functional,
                                                                              entry=(0, 0)),
                                                                      parameter_type={'transformation': (2, 2)},
                                                                      name='Transformation'),
                                           GenericParameterFunctional(mapping=
                                                                      partial(evaluate_dirichlet_data_functional,
                                                                              entry=(0, 1)),
                                                                      parameter_type={'transformation': (2, 2)},
                                                                      name='Transformation'),
                                           GenericParameterFunctional(mapping=
                                                                      partial(evaluate_dirichlet_data_functional,
                                                                              entry=(1, 0)),
                                                                      parameter_type={'transformation': (2, 2)},
                                                                      name='Transformation'),
                                           GenericParameterFunctional(mapping=
                                                                      partial(evaluate_dirichlet_data_functional,
                                                                              entry=(1, 1)),
                                                                      parameter_type={'transformation': (2, 2)},
                                                                      name='Transformation')]
        self.domain = problem.domain
        self.rhs = problem.rhs
        self.dirichlet_data = problem.dirichlet_data
        self.neumann_data = problem.neumann_data
        self.robin_data = problem.robin_data
        self.parameter_space = problem.parameter_space
