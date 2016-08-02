from __future__ import absolute_import, division, print_function

from stokes.analyticalproblems.stokes import StokesProblem
from stokes.functions.affine_transformation import AffineTransformation
from stokes.functions.piecewise_affine_transformation import PiecewiseAffineTransformation


class AffineTransformedStokes(StokesProblem):

    def __init__(self, problem, transformation, name=None):

        # problem must be a stokes problem
        assert isinstance(problem, StokesProblem)

        # transformation must be a AffineTransformation
        assert isinstance(transformation, AffineTransformation) or\
            isinstance(transformation, PiecewiseAffineTransformation)

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
        diffusion_functions = transformation.diffusion_functions()
        diffusion_functionals = transformation.diffusion_functionals()

        # advection
        advection_functions = transformation.advection_functions()
        advection_functionals = transformation.advection_functionals()

        # rhs
        rhs_functions = transformation.rhs_functions()
        rhs_functionals = transformation.rhs_functionals()

        # dirichlet_data
        dirichlet_data_functions = transformation.dirichlet_data_functions()
        dirichlet_data_functionals = transformation.dirichlet_data_functionals()

        # mass
        #mass_functions = transformation.mass_functions()
        #mass_functionals = transformation.mass_functionals()

        # domain = problem.domain
        # rhs = problem.rhs
        # dirichlet_data = problem.dirichlet_data
        # neumann_data = problem.neumann_data
        # robin_data = problem.robin_data

        super(AffineTransformedStokes, self).__init__(domain=problem.domain,
                                                      rhs=problem.rhs,
                                                      rhs_functions=rhs_functions,
                                                      rhs_functionals=rhs_functionals,
                                                      diffusion_functions=diffusion_functions,
                                                      diffusion_functionals=diffusion_functionals,
                                                      advection_functions=advection_functions,
                                                      advection_functionals=advection_functionals,
                                                      dirichlet_data=problem.dirichlet_data,
                                                      dirichlet_data_functions=dirichlet_data_functions,
                                                      dirichlet_data_functionals=dirichlet_data_functionals,
                                                      #mass_functions=mass_functions,
                                                      #mass_functionals=mass_functionals,
                                                      neumann_data=problem.neumann_data,
                                                      robin_data=problem.robin_data,
                                                      viscosity=problem.viscosity,
                                                      #parameter_space=problem.parameter_space,
                                                      parameter_space=transformation.parameter_space,
                                                      name=name)
        self.transformation = transformation
