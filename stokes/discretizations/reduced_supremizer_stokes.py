from __future__ import absolute_import, division, print_function

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.discretizations.basic import StationaryDiscretization
from pymor.parameters.base import Parameter
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace

from stokes.operators.block import StokesLhsBlockOperator, StokesRhsBlockOperator
from stokes.operators.cg import TransposedOperator
from stokes.reductors.basic import GenericStokesRBReconstructor


class ReducedSupremizerStokesDiscretization(StationaryDiscretization):
    """
    A discretization so solve a reduced Stokes problem with online supremizer enrichment.
    """

    def __init__(self, discretization, velocity_rb, pressure_rb, orthonormalize=False):
        assert isinstance(discretization, StationaryDiscretization)
        assert isinstance(velocity_rb, NumpyVectorArray)
        assert isinstance(pressure_rb, NumpyVectorArray)

        self.discretization = discretization
        self.velocity_rb = velocity_rb
        self.pressure_rb = pressure_rb
        self.orthonormalize = orthonormalize
        self.reconstructor = GenericStokesRBReconstructor(velocity_rb=velocity_rb, pressure_rb=pressure_rb)

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu)
        mu_id = Parameter({'scale_x': 1, 'scale_y': 1, 'shear': 0})

        mass_supremizer_operator = self.discretization.operators['supremizer_mass']
        advection_supremizer_operator = self.discretization.operators['supremizer_advection']

        supremizer_rb = NumpyVectorSpace(self.velocity_rb.dim).empty()
        for i in range(len(self.pressure_rb)):
            p = NumpyVectorArray(self.pressure_rb.data[i, :])
            supremizer = mass_supremizer_operator.apply_inverse(advection_supremizer_operator.apply(p, mu=mu), mu=mu_id)
            supremizer_rb.append(supremizer)

        if self.orthonormalize:
            supremizer_rb = gram_schmidt(supremizer_rb, self.discretization.products['h1_uv_single'])

        # velocity rb must be copied
        velocity_rb = self.velocity_rb.copy()

        # append supremizer rb to velocity rb
        velocity_rb.append(supremizer_rb)

        # get operator an functional
        lhs_operator = self.discretization.operators['operator']
        rhs_functional = self.discretization.functionals['rhs']

        # get operators
        a = lhs_operator.blocks[0]
        b = lhs_operator.blocks[1]
        bt = lhs_operator.blocks[2]
        c = lhs_operator.blocks[3]

        # get functionals
        f1 = rhs_functional.blocks[0]
        f2 = rhs_functional.blocks[1]

        # project every operator
        projected_a = a.projected(range_basis=velocity_rb, source_basis=velocity_rb, product=None)
        projected_b = b.projected(range_basis=velocity_rb, source_basis=self.pressure_rb, product=None)
        projected_bt = TransposedOperator(
            bt.operator.projected(range_basis=velocity_rb, source_basis=self.pressure_rb, product=None))
        projected_c = c.projected(range_basis=self.pressure_rb, source_basis=self.pressure_rb, product=None)

        # projected operator
        projected_lhs = StokesLhsBlockOperator([projected_a, projected_b, projected_bt, projected_c])

        # project every functional
        projected_f1 = f1.projected(range_basis=None, source_basis=velocity_rb, product=None)
        projected_f2 = f2.projected(range_basis=None, source_basis=self.pressure_rb, product=None)

        # projected functional
        projected_rhs = StokesRhsBlockOperator([projected_f1, projected_f2])

        cache_region = self.discretization.cache_region

        # reduced online supremizer discretization
        rd = self.discretization.with_(operator=projected_lhs,
                                       rhs=projected_rhs,
                                       vector_operators=None,
                                       products=None,
                                       visualizer=None,
                                       estimator=None,
                                       cache_region=cache_region,
                                       name=self.discretization.name + '_reduced_with_online_supremizers')

        # reconstructor
        rc = GenericStokesRBReconstructor(velocity_rb=velocity_rb, pressure_rb=self.pressure_rb)

        return rd.solve(mu), rc
