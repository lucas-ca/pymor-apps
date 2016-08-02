from __future__ import absolute_import, division, print_function

from stokes.operators.block import StokesLhsBlockOperator, StokesRhsBlockOperator, DiagonalBlockOperator
from stokes.operators.cg import TransposedOperator
from stokes.reductors.basic import GenericStokesRBReconstructor


def reduce_generic_rb_stokes(discretization, velocity_rb, pressure_rb, vector_product=None, disable_caching=True,
                             extends=None):
    """
    Replaces each |Operator| of the given |Discretization| with the projection
    onto the span of the given reduced basis. Just for Stokes problems.

    Parameters
    ----------
    discretization
        The discretization the snapshots were calculated with.
    velocity_rb
        The reduced basis for velocity.
    pressure_rb
        The reduced basis for pressure.
    vector_product
    disable_caching
     Whether caching is disabled or not.
    extends

    Returns
    -------
    rd
        The reduced discretization.
    rc
        The reconstructor for reduced solutions.

    """
    assert extends is None or len(extends) == 3

    assert velocity_rb is not None
    assert pressure_rb is not None

    lhs_operator = discretization.operators['operator']
    rhs_functional = discretization.functionals['rhs']

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
    projected_b = b.projected(range_basis=velocity_rb, source_basis=pressure_rb, product=None)
    projected_bt = TransposedOperator(
        bt.operator.projected(range_basis=velocity_rb, source_basis=pressure_rb, product=None))
    projected_c = c.projected(range_basis=pressure_rb, source_basis=pressure_rb, product=None)

    # projected lhs
    projected_lhs = StokesLhsBlockOperator([projected_a, projected_b, projected_bt, projected_c])

    # project every functional
    projected_f1 = f1.projected(range_basis=None, source_basis=velocity_rb, product=None)
    projected_f2 = f2.projected(range_basis=None, source_basis=pressure_rb, product=None)

    # projected functional
    projected_rhs = StokesRhsBlockOperator([projected_f1, projected_f2])

    cache_region = None if disable_caching else discretization.cache_region

    # reduced discretization
    rd = discretization.with_(operator=projected_lhs,
                              rhs=projected_rhs,
                              vector_operators=None,
                              products=None,
                              visualizer=None,
                              estimator=None,
                              cache_region=cache_region,
                              name=discretization.name + '_reduced')

    # reconstructor
    rc = GenericStokesRBReconstructor(velocity_rb=velocity_rb,
                                      pressure_rb=pressure_rb)

    return rd, rc, {}
