from __future__ import absolute_import, division, print_function

from pymor.operators.basic import OperatorBase
from pymor.vectorarrays.numpy import NumpyVectorArray

import numpy as np


def absolute_error(reference_solution, solution, product):
    assert isinstance(reference_solution, NumpyVectorArray)
    assert isinstance(solution, NumpyVectorArray)
    assert isinstance(product, OperatorBase)
    assert len(reference_solution) == 1
    assert len(solution) == 1
    assert reference_solution.dim == solution.dim

    e = reference_solution - solution

    return np.sqrt(product.pairwise_apply2(e, e))


def relative_error(reference_solution, solution, product):
    assert isinstance(reference_solution, NumpyVectorArray)
    assert isinstance(solution, NumpyVectorArray)
    assert isinstance(product, OperatorBase)
    assert len(reference_solution) == 1
    assert len(solution) == 1
    assert reference_solution.dim == solution.dim

    e = reference_solution - solution

    return np.sqrt(product.pairwise_apply2(e, e)) / np.sqrt(product.pairwise_apply2(reference_solution,
                                                                                    reference_solution))
